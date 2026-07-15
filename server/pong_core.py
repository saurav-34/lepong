"""Env + model + policy for human-vs-JEPA Pong, with no web framework attached.

server/play_pong.py wraps this in a websocket; scripts can drive it headlessly.

The model never reads RAM to *decide*. Each tick it encodes the rendered frame,
reads a predicted state out of state_head(predict_next(...)), and applies the
same align-then-move rule collect_pong.py used to generate the data -- on the
PREDICTED ball_y. RAM is touched only for two things the pixels cannot reliably
express: the JEPA paddle's own y (proprioception) and whether a ball is in play
(serve detection -- a parked ball reads ball_y==0 in RAM while its pixels are
absent/ambiguous, exactly the contaminated case that wrecks the ball_x readout).

Convention (collect_pong.py): JEPA drives `second_0` (LEFT paddle), the human
drives `first_0` (RIGHT paddle, classic seat). Actions are held for `frameskip`
ALE frames so the play stride equals the training stride. Unlike Tennis, the
paddle<->agent mapping does NOT swap across points, so there is no end-aware
bookkeeping here.
"""
from __future__ import annotations

import base64
import io
import logging
import os

import cv2
import numpy as np
import torch
from PIL import Image

from model.jepa_pool import JEPAPool, EMBED_DIM, HISTORY_SIZE, resolve_state_meta

logger = logging.getLogger("pong")

# Pong minimal action ids (calibrated on this ROM in collect_pong.py):
# 2 = up (RAM y decreases), 3 = down. FIRE serves the ball.
NOOP, FIRE, UP, DOWN = 0, 1, 2, 3
AGENT_JEPA, AGENT_HUMAN = "second_0", "first_0"   # JEPA=left paddle, human=right

# RAM addresses (verified in collect_pong.py, no end-swap):
JEPA_Y_ADDR = 50        # second_0 / left paddle top-y  (= state head's enemy_y)
BALL_Y_ADDR = 54        # ball_y; ==0 means no ball in play (serve needed)
PADDLE_HALF = 10        # RAM y is the paddle top; aim its centre at the ball
ALIGN_DEADZONE = 4      # px: within this of the ball's y -> hold position


def rom_dir() -> str:
    import ale_py
    return os.path.join(os.path.dirname(ale_py.__file__), "roms")


def load_frozen_model(checkpoint: str, device: torch.device):
    """Load a pong state-head checkpoint, rebuilding the discrete ActionEncoder.

    Returns (model, state_mean, state_std, state_idx).
    """
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)

    # Accept both conventions (stage-2 state_dim/state_names, stage-1 state_cols).
    state_dim, state_names, state_mean, state_std = resolve_state_meta(ckpt)
    if state_dim <= 0:
        raise RuntimeError(f"needs a state-head checkpoint, got state_dim={state_dim}")
    if state_mean is None or state_std is None:
        raise RuntimeError("checkpoint has no state_mean/state_std; cannot denormalize")

    # num_actions decides which ActionEncoder branch exists: >0 builds the
    # Embedding, 0 builds an MLP that cannot accept this state_dict.
    num_actions = ckpt.get("num_actions")
    if num_actions is None:
        w = ckpt["model"].get("action_encoder.embed.weight")
        if w is None:
            raise RuntimeError(
                "checkpoint has no num_actions and no action_encoder.embed.weight; "
                "it was trained with a continuous ActionEncoder and cannot drive pong")
        num_actions = int(w.shape[0])
        logger.warning("num_actions absent; inferred %d from embed.weight", num_actions)

    if not state_names:
        raise RuntimeError("checkpoint has no state_names/state_cols; cannot map head outputs")
    state_idx = {n: i for i, n in enumerate(state_names)}
    if "ball_y" not in state_idx:
        raise RuntimeError(f"state_names missing 'ball_y' (pong aligns on it): {state_names}")

    model = JEPAPool(embed_dim=ckpt.get("embed_dim", EMBED_DIM),
                     state_dim=state_dim, num_actions=num_actions)
    model.load_state_dict(ckpt["model"])
    for p in model.parameters():
        p.requires_grad = False
    model.eval().to(device)

    return (model,
            state_mean.to(device),
            state_std.to(device),
            state_idx)


def choose_action(state: np.ndarray, state_idx: dict[str, int],
                  own_y: float) -> tuple[int, float]:
    """Align-then-move: the ball_y is PERCEIVED, the own paddle is PROPRIOCEPTED.

    Only ball_y comes from the state head. The own paddle y is read from RAM for
    the same reason server/infer.py and tennis_core.py do it: it is our own
    actuator state, not something we must see, and it does not survive the
    feedback loop (both paddles chased identically during collection, so a probe
    reads our paddle partly *through* the ball; once our policy diverges from the
    collector's that correlation breaks).

    ball_y is the reliable readout here (corr ~0.68 in-val); ball_x is the
    contaminated one and is deliberately not used to steer.
    """
    ball_y = float(state[state_idx["ball_y"]])
    dy = ball_y - (own_y + PADDLE_HALF)     # aim the paddle centre at the ball
    if abs(dy) <= ALIGN_DEADZONE:
        return NOOP, dy
    return (DOWN if dy > 0 else UP), dy


def png_b64(rgb: np.ndarray, scale: int = 1) -> str:
    if scale != 1:
        rgb = cv2.resize(rgb, (rgb.shape[1] * scale, rgb.shape[0] * scale),
                         interpolation=cv2.INTER_NEAREST)
    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


class PongSession:
    """One pong_v3 match driven by the frozen world model."""

    def __init__(self, checkpoint: str, device: torch.device, frameskip: int = 4,
                 img_size: int = 128, seed: int = 0):
        from pettingzoo.atari import pong_v3

        self.device = device
        self.frameskip = frameskip
        self.img_size = img_size
        self.model, self.state_mean, self.state_std, self.state_idx = \
            load_frozen_model(checkpoint, device)

        self.env = pong_v3.parallel_env(render_mode="rgb_array",
                                        auto_rom_install_path=rom_dir())
        self.env.reset(seed=seed)
        self.ale = self.env.unwrapped.ale

        self.hist_emb: list[torch.Tensor] = []
        self.hist_act: list[int] = []
        self.prev_action = NOOP
        self.score = {AGENT_JEPA: 0.0, AGENT_HUMAN: 0.0}
        self.tick = 0

    def _reset_point(self):
        self.env.reset()
        self.hist_emb.clear()
        self.hist_act.clear()
        self.prev_action = NOOP

    def _render(self):
        raw = self.env.render()                                   # (210,160,3) uint8 RGB
        n = self.img_size
        interp = cv2.INTER_AREA if n * n < raw.shape[0] * raw.shape[1] else cv2.INTER_LINEAR
        return cv2.resize(raw, (n, n), interpolation=interp), raw

    @torch.no_grad()
    def _predict_state(self) -> np.ndarray | None:
        if len(self.hist_emb) < HISTORY_SIZE:
            return None
        ctx = torch.stack(self.hist_emb).unsqueeze(0)                    # (1,H,D)
        a_idx = torch.tensor(self.hist_act, device=self.device).unsqueeze(0)
        ctx_a = self.model.action_encoder(a_idx)                         # (1,H,D)
        pred = self.model.predict_next(ctx, ctx_a)                       # (1,D)
        s_norm = self.model.state_head(pred)[0]
        return (s_norm * self.state_std + self.state_mean).cpu().numpy()

    @torch.no_grad()
    def step(self, human_action: int = NOOP) -> dict:
        """Advance one decision tick (= frameskip ALE frames). Returns telemetry."""
        if not self.env.agents:
            self._reset_point()

        small, raw = self._render()

        frame = torch.from_numpy(small.astype(np.float32) / 255.0)
        frame = frame.permute(2, 0, 1).unsqueeze(0).unsqueeze(0).to(self.device)
        self.hist_emb.append(self.model.encode(frame)[0, 0])

        # The predictor conditions position k on the action taken AT frame k, so
        # predicting the next state needs the action we are about to choose.
        # Break the circularity with persistence (a_t := a_{t-1}); actions are
        # held for `frameskip` frames anyway, so this is usually exact.
        self.hist_act.append(self.prev_action)

        if len(self.hist_emb) > HISTORY_SIZE:
            self.hist_emb = self.hist_emb[-HISTORY_SIZE:]
            self.hist_act = self.hist_act[-HISTORY_SIZE:]

        ram = self.ale.getRAM()
        own_y = float(ram[JEPA_Y_ADDR])
        ball_in_play = int(ram[BALL_Y_ADDR]) != 0
        state = self._predict_state()

        if not ball_in_play:
            jepa_action, dy = FIRE, None            # serve; a parked ball can't be aligned to
        elif state is None:
            jepa_action, dy = NOOP, None            # still filling history
        else:
            jepa_action, dy = choose_action(state, self.state_idx, own_y)

        # Hold both actions for `frameskip` ALE frames -> matches training stride.
        for _ in range(self.frameskip):
            if not self.env.agents:
                break
            acts = {AGENT_JEPA: jepa_action, AGENT_HUMAN: human_action}
            _, rewards, _, _, _ = self.env.step({a: acts[a] for a in self.env.agents})
            for a, r in rewards.items():
                self.score[a] += float(r)

        self.prev_action = jepa_action
        self.tick += 1

        return {
            "raw": raw,
            "jepa_action": jepa_action,
            "human_action": human_action,
            "dy": dy,
            "own_y": own_y,
            "ball_in_play": ball_in_play,
            "pred_state": None if state is None
                          else {n: float(state[i]) for n, i in self.state_idx.items()},
            "score": {"jepa": self.score[AGENT_JEPA], "human": self.score[AGENT_HUMAN]},
            "tick": self.tick,
            "history_ready": len(self.hist_emb) == HISTORY_SIZE,
        }

    def close(self):
        self.env.close()
