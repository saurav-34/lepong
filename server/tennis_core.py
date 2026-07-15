"""Env + model + policy for human-vs-JEPA Tennis, with no web framework attached.

server/play_tennis.py wraps this in a websocket; scripts can drive it headlessly.

The model never reads RAM to decide. Each tick it encodes the rendered frame,
reads a predicted state out of state_head(predict_next(...)), and applies the
same align-then-swing rule collect_tennis.py used to generate the data -- on
PREDICTED coordinates. RAM is touched only for the court-ends bookkeeping the
pixels genuinely cannot express.

Convention (collect_tennis.py): JEPA drives `second_0` (top sprite), the human
drives `first_0` (bottom). Actions are held for `frameskip` ALE frames so the
play stride equals the training stride.
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

logger = logging.getLogger("tennis")

# ALE full-action-set ids, calibrated on this ROM in collect_tennis.py:
# RIGHT(3) -> paddle x +40, LEFT(4) -> x -32, for BOTH agents. FIRE locks
# movement, so align on x with RIGHT/LEFT, then FIRE inside the deadzone.
NOOP, FIRE, UP, RIGHT, LEFT, DOWN = 0, 1, 2, 3, 4, 5
ALIGN_DEADZONE = 6                       # px, same as the collector
AGENT_JEPA, AGENT_HUMAN = "second_0", "first_0"



def rom_dir() -> str:
    import ale_py
    return os.path.join(os.path.dirname(ale_py.__file__), "roms")


def load_frozen_model(checkpoint: str, device: torch.device):
    """Load a tennis state-head checkpoint, rebuilding the discrete ActionEncoder.

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
    # Embedding, 0 builds an MLP that cannot accept this state_dict. Checkpoints
    # written before train_statehead.py saved the key fall back to the table shape.
    num_actions = ckpt.get("num_actions")
    if num_actions is None:
        w = ckpt["model"].get("action_encoder.embed.weight")
        if w is None:
            raise RuntimeError(
                "checkpoint has no num_actions and no action_encoder.embed.weight; "
                "it was trained with a continuous ActionEncoder and cannot drive tennis")
        num_actions = int(w.shape[0])
        logger.warning("num_actions absent; inferred %d from embed.weight", num_actions)

    if not state_names:
        raise RuntimeError("checkpoint has no state_names/state_cols; cannot map head outputs")
    state_idx = {n: i for i, n in enumerate(state_names)}
    for required in ("ball_x", "player_x", "enemy_x"):
        if required not in state_idx:
            raise RuntimeError(f"state_names missing {required!r}: {state_names}")

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


def second0_is_top(ram: np.ndarray) -> bool:
    """Players change court ends after odd cumulative games (RAM[71]+RAM[72]).

    collect_tennis.py always stored RAM[26]->player_x (bottom sprite) and
    RAM[27]->enemy_x (top sprite), so the head's `enemy_x` output always names
    the top sprite. Which output is OURS therefore flips with the ends.
    """
    games = int(ram[71]) + int(ram[72])
    first0_bottom = ((games + 1) // 2) % 2 == 0
    return first0_bottom


def own_x_addr(ram: np.ndarray, agent: str) -> int:
    """End-aware RAM address of `agent`'s own paddle x (26=bottom, 27=top)."""
    return (26 if second0_is_top(ram) else 27) if agent == AGENT_HUMAN else \
           (27 if second0_is_top(ram) else 26)


def choose_action(state: np.ndarray, state_idx: dict[str, int],
                  own_x: float) -> tuple[int, float]:
    """Align-then-swing: the ball is PERCEIVED, the own paddle is PROPRIOCEPTED.

    Only ball_x comes from the state head. Two reasons not to also predict our
    own paddle:

      1. It is our own actuator state, not something we must see. lepong does the
         same -- server/infer.py predicts only the ball and lets the client move
         its own paddle toward it.
      2. It does not survive the feedback loop. Both agents ran the same chasing
         heuristic during collection, so corr(enemy_x, ball_x) = +0.70 and
         corr(player_x, enemy_x) = +0.86 in the data. A linear probe reaches
         corr 0.90 on enemy_x partly by reading the ball. Once our policy stops
         chasing the way the collector did, that correlation breaks and the
         readout falls to corr 0.23 -- measured, not hypothesised.

    ball_y is never used: collect_tennis.py flags RAM[17] as ball ARC-HEIGHT
    rather than court-y.
    """
    ball_x = float(state[state_idx["ball_x"]])
    dx = ball_x - own_x
    if abs(dx) <= ALIGN_DEADZONE:
        return FIRE, dx
    return (RIGHT if dx > 0 else LEFT), dx


def png_b64(rgb: np.ndarray, scale: int = 1) -> str:
    if scale != 1:
        rgb = cv2.resize(rgb, (rgb.shape[1] * scale, rgb.shape[0] * scale),
                         interpolation=cv2.INTER_NEAREST)
    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


class TennisSession:
    """One tennis_v3 match driven by the frozen world model."""

    def __init__(self, checkpoint: str, device: torch.device, frameskip: int = 4,
                 img_size: int = 128, seed: int = 0, predict_k: int = 1):
        from pettingzoo.atari import tennis_v3

        self.device = device
        self.frameskip = frameskip
        self.img_size = img_size
        # How many steps to roll the world model forward before reading the ball.
        # k=1 re-anchors to real pixels every tick; k>1 predicts further ahead so
        # JEPA aims where the ball WILL be, trading lead time for compounding error.
        self.predict_k = max(1, int(predict_k))
        self.model, self.state_mean, self.state_std, self.state_idx = \
            load_frozen_model(checkpoint, device)

        self.env = tennis_v3.parallel_env(render_mode="rgb_array",
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
        # Roll the predictor forward predict_k steps, feeding each predicted
        # embedding back into the context and holding the last action (the same
        # persistence approximation step() uses). k=1 is a single one-step
        # prediction re-anchored to real pixels; k>1 keeps extrapolating from its
        # own predictions, so error compounds but the read leads further ahead.
        emb = list(self.hist_emb)                                        # each (D,)
        act = list(self.hist_act)                                        # each action id
        pred = None
        for _ in range(self.predict_k):
            ctx = torch.stack(emb[-HISTORY_SIZE:]).unsqueeze(0)          # (1,H,D)
            a_idx = torch.tensor(act[-HISTORY_SIZE:], device=self.device).unsqueeze(0)
            ctx_a = self.model.action_encoder(a_idx)                     # (1,H,D)
            pred = self.model.predict_next(ctx, ctx_a)[0]               # (D,)
            emb.append(pred)
            act.append(self.prev_action)                                # persist action
        s_norm = self.model.state_head(pred.unsqueeze(0))[0]
        return (s_norm * self.state_std + self.state_mean).cpu().numpy()

    @torch.no_grad()
    def step(self, human_action: int = NOOP) -> dict:
        """Advance one decision tick (= frameskip ALE frames). Returns telemetry."""
        if not self.env.agents:
            self._reset_point()

        # Anchor JEPA to its native (top) end. The ROM swaps court ends every
        # couple games (second0_is_top flips), but the reactive align-then-swing
        # policy only rallies from the top -- on the bottom it chases a
        # mispredicted ball into a corner and stalls. When the ends flip off top,
        # reset the match to the native orientation so JEPA keeps playing its good
        # end. The running tally in self.score is not touched, so the score the
        # client shows carries across the restart.
        if not second0_is_top(self.ale.getRAM()):
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
        jepa_top = second0_is_top(ram)
        own_x = float(ram[own_x_addr(ram, AGENT_JEPA)])
        state = self._predict_state()
        if state is None:
            jepa_action, dx = NOOP, None
        else:
            jepa_action, dx = choose_action(state, self.state_idx, own_x)

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
            "dx": dx,
            "pred_state": None if state is None
                          else {n: float(state[i]) for n, i in self.state_idx.items()},
            "jepa_is_top": jepa_top,
            "score": {"jepa": self.score[AGENT_JEPA], "human": self.score[AGENT_HUMAN]},
            "tick": self.tick,
            "history_ready": len(self.hist_emb) == HISTORY_SIZE,
            "predict_k": self.predict_k,
        }

    def close(self):
        self.env.close()
