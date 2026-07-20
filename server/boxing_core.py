"""Env + model + policy for JEPA Boxing, with no web framework attached.

scripts/play_boxing.py drives this headlessly; a websocket server can wrap it the
way server/play_tennis.py wraps tennis_core.

The model never reads RAM to decide. Each tick it encodes the rendered frame,
reads the predicted (x, y) of both boxers out of state_head(predict_next(...)),
and applies the same chase-then-punch rule collect_pettingzoo.py's `boxing_chase`
used to generate the data -- on PREDICTED coordinates.

Boxing is the clean case: both boxers' (x, y) live in the SAME RAM coordinate
frame (x right, y down), so `opponent - self` is a correct chase with no pixel
calibration, and the state head is trained on those raw RAM units -- the
collector's thresholds carry over unchanged.

Convention (scripts/configs/boxing.yml): JEPA drives `first_0`, the opponent
drives `second_0`. Actions are held for `frameskip` ALE frames so the play
stride equals the training stride.
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

logger = logging.getLogger("boxing")

# ALE full-action-set ids for boxing_v2, as calibrated in configs/boxing.yml.
NOOP, FIRE = 0, 1
UP, RIGHT, LEFT, DOWN = 2, 3, 4, 5
UPRIGHT, UPLEFT, DOWNRIGHT, DOWNLEFT = 6, 7, 8, 9
RIGHTFIRE, LEFTFIRE = 11, 12

AGENT_JEPA, AGENT_OPPONENT = "first_0", "second_0"

# RAM addresses -- used ONLY for the opponent's scripted policy and for the
# proprioception option below. JEPA's own perception never touches these.
RAM_POS = {"first_0": {"x": 32, "y": 34}, "second_0": {"x": 33, "y": 35}}

# Policy thresholds, in RAM units, identical to the collector's heuristic block.
PUNCH_RANGE_X = 16      # horizontal reach to throw a punch
ALIGN_Y = 8             # must be roughly level in y to land it
DEADZONE = 3            # px within which an axis is "close enough" (don't jitter)


def rom_dir() -> str:
    import ale_py
    return os.path.join(os.path.dirname(ale_py.__file__), "roms")


def load_frozen_model(checkpoint: str, device: torch.device):
    """Load a boxing state-head checkpoint, rebuilding the discrete ActionEncoder.

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
                "it was trained with a continuous ActionEncoder and cannot drive boxing")
        num_actions = int(w.shape[0])
        logger.warning("num_actions absent; inferred %d from embed.weight", num_actions)

    if not state_names:
        raise RuntimeError("checkpoint has no state_names/state_cols; cannot map head outputs")
    state_idx = {n: i for i, n in enumerate(state_names)}
    for required in ("first_x", "first_y", "second_x", "second_y"):
        if required not in state_idx:
            raise RuntimeError(f"state_names missing {required!r}: {state_names}")

    model = JEPAPool(embed_dim=ckpt.get("embed_dim", EMBED_DIM),
                     state_dim=state_dim, num_actions=num_actions)
    model.load_state_dict(ckpt["model"])
    for p in model.parameters():
        p.requires_grad = False
    model.eval().to(device)

    return model, state_mean.to(device), state_std.to(device), state_idx


def choose_action(me_x: float, me_y: float, opp_x: float, opp_y: float) -> int:
    """Chase-then-punch, on the same thresholds the collector used.

    Port of `_boxing_chase` (scripts/collect_pettingzoo.py) with `ram[addr]`
    reads replaced by coordinates. Both boxers share one coordinate frame, so
    `opp - me` needs no calibration.
    """
    dx, dy = opp_x - me_x, opp_y - me_y
    if abs(dx) <= PUNCH_RANGE_X and abs(dy) <= ALIGN_Y:
        return RIGHTFIRE if dx >= 0 else LEFTFIRE          # punch toward opponent
    h = None if abs(dx) <= DEADZONE else ("right" if dx > 0 else "left")
    v = None if abs(dy) <= DEADZONE else ("down" if dy > 0 else "up")
    if h and v:
        return {("right", "down"): DOWNRIGHT, ("right", "up"): UPRIGHT,
                ("left", "down"): DOWNLEFT, ("left", "up"): UPLEFT}[(h, v)]
    if h:
        return RIGHT if h == "right" else LEFT
    if v:
        return DOWN if v == "down" else UP
    return NOOP                                            # already on top of them


def opponent_action(ram: np.ndarray, rng: np.random.Generator, epsilon: float = 0.0) -> int:
    """The scripted seat: the collector's heuristic, read straight from RAM."""
    if epsilon > 0 and rng.random() < epsilon:
        return int(rng.integers(0, 18))
    me, opp = RAM_POS[AGENT_OPPONENT], RAM_POS[AGENT_JEPA]
    return choose_action(float(ram[me["x"]]), float(ram[me["y"]]),
                         float(ram[opp["x"]]), float(ram[opp["y"]]))


def png_b64(rgb: np.ndarray, scale: int = 1) -> str:
    if scale != 1:
        rgb = cv2.resize(rgb, (rgb.shape[1] * scale, rgb.shape[0] * scale),
                         interpolation=cv2.INTER_NEAREST)
    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


class BoxingSession:
    """One boxing_v2 match driven by the frozen world model.

    `own_source` picks where JEPA's own (x, y) comes from:
      "pred" -- the state head, i.e. pure pixels in / action out (the default).
      "ram"  -- proprioception, mirroring tennis_core's choice. Tennis had a
                measured reason: both seats ran the same chasing heuristic during
                collection, so the own-paddle readout leaned on the ball and fell
                apart once the policy stopped matching the collector. The same
                feedback-loop risk applies here, so this is kept as a comparison
                knob rather than assumed away.
    """

    def __init__(self, checkpoint: str, device: torch.device, frameskip: int = 4,
                 img_size: int = 128, seed: int = 0, predict_k: int = 1,
                 own_source: str = "pred", opponent_epsilon: float = 0.0):
        from pettingzoo.atari import boxing_v2

        if own_source not in ("pred", "ram"):
            raise ValueError(f"own_source must be 'pred' or 'ram', got {own_source!r}")

        self.device = device
        self.frameskip = frameskip
        self.img_size = img_size
        self.own_source = own_source
        self.opponent_epsilon = opponent_epsilon
        # How many steps to roll the world model forward before reading state.
        # k=1 re-anchors to real pixels every tick; k>1 predicts further ahead,
        # trading lead time for compounding error.
        self.predict_k = max(1, int(predict_k))
        self.model, self.state_mean, self.state_std, self.state_idx = \
            load_frozen_model(checkpoint, device)

        self.env = boxing_v2.parallel_env(render_mode="rgb_array",
                                          auto_rom_install_path=rom_dir())
        self.env.reset(seed=seed)
        self.ale = self.env.unwrapped.ale
        self.rng = np.random.default_rng(seed)

        self.hist_emb: list[torch.Tensor] = []
        self.hist_act: list[int] = []
        self.prev_action = NOOP
        self.score = {AGENT_JEPA: 0.0, AGENT_OPPONENT: 0.0}
        self.tick = 0

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
        # persistence approximation step() uses).
        emb = list(self.hist_emb)
        act = list(self.hist_act)
        pred = None
        for _ in range(self.predict_k):
            ctx = torch.stack(emb[-HISTORY_SIZE:]).unsqueeze(0)          # (1,H,D)
            a_idx = torch.tensor(act[-HISTORY_SIZE:], device=self.device).unsqueeze(0)
            ctx_a = self.model.action_encoder(a_idx)                     # (1,H,D)
            pred = self.model.predict_next(ctx, ctx_a)[0]                # (D,)
            emb.append(pred)
            act.append(self.prev_action)                                 # persist action
        s_norm = self.model.state_head(pred.unsqueeze(0))[0]
        return (s_norm * self.state_std + self.state_mean).cpu().numpy()

    @torch.no_grad()
    def step(self, opponent_override: int | None = None) -> dict:
        """Advance one decision tick (= frameskip ALE frames). Returns telemetry."""
        done = not self.env.agents
        if done:
            self.env.reset()
            self.hist_emb.clear()
            self.hist_act.clear()
            self.prev_action = NOOP

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
        state = self._predict_state()
        if state is None:
            jepa_action = NOOP
            err = None
        else:
            opp_x = float(state[self.state_idx["second_x"]])
            opp_y = float(state[self.state_idx["second_y"]])
            if self.own_source == "ram":
                me_x = float(ram[RAM_POS[AGENT_JEPA]["x"]])
                me_y = float(ram[RAM_POS[AGENT_JEPA]["y"]])
            else:
                me_x = float(state[self.state_idx["first_x"]])
                me_y = float(state[self.state_idx["first_y"]])
            jepa_action = choose_action(me_x, me_y, opp_x, opp_y)
            # Perception error against RAM ground truth, for diagnostics only.
            err = {n: float(state[self.state_idx[n]]) - float(ram[a])
                   for n, a in (("first_x", 32), ("first_y", 34),
                                ("second_x", 33), ("second_y", 35))}

        opp_action = (opponent_override if opponent_override is not None
                      else opponent_action(ram, self.rng, self.opponent_epsilon))

        # Hold both actions for `frameskip` ALE frames -> matches training stride.
        for _ in range(self.frameskip):
            if not self.env.agents:
                break
            acts = {AGENT_JEPA: jepa_action, AGENT_OPPONENT: opp_action}
            _, rewards, _, _, _ = self.env.step({a: acts[a] for a in self.env.agents})
            for a, r in rewards.items():
                self.score[a] += float(r)

        self.prev_action = jepa_action
        self.tick += 1

        return {
            "raw": raw,
            "jepa_action": jepa_action,
            "opponent_action": opp_action,
            "pred_state": None if state is None
                          else {n: float(state[i]) for n, i in self.state_idx.items()},
            "pred_err": err,
            "score": {"jepa": self.score[AGENT_JEPA], "opponent": self.score[AGENT_OPPONENT]},
            "tick": self.tick,
            "history_ready": len(self.hist_emb) == HISTORY_SIZE,
            "predict_k": self.predict_k,
            "episode_done": done,
        }

    def close(self):
        self.env.close()
