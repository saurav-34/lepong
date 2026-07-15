"""Generic PettingZoo-Atari 2-player frame collector, driven by a YAML config.

Add a new game by writing `configs/<game>.yml` — no code changes are needed for the
random-action path. This is a faithful superset of the hand-written collectors:
`configs/pong.yml` and `configs/tennis.yml` reproduce collect_pong.py /
collect_tennis.py exactly (same env, agent seat, RAM state map, velocities, and
ball-chasing heuristic). `configs/space_war.yml` is the new game.

What a config controls (see the shipped configs for full examples):
  env         module name under `pettingzoo.atari` (e.g. space_war_v2)
  agents      jepa/human seat -> agent id. JEPA's action is stored as `action`
              (what the world model is conditioned on); the other seat as `action_human`.
  frameskip   ALE frames per recorded transition. These envs step 1 ALE frame each,
              so we HOLD the chosen action this many env-steps -> the recorded stride
              equals the play stride (the train==play-stride lesson from Pong/Tennis).
  state       RAM byte -> state-field name. Leave `{}` for a game whose RAM has not
              been reverse-engineered yet (Space War): only pixels+actions are stored.
  velocity    output field -> {source: <state field>, cap: <px>}. Finite-difference of
              the source per transition; jumps larger than `cap` (serve teleports /
              off-screen) are written as 0.
  policy      sticky (DEFAULT; standard Atari data-collection policy — sticky-action
              random, repeat previous action w.p. `repeat_prob`=0.25, Machado et al.
              2018; game-agnostic, builds momentum so ships/paddles actually move and
              fire, unlike i.i.d. `random`) | random (i.i.d. uniform) | heuristic (a
              registered ball-chasing policy + its params; see HEURISTICS below).

Usage:
  python scripts/collect_pettingzoo.py --config scripts/configs/space_war.yml
  python scripts/collect_pettingzoo.py --config scripts/configs/pong.yml --frames 400000
CLI flags override the matching config keys (handy for sweeps without editing YAML).
"""

import argparse
import importlib
import io
import os

import cv2
import lance
import numpy as np
import pyarrow as pa
import yaml
from PIL import Image


def _rom_dir() -> str:
    import ale_py
    return os.path.join(os.path.dirname(ale_py.__file__), "roms")


def encode_frame(frame: np.ndarray) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(frame.astype(np.uint8)).save(buf, format="PNG")
    return buf.getvalue()


# --- Ball-chasing heuristics -------------------------------------------------
# A heuristic is fn(ram, agent, hp, rng) -> action, where `agent` is the concrete
# PettingZoo id ("first_0"/"second_0") and `hp` is the config's `heuristic` block.
# Register a new one here once a game's RAM/actions are verified; until then use
# policy: random. These two reproduce collect_pong.py / collect_tennis.py exactly.

def _pong_chase(ram, agent, hp, rng):
    """Move the paddle centre toward the ball's y; FIRE to serve when no ball is in
    play (ball_y==0 -> holding NOOP deadlocks the point)."""
    a = hp["actions"]
    if int(ram[hp["ball_y_addr"]]) == 0:
        return a["serve"]
    paddle_top = int(ram[hp["paddle_y_addr"][agent]])
    dy = int(ram[hp["ball_y_addr"]]) - (paddle_top + hp["paddle_half"])
    if abs(dy) <= hp["deadzone"]:
        return a["noop"]
    return a["down"] if dy > 0 else a["up"]


def _tennis_chase(ram, agent, hp, rng):
    """Move x toward the ball's x, FIRE to swing when aligned. End-aware: players
    change court ends after odd cumulative games (RAM sum of `games_addrs`), which
    swaps which RAM x-addr each seat controls (same rule the play script handles)."""
    a = hp["actions"]
    games = sum(int(ram[addr]) for addr in hp["games_addrs"])
    first0_bottom = ((games + 1) // 2) % 2 == 0
    bottom, top = hp["bottom_x_addr"], hp["top_x_addr"]
    if agent == "first_0":
        own = bottom if first0_bottom else top
    else:
        own = top if first0_bottom else bottom
    dx = int(ram[hp["ball_x_addr"]]) - int(ram[own])
    if abs(dx) <= hp["deadzone"]:
        return a["fire"]
    return a["right"] if dx > 0 else a["left"]


def _boxing_chase(ram, agent, hp, rng):
    """Close the distance to the opponent and punch when aligned. Both boxers' (x,y)
    are in the SAME RAM frame (x increases right, y increases down), so `opp - self`
    is a correct chase with no calibration. When within horizontal punch range and
    roughly level in y, throw a directional punch toward the opponent; otherwise step
    (8-directional) toward them."""
    a = hp["actions"]
    me, opp = hp["pos"][agent], hp["pos"][hp["opponent"][agent]]
    dx = int(ram[opp["x"]]) - int(ram[me["x"]])
    dy = int(ram[opp["y"]]) - int(ram[me["y"]])
    if abs(dx) <= hp["punch_range_x"] and abs(dy) <= hp["align_y"]:
        return a["rightfire"] if dx >= 0 else a["leftfire"]   # punch toward opponent
    h = None if abs(dx) <= hp["deadzone"] else ("right" if dx > 0 else "left")
    v = None if abs(dy) <= hp["deadzone"] else ("down" if dy > 0 else "up")
    if h and v:
        return a[{"up": "upright", "down": "downright"}[v] if h == "right"
                 else {"up": "upleft", "down": "downleft"}[v]]
    return a[h or v]


HEURISTICS = {"pong_chase": _pong_chase, "tennis_chase": _tennis_chase,
              "boxing_chase": _boxing_chase}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Path to a game YAML config.")
    # Optional overrides of the matching config keys.
    ap.add_argument("--frames", type=int)
    ap.add_argument("--out", type=str)
    ap.add_argument("--img-size", type=int)
    ap.add_argument("--frameskip", type=int)
    ap.add_argument("--seed", type=int)
    ap.add_argument("--policy", choices=["heuristic", "sticky", "random"])
    ap.add_argument("--epsilon", type=float)
    ap.add_argument("--repeat-prob", type=float)
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    for key in ("frames", "out", "img_size", "frameskip", "seed", "policy",
                "epsilon", "repeat_prob"):
        v = getattr(args, key)
        if v is not None:
            cfg[key] = v

    cfg.setdefault("img_size", 128)
    cfg.setdefault("frameskip", 4)
    cfg.setdefault("seed", 0)
    cfg.setdefault("policy", "sticky")
    cfg.setdefault("epsilon", 0.30)
    cfg.setdefault("repeat_prob", 0.25)   # sticky-action repeat prob (Machado et al. 2018)
    cfg.setdefault("state", {})
    cfg.setdefault("velocity", {})

    IMG_SIZE = int(cfg["img_size"])
    FRAMES = int(cfg["frames"])
    FRAMESKIP = int(cfg["frameskip"])
    JEPA = cfg["agents"]["jepa"]
    HUMAN = cfg["agents"]["human"]
    # RAM byte -> field name; field names (the values) are the schema columns.
    state_map = dict(cfg["state"])
    state_fields = list(state_map.values())
    vel_map = dict(cfg["velocity"])          # out_name -> {source, cap}
    vel_fields = list(vel_map.keys())

    if cfg["policy"] == "heuristic":
        hp = cfg["heuristic"]
        heuristic = HEURISTICS[hp["name"]]
    else:
        hp = None
        heuristic = None

    rng = np.random.default_rng(cfg["seed"])

    mod = importlib.import_module(f"pettingzoo.atari.{cfg['env']}")
    env = mod.parallel_env(render_mode="rgb_array", auto_rom_install_path=_rom_dir())
    env.reset(seed=cfg["seed"])
    ale = env.unwrapped.ale
    n_actions = {a: int(env.action_space(a).n) for a in env.agents}
    print(f"{cfg['env']} ready. agents={env.agents}  action_space={n_actions}  "
          f"frameskip={FRAMESKIP}  jepa={JEPA} human={HUMAN}")

    prev_action = {}  # per-agent last action, for the sticky policy

    def choose_action(ram, agent):
        n = n_actions[agent]
        if cfg["policy"] == "heuristic" and rng.random() >= cfg["epsilon"]:
            return int(heuristic(ram, agent, hp, rng))
        if cfg["policy"] == "sticky" and agent in prev_action \
                and rng.random() < cfg["repeat_prob"]:
            return prev_action[agent]        # repeat previous action (build momentum)
        a = int(rng.integers(0, n))
        prev_action[agent] = a
        return a

    # --- Dynamic schema: fixed columns + one float32 per state and velocity field.
    schema = pa.schema(
        [pa.field("episode_idx", pa.int32()),
         pa.field("step_idx", pa.int32()),
         pa.field("action", pa.int32()),        # JEPA seat (world-model conditioning)
         pa.field("action_human", pa.int32()),  # other seat (reference)
         pa.field("pixels", pa.binary())]
        + [pa.field(name, pa.float32()) for name in state_fields]
        + [pa.field(name, pa.float32()) for name in vel_fields]
    )

    os.makedirs(os.path.dirname(cfg["out"]) or ".", exist_ok=True)
    batches = []
    buf = {k: [] for k in ["action", "action_human", "pixels"] + state_fields}
    ep = 0
    current = 0

    def vel(seq, cap):
        v = [0.0] + [seq[i] - seq[i - 1] for i in range(1, len(seq))]
        return [x if abs(x) <= cap else 0.0 for x in v]

    def flush_episode():
        nonlocal ep
        n = len(buf["action"])
        if n == 0:
            return
        arrays = [
            pa.array([ep] * n, type=pa.int32()),
            pa.array(list(range(n)), type=pa.int32()),
            pa.array(buf["action"], type=pa.int32()),
            pa.array(buf["action_human"], type=pa.int32()),
            pa.array(buf["pixels"], type=pa.binary()),
        ]
        arrays += [pa.array(buf[f], type=pa.float32()) for f in state_fields]
        arrays += [pa.array(vel(buf[vel_map[f]["source"]], float(vel_map[f]["cap"])),
                            type=pa.float32()) for f in vel_fields]
        batches.append(pa.RecordBatch.from_arrays(arrays, schema=schema))
        ep += 1
        for k in buf:
            buf[k].clear()

    pol = cfg["policy"]
    tag = ({"heuristic": f" (epsilon={cfg['epsilon']})",
            "sticky": f" (repeat_prob={cfg['repeat_prob']})"}).get(pol, "")
    print(f"Collecting {FRAMES} transitions at {IMG_SIZE}px, policy={pol}" + tag
          + f", {len(state_fields)} state fields (both agents)...")

    while current < FRAMES:
        if not env.agents:                       # game over -> new episode
            flush_episode()
            env.reset()
            if ep % 5 == 0:
                print(f"  {current}/{FRAMES} transitions, {ep} episodes")
            continue

        # 1. Capture frame + state BEFORE acting (keeps (s_t, a_t) aligned).
        raw = env.render()
        interp = (cv2.INTER_AREA if IMG_SIZE * IMG_SIZE < raw.shape[0] * raw.shape[1]
                  else cv2.INTER_LINEAR)
        frame = cv2.resize(raw, (IMG_SIZE, IMG_SIZE), interpolation=interp)
        ram = ale.getRAM()

        # 2. Choose both seats' actions.
        a_jepa = choose_action(ram, JEPA)
        a_human = choose_action(ram, HUMAN)

        # 3. Save aligned transition.
        buf["action"].append(a_jepa)
        buf["action_human"].append(a_human)
        buf["pixels"].append(encode_frame(frame))
        for addr, name in state_map.items():
            buf[name].append(float(int(ram[int(addr)])))

        # 4. Advance FRAMESKIP ALE frames holding the same actions (-> play stride).
        for _ in range(FRAMESKIP):
            if not env.agents:
                break
            env.step({a: (a_jepa if a == JEPA else a_human) for a in env.agents})
        current += 1

    flush_episode()
    lance.write_dataset(batches, cfg["out"], schema=schema, mode="overwrite")
    print(f"Done! {lance.dataset(cfg['out']).count_rows()} transitions saved to {cfg['out']}")


if __name__ == "__main__":
    main()
