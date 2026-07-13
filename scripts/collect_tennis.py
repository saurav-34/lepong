"""Collect Tennis frames from the TRUE 2-player env (PettingZoo tennis_v3).

Why this exists (vs collect_tennis_with_state.py which uses single-agent ALE/Tennis-v5):
  The human-vs-JEPA game is played in PettingZoo `tennis_v3` (backend
  `multi_agent_ale_py`), where BOTH players take independent actions — `first_0`
  and `second_0`, 18 actions each. Collecting from the SAME env we play in removes
  any train/play backend/mode gap. (Frames match the single-agent env to ~0.4/255,
  so the two datasets are compatible, but this one is collected in genuine 2-player
  dynamics: no built-in CPU opponent, both sides agent-driven.)

Convention: **JEPA controls `second_0` = the TOP player** — its action is stored as
`action` (what the world model is conditioned on). The human is `first_0` = the BOTTOM
player, stored as `action_human` for reference. At play, send JEPA's action as `second_0`.
(Verified on this machine: driving first_0 moves the bottom sprite, second_0 the top.)

Frameskip: tennis_v3 advances 1 ALE frame per env.step. We hold each action for
FRAMESKIP env-steps so one recorded transition = FRAMESKIP ALE frames, matching the
fs4 training/play stride (same lesson as Pong: train stride must equal play stride).

RAM addresses (verified on this machine, same as single-agent Tennis):
  26=player_x 24=player_y (bottom) · 27=enemy_x 25=enemy_y (top) · 16=ball_x
  NOTE 17 is ball ARC-HEIGHT, not court-y — stored as ball_y but treat as suspect
  (resolve the real court-y before using it in a planner/state-head).
"""

import argparse
import io
import os
import random

import cv2
import lance
import numpy as np
import pyarrow as pa
from PIL import Image
from pettingzoo.atari import tennis_v3


def _rom_dir() -> str:
    import ale_py
    return os.path.join(os.path.dirname(ale_py.__file__), "roms")


def encode_frame(frame: np.ndarray) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(frame.astype(np.uint8)).save(buf, format="PNG")
    return buf.getvalue()


parser = argparse.ArgumentParser()
parser.add_argument("--frames", type=int, default=50000)
parser.add_argument("--out", type=str, default="datasets/tennis_ma_128x128.lance")
parser.add_argument("--img-size", type=int, default=128)
parser.add_argument("--frameskip", type=int, default=4,
                    help="ALE frames per recorded transition (tennis_v3 is 1/step, so "
                         "we hold each action this many env-steps). Match the play stride.")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--policy", choices=["heuristic", "random"], default="heuristic",
                    help="heuristic = both players chase the ball's x and swing "
                         "(produces real rallies/interceptions); random = uniform actions.")
parser.add_argument("--epsilon", type=float, default=0.35,
                    help="Under --policy heuristic, fraction of random actions mixed in. "
                         "0.35 balances rallies/swings (interception states) with full "
                         "18-action coverage the WM planner needs. Lower=tighter tracking "
                         "but sparse action coverage; higher=more coverage, fewer rallies.")
args = parser.parse_args()

IMG_SIZE = args.img_size
MAX_BALL_SPEED = 50.0
AGENT_JEPA, AGENT_OPP = "second_0", "first_0"   # JEPA=second_0=top, human=first_0=bottom

# ALE full-action-set ids used by the ball-chasing heuristic. Calibrated on this ROM:
# RIGHT(3) -> player_x +40, LEFT(4) -> player_x -32 (same for both agents). Firing LOCKS
# movement, so we MOVE with pure RIGHT/LEFT to align x, then FIRE(1) to swing once aligned.
FIRE, RIGHT, LEFT = 1, 3, 4
BALL_X_ADDR = 16
PLAYER_X_ADDR = {"first_0": 26, "second_0": 27}   # RAM x-addr each agent controls
# (verified by direct wiggle-probe 2026-07-10: holding RIGHT on first_0 moves
#  RAM[26]=player_x/bottom, holding RIGHT on second_0 moves RAM[27]=enemy_x/top;
#  reproducible across seeds. The previous swapped mapping made both heuristic
#  agents chase the ball relative to the OPPONENT's paddle, which produced the
#  v2 dataset's degenerate anti-correlated paddle data — corr(player_x,enemy_x)
#  = -0.98, corr(paddle,ball) ~= 0.07, paddles pinned at the walls.
#  NOTE: this mapping holds at game 1; players change court ends after odd
#  cumulative games (RAM[71]+RAM[72]), same rule the play script handles.)
ALIGN_DEADZONE = 6   # px: within this of the ball's x -> swing instead of moving


def own_x_addr(ram, agent):
    """End-aware paddle address: RAM[26]/RAM[27] track court sides (bottom/top),
    and players change ends after odd cumulative games (RAM[71]+RAM[72])."""
    games = int(ram[71]) + int(ram[72])
    first0_bottom = ((games + 1) // 2) % 2 == 0
    if agent == "first_0":
        return 26 if first0_bottom else 27
    return 27 if first0_bottom else 26


def choose_action(ram, agent):
    """Ball-chasing heuristic (with epsilon-random exploration): move the player's x
    toward the ball's x, and swing when aligned. Random for --policy random or epsilon."""
    if args.policy == "random" or random.random() < args.epsilon:
        return random.randint(0, 17)
    dx = int(ram[BALL_X_ADDR]) - int(ram[own_x_addr(ram, agent)])
    if abs(dx) <= ALIGN_DEADZONE:
        return FIRE
    return RIGHT if dx > 0 else LEFT

env = tennis_v3.parallel_env(render_mode="rgb_array",
                             auto_rom_install_path=_rom_dir())
env.reset(seed=args.seed)
ale = env.unwrapped.ale
print(f"tennis_v3 ready. agents={env.agents}  action_space="
      f"{ {a: env.action_space(a).n for a in env.agents} }  frameskip={args.frameskip}")


def get_state():
    r = ale.getRAM()
    return {"player_x": int(r[26]), "player_y": int(r[24]),
            "enemy_x": int(r[27]), "enemy_y": int(r[25]),
            "ball_x": int(r[16]), "ball_y": int(r[17])}


def _vel(seq, cap):
    v = [0.0] + [seq[i] - seq[i - 1] for i in range(1, len(seq))]
    return [x if abs(x) <= cap else 0.0 for x in v]


schema = pa.schema([
    pa.field("episode_idx", pa.int32()),
    pa.field("step_idx", pa.int32()),
    pa.field("action", pa.int32()),        # second_0 = JEPA/top (what the WM is conditioned on)
    pa.field("action_human", pa.int32()),  # first_0 = human/bottom (reference)
    pa.field("pixels", pa.binary()),
    pa.field("player_x", pa.float32()),
    pa.field("player_y", pa.float32()),
    pa.field("enemy_x", pa.float32()),
    pa.field("enemy_y", pa.float32()),
    pa.field("ball_x", pa.float32()),
    pa.field("ball_y", pa.float32()),
    pa.field("ball_vx", pa.float32()),
    pa.field("ball_vy", pa.float32()),
])

os.makedirs(os.path.dirname(args.out), exist_ok=True)
batches = []
current_frames = 0
ep = 0
buf = {k: [] for k in ["action", "action_human", "pixels",
                       "player_x", "player_y", "enemy_x", "enemy_y", "ball_x", "ball_y"]}

print(f"Collecting {args.frames} transitions at {IMG_SIZE}px, "
      f"policy={args.policy}" + (f" (epsilon={args.epsilon})" if args.policy == "heuristic" else "")
      + " (both agents)...")


def flush_episode():
    global ep
    n = len(buf["action"])
    if n == 0:
        return
    batch = pa.RecordBatch.from_arrays([
        pa.array([ep] * n, type=pa.int32()),
        pa.array(list(range(n)), type=pa.int32()),
        pa.array(buf["action"], type=pa.int32()),
        pa.array(buf["action_human"], type=pa.int32()),
        pa.array(buf["pixels"], type=pa.binary()),
        pa.array(buf["player_x"], type=pa.float32()),
        pa.array(buf["player_y"], type=pa.float32()),
        pa.array(buf["enemy_x"], type=pa.float32()),
        pa.array(buf["enemy_y"], type=pa.float32()),
        pa.array(buf["ball_x"], type=pa.float32()),
        pa.array(buf["ball_y"], type=pa.float32()),
        pa.array(_vel(buf["ball_x"], MAX_BALL_SPEED), type=pa.float32()),
        pa.array(_vel(buf["ball_y"], MAX_BALL_SPEED), type=pa.float32()),
    ], schema=schema)
    batches.append(batch)
    ep += 1
    for k in buf:
        buf[k].clear()


while current_frames < args.frames:
    if not env.agents:                      # episode ended -> new episode
        flush_episode()
        env.reset()
        if ep % 5 == 0:
            print(f"  {current_frames}/{args.frames} transitions, {ep} episodes")
        continue

    # 1. Capture state + frame BEFORE acting (keeps (s_t, a_t) aligned).
    raw = env.render()                      # 210x160x3, shared
    interp = cv2.INTER_AREA if IMG_SIZE * IMG_SIZE < raw.shape[0] * raw.shape[1] else cv2.INTER_LINEAR
    frame = cv2.resize(raw, (IMG_SIZE, IMG_SIZE), interpolation=interp)
    s = get_state()

    # 2. Choose actions. Heuristic: both players chase the ball's x and swing, so the
    #    ball is actually returned -> rallies/interceptions in the data (with epsilon
    #    random mixed in for coverage). `s` was just read from RAM above.
    ram_now = ale.getRAM()
    a_jepa = choose_action(ram_now, AGENT_JEPA)
    a_opp = choose_action(ram_now, AGENT_OPP)

    # 3. Save aligned transition.
    buf["action"].append(a_jepa)
    buf["action_human"].append(a_opp)
    buf["pixels"].append(encode_frame(frame))
    for k in ["player_x", "player_y", "enemy_x", "enemy_y", "ball_x", "ball_y"]:
        buf[k].append(float(s[k]))

    # 4. Advance FRAMESKIP ALE frames holding the same actions (-> fs4 stride).
    for _ in range(args.frameskip):
        if not env.agents:
            break
        acts = {a: (a_jepa if a == AGENT_JEPA else a_opp) for a in env.agents}
        env.step(acts)
    current_frames += 1

flush_episode()
lance.write_dataset(batches, args.out, schema=schema, mode="overwrite")
print(f"Done! {lance.dataset(args.out).count_rows()} transitions saved to {args.out}")
