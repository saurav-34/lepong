"""Collect Pong frames from the TRUE 2-player env (PettingZoo pong_v3).

Why this exists (vs collect_pong_with_state.py which uses single-agent ALE/Pong-v5
via SB3's make_atari_env):
  1. The old collector double-frameskips: ALE/Pong-v5 has builtin frameskip 4 +
     sticky actions, and AtariWrapper stacks MaxAndSkip(4) on top -> one recorded
     transition = 16 ALE frames with corrupted action labels. Play runs at 4
     frames/step, so the predictor's learned horizon was 4x off. Here pong_v3
     advances exactly 1 ALE frame per env.step (verified) and we hold each action
     for FRAMESKIP steps -> stride is explicit and equals the play stride.
  2. The human-vs-JEPA game is played in pong_v3, where BOTH paddles are
     agent-driven (no builtin CPU opponent). Collecting from the same env removes
     any train/play backend gap.

Convention: **JEPA controls `second_0` = the LEFT paddle** — its action is stored
as `action` (what the world model is conditioned on). The human plays `first_0` =
the RIGHT paddle (classic Pong seat), stored as `action_human`.

Verified on this machine (wiggle-probe, 2026-07-10):
  first_0 drives RAM[51]=player_y (right paddle), second_0 drives RAM[50]=enemy_y
  (left paddle), zero cross-talk, and the mapping does NOT swap across points or
  games (unlike Tennis). Action space is Discrete(6); RIGHT(2)=up (RAM y
  decreases), LEFT(3)=down.

Serves require FIRE. Holding NOOP leaves the ball parked: it advances on 6.6% of
fs4 steps under a pure chase heuristic vs 56.3% when FIRE is pressed. The heuristic
must emit FIRE itself — relying on epsilon-random to stumble onto it pins the
paddles at the wall (see below).

RAM addresses (same as single-agent Pong):
  49=ball_x  54=ball_y  51=player_y (right)  50=enemy_y (left)
  13/14=scores. **ball_y==0** — not ball_x — means no ball is in play (between
  point and serve); ball_x is nonzero on 99.5% of frames, so it is useless as an
  off-screen guard. Chasing a phantom ball at y=0 drives the paddle to the top
  wall and keeps it there.
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
from pettingzoo.atari import pong_v3


def _rom_dir() -> str:
    import ale_py
    return os.path.join(os.path.dirname(ale_py.__file__), "roms")


def encode_frame(frame: np.ndarray) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(frame.astype(np.uint8)).save(buf, format="PNG")
    return buf.getvalue()


parser = argparse.ArgumentParser()
parser.add_argument("--frames", type=int, default=400000)
parser.add_argument("--out", type=str, default="datasets/pong_ma_128x128.lance")
parser.add_argument("--img-size", type=int, default=128)
parser.add_argument("--frameskip", type=int, default=4,
                    help="ALE frames per recorded transition (pong_v3 is 1/step, so "
                         "we hold each action this many env-steps). Match the play stride.")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--policy", choices=["heuristic", "random"], default="heuristic",
                    help="heuristic = both paddles chase the ball's y (produces real "
                         "rallies/interceptions); random = uniform actions.")
parser.add_argument("--epsilon", type=float, default=0.30,
                    help="Under --policy heuristic, fraction of random actions mixed in "
                         "for the 6-action coverage the WM needs.")
args = parser.parse_args()

IMG_SIZE = args.img_size
# Real ball motion is ~4-5 RAM px per fs4 step (p50, measured); serve teleports and
# off-screen jumps are >70. Cap in between so teleports become v=0.
MAX_BALL_SPEED = 24.0

AGENT_JEPA, AGENT_HUMAN = "second_0", "first_0"   # JEPA=left paddle, human=right

NOOP, FIRE, UP, DOWN = 0, 1, 2, 3                 # minimal set: 2=RIGHT moves up, 3=LEFT moves down
PADDLE_Y_ADDR = {"first_0": 51, "second_0": 50}   # RAM y-addr each agent controls (fixed, no end swap)
BALL_X_ADDR, BALL_Y_ADDR = 49, 54
# Ball and paddle live in different RAM origins: fitting both against pixel position
# gives ball_y ~= 0.962*paddle_y + 10.0 when the ball is level with the paddle centre.
PADDLE_HALF = 10    # RAM y is the paddle top; aim its centre at the ball
ALIGN_DEADZONE = 4  # px: within this of the ball's y -> hold position


def choose_action(ram, agent):
    """Ball-chasing heuristic (with epsilon-random exploration): move the paddle's
    centre toward the ball's y, and FIRE to serve when no ball is in play. Random for
    --policy random or epsilon."""
    if args.policy == "random" or random.random() < args.epsilon:
        return random.randint(0, 5)
    if int(ram[BALL_Y_ADDR]) == 0:   # no ball in play -> serve it (NOOP would deadlock)
        return FIRE
    dy = int(ram[BALL_Y_ADDR]) - (int(ram[PADDLE_Y_ADDR[agent]]) + PADDLE_HALF)
    if abs(dy) <= ALIGN_DEADZONE:
        return NOOP
    return DOWN if dy > 0 else UP


random.seed(args.seed)
env = pong_v3.parallel_env(render_mode="rgb_array", auto_rom_install_path=_rom_dir())
env.reset(seed=args.seed)
ale = env.unwrapped.ale
print(f"pong_v3 ready. agents={env.agents}  action_space="
      f"{ {a: int(env.action_space(a).n) for a in env.agents} }  frameskip={args.frameskip}")


def get_state():
    r = ale.getRAM()
    return {"player_y": int(r[51]), "enemy_y": int(r[50]),
            "ball_x": int(r[49]), "ball_y": int(r[54])}


def _vel(seq, cap):
    v = [0.0] + [seq[i] - seq[i - 1] for i in range(1, len(seq))]
    return [x if abs(x) <= cap else 0.0 for x in v]


schema = pa.schema([
    pa.field("episode_idx", pa.int32()),
    pa.field("step_idx", pa.int32()),
    pa.field("action", pa.int32()),        # second_0 = JEPA/left (what the WM is conditioned on)
    pa.field("action_human", pa.int32()),  # first_0 = human/right (reference)
    pa.field("pixels", pa.binary()),
    pa.field("player_y", pa.float32()),
    pa.field("enemy_y", pa.float32()),
    pa.field("ball_x", pa.float32()),
    pa.field("ball_y", pa.float32()),
    pa.field("ball_vx", pa.float32()),
    pa.field("ball_vy", pa.float32()),
])

os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
batches = []
current_frames = 0
ep = 0
buf = {k: [] for k in ["action", "action_human", "pixels",
                       "player_y", "enemy_y", "ball_x", "ball_y"]}

print(f"Collecting {args.frames} transitions at {IMG_SIZE}px, "
      f"policy={args.policy}" + (f" (epsilon={args.epsilon})" if args.policy == "heuristic" else "")
      + " (both paddles)...")


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
        pa.array(buf["player_y"], type=pa.float32()),
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
    if not env.agents:                      # game over (someone hit 21) -> new game
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

    # 2. Choose actions. Heuristic: both paddles chase the ball's y, so the ball is
    #    actually returned -> rallies/interceptions in the data (with epsilon random
    #    mixed in for coverage).
    ram_now = ale.getRAM()
    a_jepa = choose_action(ram_now, AGENT_JEPA)
    a_human = choose_action(ram_now, AGENT_HUMAN)

    # 3. Save aligned transition.
    buf["action"].append(a_jepa)
    buf["action_human"].append(a_human)
    buf["pixels"].append(encode_frame(frame))
    for k in ["player_y", "enemy_y", "ball_x", "ball_y"]:
        buf[k].append(float(s[k]))

    # 4. Advance FRAMESKIP ALE frames holding the same actions (-> fs4 stride).
    for _ in range(args.frameskip):
        if not env.agents:
            break
        env.step({a: (a_jepa if a == AGENT_JEPA else a_human) for a in env.agents})
    current_frames += 1

flush_episode()
lance.write_dataset(batches, args.out, schema=schema, mode="overwrite")
print(f"Done! {lance.dataset(args.out).count_rows()} transitions saved to {args.out}")
