"""Render frames from the Lance dataset to an MP4 for visual inspection.

Decodes the PNG `pixels` for one episode, upscales, and (optionally) overlays the
RAM-derived ball position + action so you can see flicker vs. the true ball.

    python -m scripts.make_video --data datasets/pong_ma_128x128.lance \
        --episode 23 --max-frames 600 --out results/pong_ep23.mp4
"""
import argparse
import io

import cv2
import lance
import numpy as np
from PIL import Image

ACTION_NAMES = {0: "NOOP", 1: "FIRE", 2: "UP", 3: "DOWN", 4: "?", 5: "?"}


def main():
    pa_ = argparse.ArgumentParser()
    pa_.add_argument("--data", default="datasets/pong_ma_128x128.lance")
    pa_.add_argument("--episode", type=int, default=23)
    pa_.add_argument("--max-frames", type=int, default=600)
    pa_.add_argument("--scale", type=int, default=4, help="upscale factor for the video")
    pa_.add_argument("--fps", type=int, default=15)
    pa_.add_argument("--out", default="results/pong_ep.mp4")
    pa_.add_argument("--no-overlay", action="store_true",
                     help="disable the ball-position/action overlay (raw frames only)")
    args = pa_.parse_args()

    ds = lance.dataset(args.data)
    tbl = ds.to_table(
        filter=f"episode_idx == {args.episode}",
        columns=["step_idx", "pixels", "action", "ball_x", "ball_y"],
    ).sort_by("step_idx")

    pixels = tbl["pixels"].to_pylist()
    actions = tbl["action"].to_pylist()
    ball_x = np.asarray(tbl["ball_x"].to_pylist())
    ball_y = np.asarray(tbl["ball_y"].to_pylist())
    n = min(len(pixels), args.max_frames)
    if n == 0:
        raise SystemExit(f"episode {args.episode} has no frames in {args.data}")

    # Decode first frame to get native size.
    first = np.array(Image.open(io.BytesIO(pixels[0])))
    h, w = first.shape[:2]
    S = args.scale
    W, H = w * S, h * S

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(args.out, fourcc, args.fps, (W, H))

    # RAM ball_x/ball_y are in the 160x210 ALE frame; map to the 128px frame.
    # ball_x: 0..160 -> 0..w ; ball_y: 0..210 -> 0..h (ball_y==0 means "no ball").
    for i in range(n):
        img = np.array(Image.open(io.BytesIO(pixels[i])))  # RGB
        big = cv2.resize(img, (W, H), interpolation=cv2.INTER_NEAREST)
        big = cv2.cvtColor(big, cv2.COLOR_RGB2BGR)

        if not args.no_overlay:
            a = actions[i]
            cv2.putText(big, f"ep{args.episode} f{i}  a={ACTION_NAMES.get(a, a)}",
                        (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
            bx, by = ball_x[i], ball_y[i]
            if by > 0:  # ball in play -> draw the RAM position as a circle
                px = int(bx / 160.0 * W)
                py = int(by / 210.0 * H)
                cv2.circle(big, (px, py), max(3, S), (0, 165, 255), 1, cv2.LINE_AA)
            else:
                cv2.putText(big, "no ball (serve)", (6, H - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 1, cv2.LINE_AA)

        out.write(big)

    out.release()
    print(f"Wrote {n} frames -> {args.out} ({W}x{H} @ {args.fps}fps)")


if __name__ == "__main__":
    main()
