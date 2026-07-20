"""Drive the frozen JEPA boxing checkpoint against the scripted opponent, headlessly.

Plays `--ticks` decision steps of boxing_v2 with JEPA on `first_0`, renders the
match to an MP4, and reports the score plus the state head's perception error
against RAM ground truth.

    python -m scripts.play_boxing --checkpoint checkpoints/boxing_R1.pt \
        --ticks 500 --out results/boxing_play.mp4

`--own ram` swaps JEPA's own (x, y) from the state head to proprioception; see
BoxingSession's docstring for why that comparison exists.
"""
import argparse

import cv2
import numpy as np
import torch

from server.boxing_core import BoxingSession


def main():
    pa_ = argparse.ArgumentParser()
    pa_.add_argument("--checkpoint", default="checkpoints/boxing_R1.pt")
    pa_.add_argument("--ticks", type=int, default=500, help="decision steps to play")
    pa_.add_argument("--out", default="results/boxing_play.mp4")
    pa_.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    pa_.add_argument("--frameskip", type=int, default=4)
    pa_.add_argument("--predict-k", type=int, default=1)
    pa_.add_argument("--own", choices=["pred", "ram"], default="pred",
                     help="source of JEPA's own (x,y): state head or RAM")
    pa_.add_argument("--opponent-epsilon", type=float, default=0.0,
                     help="random-action rate for the scripted opponent")
    pa_.add_argument("--seed", type=int, default=0)
    pa_.add_argument("--scale", type=int, default=3, help="upscale factor for the video")
    pa_.add_argument("--fps", type=int, default=15)
    pa_.add_argument("--no-video", action="store_true")
    args = pa_.parse_args()

    sess = BoxingSession(args.checkpoint, torch.device(args.device),
                         frameskip=args.frameskip, predict_k=args.predict_k,
                         seed=args.seed, own_source=args.own,
                         opponent_epsilon=args.opponent_epsilon)
    print(f"boxing_v2: JEPA=first_0 vs scripted second_0 | own={args.own} "
          f"k={args.predict_k} device={args.device}")

    writer = None
    errs = {k: [] for k in ("first_x", "first_y", "second_x", "second_y")}
    punches = 0

    for _ in range(args.ticks):
        info = sess.step()

        if info["jepa_action"] in (11, 12):
            punches += 1
        if info["pred_err"]:
            for k, v in info["pred_err"].items():
                errs[k].append(abs(v))

        if not args.no_video:
            raw = info["raw"]
            S = args.scale
            big = cv2.resize(raw, (raw.shape[1] * S, raw.shape[0] * S),
                             interpolation=cv2.INTER_NEAREST)
            big = cv2.cvtColor(big, cv2.COLOR_RGB2BGR)
            cv2.putText(big, f"JEPA {info['score']['jepa']:.0f} - "
                             f"{info['score']['opponent']:.0f} OPP",
                        (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            if writer is None:
                writer = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"),
                                         args.fps, (big.shape[1], big.shape[0]))
            writer.write(big)

    score = sess.score
    print(f"\nScore after {args.ticks} ticks: JEPA {score['first_0']:.0f} "
          f"- {score['second_0']:.0f} opponent  (net {score['first_0'] - score['second_0']:+.0f})")
    print(f"Punches thrown: {punches}/{args.ticks} ticks ({100 * punches / args.ticks:.1f}%)")
    print("\nState-head perception error vs RAM (mean abs, RAM units):")
    for k, v in errs.items():
        if v:
            print(f"  {k:10s}: {np.mean(v):5.2f}")

    if writer is not None:
        writer.release()
        print(f"\nVideo -> {args.out}")
    sess.close()


if __name__ == "__main__":
    main()
