"""Goal-conditioned controller rollout test.

For each starting state drawn from a trajectory, runs the JEPA paddle policy
for N server ticks and measures whether the AI paddle successfully returns
the ball.

Success criterion: ball_vx sign flips from negative to positive (AI made
contact) before ball_x <= 0 (scored against AI).

Reports success rate in-distribution (AI-tracked policy data) vs
out-of-distribution (random-policy data).

Usage:
    python scripts/eval_controller.py \
        --checkpoint checkpoints/lepong_statehead_frozen.pt \
        --trials 60 --horizon 20 --output results/eval_controller.json
"""
import argparse
import json
import pathlib
import time

import numpy as np
import torch

from model.jepa_pool import JEPAPool, EMBED_DIM, HISTORY_SIZE
from model.pong_world import PongWorld, PADDLE_H, PADDLE_W, PADDLE_MARGIN


COURT_H = 0.6
BALL_SPEED_MAX = 0.025


def clone_env_state(env: PongWorld) -> dict:
    return {
        "ball_x": float(env.ball_x),
        "ball_y": float(env.ball_y),
        "ball_vx": float(env.ball_vx),
        "ball_vy": float(env.ball_vy),
        "paddle_l": float(env.paddle_l),
        "paddle_r": float(env.paddle_r),
        "score_l": int(env.score_l),
        "score_r": int(env.score_r),
        "rally": int(env.rally),
    }


def restore_env_state(env: PongWorld, state: dict) -> None:
    env.ball_x = state["ball_x"]
    env.ball_y = state["ball_y"]
    env.ball_vx = state["ball_vx"]
    env.ball_vy = state["ball_vy"]
    env.paddle_l = state["paddle_l"]
    env.paddle_r = state["paddle_r"]
    env.score_l = state["score_l"]
    env.score_r = state["score_r"]
    env.rally = state["rally"]


def collect_test_states(
    policy: str,
    n_episodes: int,
    steps_per_ep: int,
    frameskip: int,
    base_seed: int,
    hard_mode: bool = True,
) -> list:
    """Generate trajectories and sample hard test states.

    A test state is "hard" when the paddle quality actually matters:
      1. Ball is moving toward the AI paddle (ball_vx < 0)
      2. Ball is close enough that tracking time is limited (ball_x < 0.5)
      3. The paddle is far from the ball's current y (|padL - ball_y| > 0.08)
      4. Ball is not already past the paddle (ball_x > 0.07)
    """
    env = PongWorld()
    rng = np.random.default_rng(base_seed)
    test_states = []

    for ep in range(n_episodes):
        env.reset(seed=base_seed + ep)
        noise = rng.uniform(0.0, 0.15)

        frame_history = []

        for step in range(steps_per_ep):
            if policy == "ai":
                action = env.ai_action(noise=noise)
            elif policy == "random":
                action = [float(rng.uniform(-1, 1)), float(rng.uniform(-1, 1))]
            else:
                raise ValueError(f"unknown policy: {policy}")

            for _ in range(frameskip):
                env.step(action)

            frame_history.append(env.render(128))
            if len(frame_history) > HISTORY_SIZE:
                frame_history.pop(0)

            if len(frame_history) != HISTORY_SIZE:
                continue
            if env.ball_vx >= 0:
                continue
            if env.ball_x <= 0.07 or env.ball_x >= 0.95:
                continue
            if hard_mode:
                if env.ball_x > 0.5:
                    continue
                if abs(env.paddle_l - env.ball_y) < 0.08:
                    continue

            test_states.append({
                "env_state": clone_env_state(env),
                "frames": [f.copy() for f in frame_history],
                "episode": ep,
                "step": step,
                "policy": policy,
            })

    return test_states


def load_frozen_model(checkpoint_path: str, device: torch.device):
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dim = ckpt.get("state_dim", 10)
    model = JEPAPool(embed_dim=ckpt.get("embed_dim", EMBED_DIM), state_dim=state_dim)
    model.load_state_dict(ckpt["model"])
    for p in model.parameters():
        p.requires_grad = False
    model.eval()
    model = model.to(device)
    state_mean = ckpt.get("state_mean", torch.zeros(state_dim)).to(device)
    state_std = ckpt.get("state_std", torch.ones(state_dim)).to(device)


def jepa_paddle_policy(
    model: JEPAPool,
    device: torch.device,
    state_mean: torch.Tensor,
    state_std: torch.Tensor,
    ctx_frames: list,
    ctx_action_embs: list,
) -> float:
    """Given the current HISTORY_SIZE frames, return predicted ball_y."""
    ctx_np = np.stack(ctx_frames, axis=0)
    ctx_tensor = (
        torch.from_numpy(ctx_np)
        .float()
        .permute(0, 3, 1, 2)
        .unsqueeze(0)
        / 255.0
    ).to(device)
    with torch.no_grad():
        emb = model.encode(ctx_tensor)
        action_emb = torch.stack(ctx_action_embs, dim=0).unsqueeze(0)
        pred = model.predict_next(emb, action_emb)
        s_norm = model.state_head(pred)[0]
        s = s_norm * state_std + state_mean
        pred_ball_y_norm = float(s[1].cpu().item())
    return pred_ball_y_norm * COURT_H


def run_rollout(
    test_state: dict,
    model: JEPAPool,
    device: torch.device,
    state_mean: torch.Tensor,
    state_std: torch.Tensor,
    horizon: int,
    frameskip: int,
) -> dict:
    """Replay the JEPA policy from a starting state for `horizon` server ticks."""
    env = PongWorld()
    env.reset(seed=0)
    restore_env_state(env, test_state["env_state"])

    ctx_frames = [f.copy() for f in test_state["frames"]]

    zero_action_emb = None
    with torch.no_grad():
        if zero_action_emb is None:
            zero_action = torch.zeros(1, 2, device=device)
            zero_action_emb = model.action_encoder(zero_action)[0]
    ctx_action_embs = [zero_action_emb.clone() for _ in range(HISTORY_SIZE)]

    start_ball_vx = float(env.ball_vx)
    start_rally = int(env.rally)

    returned = False
    scored_against_ai = False
    ball_never_reached = False

    for tick in range(horizon):
        target_y = jepa_paddle_policy(
            model, device, state_mean, state_std, ctx_frames, ctx_action_embs
        )
        target_y = max(PADDLE_H / 2, min(COURT_H - PADDLE_H / 2, target_y))

        for _ in range(frameskip):
            PAD_SPEED = 0.018
            delta = max(-1, min(1, (target_y - env.paddle_l) / PAD_SPEED))
            dl = delta * PAD_SPEED * 0.95
            env.paddle_l = max(
                PADDLE_H / 2,
                min(COURT_H - PADDLE_H / 2, env.paddle_l + dl),
            )
            env.paddle_r = max(
                PADDLE_H / 2,
                min(COURT_H - PADDLE_H / 2, float(env.ball_y)),
            )
            env.step([0.0, 0.0])

            if env.ball_x < 0:
                scored_against_ai = True
                break
            if env.ball_vx > 0 and start_ball_vx < 0:
                returned = True
                break

        ctx_frames.append(env.render(128))
        if len(ctx_frames) > HISTORY_SIZE:
            ctx_frames.pop(0)

        if scored_against_ai or returned:
            break

    if not returned and not scored_against_ai:
        ball_never_reached = True

    return {
        "returned": returned,
        "scored_against_ai": scored_against_ai,
        "ball_never_reached": ball_never_reached,
        "final_ball_x": float(env.ball_x),
        "final_ball_y": float(env.ball_y),
        "final_padL": float(env.paddle_l),
        "rally_delta": int(env.rally) - start_rally,
    }


def summarize_rollouts(results: list) -> dict:
    total = len(results)
    returned = sum(1 for r in results if r["returned"])
    scored = sum(1 for r in results if r["scored_against_ai"])
    timed_out = sum(1 for r in results if r["ball_never_reached"])
    return {
        "total": total,
        "returned": returned,
        "returned_pct": returned / max(total, 1) * 100,
        "scored_against_ai": scored,
        "scored_against_ai_pct": scored / max(total, 1) * 100,
        "timed_out": timed_out,
        "timed_out_pct": timed_out / max(total, 1) * 100,
    }


def main():
    pa = argparse.ArgumentParser()
    pa.add_argument("--checkpoint", default="checkpoints/lepong_statehead_frozen.pt")
    pa.add_argument("--episodes", type=int, default=40)
    pa.add_argument("--steps", type=int, default=100)
    pa.add_argument("--frameskip", type=int, default=5)
    pa.add_argument("--horizon", type=int, default=20,
                    help="Server ticks (6 Hz) to run the policy per trial")
    pa.add_argument("--max-trials", type=int, default=200)
    pa.add_argument("--seed", type=int, default=2349867)
    pa.add_argument("--output", default="results/eval_controller.json")
    args = pa.parse_args()

    device = torch.device("cpu")
    print("=== Controller rollout OOD evaluation ===", flush=True)
    print(f"  checkpoint: {args.checkpoint}", flush=True)
    print(f"  episodes:   {args.episodes}", flush=True)
    print(f"  steps/ep:   {args.steps}", flush=True)
    print(f"  horizon:    {args.horizon} server ticks = "
          f"{args.horizon * args.frameskip} physics steps", flush=True)

    # 1. Collect test states from both policies
    print("\nCollecting in-distribution test states (AI-tracked paddles)...", flush=True)
    t0 = time.time()
    indist_states = collect_test_states(
        policy="ai",
        n_episodes=args.episodes,
        steps_per_ep=args.steps,
        frameskip=args.frameskip,
        base_seed=args.seed,
    )
    print(f"  {len(indist_states)} viable test states in {time.time() - t0:.1f}s", flush=True)

    print("\nCollecting out-of-distribution test states (random paddles)...", flush=True)
    t1 = time.time()
    ood_states = collect_test_states(
        policy="random",
        n_episodes=args.episodes,
        steps_per_ep=args.steps,
        frameskip=args.frameskip,
        base_seed=args.seed + 10000,
    )
    print(f"  {len(ood_states)} viable test states in {time.time() - t1:.1f}s", flush=True)

    # Subsample to --max-trials per condition for runtime
    rng = np.random.default_rng(args.seed + 20000)
    if len(indist_states) > args.max_trials:
        indices = rng.choice(len(indist_states), args.max_trials, replace=False)
        indist_states = [indist_states[i] for i in indices]
    if len(ood_states) > args.max_trials:
        indices = rng.choice(len(ood_states), args.max_trials, replace=False)
        ood_states = [ood_states[i] for i in indices]
    print(f"\nRunning {len(indist_states)} in-dist rollouts + "
          f"{len(ood_states)} OOD rollouts at horizon={args.horizon}", flush=True)

    # 2. Load checkpoint
    print("\nLoading frozen checkpoint...", flush=True)
    model, state_mean, state_std = load_frozen_model(args.checkpoint, device)

    # 3. Run rollouts
    print("\nRunning in-distribution controller rollouts...", flush=True)
    t2 = time.time()
    indist_results = []
    for i, ts in enumerate(indist_states):
        r = run_rollout(ts, model, device, state_mean, state_std,
                        args.horizon, args.frameskip)
        indist_results.append(r)
        if (i + 1) % 20 == 0:
            so_far = summarize_rollouts(indist_results)
            print(f"  {i + 1}/{len(indist_states)}: returned={so_far['returned_pct']:.0f}%", flush=True)
    print(f"  done in {time.time() - t2:.1f}s", flush=True)

    print("\nRunning OOD controller rollouts...", flush=True)
    t3 = time.time()
    ood_results = []
    for i, ts in enumerate(ood_states):
        r = run_rollout(ts, model, device, state_mean, state_std,
                        args.horizon, args.frameskip)
        ood_results.append(r)
        if (i + 1) % 20 == 0:
            so_far = summarize_rollouts(ood_results)
            print(f"  {i + 1}/{len(ood_states)}: returned={so_far['returned_pct']:.0f}%", flush=True)
    print(f"  done in {time.time() - t3:.1f}s", flush=True)

    # 4. Report
    ind_summary = summarize_rollouts(indist_results)
    ood_summary = summarize_rollouts(ood_results)

    print("\n=== IN-DISTRIBUTION (AI-tracked trajectories) ===", flush=True)
    print(f"  total trials:       {ind_summary['total']}", flush=True)
    print(f"  returned the ball:  {ind_summary['returned']} ({ind_summary['returned_pct']:.1f}%)", flush=True)
    print(f"  scored against AI:  {ind_summary['scored_against_ai']} ({ind_summary['scored_against_ai_pct']:.1f}%)", flush=True)
    print(f"  timed out:          {ind_summary['timed_out']} ({ind_summary['timed_out_pct']:.1f}%)", flush=True)

    print("\n=== OUT-OF-DISTRIBUTION (random-policy trajectories) ===", flush=True)
    print(f"  total trials:       {ood_summary['total']}", flush=True)
    print(f"  returned the ball:  {ood_summary['returned']} ({ood_summary['returned_pct']:.1f}%)", flush=True)
    print(f"  scored against AI:  {ood_summary['scored_against_ai']} ({ood_summary['scored_against_ai_pct']:.1f}%)", flush=True)
    print(f"  timed out:          {ood_summary['timed_out']} ({ood_summary['timed_out_pct']:.1f}%)", flush=True)

    drop_pct = ind_summary['returned_pct'] - ood_summary['returned_pct']
    drop_rel = (
        (ood_summary['returned_pct'] - ind_summary['returned_pct'])
        / max(ind_summary['returned_pct'], 1e-9) * 100
    )
    print(f"\nOOD drop: in-dist {ind_summary['returned_pct']:.1f}% -> "
          f"OOD {ood_summary['returned_pct']:.1f}%  "
          f"(absolute {-drop_pct:+.1f} pts, relative {drop_rel:+.1f}%)", flush=True)

    results = {
        "checkpoint": args.checkpoint,
        "horizon": args.horizon,
        "frameskip": args.frameskip,
        "episodes_per_condition": args.episodes,
        "steps_per_ep": args.steps,
        "max_trials_per_condition": args.max_trials,
        "seed": args.seed,
        "indist": ind_summary,
        "ood": ood_summary,
        "ood_drop_absolute_pts": -drop_pct,
        "ood_drop_relative_pct": drop_rel,
        "methodology": (
            "Goal-conditioned controller test. Test states drawn from "
            "trajectories under two policies on the same PongWorld.render(128) "
            "renderer: in-dist = AI-tracked paddles with noise(0, 0.15), "
            "OOD = uniform random paddle actions. For each viable state "
            "(ball heading toward AI, not past paddle), restore env, run "
            "JEPA paddle policy for horizon*frameskip physics steps, "
            "measure success = ball's ball_vx flipped positive (AI returned) "
            "before ball.x < 0 (scored against AI)."
        ),
    }
    pathlib.Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    pathlib.Path(args.output).write_text(json.dumps(results, indent=2))
    print(f"\nSaved -> {args.output}", flush=True)


if __name__ == "__main__":
    main()
