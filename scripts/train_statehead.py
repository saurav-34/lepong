"""Train a frozen state head on Pong data.

Loads a pretrained JEPAPool checkpoint and freezes the encoder + predictor.
Only the Linear(192, 10) state head trains (~1,930 parameters).

Usage:
    python scripts/train_statehead.py \
        --data /path/to/pong_v1.npz \
        --init /path/to/lepong_v1.pt \
        --output /path/to/lepong_statehead_frozen.pt \
        --epochs 20 --batch 128 --lr 1e-3
"""
import argparse
import pathlib
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from model.jepa_pool import JEPAPool, EMBED_DIM, HISTORY_SIZE
from model.lance_data import train_val_split


def _init_worker(_worker_id: int) -> None:
    """One intra-op thread per worker; 16 workers x N threads only thrashes."""
    torch.set_num_threads(1)


def make_loader(ds, batch_size: int, shuffle: bool, num_workers: int) -> DataLoader:
    """Build a DataLoader whose workers can safely re-enter Lance.

    Reading the dataset in the parent starts Lance's global Tokio runtime, and
    fork() clones only the calling thread: a mutex held by a runtime thread at
    fork time stays locked forever in the child, which then deadlocks on its
    first read. Opening a fresh lance.dataset() per worker does not help, since
    the runtime is process-global. Spawn gives each worker its own runtime.
    """
    kwargs = {}
    if num_workers > 0:
        kwargs = dict(
            persistent_workers=True,
            multiprocessing_context="spawn",
            worker_init_fn=_init_worker,
            prefetch_factor=4,
        )
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                      num_workers=num_workers, pin_memory=True, **kwargs)


def freeze_everything_except_state_head(model: JEPAPool) -> list[str]:
    """Flip requires_grad off on all modules except state_head.

    Returns the list of modules that were frozen, for logging and
    for saving in the checkpoint as an audit trail.
    """
    frozen_modules = []
    for p in model.encoder.parameters():
        p.requires_grad = False
    frozen_modules.append("encoder")
    for p in model.projector.parameters():
        p.requires_grad = False
    frozen_modules.append("projector")
    for p in model.action_encoder.parameters():
        p.requires_grad = False
    frozen_modules.append("action_encoder")
    for p in model.predictor.parameters():
        p.requires_grad = False
    frozen_modules.append("predictor")
    for p in model.pred_projector.parameters():
        p.requires_grad = False
    frozen_modules.append("pred_projector")
    for p in model.sigreg.parameters():
        p.requires_grad = False
    frozen_modules.append("sigreg")
    return frozen_modules


@torch.no_grad()
def precompute_features(model: JEPAPool, loader, device):
    """Run the frozen backbone once and cache exactly what the state head consumes.

    With the backbone frozen and in eval mode, and no augmentation in the loader,
    pred_emb is a pure function of the window. Recomputing it every epoch decodes
    the same PNGs and re-runs the same 13M-param CNN to get the same numbers, so
    we do it once. Also skips SIGReg, which model.forward computes and discards.

    Returns (feats (N, embed_dim), targets (N, state_dim), mean pred_loss) --
    one row per window, the last predicted position only (matching predict_next).
    pred_loss is constant, so it is reported once.
    """
    feats, targets = [], []
    pred_sum, n = 0.0, 0
    total = len(loader)
    for step, (xb, ab, sb) in enumerate(loader, 1):
        xb = xb.to(device, non_blocking=True)
        ab = ab.to(device, non_blocking=True)
        sb = sb.to(device, non_blocking=True)
        b = xb.size(0)

        emb = model.encode(xb)
        action_emb = model.encode_actions(ab)
        # Match the deployed readout (predict_next -> state_head): keep ONLY the
        # last predicted position. pred_raw[:, :-1] are predictions made from a
        # shorter causal context (1..HISTORY_SIZE-1 frames) and give noisier state
        # readouts; training/scoring on them dragged corr below jepa_pool's joint
        # diagnostic, which uses pred_raw[:, -1] alone -- and left the shipped head
        # fit to positions the play path never sees.
        pred_raw = model.predictor(emb[:, :HISTORY_SIZE], action_emb[:, :HISTORY_SIZE])
        pred_emb = model.pred_projector(pred_raw[:, -1])              # (b, D)

        pred_sum += (pred_emb - emb[:, -1]).pow(2).mean().item() * b
        n += b

        feats.append(pred_emb)
        targets.append(sb[:, -1])                                    # state at last frame

        if step == 1 or step == total or step % max(1, total // 10) == 0:
            print(f"    encoding {step}/{total} ({100.0 * step / total:5.1f}%)", flush=True)

    return torch.cat(feats), torch.cat(targets), pred_sum / n


def main():
    pa = argparse.ArgumentParser()
    pa.add_argument("--data", required=True, help="Path to a .lance dataset")
    pa.add_argument("--init", required=True, help="Init checkpoint (lepong_v1.pt)")
    pa.add_argument("--output", required=True, help="Output checkpoint path")
    pa.add_argument("--epochs", type=int, default=20)
    pa.add_argument("--batch", type=int, default=128)
    pa.add_argument("--lr", type=float, default=1e-3,
                    help="Higher than full-model lr because only ~2K params train")
    pa.add_argument("--limit", type=int, default=None,
                    help="Train on only the first N rows (e.g. 100000)")
    pa.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    pa.add_argument("--num-workers", type=int, default=8)
    pa.add_argument("--val-frac", type=float, default=0.1)
    pa.add_argument("--min-ball-x", type=float, default=None,
                    help="Drop target frames whose ball_x is below this (ALE parks the "
                         "ball at ~1-4 between points). On tennis 59%% of frames are "
                         "parked, and training on them lets the head score corr 0.81 by "
                         "separating dead-from-live while resolving the ball during a "
                         "rally at only corr 0.31. Try 5.")
    args = pa.parse_args()

    device = torch.device(args.device)
    print(f"=== train_statehead (frozen backbone) ===", flush=True)
    print(f"  device:  {device}", flush=True)
    print(f"  data:    {args.data}", flush=True)
    print(f"  init:    {args.init}", flush=True)
    print(f"  output:  {args.output}", flush=True)
    print(f"  epochs:  {args.epochs}", flush=True)
    print(f"  batch:   {args.batch}", flush=True)
    print(f"  lr:      {args.lr}", flush=True)
    print(f"  limit:   {args.limit}", flush=True)

    # Load data (lazy: PNG frames are decoded per-window in the loader workers)
    print("\nLoading data...", flush=True)
    train_ds, val_ds = train_val_split(args.data, HISTORY_SIZE,
                                       val_frac=args.val_frac, limit=args.limit)
    args.state_dim = train_ds.state_dim
    state_mean, state_std = train_ds.state_mean, train_ds.state_std
    print(f"  game: {train_ds.game}  num_actions: {train_ds.num_actions}", flush=True)
    print(f"  state cols: {train_ds.state_cols}", flush=True)
    print(f"  state mean: {state_mean.tolist()}", flush=True)
    print(f"  state std:  {state_std.tolist()}", flush=True)

    train_loader = make_loader(train_ds, args.batch, True, args.num_workers)
    val_loader = make_loader(val_ds, args.batch, False, args.num_workers)
    print(f"  train windows: {len(train_ds)}, val windows: {len(val_ds)}", flush=True)

    # Model
    print(f"\nBuilding JEPAPool with state_dim={args.state_dim}...", flush=True)
    model = JEPAPool(state_dim=args.state_dim, num_actions=train_ds.num_actions).to(device)
    n_params_total = sum(p.numel() for p in model.parameters())
    print(f"  total params: {n_params_total:,}", flush=True)

    # Init from existing checkpoint (encoder + predictor already pretrained)
    print(f"\nLoading init checkpoint: {args.init}", flush=True)
    ckpt = torch.load(args.init, map_location=device, weights_only=False)
    msg = model.load_state_dict(ckpt["model"], strict=False)
    print(f"  missing: {len(msg.missing_keys)} keys (expected: state_head)", flush=True)
    print(f"  unexpected: {len(msg.unexpected_keys)} keys", flush=True)

    # Freeze the encoder + predictor + everything except the state head
    print("\nFreezing encoder + predictor + everything except state_head...", flush=True)
    frozen_modules = freeze_everything_except_state_head(model)
    for name in frozen_modules:
        print(f"  [frozen] model.{name}", flush=True)

    # Sanity check: count trainable params
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    n_trainable = sum(p.numel() for p in trainable_params)
    n_frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    print(f"\n  trainable params: {n_trainable:,}  (should be ~2K for Linear(192, 10))", flush=True)
    print(f"  frozen params:    {n_frozen:,}  (should be ~13M)", flush=True)

    if n_trainable > 5000:
        print(f"\n  ERROR: expected ~1930 trainable params for a Linear(192, 10)"
              f" state head, got {n_trainable}. Something is wrong with the freezing.",
              flush=True)
        sys.exit(1)

    # Optimizer only sees trainable params
    opt = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=1e-3,
                            betas=(0.9, 0.95))

    state_names = train_ds.state_cols

    # Put the frozen modules into eval mode so BatchNorm running stats don't update
    model.encoder.eval()
    model.projector.eval()
    model.action_encoder.eval()
    model.predictor.eval()
    model.pred_projector.eval()

    # One pass over the pixels. After this the PNGs and the CNN are never touched
    # again, so each epoch below is a linear pass over cached embeddings.
    print("\n=== ENCODING (frozen backbone, one pass) ===", flush=True)
    t_pre = time.time()
    print("  train:", flush=True)
    tr_feats, tr_tgts, train_pred_loss = precompute_features(model, train_loader, device)
    print("  val:", flush=True)
    va_feats, va_tgts, val_pred_loss = precompute_features(model, val_loader, device)
    print(f"  train feats: {tuple(tr_feats.shape)} ({tr_feats.numel() * 4 / 1e6:.0f} MB)", flush=True)
    print(f"  val   feats: {tuple(va_feats.shape)} ({va_feats.numel() * 4 / 1e6:.0f} MB)", flush=True)
    print(f"  pred loss (frozen, constant): train={train_pred_loss:.4f} val={val_pred_loss:.4f}", flush=True)
    print(f"  encoded in {time.time() - t_pre:.0f}s", flush=True)

    if args.min_ball_x is not None:
        if "ball_x" not in state_names:
            sys.exit(f"--min-ball-x needs a ball_x column; have {state_names}")
        bi = state_names.index("ball_x")
        mu, sd = state_mean[bi].to(device), state_std[bi].to(device)

        def keep_live(feats, tgts):
            """Flatten (N, H, ·) to rows and drop parked-ball target frames."""
            f = feats.reshape(-1, feats.size(-1))
            t = tgts.reshape(-1, tgts.size(-1))
            m = (t[:, bi] * sd + mu) >= args.min_ball_x
            return f[m], t[m], int(m.sum()), m.numel()

        tr_feats, tr_tgts, ntr, dtr = keep_live(tr_feats, tr_tgts)
        va_feats, va_tgts, nva, dva = keep_live(va_feats, va_tgts)
        print(f"\n  live filter (ball_x >= {args.min_ball_x}): "
              f"train {ntr}/{dtr} ({100 * ntr / dtr:.1f}%) "
              f"val {nva}/{dva} ({100 * nva / dva:.1f}%) rows kept", flush=True)
        if ntr == 0 or nva == 0:
            sys.exit("live filter removed every row; lower --min-ball-x")

    print("\n=== TRAINING (state head only) ===", flush=True)
    t0 = time.time()
    n_train = tr_feats.size(0)

    for epoch in range(args.epochs):
        model.state_head.train()
        perm = torch.randperm(n_train, device=device)
        sum_state, n_b = 0.0, 0

        for i in range(0, n_train, args.batch):
            idx = perm[i:i + args.batch]
            fb, tb = tr_feats[idx], tr_tgts[idx]

            state_loss = (model.state_head(fb) - tb).pow(2).mean()

            opt.zero_grad()
            state_loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
            opt.step()

            sum_state += state_loss.item() * len(fb)
            n_b += len(fb)

        avg_pred = train_pred_loss
        avg_state = sum_state / n_b

        # Validation -- the whole cached val set fits in one forward
        model.state_head.eval()
        with torch.no_grad():
            val_pred_states = model.state_head(va_feats)
            val_state_loss = (val_pred_states - va_tgts).pow(2).mean().item()

        all_p = val_pred_states.reshape(-1, args.state_dim).cpu()
        all_t = va_tgts.reshape(-1, args.state_dim).cpu()
        corrs = []
        for i in range(args.state_dim):
            if all_t[:, i].std() < 1e-6:
                corrs.append(float("nan"))
            else:
                c = float(np.corrcoef(all_p[:, i].numpy(), all_t[:, i].numpy())[0, 1])
                corrs.append(c)

        elapsed = time.time() - t0
        print(f"Epoch {epoch+1:3d}/{args.epochs}  "
              f"train: pred={avg_pred:.4f} (frozen, constant) state={avg_state:.4f}  "
              f"val: pred={val_pred_loss:.4f} state={val_state_loss:.4f}  "
              f"({elapsed:.0f}s)", flush=True)
        if (epoch + 1) % 5 == 0 or epoch == 0 or epoch == args.epochs - 1:
            corr_str = " ".join(f"{n[:5]}={c:+.3f}" for n, c in zip(state_names, corrs))
            print(f"        val corrs: {corr_str}", flush=True)

    elapsed = time.time() - t0
    print(f"\n=== TRAINING DONE in {elapsed:.0f}s "
          f"(+{t0 - t_pre:.0f}s encoding) ===", flush=True)

    # Final eval
    print("\nFinal val correlations (state head reading frozen encoder + predictor):",
          flush=True)
    for i, name in enumerate(state_names):
        c = corrs[i]
        status = "GOOD" if c > 0.85 else ("OK" if c > 0.5 else "WEAK")
        print(f"  {name:10s}: {c:+.4f} [{status}]", flush=True)

    # Save checkpoint
    save_dict = {
        "model": model.state_dict(),
        "embed_dim": EMBED_DIM,
        "state_dim": args.state_dim,
        "state_mean": state_mean,
        "state_std": state_std,
        "state_names": state_names,
        # Needed to rebuild ActionEncoder: >0 selects the discrete Embedding
        # branch. Without it a loader defaults to 0 and builds the MLP branch,
        # which cannot accept this state_dict.
        "num_actions": train_ds.num_actions,
        "game": train_ds.game,
        "val_correlations": dict(zip(state_names, corrs)),
        "epochs": args.epochs,
        "lr": args.lr,
        "frozen_modules": frozen_modules,
        "trainable_param_count": n_trainable,
        "frozen_param_count": n_frozen,
        "init_checkpoint": args.init,
        "training_script": "scripts/train_statehead.py",
    }
    torch.save(save_dict, args.output)
    print(f"\nSaved checkpoint: {args.output}", flush=True)
    print(f"  size: {pathlib.Path(args.output).stat().st_size / 1e6:.1f} MB", flush=True)
    print(f"  frozen modules: {frozen_modules}", flush=True)


if __name__ == "__main__":
    main()
