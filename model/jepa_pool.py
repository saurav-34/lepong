"""CNN JEPA world model for Pong — pixel encoder for 128x128 RGB frames.

Architecture:
  - PixelEncoder: CNN for 128x128 RGB images -> 192-dim embedding
  - ActionEncoder: 2-float [left_dy, right_dy] -> 192-dim embedding
  - ARPredictor: 6-layer causal transformer with AdaLN-zero conditioning
  - state_head: Linear(192, state_dim) readout, trained jointly (--state-lambda)
    and refit on the frozen backbone in stage 2 (scripts/train_statehead.py)

13M parameters total. Runs at ~20fps on CPU.
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

EMBED_DIM = 192
HISTORY_SIZE = 3
PROJ_HIDDEN = 2048
ACTION_DIM = 2    # [left_paddle_dy, right_paddle_dy]
STATE_DIM = 28    # default state_head output dim (overridden per-game, e.g. Pong=10)


def resolve_state_meta(ckpt, default_dim=0):
    """Normalize state-head metadata across the two checkpoint conventions.

    train_statehead.py (stage 2) saves ``state_dim`` and ``state_names``; the
    jointly-trained checkpoint from this module (stage 1) omits ``state_dim`` and
    stores the column list as ``state_cols``. Loaders must accept either so a raw
    ``lepong_*_v1.pt`` and a refit ``*_statehead.pt`` are interchangeable.

    Returns (state_dim, state_names_or_None, state_mean_or_None, state_std_or_None).
    ``state_dim`` is taken verbatim when present, else inferred from the length of
    the saved stats / names, else ``default_dim``.
    """
    mean = ckpt.get("state_mean")
    std = ckpt.get("state_std")
    names = ckpt.get("state_names") or ckpt.get("state_cols")
    dim = ckpt.get("state_dim")
    if not dim:
        if mean is not None:
            dim = len(mean)
        elif names is not None:
            dim = len(names)
        else:
            dim = default_dim
    return dim, names, mean, std


# ---------------------------------------------------------------------------
# SIGReg — spectral implicit Gaussian regularization
# ---------------------------------------------------------------------------

class SIGReg(nn.Module):
    def __init__(self, knots=17, num_proj=1024):
        super().__init__()
        self.num_proj = num_proj
        t = torch.linspace(0, 3, knots)
        dt = 3 / (knots - 1)
        weights = torch.full((knots,), 2 * dt)
        weights[[0, -1]] = dt
        window = torch.exp(-t.square() / 2.0)
        self.register_buffer("t", t)
        self.register_buffer("phi", window)
        self.register_buffer("weights", weights * window)

    def forward(self, proj):
        A = torch.randn(proj.size(-1), self.num_proj, device=proj.device)
        A = A.div_(A.norm(p=2, dim=0))
        x_t = (proj @ A).unsqueeze(-1) * self.t
        err = (x_t.cos().mean(-3) - self.phi).square() + x_t.sin().mean(-3).square()
        statistic = (err @ self.weights) * proj.size(-2)
        return statistic.mean()


# ---------------------------------------------------------------------------
# Projector MLP with BatchNorm
# ---------------------------------------------------------------------------

class ProjectorMLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=PROJ_HIDDEN):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        return self.net(x)


# ---------------------------------------------------------------------------
# Custom Attention (heads=16, dim_head=64, inner_dim=1024)
# ---------------------------------------------------------------------------

class Attention(nn.Module):
    def __init__(self, dim, heads=16, dim_head=64, dropout=0.1):
        super().__init__()
        self.heads = heads
        self.dim_head = dim_head
        inner_dim = heads * dim_head
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Linear(inner_dim, dim)
        self.dropout = nn.Dropout(dropout)
        self.scale = dim_head ** -0.5

    def forward(self, x, causal=False):
        B, T, _ = x.shape
        qkv = self.to_qkv(x).reshape(B, T, 3, self.heads, self.dim_head)
        q, k, v = qkv.unbind(2)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=causal,
                                              dropout_p=self.dropout.p if self.training else 0.0)
        out = out.transpose(1, 2).reshape(B, T, -1)
        return self.to_out(out)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim), nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class ConditionalBlock(nn.Module):
    def __init__(self, dim, heads=16, dim_head=64, ff_dim=2048, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False)
        self.attn = Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False)
        self.ff = FeedForward(dim, ff_dim, dropout)
        self.adaLN = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim))
        nn.init.zeros_(self.adaLN[-1].weight)
        nn.init.zeros_(self.adaLN[-1].bias)

    def forward(self, x, c, causal=False):
        s1, sh1, g1, s2, sh2, g2 = self.adaLN(c).chunk(6, dim=-1)
        h = self.norm1(x) * (1 + s1) + sh1
        h = self.attn(h, causal=causal)
        x = x + g1 * h
        h = self.norm2(x) * (1 + s2) + sh2
        h = self.ff(h)
        x = x + g2 * h
        return x


# ---------------------------------------------------------------------------
# Pixel Encoder — CNN for 128x128 RGB
# ---------------------------------------------------------------------------

class PixelEncoder(nn.Module):
    """Encode 128x128 RGB image -> hidden_dim vector."""

    def __init__(self, hidden_dim=EMBED_DIM):
        super().__init__()
        self.convs = nn.Sequential(
            nn.Conv2d(3, 32, 4, stride=2, padding=1),    # 64x64
            nn.BatchNorm2d(32), nn.GELU(),
            nn.Conv2d(32, 64, 4, stride=2, padding=1),   # 32x32
            nn.BatchNorm2d(64), nn.GELU(),
            nn.Conv2d(64, 128, 4, stride=2, padding=1),  # 16x16
            nn.BatchNorm2d(128), nn.GELU(),
            nn.Conv2d(128, hidden_dim, 4, stride=2, padding=1),  # 8x8
            nn.BatchNorm2d(hidden_dim), nn.GELU(),
        )
        # Global average pooling alone dilutes a small bright object (e.g. a
        # 1-2px Atari ball) across the full 8x8=64-cell grid: its contribution
        # to the mean is ~1/64 of one cell's activation. AdaptiveMaxPool2d
        # keeps the single strongest cell's response, then pool_fuse lets the
        # model combine the peak (ball/paddle location) with the average
        # (broader scene context) instead of only ever seeing the blurred mean.
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.maxpool = nn.AdaptiveMaxPool2d(1)
        self.pool_fuse = nn.Linear(2 * hidden_dim, hidden_dim)

    def forward(self, img):
        """img: (B, 3, 128, 128) float [0,1] -> (B, hidden_dim)"""
        x = self.convs(img)
        avg = self.avgpool(x).flatten(1)
        mx = self.maxpool(x).flatten(1)
        return self.pool_fuse(torch.cat([avg, mx], dim=1))


# ---------------------------------------------------------------------------
# Action Encoder
# ---------------------------------------------------------------------------

class ActionEncoder(nn.Module):
    """Discrete when num_actions > 0 (ALE: 6 for pong_v3, 18 for tennis_v3),
    otherwise an MLP over a continuous action vector (synthetic PongWorld)."""

    def __init__(self, action_dim=ACTION_DIM, embed_dim=EMBED_DIM, num_actions=0):
        super().__init__()
        self.num_actions = num_actions
        if num_actions > 0:
            self.embed = nn.Embedding(num_actions, embed_dim)
        else:
            self.net = nn.Sequential(
                nn.Linear(action_dim, 4 * embed_dim), nn.SiLU(),
                nn.Linear(4 * embed_dim, embed_dim),
            )

    def forward(self, action):
        """action: (..., action_dim) float, or (...) long when discrete."""
        if self.num_actions > 0:
            return self.embed(action.long())
        return self.net(action)


# ---------------------------------------------------------------------------
# ARPredictor — causal transformer with AdaLN-zero
# ---------------------------------------------------------------------------

class ARPredictor(nn.Module):
    def __init__(self, embed_dim=EMBED_DIM, heads=16, dim_head=64, n_layers=6,
                 ff_dim=2048, dropout=0.1, max_len=HISTORY_SIZE):
        super().__init__()
        self.pos_embed = nn.Parameter(torch.randn(1, max_len, embed_dim) * 0.02)
        self.blocks = nn.ModuleList([
            ConditionalBlock(embed_dim, heads=heads, dim_head=dim_head,
                             ff_dim=ff_dim, dropout=dropout)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x, cond):
        T = x.size(1)
        x = x + self.pos_embed[:, :T]
        for block in self.blocks:
            x = block(x, cond, causal=True)
        return self.norm(x)


# ---------------------------------------------------------------------------
# Full JEPA World Model
# ---------------------------------------------------------------------------

class JEPAPool(nn.Module):
    def __init__(self, embed_dim=EMBED_DIM, heads=16, dim_head=64,
                 n_layers=6, ff_dim=2048, state_dim=0, num_actions=0):
        """
        state_dim: if > 0, adds an auxiliary state head that maps the predictor's
                   output embedding directly to a structured state vector. Trained
                   jointly with the predictor via L_total = L_embed + lambda * L_state.
                   When state_dim = 0 the model has no state head.
        num_actions: > 0 selects a discrete action embedding (ALE envs).
        """
        super().__init__()
        self.encoder = PixelEncoder(hidden_dim=embed_dim)
        self.projector = ProjectorMLP(embed_dim, embed_dim, hidden_dim=PROJ_HIDDEN)
        self.action_encoder = ActionEncoder(embed_dim=embed_dim, num_actions=num_actions)
        self.predictor = ARPredictor(embed_dim, heads=heads, dim_head=dim_head,
                                      n_layers=n_layers, ff_dim=ff_dim)
        self.pred_projector = ProjectorMLP(embed_dim, embed_dim, hidden_dim=PROJ_HIDDEN)
        self.sigreg = SIGReg()

        # Auxiliary state head
        self.state_dim = state_dim
        if state_dim > 0:
            self.state_head = nn.Linear(embed_dim, state_dim)
        else:
            self.state_head = None

    def encode(self, frames):
        """frames: (B, T, 3, H, W) float [0,1] -> (B, T, embed_dim)"""
        B, T = frames.shape[:2]
        flat = frames.reshape(B * T, *frames.shape[2:])
        h = self.encoder(flat)
        e = self.projector(h)
        return e.reshape(B, T, -1)

    def encode_actions(self, actions):
        """actions: (B, T, action_dim) float, or (B, T) long when discrete -> (B, T, embed_dim)"""
        if self.action_encoder.num_actions > 0:
            return self.action_encoder(actions)
        B, T = actions.shape[:2]
        return self.action_encoder(actions.reshape(B * T, -1)).reshape(B, T, -1)

    def forward(self, frames, actions, states=None):
        """
        frames: (B, T, 3, H, W) -- T = history_size + 1 = 4
        actions: (B, T, 2) float, or (B, T) long when discrete
        states: (B, T, state_dim) -- ground-truth state per frame (optional)
        """
        B, T = frames.shape[:2]
        emb = self.encode(frames)

        action_emb = self.encode_actions(actions)

        ctx_emb = emb[:, :HISTORY_SIZE]
        ctx_action = action_emb[:, :HISTORY_SIZE]

        pred_raw = self.predictor(ctx_emb, ctx_action)
        pred_proj = self.pred_projector(pred_raw.reshape(B * HISTORY_SIZE, -1))
        pred_emb = pred_proj.reshape(B, HISTORY_SIZE, -1)

        tgt_emb = emb[:, 1:HISTORY_SIZE + 1]

        pred_loss = (pred_emb - tgt_emb).pow(2).mean()
        sigreg_loss = self.sigreg(emb.transpose(0, 1))

        # Auxiliary state head -- predicts NEXT-frame state from the predictor output
        if self.state_head is not None and states is not None:
            state_pred = self.state_head(pred_emb.reshape(B * HISTORY_SIZE, -1))
            state_pred = state_pred.reshape(B, HISTORY_SIZE, -1)
            tgt_states = states[:, 1:HISTORY_SIZE + 1]
            state_loss = (state_pred - tgt_states).pow(2).mean()
            return pred_loss, sigreg_loss, pred_emb, tgt_emb, state_loss, state_pred, tgt_states

        return pred_loss, sigreg_loss, pred_emb, tgt_emb

    def predict_next(self, ctx_emb, ctx_action):
        was_training = self.pred_projector.training
        self.pred_projector.eval()
        pred_raw = self.predictor(ctx_emb, ctx_action)
        pred_proj = self.pred_projector(pred_raw[:, -1])
        if was_training:
            self.pred_projector.train()
        return pred_proj

    def predict_state(self, ctx_emb, ctx_action):
        """Predict next embedding via predictor, then read state via state_head.

        ctx_emb: (B, HISTORY_SIZE, embed_dim)
        ctx_action: (B, HISTORY_SIZE, embed_dim)
        Returns: (B, state_dim) state predictions
        """
        if self.state_head is None:
            raise RuntimeError("predict_state requires state_head -- instantiate JEPAPool with state_dim > 0")
        pred = self.predict_next(ctx_emb, ctx_action)  # (B, embed_dim)
        return self.state_head(pred)                    # (B, state_dim)

    def rollout(self, seed_frames, seed_actions, future_actions, n_steps):
        emb = self.encode(seed_frames)

        action_emb_list = list(self.encode_actions(seed_actions).unbind(1))
        emb_list = list(emb.unbind(1))

        predictions = []
        for t in range(n_steps):
            ctx = torch.stack(emb_list[-HISTORY_SIZE:], dim=1)
            ctx_a = torch.stack(action_emb_list[-HISTORY_SIZE:], dim=1)
            pred = self.predict_next(ctx, ctx_a)
            predictions.append(pred)
            emb_list.append(pred)
            fa = self.action_encoder(future_actions[:, t])
            action_emb_list.append(fa)

        return torch.stack(predictions, dim=1)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import time

    from model.lance_data import WindowBatcher, load_arrays, make_windows

    pa = argparse.ArgumentParser(description="Train JEPA arcade world model (stage 1)")
    pa.add_argument("--data", required=True, help="Path to a .lance dataset")
    pa.add_argument("--limit", type=int, default=None, help="Use only the first N rows")
    pa.add_argument("--epochs", type=int, default=100)
    pa.add_argument("--batch-size", type=int, default=256)
    pa.add_argument("--checkpoint", default="checkpoints/jepa_arcade.pt")
    pa.add_argument("--device", default="cuda")
    pa.add_argument("--lr", type=float, default=5e-5)
    pa.add_argument("--decode-threads", type=int, default=16)
    pa.add_argument("--sigreg-lambda", type=float, default=0.05,
                    help="Weight on the SIGReg term (was hardcoded 0.09)")
    pa.add_argument("--state-lambda", type=float, default=1.0,
                    help="Weight on the auxiliary state-head loss. Ground-truth ball/paddle "
                         "state is known at collection time -- supervising it jointly during "
                         "stage 1 gives the encoder a direct gradient for small/sparse objects "
                         "like the ball, instead of relying on the self-supervised pred+SIGReg "
                         "objective to discover them incidentally.")
    pa.add_argument("--wandb", action="store_true", help="Log metrics to Weights & Biases")
    pa.add_argument("--wandb-project", default="lepong")
    pa.add_argument("--wandb-entity", default="ssaurav3425-iiser-bhopal")
    pa.add_argument("--wandb-run-name", default=None)
    args = pa.parse_args()

    device = torch.device(args.device if args.device != "auto" else
                          ("cuda" if torch.cuda.is_available() else "cpu"))
    amp = device.type == "cuda"

    wandb = None
    if args.wandb:
        import wandb
        wandb.init(project=args.wandb_project, entity=args.wandb_entity,
                   name=args.wandb_run_name, config=vars(args))

    t0 = time.time()
    frames, actions, states, episode, meta = load_arrays(
        args.data, limit=args.limit, decode_threads=args.decode_threads)
    windows = make_windows(episode, HISTORY_SIZE)
    state_dim = states.shape[1]
    print(f"{meta['game']}: {len(windows)} windows from {len(frames)} frames "
          f"({frames.numel() / 1e9:.1f} GB uint8, decoded in {time.time() - t0:.0f}s)")
    print(f"  num_actions={meta['num_actions']}  state_dim={state_dim} {meta['state_cols']}")
    if wandb is not None:
        wandb.config.update({"game": meta["game"], "num_actions": meta["num_actions"],
                             "state_dim": state_dim, "state_cols": meta["state_cols"],
                             "n_windows": len(windows), "n_frames": len(frames)})

    # Normalise states once, on the frames the windows actually touch.
    s_mean = states[windows].mean(0)
    s_std = states[windows].std(0).clamp(min=1e-6)
    states_n = (states - s_mean) / s_std

    batcher = WindowBatcher(frames, actions, states_n, windows, HISTORY_SIZE, device)

    model = JEPAPool(num_actions=meta["num_actions"], state_dim=state_dim).to(device)
    model.encoder = model.encoder.to(memory_format=torch.channels_last)
    print(f"JEPAPool: {sum(p.numel() for p in model.parameters()):,} params on {device}")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-3, fused=amp)
    warmup_epochs = 5
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs - warmup_epochs)
    SIGREG_LAMBDA = args.sigreg_lambda
    STATE_LAMBDA = args.state_lambda
    bs = args.batch_size
    n_win = len(windows)

    print(f"\n=== JEPA Training: {args.epochs} epochs  sigreg_lambda={SIGREG_LAMBDA}  "
          f"state_lambda={STATE_LAMBDA} ===")
    for epoch in range(args.epochs):
        perm = torch.randperm(n_win)
        e_pred, e_reg, e_state, n = 0.0, 0.0, 0.0, 0
        model.train()
        t_ep = time.time()

        for i in range(0, n_win, bs):
            sel = perm[i:i + bs]
            xb, ab, sb = batcher.batch(sel)

            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
                pred_loss, sigreg_loss, _, _, state_loss, _, _ = model(xb, ab, sb)
                loss = pred_loss + SIGREG_LAMBDA * sigreg_loss + STATE_LAMBDA * state_loss

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            e_pred += pred_loss.item() * len(sel)
            e_reg += sigreg_loss.item() * len(sel)
            e_state += state_loss.item() * len(sel)
            n += len(sel)

        if epoch < warmup_epochs:
            for pg in opt.param_groups:
                pg["lr"] = args.lr * (epoch + 1) / warmup_epochs
        else:
            scheduler.step()

        print(f"Epoch {epoch+1}/{args.epochs}  L_pred={e_pred/n:.4f}  "
              f"SIGReg={e_reg/n:.4f}  L_state={e_state/n:.4f}  lr={opt.param_groups[0]['lr']:.2e}  "
              f"({time.time() - t_ep:.0f}s)", flush=True)
        if wandb is not None:
            wandb.log({"epoch": epoch + 1, "train/pred_loss": e_pred / n,
                       "train/sigreg_loss": e_reg / n, "train/state_loss": e_state / n,
                       "train/total_loss": e_pred / n + SIGREG_LAMBDA * e_reg / n + STATE_LAMBDA * e_state / n,
                       "lr": opt.param_groups[0]["lr"], "epoch_time_s": time.time() - t_ep})

        if (epoch + 1) % 10 == 0:
            os.makedirs(os.path.dirname(args.checkpoint) or ".", exist_ok=True)
            torch.save({"model": model.state_dict(), "epoch": epoch + 1,
                        "embed_dim": EMBED_DIM, "num_actions": meta["num_actions"],
                        "game": meta["game"]}, args.checkpoint)

    os.makedirs(os.path.dirname(args.checkpoint) or ".", exist_ok=True)
    torch.save({"model": model.state_dict(), "epoch": args.epochs,
                "embed_dim": EMBED_DIM, "num_actions": meta["num_actions"],
                "game": meta["game"]}, args.checkpoint)

    # =====================================================
    # State readout diagnostic -- corr of the (already-trained) joint state_head
    # on frozen predicted latents. This is the SAME linear head deployed at play
    # time (predict_state -> state_head), so its per-column corr is the honest
    # signal; no separate probe is trained. Stage 2 (train_statehead.py) is the
    # one place that refits this head to convergence on the frozen backbone.
    # =====================================================
    print("\n=== State readout diagnostic (joint state_head, frozen backbone) ===")
    model.eval()
    print("Encoding predicted latents...")
    all_emb, all_tgt = [], []
    with torch.no_grad():
        for i in range(0, n_win, bs):
            sel = torch.arange(i, min(i + bs, n_win))
            xb, ab, sb = batcher.batch(sel)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
                emb = model.encode(xb)
                ctx_a = model.encode_actions(ab[:, :HISTORY_SIZE])
                pred = model.predict_next(emb[:, :HISTORY_SIZE], ctx_a)
            all_emb.append(pred.float())
            all_tgt.append(sb[:, -1])          # state at the window's last frame
    all_emb = torch.cat(all_emb)
    all_tgt = torch.cat(all_tgt)

    # Held-out split, comparable to stage 2's val corrs.
    n_tr = int(len(all_emb) * 0.9)
    with torch.no_grad():
        p = model.state_head(all_emb[n_tr:]).cpu().numpy()
        t = all_tgt[n_tr:].cpu().numpy()
    for i, name in enumerate(meta["state_cols"]):
        print(f"  {name:10s}: corr={np.corrcoef(p[:, i], t[:, i])[0, 1]:+.3f}")
    if wandb is not None:
        wandb.log({f"corr/{name}": np.corrcoef(p[:, i], t[:, i])[0, 1]
                   for i, name in enumerate(meta["state_cols"])})

    # Save the deploy-format keys the servers require (tennis_core/pong_core/infer
    # load_frozen_model reads state_dim + state_names + state_mean/std). The joint
    # state_head is trained here on states normalized by exactly these s_mean/s_std,
    # so this checkpoint is directly playable WITHOUT a stage-2 refit -- stage 2
    # (train_statehead.py) only adds a frozen-backbone refit + keep_live filtering.
    ckpt = torch.load(args.checkpoint, weights_only=False)
    ckpt.update({"state_mean": s_mean, "state_std": s_std,
                 "state_dim": state_dim,
                 "state_names": meta["state_cols"],
                 "state_cols": meta["state_cols"]})
    torch.save(ckpt, args.checkpoint)
    print(f"State stats saved -> {args.checkpoint} (directly playable)")

    print("\n=== Rollout Test ===")
    with torch.no_grad():
        xb, ab, _ = batcher.batch(torch.arange(1))
        future_a = torch.randint(0, meta["num_actions"], (1, 10), device=device)
        preds = model.rollout(xb[:, :HISTORY_SIZE], ab[:, :HISTORY_SIZE], future_a, n_steps=10)
        print(f"Rollout: {tuple(preds.shape)}  std={preds.std():.4f}  "
              f"drift={(preds[:, 0] - preds[:, -1]).norm():.4f}")

    if wandb is not None:
        wandb.log({"rollout/std": preds.std().item(),
                   "rollout/drift": (preds[:, 0] - preds[:, -1]).norm().item()})
        wandb.finish()
