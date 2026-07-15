# lepong

A 13M-parameter JEPA world model plays Pong by watching pixels.

```
128x128 PNG ──> CNN Encoder ──> 192-dim embedding
                                      │
                 action (2-d) ──> ActionEncoder ──> conditioning
                                      │
                              6-layer Transformer
                              (AdaLN-zero, causal)
                                      │
                              predicted embedding
                                      │
                           Linear(192, 10) state head
                                      │
                         ball_x, ball_y, vx, vy, paddles...
```

## What this is

A Joint Embedding Predictive Architecture (JEPA) trained on 30,000 Pong frames.
The model watches 128x128 pixel screenshots of the game, predicts where the ball
will be next, and moves the left paddle to intercept it. The model receives no
game state -- only the raw pixel screenshot. Pixels in, paddle target out.

The encoder and predictor are frozen (13M params). Only a `Linear(192, 10)` state
head is trained (1,930 parameters).

## Quick start

```bash
pip install torch torchvision numpy pillow fastapi uvicorn websockets h5py

# Download checkpoints from HuggingFace
# huggingface-cli download <your-hf-user>/lepong --local-dir checkpoints/

# Run the server
python -m server.infer --checkpoint checkpoints/lepong_statehead_occ_aug.pt --port 8791

# Open http://localhost:8791 in your browser
```

## Training from scratch

```bash
# 1. Generate training data (30K frames, ~5 min)
python -m model.pong_world --episodes 300 --steps 100 --output data/pong_v1.npz

# 2. Train the JEPA encoder + predictor (~12 min on an L4 GPU)
python -m model.jepa_pool --data data/pong_v1.npz --epochs 100 \
    --checkpoint checkpoints/lepong_v1.pt --device cuda

# 3. Train the frozen state head (~3 min on an L4 GPU)
python -m scripts.train_statehead \
    --data data/pong_v1.npz \
    --init checkpoints/lepong_v1.pt \
    --output checkpoints/lepong_statehead_frozen.pt \
    --epochs 20 --batch 128 --lr 1e-3

# 4. Evaluate
python -m scripts.eval_ood --checkpoint checkpoints/lepong_statehead_frozen.pt
python -m scripts.eval_controller --checkpoint checkpoints/lepong_statehead_frozen.pt
python -m scripts.eval_occlusion --checkpoints checkpoints/lepong_statehead_frozen.pt
```

## Results

### State head prediction error (in-distribution vs OOD)

| Dimension | In-dist median | OOD median | OOD drop |
|-----------|---------------|------------|----------|
| ball_x    | 0.042         | 0.059      | +42%     |
| ball_y    | 0.028         | 0.031      | +10%     |
| ball_vx   | 0.741         | 0.816      | +10%     |
| ball_vy   | 0.194         | 0.202      | +4%      |
| pad_l     | 0.014         | 0.029      | +108%    |
| pad_r     | 0.015         | 0.058      | +285%    |

In-dist = AI-tracked paddles (matches training). OOD = random paddle actions.

### Controller rollout (ball return rate)

| Condition       | Returned | Scored against | Timed out |
|-----------------|----------|----------------|-----------|
| In-distribution | 99.3%    | 0.0%           | 0.7%      |
| OOD             | 88.7%    | 0.0%           | 11.3%     |

OOD drop: -10.7 percentage points (relative -10.7%).

## Architecture

- **PixelEncoder**: 4-layer CNN (3->32->64->128->192, stride-2 convolutions with BatchNorm + GELU, AdaptiveAvgPool)
- **ActionEncoder**: 2-layer MLP (2->768->192, SiLU activation)
- **ARPredictor**: 6-layer causal transformer with AdaLN-zero conditioning (16 heads, dim_head=64)
- **ProjectorMLP**: 2-layer MLP with BatchNorm (192->2048->192) for embedding and prediction projection
- **SIGReg**: Spectral implicit Gaussian regularization loss
- **StateHead**: `Linear(192, 10)` -- the only trainable component at inference time

Total: 13,084,010 parameters. State head: 1,930 parameters.

## Occlusion experiment

The browser demo includes an occlusion slider (0% / 20% / 40% / 60%) that blacks out
a vertical strip on the right side of the 128x128 canvas before sending it to the
server. This forces the model to predict ball position from partial observations.

The occlusion-augmented checkpoint (`lepong_statehead_occ_aug.pt`) is trained with
random occlusion during state head training and maintains stable performance up to
40% occlusion. The baseline checkpoint collapses on ball_x at 40%+ occlusion.

## Model

Available on HuggingFace (upload after training):

- `lepong_statehead_occ_aug.pt` -- shipping checkpoint (occlusion-robust)
- `lepong_statehead_frozen.pt` -- baseline (no augmentation)
- `lepong_v1.pt` -- init checkpoint (encoder + predictor only, no state head)

## License

MIT
