"""Lance-backed windowed dataset for the PettingZoo collectors.

The collectors (scripts/collect_pong.py, scripts/collect_tennis.py) write Lance with
PNG-encoded `pixels` and a discrete int `action`. Decoding is lazy: 400K frames is
19.7 GB as uint8 and 78.6 GB as float32, so nothing is materialised up front.

Windows are `history + 1` frames that never straddle an episode boundary.
"""
from __future__ import annotations

import io
from concurrent.futures import ThreadPoolExecutor

import cv2
import lance
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

# Non-pixel float columns, in the order the state head reads them.
STATE_COLUMNS = {
    "pong": ["player_y", "enemy_y", "ball_x", "ball_y", "ball_vx", "ball_vy"],
    "tennis": ["player_x", "player_y", "enemy_x", "enemy_y",
               "ball_x", "ball_y", "ball_vx", "ball_vy"],
    "boxing": ["first_x", "first_y", "second_x", "second_y"],
}
NUM_ACTIONS = {"pong": 6, "tennis": 18, "boxing": 18}


def infer_game(path: str) -> str:
    names = set(lance.dataset(path).schema.names)
    if "first_x" in names:
        return "boxing"
    return "tennis" if "player_x" in names else "pong"


def make_windows(episode: np.ndarray, history: int) -> np.ndarray:
    """End indices i whose window [i-history, i] stays inside one episode."""
    cand = np.arange(history, len(episode))
    return cand[episode[cand] == episode[cand - history]]


def load_arrays(path: str, limit: int | None = None, decode_threads: int = 16):
    """Decode the whole (sliced) dataset once into RAM.

    Frames stay uint8 (128x128x3 = 49 KB each: 4.9 GB at 100K rows, 19.7 GB at 400K).
    Converting to float32 up front would cost 4x that, so the cast is deferred to the
    GPU. cv2.imdecode drops the GIL, so threads give a real speedup here.
    """
    game = infer_game(path)
    cols = STATE_COLUMNS[game]
    ds = lance.dataset(path)
    tbl = ds.to_table(columns=["episode_idx", "action", "pixels", *cols],
                      limit=limit)

    episode = np.asarray(tbl["episode_idx"].to_pylist(), dtype=np.int64)
    actions = torch.from_numpy(np.asarray(tbl["action"].to_pylist(), dtype=np.int64))
    states = torch.from_numpy(np.stack(
        [np.asarray(tbl[c].to_pylist(), dtype=np.float32) for c in cols], axis=1))

    blobs = tbl["pixels"].to_pylist()
    probe = cv2.imdecode(np.frombuffer(blobs[0], np.uint8), cv2.IMREAD_COLOR)
    frames = torch.empty((len(blobs), *probe.shape), dtype=torch.uint8)
    buf = frames.numpy()

    def _decode(k):
        # collectors write RGB PNGs via PIL; imdecode returns BGR -> flip back
        buf[k] = cv2.imdecode(np.frombuffer(blobs[k], np.uint8), cv2.IMREAD_COLOR)[..., ::-1]

    with ThreadPoolExecutor(decode_threads) as pool:
        list(pool.map(_decode, range(len(blobs)), chunksize=512))

    return frames, actions, states, episode, {
        "game": game, "num_actions": NUM_ACTIONS[game], "state_cols": cols,
    }


class WindowBatcher:
    """Gathers uint8 windows on CPU, normalises on GPU.

    frames[rows] materialises only (B, T, H, W, 3) uint8 (~12 MB at B=64, T=4), so the
    float32 explosion never touches host RAM.
    """

    def __init__(self, frames, actions, states, windows, history, device):
        self.frames, self.actions, self.states = frames, actions, states
        self.windows = torch.as_tensor(windows)
        self.device = device
        self.offsets = torch.arange(-history, 1)

    def __len__(self):
        return len(self.windows)

    def batch(self, sel):
        rows = self.windows[sel].unsqueeze(1) + self.offsets      # (B, T)
        f = self.frames[rows].pin_memory().to(self.device, non_blocking=True)
        f = f.permute(0, 1, 4, 2, 3).float().div_(255.0)          # (B, T, 3, H, W)
        a = self.actions[rows].to(self.device, non_blocking=True)
        s = self.states[rows].to(self.device, non_blocking=True)
        return f, a, s


class LanceWindowDataset(Dataset):
    def __init__(self, path: str, history: int, game: str | None = None,
                 indices: np.ndarray | None = None,
                 state_mean: torch.Tensor | None = None,
                 state_std: torch.Tensor | None = None):
        self.path = path
        self.history = history
        self.game = game or infer_game(path)
        self.state_cols = STATE_COLUMNS[self.game]
        self.num_actions = NUM_ACTIONS[self.game]

        ds = lance.dataset(path)
        tbl = ds.to_table(columns=["episode_idx", "action", *self.state_cols])
        self.episode = np.asarray(tbl["episode_idx"].to_pylist(), dtype=np.int64)
        self.actions = np.asarray(tbl["action"].to_pylist(), dtype=np.int64)
        self.states = np.stack(
            [np.asarray(tbl[c].to_pylist(), dtype=np.float32) for c in self.state_cols], axis=1
        )

        if indices is None:
            h = self.history
            cand = np.arange(h, len(self.episode))
            indices = cand[self.episode[cand] == self.episode[cand - h]]
        self.indices = indices

        if state_mean is None:
            s = torch.from_numpy(self.states[self.indices])
            state_mean, state_std = s.mean(0), s.std(0).clamp(min=1e-6)
        self.state_mean, self.state_std = state_mean, state_std
        self._ds = None  # opened per worker

    @property
    def state_dim(self) -> int:
        return len(self.state_cols)

    def __getstate__(self):
        # A lance.Dataset handle must never cross a process boundary: the Tokio
        # runtime backing it belongs to the process that opened it. Workers
        # reopen lazily in _pixels().
        return {**self.__dict__, "_ds": None}

    def __len__(self) -> int:
        return len(self.indices)

    def _pixels(self, rows: np.ndarray) -> torch.Tensor:
        if self._ds is None:
            self._ds = lance.dataset(self.path)
        blobs = self._ds.take(rows, columns=["pixels"])["pixels"].to_pylist()
        arr = np.stack([np.asarray(Image.open(io.BytesIO(b)).convert("RGB")) for b in blobs])
        return torch.from_numpy(arr).permute(0, 3, 1, 2).float().div_(255.0)

    def __getitem__(self, k: int):
        i = int(self.indices[k])
        rows = np.arange(i - self.history, i + 1)
        frames = self._pixels(rows)
        actions = torch.from_numpy(self.actions[rows])
        states = torch.from_numpy(self.states[rows])
        states = (states - self.state_mean) / self.state_std
        return frames, actions, states


def train_val_split(path: str, history: int, val_frac: float = 0.1, game: str | None = None,
                    limit: int | None = None):
    """Chronological split at a row cut, dropping the `history` windows that would
    straddle it so no val window shares a frame with train. Splitting on episode_idx
    is not viable: these datasets run tens of thousands of rows per episode, so a
    100K slice can contain a single episode.

    limit: use only the first N rows (e.g. 100_000 of a 400K dataset).
    """
    game = game or infer_game(path)
    ds = lance.dataset(path)
    episode = np.asarray(ds.to_table(columns=["episode_idx"])["episode_idx"].to_pylist(), dtype=np.int64)
    if limit is not None:
        episode = episode[:limit]
    cand = np.arange(history, len(episode))
    cand = cand[episode[cand] == episode[cand - history]]

    cut = int(len(episode) * (1 - val_frac))
    tr = cand[cand < cut]
    va = cand[cand - history >= cut]

    train = LanceWindowDataset(path, history, game=game, indices=tr)
    val = LanceWindowDataset(path, history, game=game, indices=va,
                             state_mean=train.state_mean, state_std=train.state_std)
    return train, val
