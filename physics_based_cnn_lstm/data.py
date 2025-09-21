# data.py
from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Tuple
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import random
from collections import defaultdict
import numpy as np
import os

from config import (
    DATA_ROOT, SENSORS, SENSOR_MODE_SUBDIR, CLASSES,
    TRAIN_PCT, VAL_PCT, TEST_PCT, SEED, SEQ_LEN, IMG_SEG_SIZE, BATCH_SIZE,
    AUG_ENABLE, TIME_MASKS, TIME_MASK_WIDTH, FREQ_MASKS, FREQ_MASK_HEIGHT
)
from utils import normalize_img, slice_time_axis, spec_augment_

# ----------------------------
# Configurable loader settings
# ----------------------------
NUM_WORKERS = int(os.environ.get("NUM_WORKERS", 4))
PIN_MEMORY  = os.environ.get("PIN_MEMORY", "1") not in {"0", "false", "False"}
PERSISTENT  = os.environ.get("PERSISTENT_WORKERS", "1") not in {"0", "false", "False"}

IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def list_files(cdir: Path):
    if not cdir.exists():
        raise FileNotFoundError(f"Missing directory: {cdir}")
    return [p for p in cdir.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS]


def _sensor_class_dir(sensor_root: Path, cname: str) -> Path:
    # Support optional deeper subdir (e.g., "Looped"); if empty -> direct class folder
    if SENSOR_MODE_SUBDIR:
        return sensor_root / SENSOR_MODE_SUBDIR / cname
    return sensor_root / cname


def collect_per_class_paths(root: Path, classes: List[str]) -> Dict[int, List[Path]]:
    """
    Aggregate ALL images from ALL sensors into each class bucket.
    Result: class_to_paths[ci] = [path1, path2, ...]
    """
    class_to_paths: Dict[int, List[Path]] = defaultdict(list)

    if not root.exists():
        raise FileNotFoundError(f"DATA_ROOT not found: {root}")

    # Verify each sensor root exists; guide user if not
    sensor_roots = []
    for s in SENSORS:
        sr = root / s
        if not sr.exists():
            raise FileNotFoundError(
                f"Sensor folder missing: {sr}\n"
                f"Expected layout:\n"
                f"  {root}/<SensorName>/{SENSOR_MODE_SUBDIR or '<ClassName>'}/<ClassName>/*.png"
            )
        sensor_roots.append(sr)

    # Collect paths
    for ci, cname in enumerate(classes):
        per_class = []
        for sr in sensor_roots:
            cdir = _sensor_class_dir(sr, cname)
            per_class.extend(sorted(list_files(cdir)))
        if not per_class:
            raise FileNotFoundError(f"No images found for class '{cname}'. Checked sensors: {SENSORS}")
        class_to_paths[ci] = per_class

    return class_to_paths


def train_val_test_indices(per_class_counts: Dict[int, int], seed=SEED):
    """
    Deterministic split per class using percentages from config.
    """
    rng = random.Random(seed)
    splits = {}
    for ci, n in per_class_counts.items():
        idxs = list(range(n))
        rng.shuffle(idxs)
        n_tr = int(round(n * TRAIN_PCT))
        n_va = int(round(n * VAL_PCT))
        # Ensure we don't exceed n
        n_tr = min(n_tr, n)
        n_va = min(n_va, max(0, n - n_tr))
        splits[ci] = {
            "train": idxs[:n_tr],
            "val":   idxs[n_tr:n_tr + n_va],
            "test":  idxs[n_tr + n_va:],
        }
    return splits


def load_gray(path: Path):
    return Image.open(path).convert("L")


def compute_mean_std(class_paths: Dict[int, List[Path]], train_idx: List[Tuple[int, int]]) -> Tuple[float, float]:
    """
    Global grayscale mean/std computed on TRAIN images only, in [0,1].
    """
    n_pixels = 0
    s1 = 0.0
    s2 = 0.0
    for ci, si in train_idx:
        arr = np.array(load_gray(class_paths[ci][si]), dtype=np.float64) / 255.0
        n = arr.size
        n_pixels += n
        s1 += float(arr.sum())
        s2 += float((arr**2).sum())
    if n_pixels == 0:
        # Fallback: neutral normalization
        return 0.5, 0.25
    mean = s1 / n_pixels
    var = (s2 / n_pixels) - mean**2
    std = float(np.sqrt(max(var, 1e-10)))
    return float(mean), std


class CWTSingleImageDataset(Dataset):
    """
    Each grayscale spectrogram image (from any sensor) is ONE sample.
    Returns: seq (T,1,Hs,Ws), target (int)
    """
    def __init__(self, class_paths: Dict[int, List[Path]], indices: List[Tuple[int, int]],
                 seq_len=SEQ_LEN, seg_hw=(IMG_SEG_SIZE, IMG_SEG_SIZE),
                 train_mode: bool = False, norm_mean: float = 0.5, norm_std: float = 0.5):
        self.class_paths = class_paths
        self.indices = indices
        self.seq_len = seq_len
        self.seg_hw = seg_hw
        self.train_mode = train_mode
        self.norm_mean = norm_mean
        self.norm_std = norm_std

    def __len__(self): 
        return len(self.indices)

    def __getitem__(self, idx: int):
        ci, si = self.indices[idx]
        img_path = self.class_paths[ci][si]
        arr = np.array(load_gray(img_path), dtype=np.float32) / 255.0  # (H,W)
        t = torch.from_numpy(arr[None, ...])  # (1,H,W)

        if self.train_mode and AUG_ENABLE:
            spec_augment_(t, time_masks=TIME_MASKS, time_w=TIME_MASK_WIDTH,
                          freq_masks=FREQ_MASKS, freq_h=FREQ_MASK_HEIGHT)

        t = normalize_img(t, mean=self.norm_mean, std=self.norm_std)
        seq = slice_time_axis(t, self.seq_len, self.seg_hw)  # (T,1,Hs,Ws)
        return seq, ci


def _build_indices(class_paths: Dict[int, List[Path]]):
    per_class_counts = {ci: len(v) for ci, v in class_paths.items()}
    splits = train_val_test_indices(per_class_counts)

    # indices per split
    train_idx, val_idx, test_idx = [], [], []
    for ci, paths in class_paths.items():
        train_idx += [(ci, si) for si in splits[ci]["train"]]
        val_idx   += [(ci, si) for si in splits[ci]["val"]]
        test_idx  += [(ci, si) for si in splits[ci]["test"]]

    rnd = random.Random(SEED)
    rnd.shuffle(train_idx); rnd.shuffle(val_idx); rnd.shuffle(test_idx)

    return (train_idx, val_idx, test_idx)


def _seed_worker(worker_id: int):
    """
    Reproducible dataloader workers.
    """
    worker_seed = SEED + worker_id
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def make_dataloaders():
    """
    Returns:
      train_loader, val_loader, test_loader,
      (train_idx, val_idx, test_idx),
      (norm_mean, norm_std)
    """
    class_paths = collect_per_class_paths(DATA_ROOT, CLASSES)
    train_idx, val_idx, test_idx = _build_indices(class_paths)

    # Normalization stats on TRAIN only
    norm_mean, norm_std = compute_mean_std(class_paths, train_idx)

    # datasets
    train_ds = CWTSingleImageDataset(class_paths, train_idx, train_mode=True,
                                     norm_mean=norm_mean, norm_std=norm_std)
    val_ds   = CWTSingleImageDataset(class_paths, val_idx,   train_mode=False,
                                     norm_mean=norm_mean, norm_std=norm_std)
    test_ds  = CWTSingleImageDataset(class_paths, test_idx,  train_mode=False,
                                     norm_mean=norm_mean, norm_std=norm_std)

    # class-balanced sampler for training
    train_class_counts = defaultdict(int)
    for ci, _ in train_idx:
        train_class_counts[ci] += 1

    # Avoid zero-division; if a class has 0 in train split, skip balancing for that class
    weights = []
    for (ci, _) in train_idx:
        cnt = max(1, train_class_counts[ci])
        weights.append(1.0 / cnt)
    sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)

    # DataLoader common kwargs
    common = dict(num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY, persistent_workers=PERSISTENT)
    # On Windows with NUM_WORKERS>0, persistent_workers requires pin_memory True; we already default to True.

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, sampler=sampler,
        worker_init_fn=_seed_worker, **common
    )
    val_loader   = DataLoader(
        val_ds,   batch_size=BATCH_SIZE, shuffle=False,
        worker_init_fn=_seed_worker, **common
    )
    test_loader  = DataLoader(
        test_ds,  batch_size=BATCH_SIZE, shuffle=False,
        worker_init_fn=_seed_worker, **common
    )

    return train_loader, val_loader, test_loader, (train_idx, val_idx, test_idx), (norm_mean, norm_std)
