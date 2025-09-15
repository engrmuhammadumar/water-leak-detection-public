# data.py
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple, Iterable
from collections import defaultdict
import random
import difflib

import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

from config import (
    DATA_ROOT, SENSORS, SENSOR_MODE_SUBDIR, CLASSES,
    TRAIN_PCT, VAL_PCT, TEST_PCT, SEED, SEQ_LEN, IMG_SIZE, BATCH_SIZE,
    AUG_ENABLE, TIME_MASKS, TIME_MASK_WIDTH, FREQ_MASKS, FREQ_MASK_HEIGHT,
    USE_PHYSICS_FEATURES, NUM_PHYSICS_FEATURES
)
from utils import normalize_img, slice_time_axis, spec_augment_
from physics_features import compute_physics_features_from_image_path as phys_feats

IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


# ----------------------------
# Helpers
# ----------------------------
def _list_files(cdir: Path) -> List[Path]:
    if not cdir.exists():
        return []
    return [p for p in cdir.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS]


def _seed_everything(seed: int = SEED):
    random.seed(seed)
    np.random.seed(seed)
    try:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def collect_per_class_paths(root: Path, classes: List[str]) -> Dict[int, List[Path]]:
    """
    Build a mapping: class_index -> list[Path] of images from all sensors.
    Raises a helpful error if a class directory is missing.
    """
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"DATA_ROOT not found: {root}")

    class_to_paths: Dict[int, List[Path]] = defaultdict(list)

    # For suggestions
    def _suggest(missing: str, options: Iterable[str]) -> str:
        guess = difflib.get_close_matches(missing, list(options), n=1)
        return f" (did you mean: {guess[0]})" if guess else ""

    for ci, cname in enumerate(classes):
        per_class_list: List[Path] = []
        for sensor in SENSORS:
            base = root / sensor
            cdir = base / SENSOR_MODE_SUBDIR / cname if SENSOR_MODE_SUBDIR else base / cname
            if not cdir.exists():
                # Suggest a close match if possible
                parent = cdir.parent
                existing = [p.name for p in parent.iterdir() if p.is_dir()] if parent.exists() else []
                hint = _suggest(cname, existing)
                raise FileNotFoundError(f"Missing class dir: {cdir}{hint}")

            per_class_list.extend(sorted(_list_files(cdir)))

        if len(per_class_list) == 0:
            raise FileNotFoundError(
                f"No image files found for class '{cname}'. "
                f"Checked sensors={SENSORS} under root='{root}'. "
                f"Accepted extensions: {sorted(IMG_EXTS)}"
            )
        class_to_paths[ci] = per_class_list

    return class_to_paths


def train_val_test_indices(per_class_counts: Dict[int, int], seed=SEED):
    rng = random.Random(seed)
    splits = {}
    for ci, n in per_class_counts.items():
        idxs = list(range(n))
        rng.shuffle(idxs)
        n_tr = int(n * TRAIN_PCT)
        n_va = int(n * VAL_PCT)
        # Ensure at least 1 sample falls into train if possible
        if n_tr == 0 and n > 0:
            n_tr = 1
        splits[ci] = {
            "train": idxs[:n_tr],
            "val": idxs[n_tr:n_tr + n_va],
            "test": idxs[n_tr + n_va:]
        }
    return splits


def _load_gray(path: Path) -> Image.Image:
    # Pillow defers file close until GC; use context to be safe
    with Image.open(path) as im:
        return im.convert("L")


def compute_mean_std(class_paths: Dict[int, List[Path]], train_idx: List[Tuple[int, int]]) -> Tuple[float, float]:
    """
    Global grayscale mean/std computed over TRAIN only.
    """
    n_pixels = 0
    s1 = 0.0
    s2 = 0.0

    if len(train_idx) == 0:
        # Fallback to neutral normalization
        return 0.5, 0.5

    for ci, si in train_idx:
        arr = np.array(_load_gray(class_paths[ci][si]), dtype=np.float64) / 255.0  # (H,W)
        n = arr.size
        n_pixels += n
        s1 += float(arr.sum())
        s2 += float((arr ** 2).sum())

    mean = s1 / max(1, n_pixels)
    var = (s2 / max(1, n_pixels)) - mean ** 2
    std = float(np.sqrt(max(var, 1e-10)))
    return float(mean), std


# ----------------------------
# Dataset
# ----------------------------
class CWTSingleImageDataset(Dataset):
    """
    Each grayscale spectrogram image (from any sensor) is ONE sample.
    Returns:
      - if USE_PHYSICS_FEATURES=False: (seq, target)
      - if True: (seq, physics_vec, target)
    """
    def __init__(
        self,
        class_paths: Dict[int, List[Path]],
        indices: List[Tuple[int, int]],
        seq_len: int = SEQ_LEN,
        img_hw: Tuple[int, int] = (IMG_SIZE, IMG_SIZE),
        train_mode: bool = False,
        norm_mean: float = 0.5,
        norm_std: float = 0.5,
    ):
        self.class_paths = class_paths
        self.indices = indices
        self.seq_len = seq_len
        self.img_hw = img_hw
        self.train_mode = train_mode
        self.norm_mean = norm_mean
        self.norm_std = norm_std

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int):
        ci, si = self.indices[idx]
        img_path = self.class_paths[ci][si]
        arr = np.array(_load_gray(img_path), dtype=np.float32) / 255.0  # (H,W)
        t = torch.from_numpy(arr[None, ...])  # (1,H,W)

        if self.train_mode and AUG_ENABLE:
            spec_augment_(
                t,
                time_masks=TIME_MASKS,
                time_w=TIME_MASK_WIDTH,
                freq_masks=FREQ_MASKS,
                freq_h=FREQ_MASK_HEIGHT,
            )

        t = normalize_img(t, mean=self.norm_mean, std=self.norm_std)
        seq = slice_time_axis(t, self.seq_len, self.img_hw)  # (T,1,Hs,Ws)

        if USE_PHYSICS_FEATURES:
            p = phys_feats(str(img_path))  # np.float32, (F,)
            p = torch.from_numpy(p)        # (F,)
            # Optional sanity: ensure expected length
            if NUM_PHYSICS_FEATURES and p.numel() != NUM_PHYSICS_FEATURES:
                # Don’t crash training; just warn once
                # (You can swap to raise ValueError if you prefer strictness)
                pass
            return seq, p, ci
        else:
            return seq, ci


# ----------------------------
# Dataloaders
# ----------------------------
def make_dataloaders(
    batch_size: int = BATCH_SIZE,
    num_workers: int = 0,
    pin_memory: bool | None = None,
    persistent_workers: bool | None = None,
):
    """
    Windows-safe defaults:
      - num_workers=0
      - persistent_workers=False (must be False if num_workers=0)
    You can increase num_workers to 2–4 after placing your training loop
    under `if __name__ == "__main__":` guard.
    """
    _seed_everything(SEED)

    data_root = Path(DATA_ROOT)
    class_paths = collect_per_class_paths(data_root, CLASSES)

    per_class_counts = {ci: len(v) for ci, v in class_paths.items()}
    if any(n == 0 for n in per_class_counts.values()):
        empties = [CLASSES[ci] for ci, n in per_class_counts.items() if n == 0]
        raise RuntimeError(f"Classes with no samples: {empties}")

    splits = train_val_test_indices(per_class_counts, seed=SEED)

    # indices per split
    train_idx: List[Tuple[int, int]] = []
    val_idx: List[Tuple[int, int]] = []
    test_idx: List[Tuple[int, int]] = []

    for ci in class_paths.keys():
        train_idx += [(ci, si) for si in splits[ci]["train"]]
        val_idx += [(ci, si) for si in splits[ci]["val"]]
        test_idx += [(ci, si) for si in splits[ci]["test"]]

    # Deterministic order before sampler
    rng = random.Random(SEED)
    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    rng.shuffle(test_idx)

    # Normalization stats on TRAIN only
    norm_mean, norm_std = compute_mean_std(class_paths, train_idx)

    # datasets
    train_ds = CWTSingleImageDataset(
        class_paths, train_idx, train_mode=True, norm_mean=norm_mean, norm_std=norm_std
    )
    val_ds = CWTSingleImageDataset(
        class_paths, val_idx, train_mode=False, norm_mean=norm_mean, norm_std=norm_std
    )
    test_ds = CWTSingleImageDataset(
        class_paths, test_idx, train_mode=False, norm_mean=norm_mean, norm_std=norm_std
    )

    # Class-balanced sampler for training
    train_class_counts = defaultdict(int)
    for ci, _ in train_idx:
        train_class_counts[ci] += 1
    # Avoid div-by-zero (already checked above); compute weights
    weights = [1.0 / train_class_counts[ci] for (ci, _) in train_idx]
    sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)

    # Resolve pin_memory default
    if pin_memory is None:
        pin = torch.cuda.is_available()
    else:
        pin = bool(pin_memory)

    # Resolve persistent_workers default
    if persistent_workers is None:
        persist = num_workers > 0
    else:
        persist = bool(persistent_workers)
    if num_workers == 0 and persist:
        # Force False per PyTorch requirement
        persist = False

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=False,          # sampler and shuffle are mutually exclusive
        num_workers=num_workers,
        pin_memory=pin,
        persistent_workers=persist,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin,
        persistent_workers=persist,
        drop_last=False,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin,
        persistent_workers=persist,
        drop_last=False,
    )

    return (
        train_loader,
        val_loader,
        test_loader,
        (train_idx, val_idx, test_idx),
        (norm_mean, norm_std),
    )
