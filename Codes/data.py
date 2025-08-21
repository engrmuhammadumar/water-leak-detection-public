# data.py
from pathlib import Path
from typing import Dict, List, Tuple
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import random
from collections import defaultdict
import numpy as np

from config import (
    DATA_ROOT, SENSORS, SENSOR_MODE_SUBDIR, CLASSES,
    TRAIN_PCT, VAL_PCT, TEST_PCT, SEED, SEQ_LEN, IMG_SEG_SIZE, BATCH_SIZE,
    AUG_ENABLE, TIME_MASKS, TIME_MASK_WIDTH, FREQ_MASKS, FREQ_MASK_HEIGHT
)
from utils import normalize_img, slice_time_axis, spec_augment_

IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

def list_files(cdir: Path):
    return [p for p in cdir.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS]

def collect_per_class_paths(root: Path, sensors: List[str], classes: List[str]) -> Dict[int, List[Path]]:
    sensor_base = [root / s / SENSOR_MODE_SUBDIR for s in sensors]
    for sb in sensor_base:
        if not sb.exists():
            raise FileNotFoundError(f"Missing sensor subdir: {sb}")
    class_to_paths: Dict[int, List[Path]] = defaultdict(list)
    for ci, cname in enumerate(classes):
        for sb in sensor_base:
            cdir = sb / cname
            if not cdir.exists():
                raise FileNotFoundError(f"Missing class dir: {cdir}")
            class_to_paths[ci].extend(sorted(list_files(cdir)))
    return class_to_paths

def train_val_test_indices(per_class_counts: Dict[int,int], seed=SEED):
    rng = random.Random(seed); splits = {}
    for ci, n in per_class_counts.items():
        idxs = list(range(n)); rng.shuffle(idxs)
        n_tr = int(n * TRAIN_PCT); n_va = int(n * VAL_PCT)
        splits[ci] = {"train": idxs[:n_tr],
                      "val":   idxs[n_tr:n_tr+n_va],
                      "test":  idxs[n_tr+n_va:]}
    return splits

def load_gray(path: Path):
    return Image.open(path).convert("L")

def compute_mean_std(class_paths: Dict[int, List[Path]], train_idx: List[Tuple[int,int]]) -> Tuple[float, float]:
    n_pixels = 0; sum_x = 0.0; sum_x2 = 0.0
    for ci, si in train_idx:
        img_path = class_paths[ci][si]
        arr = np.array(load_gray(img_path), dtype=np.float64) / 255.0
        n = arr.size
        n_pixels += n
        sum_x  += float(arr.sum())
        sum_x2 += float((arr**2).sum())
    mean = sum_x / max(1, n_pixels)
    var  = (sum_x2 / max(1, n_pixels)) - mean**2
    std  = float(np.sqrt(max(var, 1e-10)))
    return float(mean), std

class CWTSingleSensorDataset(Dataset):
    """
    Each grayscale spectrogram image is ONE sample.
    Returns: seq (T,1,Hs,Ws), target (int)
    """
    def __init__(self, class_paths: Dict[int, List[Path]], indices: List[Tuple[int,int]],
                 seq_len=SEQ_LEN, seg_hw=(IMG_SEG_SIZE, IMG_SEG_SIZE),
                 train_mode: bool = False, norm_mean: float = 0.5, norm_std: float = 0.5):
        self.class_paths = class_paths
        self.indices = indices
        self.seq_len = seq_len
        self.seg_hw = seg_hw
        self.train_mode = train_mode
        self.norm_mean = norm_mean
        self.norm_std = norm_std

    def __len__(self): return len(self.indices)

    def __getitem__(self, idx: int):
        ci, si = self.indices[idx]
        img_path = self.class_paths[ci][si]
        arr = np.array(load_gray(img_path), dtype=np.float32) / 255.0  # (H,W)
        arr = arr[None, ...]  # (1,H,W)
        t = torch.from_numpy(arr)

        if self.train_mode and AUG_ENABLE:
            spec_augment_(t, time_masks=TIME_MASKS, time_w=TIME_MASK_WIDTH,
                          freq_masks=FREQ_MASKS, freq_h=FREQ_MASK_HEIGHT)

        t = normalize_img(t, mean=self.norm_mean, std=self.norm_std)
        seq = slice_time_axis(t, self.seq_len, self.seg_hw)  # (T,1,Hs,Ws)
        return seq, ci

def make_dataloaders():
    class_paths = collect_per_class_paths(DATA_ROOT, SENSORS, CLASSES)
    per_class_counts = {ci: len(v) for ci, v in class_paths.items()}
    splits = train_val_test_indices(per_class_counts)

    # Build index lists
    train_idx, val_idx, test_idx = [], [], []
    for ci, paths in class_paths.items():
        train_idx += [(ci, si) for si in splits[ci]["train"]]
        val_idx   += [(ci, si) for si in splits[ci]["val"]]
        test_idx  += [(ci, si) for si in splits[ci]["test"]]
    random.Random(SEED).shuffle(train_idx)
    random.Random(SEED).shuffle(val_idx)
    random.Random(SEED).shuffle(test_idx)

    # Normalization stats on TRAIN only
    norm_mean, norm_std = compute_mean_std(class_paths, train_idx)

    # Datasets
    train_ds = CWTSingleSensorDataset(class_paths, train_idx, train_mode=True,
                                      norm_mean=norm_mean, norm_std=norm_std)
    val_ds   = CWTSingleSensorDataset(class_paths, val_idx,   train_mode=False,
                                      norm_mean=norm_mean, norm_std=norm_std)
    test_ds  = CWTSingleSensorDataset(class_paths, test_idx,  train_mode=False,
                                      norm_mean=norm_mean, norm_std=norm_std)

    # Class-balanced sampler for training
    train_class_counts = defaultdict(int)
    for ci, _ in train_idx: train_class_counts[ci] += 1
    weights = [1.0 / train_class_counts[ci] for (ci, _) in train_idx]
    sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler, num_workers=4, pin_memory=False)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,   num_workers=4, pin_memory=False)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False,   num_workers=4, pin_memory=False)

    return train_loader, val_loader, test_loader, (train_idx, val_idx, test_idx), (norm_mean, norm_std)
