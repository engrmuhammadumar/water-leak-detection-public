# data.py
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict
import random
import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

from config import (
    DATA_ROOT, SENSORS, SENSOR_MODE_SUBDIR, CLASSES,
    SEED, SEQ_LEN, SEQ_HW, BATCH_SIZE, NUM_WORKERS, PIN_MEMORY,
    AUGMENT, TIME_MASKS, TIME_W, FREQ_MASKS, FREQ_H, SPLITS
)
from utils import normalize_img, slice_time_axis, spec_augment_
from physics_features import extract_physics_features

IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

def _files_in(cdir: Path):
    return [p for p in cdir.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS]

def _collect_paths() -> Dict[int, List[Path]]:
    per_class: Dict[int, List[Path]] = defaultdict(list)
    for ci, cname in enumerate(CLASSES):
        for s in SENSORS:
            base = DATA_ROOT / s
            cdir = base / SENSOR_MODE_SUBDIR / cname if SENSOR_MODE_SUBDIR else base / cname
            if not cdir.exists():
                raise FileNotFoundError(f"Missing folder: {cdir}")
            per_class[ci].extend(sorted(_files_in(cdir)))
    for ci, arr in per_class.items():
        if len(arr) == 0:
            raise RuntimeError(f"No images for class '{CLASSES[ci]}'")
    return per_class

def _split_indices(per_class: Dict[int, List[Path]], seed=SEED):
    rng = random.Random(seed)
    splits = {}
    tr_p, va_p, te_p = SPLITS
    for ci, arr in per_class.items():
        idxs = list(range(len(arr))); rng.shuffle(idxs)
        n  = len(idxs)
        nt = int(n * tr_p); nv = int(n * va_p)
        splits[ci] = {
            "train": idxs[:nt],
            "val"  : idxs[nt:nt+nv],
            "test" : idxs[nt+nv:]
        }
    return splits

def _load_gray(p: Path):
    return Image.open(p).convert("L")

def _compute_img_norm(per_class: Dict[int, List[Path]], train_idx: List[Tuple[int,int]]):
    # global grayscale mean/std on TRAIN only
    s1=s2=0.0; n_pix=0
    for ci, si in train_idx:
        arr = np.array(_load_gray(per_class[ci][si]), dtype=np.float64) / 255.0
        s1 += arr.sum(); s2 += (arr*arr).sum(); n_pix += arr.size
    mean = s1 / max(1, n_pix); var = (s2 / max(1, n_pix)) - mean*mean
    std  = float(np.sqrt(max(var, 1e-10)))
    return float(mean), float(std)

def _sample_phys_norm(per_class: Dict[int, List[Path]], train_idx: List[Tuple[int,int]],
                      seq_len=SEQ_LEN, seg_hw=SEQ_HW, max_imgs=400):
    """Estimate physics feature mean/std from a subset of training images."""
    rng = random.Random(SEED)
    sample = rng.sample(train_idx, min(max_imgs, len(train_idx)))
    feats_all = []
    for ci, si in sample:
        img = np.array(_load_gray(per_class[ci][si]), dtype=np.float32) / 255.0  # (H,W)
        # slice into T segments exactly as in dataset
        t = torch.from_numpy(img[None, ...])  # (1,H,W)
        seq = slice_time_axis(t, seq_len, seg_hw).squeeze(1).numpy()  # (T,Hs,Ws)
        # per-slice physics features
        feats = [extract_physics_features(seq[k]) for k in range(seq.shape[0])]
        feats_all.append(np.stack(feats, axis=0))
    if not feats_all:
        return np.zeros((1,)), np.ones((1,))
    feats_all = np.concatenate(feats_all, axis=0)  # (N*T, D)
    mean = feats_all.mean(axis=0)
    std  = feats_all.std(axis=0) + 1e-6
    return mean.astype(np.float32), std.astype(np.float32)

class CWTDataset(Dataset):
    """
    Each spectrogram image -> sequence (T,1,Hs,Ws) and physics features (T,Dp)
    Returns: (seq, phys, target)
    """
    def __init__(self, per_class: Dict[int, List[Path]], indices: List[Tuple[int,int]],
                 train: bool, img_mean: float, img_std: float,
                 phys_mean: np.ndarray, phys_std: np.ndarray,
                 seq_len=SEQ_LEN, seg_hw=SEQ_HW):
        self.per_class = per_class
        self.indices   = indices
        self.train     = train
        self.img_mean  = img_mean
        self.img_std   = img_std
        self.phys_mean = phys_mean
        self.phys_std  = phys_std
        self.seq_len   = seq_len
        self.seg_hw    = seg_hw

    def __len__(self): return len(self.indices)

    def __getitem__(self, idx):
        ci, si = self.indices[idx]
        p = self.per_class[ci][si]
        arr = np.array(_load_gray(p), dtype=np.float32) / 255.0  # (H,W)
        t = torch.from_numpy(arr[None, ...])  # (1,H,W)

        # light augmentation on the full spectrogram before slicing
        if self.train and AUGMENT:
            spec_augment_(t, TIME_MASKS, TIME_W, FREQ_MASKS, FREQ_H)

        # normalize + slice
        t = normalize_img(t, mean=self.img_mean, std=self.img_std)
        seq = slice_time_axis(t, self.seq_len, self.seg_hw)  # (T,1,Hs,Ws)

        # physics per slice
        seq_np = seq.squeeze(1).numpy()  # (T,Hs,Ws)
        feats = np.stack([extract_physics_features(seq_np[k]) for k in range(seq_np.shape[0])], axis=0)  # (T,Dp)
        feats = (feats - self.phys_mean) / self.phys_std

        return seq, torch.from_numpy(feats).float(), ci

def make_dataloaders():
    per_class = _collect_paths()
    splits = _split_indices(per_class)

    # build index lists
    train_idx=[]; val_idx=[]; test_idx=[]
    for ci, paths in per_class.items():
        train_idx += [(ci, i) for i in splits[ci]["train"]]
        val_idx   += [(ci, i) for i in splits[ci]["val"]]
        test_idx  += [(ci, i) for i in splits[ci]["test"]]

    rng = random.Random(SEED)
    rng.shuffle(train_idx); rng.shuffle(val_idx); rng.shuffle(test_idx)

    # normalization stats
    img_mean, img_std = _compute_img_norm(per_class, train_idx)
    phys_mean, phys_std = _sample_phys_norm(per_class, train_idx)

    # datasets
    tr_ds = CWTDataset(per_class, train_idx, True,  img_mean, img_std, phys_mean, phys_std)
    va_ds = CWTDataset(per_class, val_idx,   False, img_mean, img_std, phys_mean, phys_std)
    te_ds = CWTDataset(per_class, test_idx,  False, img_mean, img_std, phys_mean, phys_std)

    # class-balanced sampling for train
    cls_counts = defaultdict(int)
    for ci,_ in train_idx: cls_counts[ci]+=1
    weights = [1.0/cls_counts[ci] for (ci,_) in train_idx]
    sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)

    train_loader = DataLoader(tr_ds, batch_size=BATCH_SIZE, sampler=sampler,
                              num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
    val_loader   = DataLoader(va_ds, batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
    test_loader  = DataLoader(te_ds, batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)

    # useful metadata for saving
    meta = {
        "img_norm": {"mean": img_mean, "std": img_std},
        "phys_norm": {"mean": phys_mean.tolist(), "std": phys_std.tolist()},
        "train_idx": train_idx, "val_idx": val_idx, "test_idx": test_idx
    }
    return train_loader, val_loader, test_loader, meta
