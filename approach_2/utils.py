# utils.py
import random, time
import numpy as np
import torch
import torch.nn.functional as F
from torchvision.transforms import functional as TF
from typing import Tuple

def set_seed(seed: int = 42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def normalize_img(t: torch.Tensor, mean=0.5, std=0.5) -> torch.Tensor:
    # t: (C,H,W)
    return TF.normalize(t, mean=[mean]*t.shape[0], std=[std]*t.shape[0])

def slice_time_axis(img_t: torch.Tensor, seq_len: int, out_hw: Tuple[int,int]) -> torch.Tensor:
    # img_t: (C,H,W) -> (T,C,out_h,out_w), split along width
    C, H, W = img_t.shape
    if W < seq_len:
        img_t = torch.nn.functional.interpolate(img_t.unsqueeze(0), size=(H, seq_len),
                                                mode="bilinear", align_corners=False).squeeze(0)
        W = seq_len
    base = W // seq_len; extra = W % seq_len
    widths = [base + (1 if i < extra else 0) for i in range(seq_len)]
    slices, x = [], 0
    for w in widths:
        seg = img_t[:, :, x:x+w]; x += w
        seg = TF.resize(seg, out_hw, antialias=True)
        slices.append(seg.unsqueeze(0))
    return torch.cat(slices, dim=0)  # (T,C,h,w)

def spec_augment_(img_chw: torch.Tensor, time_masks: int, time_w: float,
                  freq_masks: int, freq_h: float):
    # zero masks on (C,H,W)
    _, H, W = img_chw.shape
    for _ in range(time_masks):
        w = max(1, int(round(W * time_w))); s = random.randint(0, max(0, W - w))
        img_chw[:, :, s:s+w] = 0
    for _ in range(freq_masks):
        h = max(1, int(round(H * freq_h))); s = random.randint(0, max(0, H - h))
        img_chw[:, s:s+h, :] = 0

def measure_latency_fn(forward_fn, sample, device="cpu", warmup=5, runs=50) -> float:
    """Time any callable forward_fn(sample). No .eval() requirement."""
    with torch.no_grad():
        # warmup
        for _ in range(warmup):
            _ = forward_fn(sample)
        if device.startswith("cuda"): torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(runs):
            _ = forward_fn(sample)
        if device.startswith("cuda"): torch.cuda.synchronize()
    return (time.perf_counter() - t0) / runs * 1000.0
