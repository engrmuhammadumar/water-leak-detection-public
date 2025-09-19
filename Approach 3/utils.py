# utils.py
from __future__ import annotations
import torch
import torch.nn.functional as F
import numpy as np
import random, os

# -------------------------
# Reproducibility
# -------------------------
def set_seed(seed: int = 42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


# -------------------------
# Normalization helpers
# -------------------------
def normalize_img(t: torch.Tensor, mean: float, std: float) -> torch.Tensor:
    """
    t: (1,H,W) float32 in [0,1]
    mean, std: scalars (float)
    Returns normalized tensor
    """
    return (t - mean) / (std + 1e-6)


# -------------------------
# Slice spectrogram into sequence
# -------------------------
def slice_time_axis(img: torch.Tensor, seq_len: int, seg_hw=(64, 64)) -> torch.Tensor:
    """
    img: (1,H,W) torch float tensor
    Returns: (T,1,Hs,Ws)
    """
    _, H, W = img.shape
    Hs, Ws = seg_hw
    if W < seq_len:
        raise ValueError(f"Image width {W} < seq_len {seq_len}")

    # compute crop width
    stride = W // seq_len
    seq = []
    for i in range(seq_len):
        start = i * stride
        end = min(start + Ws, W)
        crop = img[:, :, start:end]
        # resize to (Hs,Ws)
        crop = F.interpolate(crop.unsqueeze(0), size=(Hs, Ws),
                             mode="bilinear", align_corners=False).squeeze(0)
        seq.append(crop)
    return torch.stack(seq, dim=0)  # (T,1,Hs,Ws)


# -------------------------
# SpecAugment (basic)
# -------------------------
def spec_augment_(spec: torch.Tensor, time_masks=1, time_w=0.05,
                  freq_masks=1, freq_h=0.05):
    """
    In-place augmentation on (1,H,W).
    """
    _, H, W = spec.shape
    # time masks
    for _ in range(time_masks):
        t = int(W * time_w)
        t0 = np.random.randint(0, max(1, W - t))
        spec[:, :, t0:t0+t] = 0.0
    # freq masks
    for _ in range(freq_masks):
        f = int(H * freq_h)
        f0 = np.random.randint(0, max(1, H - f))
        spec[:, f0:f0+f, :] = 0.0
    return spec


# -------------------------
# Metrics
# -------------------------
def accuracy(logits: torch.Tensor, target: torch.Tensor) -> float:
    pred = logits.argmax(dim=1)
    return (pred == target).float().mean().item()


def confusion_matrix(logits: torch.Tensor, target: torch.Tensor, num_classes: int):
    pred = logits.argmax(dim=1)
    cm = torch.zeros((num_classes, num_classes), dtype=torch.int64)
    for t, p in zip(target.view(-1), pred.view(-1)):
        cm[t.long(), p.long()] += 1
    return cm


# -------------------------
# Latency
# -------------------------
import time
@torch.no_grad()
def measure_latency(model, sample, device="cpu", runs=50):
    model.eval()
    sample = sample.to(device)
    # warmup
    for _ in range(5):
        _ = model(sample)
    torch.cuda.synchronize() if device.startswith("cuda") else None

    start = time.time()
    for _ in range(runs):
        _ = model(sample)
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    end = time.time()
    return 1000.0 * (end - start) / runs  # ms per run
