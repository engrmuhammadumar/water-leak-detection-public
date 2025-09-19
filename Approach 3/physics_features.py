# physics_features.py
"""
Physics-inspired features extracted from a single spectrogram slice.
Works directly on the (1, H, W) grayscale slice used by the CNN, so we don't
need raw time-series in this stage.

Input shapes:
  - Per sample: (T, 1, H, W)
  - Batched:    (B, T, 1, H, W)

All ops are PyTorch (GPU-friendly). Returns float32 tensors on the same device.

Feature philosophy (per slice):
  Temporal marginal x_t = mean over frequency bins -> proxy time-envelope
  Spectral marginal x_f = mean over time bins     -> proxy spectrum envelope

We compute:
  1) Energy (sum), 2) Log-energy
  3) Temporal mean, 4) Temporal std, 5) Temporal slope (L→R), 6) Zero-cross rate (temporal)
  7) Spectral centroid, 8) Spectral bandwidth, 9) Spectral rolloff (85%)
  10) Spectral entropy, 11) Spectral flatness, 12) Spectral slope
  13) Band energies: low/mid/high thirds of spectrum (3 features)

Total features per slice: 13 + 3 = 16 dims.

These are deliberately small and stable so they fuse well with learned features.

Public API:
  - feature_dim() -> int
  - compute_physics_features(seq_tchw: (T,1,H,W)) -> (T, D)
  - compute_physics_features_batch(batch_btchw: (B,T,1,H,W)) -> (B,T,D)
"""

from __future__ import annotations
from typing import Tuple
import torch
import torch.nn.functional as F

_EPS = 1e-12
_FEATURE_DIM = 16  # Keep in sync with config.PHYSICS_FEAT_DIM (we'll assert at runtime in model)


def feature_dim() -> int:
    """Return the dimensionality of the physics feature vector per slice."""
    return _FEATURE_DIM


def _safe_norm(x: torch.Tensor, dim: int) -> torch.Tensor:
    s = x.sum(dim=dim, keepdim=True)
    return x / (s + _EPS)


def _temporal_features_from_marginal(x_t: torch.Tensor) -> Tuple[torch.Tensor, ...]:
    """
    x_t: (T, W) non-negative envelope (mean over freq of slice)
    Returns tuple of (T,) tensors:
      mean, std, slope, zcr
    """
    T, W = x_t.shape

    # Mean and std
    mean_t = x_t.mean(dim=1)
    std_t  = x_t.std(dim=1, unbiased=False)

    # Slope (linear regression over time index 0..W-1)
    t_idx = torch.linspace(0.0, 1.0, W, device=x_t.device).view(1, W)  # normalize to [0,1]
    t_bar = t_idx.mean(dim=1, keepdim=True)         # (1,1)
    y_bar = x_t.mean(dim=1, keepdim=True)           # (T,1)
    cov   = ((x_t - y_bar) * (t_idx - t_bar)).mean(dim=1)    # (T,)
    var_t = (t_idx - t_bar).pow(2).mean(dim=1)               # scalar
    slope = cov / (var_t + _EPS)                   # (T,)

    # Zero-crossing rate on a zero-mean standardized envelope
    x_std = x_t - y_bar
    # Prevent divisions by tiny std: just scale by (std + eps)
    x_std = x_std / (x_std.std(dim=1, keepdim=True) + _EPS)
    sign_changes = (torch.sign(x_std[:, 1:]) != torch.sign(x_std[:, :-1])).float()
    zcr = sign_changes.mean(dim=1)  # (T,)

    return mean_t, std_t, slope, zcr


def _spectral_features_from_marginal(x_f: torch.Tensor) -> Tuple[torch.Tensor, ...]:
    """
    x_f: (T, H) non-negative spectrum (mean over time of slice)
    Returns tuple of (T,) tensors:
      centroid, bandwidth, rolloff85, entropy, flatness, slope, band_low, band_mid, band_high
    """
    T, H = x_f.shape
    # Indices 0..H-1 normalized to [0,1] as a proxy frequency axis
    f_idx = torch.linspace(0.0, 1.0, H, device=x_f.device).view(1, H)

    # Normalize to a distribution
    p = _safe_norm(x_f, dim=1)  # (T,H)

    # Centroid and bandwidth
    centroid = (p * f_idx).sum(dim=1)                       # E[f]
    bandwidth = torch.sqrt(((f_idx - centroid.view(T, 1))**2 * p).sum(dim=1) + _EPS)  # std

    # Rolloff (0.85)
    csum = torch.cumsum(p, dim=1)
    thr = 0.85
    # first index where cumulative >= thr
    idx = torch.argmax((csum >= thr).to(torch.int64), dim=1)  # (T,)
    rolloff = (idx.to(x_f.dtype)) / max(H - 1, 1)

    # Entropy (Shannon) scaled to [0,1] by dividing by log(H)
    entropy = -(p * (p + _EPS).log()).sum(dim=1) / (float(H).bit_length() if False else torch.log(torch.tensor(H + 0.0, device=x_f.device)))
    # (Using ln; normalize by ln(H))
    entropy = -(p * (p + _EPS).log()).sum(dim=1) / torch.log(torch.tensor(float(H), device=x_f.device))

    # Flatness = geometric mean / arithmetic mean
    # geo = exp(mean(log(x+eps))), arith = mean(x)
    geo = torch.exp(torch.mean(torch.log(x_f + _EPS), dim=1))
    arith = torch.mean(x_f, dim=1) + _EPS
    flatness = geo / arith

    # Spectral slope via regression over f_idx
    f_bar = f_idx.mean(dim=1, keepdim=True)               # (1,1)
    y_bar = x_f.mean(dim=1, keepdim=True)                 # (T,1)
    cov   = ((x_f - y_bar) * (f_idx - f_bar)).mean(dim=1) # (T,)
    var_f = (f_idx - f_bar).pow(2).mean(dim=1)            # scalar
    spec_slope = cov / (var_f + _EPS)                     # (T,)

    # Band energies (low/mid/high thirds)
    thirds = H // 3
    if thirds == 0:
        # Degenerate tiny H: duplicate total energy across bands
        tot = x_f.sum(dim=1)
        band_low = band_mid = band_high = tot / 3.0
    else:
        l = slice(0, thirds)
        m = slice(thirds, 2 * thirds)
        h = slice(2 * thirds, H)
        band_low  = x_f[:, l].sum(dim=1)
        band_mid  = x_f[:, m].sum(dim=1)
        band_high = x_f[:, h].sum(dim=1)

        # Normalize by total to reduce scale sensitivity
        tot = x_f.sum(dim=1) + _EPS
        band_low  = band_low  / tot
        band_mid  = band_mid  / tot
        band_high = band_high / tot

    return centroid, bandwidth, rolloff, entropy, flatness, spec_slope, band_low, band_mid, band_high


def compute_physics_features(seq_tchw: torch.Tensor) -> torch.Tensor:
    """
    Compute physics features for a single sample sequence.

    Args:
        seq_tchw: (T, 1, H, W) float tensor (already normalized image slices)
    Returns:
        feats: (T, D) float32 tensor on same device
    """
    assert seq_tchw.ndim == 4 and seq_tchw.shape[1] == 1, f"Expected (T,1,H,W), got {tuple(seq_tchw.shape)}"
    T, _, H, W = seq_tchw.shape
    x = seq_tchw.squeeze(1)                   # (T, H, W)
    x = x.clamp_min(0.0)                      # ensure non-negative for log/entropy logic

    # Temporal marginal (mean over freq): (T, W)
    x_t = x.mean(dim=1)

    # Spectral marginal (mean over time): (T, H)
    x_f = x.mean(dim=2)

    # Core energies
    energy = x.sum(dim=(1, 2))                                  # (T,)
    log_energy = torch.log(energy + _EPS)                       # (T,)

    # Temporal features
    mean_t, std_t, slope_t, zcr_t = _temporal_features_from_marginal(x_t)

    # Spectral features
    centroid, bandwidth, rolloff, entropy, flatness, spec_slope, band_low, band_mid, band_high = \
        _spectral_features_from_marginal(x_f)

    # Stack to (T, D)
    feats = torch.stack([
        energy,
        log_energy,
        mean_t,
        std_t,
        slope_t,
        zcr_t,
        centroid,
        bandwidth,
        rolloff,
        entropy,
        flatness,
        spec_slope,
        band_low,
        band_mid,
        band_high,
    ], dim=1)

    # If we ever change dims, pad/trim to _FEATURE_DIM to keep contracts stable
    D = feats.shape[1]
    if D < _FEATURE_DIM:
        pad = torch.zeros((T, _FEATURE_DIM - D), device=feats.device, dtype=feats.dtype)
        feats = torch.cat([feats, pad], dim=1)
    elif D > _FEATURE_DIM:
        feats = feats[:, :_FEATURE_DIM]

    return feats.to(dtype=torch.float32)


def compute_physics_features_batch(batch_btchw: torch.Tensor) -> torch.Tensor:
    """
    Batched version.

    Args:
        batch_btchw: (B, T, 1, H, W)
    Returns:
        feats: (B, T, D)
    """
    assert batch_btchw.ndim == 5 and batch_btchw.shape[2] == 1, f"Expected (B,T,1,H,W), got {tuple(batch_btchw.shape)}"
    B, T, _, H, W = batch_btchw.shape
    x = batch_btchw.squeeze(2)  # (B, T, H, W)
    x = x.clamp_min(0.0)

    # Temporal marginal: mean over freq -> (B, T, W)
    x_t = x.mean(dim=2)

    # Spectral marginal: mean over time -> (B, T, H)
    x_f = x.mean(dim=3)

    # Energy & log-energy
    energy = x.view(B, T, -1).sum(dim=2)
    log_energy = torch.log(energy + _EPS)

    # Temporal features per (B,T)
    # We'll reshape to (B*T, W) -> compute -> reshape back
    xt_bt_w = x_t.reshape(B * T, W)
    mean_t, std_t, slope_t, zcr_t = _temporal_features_from_marginal(xt_bt_w)
    mean_t  = mean_t.view(B, T)
    std_t   = std_t.view(B, T)
    slope_t = slope_t.view(B, T)
    zcr_t   = zcr_t.view(B, T)

    # Spectral features per (B,T)
    xf_bt_h = x_f.reshape(B * T, H)
    centroid, bandwidth, rolloff, entropy, flatness, spec_slope, band_low, band_mid, band_high = \
        _spectral_features_from_marginal(xf_bt_h)
    centroid   = centroid.view(B, T)
    bandwidth  = bandwidth.view(B, T)
    rolloff    = rolloff.view(B, T)
    entropy    = entropy.view(B, T)
    flatness   = flatness.view(B, T)
    spec_slope = spec_slope.view(B, T)
    band_low   = band_low.view(B, T)
    band_mid   = band_mid.view(B, T)
    band_high  = band_high.view(B, T)

    feats = torch.stack([
        energy,
        log_energy,
        mean_t,
        std_t,
        slope_t,
        zcr_t,
        centroid,
        bandwidth,
        rolloff,
        entropy,
        flatness,
        spec_slope,
        band_low,
        band_mid,
        band_high,
    ], dim=2)  # (B,T,15)

    # Pad/trim to _FEATURE_DIM
    D = feats.shape[2]
    if D < _FEATURE_DIM:
        pad = torch.zeros((B, T, _FEATURE_DIM - D), device=feats.device, dtype=feats.dtype)
        feats = torch.cat([feats, pad], dim=2)
    elif D > _FEATURE_DIM:
        feats = feats[:, :, :_FEATURE_DIM]

    return feats.to(dtype=torch.float32)
