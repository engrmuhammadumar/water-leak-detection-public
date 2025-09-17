# physics_features.py
from __future__ import annotations
import numpy as np
from PIL import Image

# ============ Utilities ============
def _safe_norm(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    s = x.sum()
    if s <= eps:
        return np.ones_like(x) / max(1, x.size)
    return x / s

def _band_energies(img: np.ndarray, bands: int = 5) -> np.ndarray:
    """Row-wise 'frequency' bands from top (low) to bottom (high)."""
    H, W = img.shape
    bsz = H // bands
    out = []
    for b in range(bands):
        y0 = b * bsz
        y1 = H if b == bands - 1 else (b + 1) * bsz
        out.append(float(img[y0:y1, :].sum()))
    s = sum(out) or 1.0
    return np.array([v / s for v in out], dtype=np.float32)

def _spectral_centroid(img: np.ndarray) -> float:
    """Centroid along rows (frequency)."""
    H, _ = img.shape
    p = img.sum(axis=1).astype(np.float64)
    p = _safe_norm(p)
    freqs = np.arange(H, dtype=np.float64)
    return float((freqs * p).sum() / max(1e-12, p.sum()))

def _spectral_spread(img: np.ndarray, centroid: float | None = None) -> float:
    H, _ = img.shape
    p = img.sum(axis=1).astype(np.float64)
    p = _safe_norm(p)
    freqs = np.arange(H, dtype=np.float64)
    c = centroid if centroid is not None else (freqs * p).sum()
    var = ((freqs - c) ** 2 * p).sum()
    return float(np.sqrt(max(0.0, var)))

def _spectral_rolloff(img: np.ndarray, pct: float = 0.85) -> float:
    p = img.sum(axis=1).astype(np.float64)
    c = np.cumsum(p)
    thr = pct * (p.sum() or 1.0)
    idx = int(np.searchsorted(c, thr))
    return float(idx)

def _spectral_flatness(img: np.ndarray) -> float:
    p = img.sum(axis=1).astype(np.float64) + 1e-12
    gmean = np.exp(np.log(p).mean())
    amean = p.mean()
    return float(gmean / amean)

def _entropy(img: np.ndarray) -> float:
    q = img.astype(np.float64)
    q = _safe_norm(q)
    return float(-(q * (np.log(q + 1e-12))).sum())

def _temporal_envelope_stats(img: np.ndarray) -> tuple[float, float]:
    """Variance and slope of envelope across time (width)."""
    env = img.mean(axis=0).astype(np.float64)  # (W,)
    var = float(env.var())
    # slope via least squares
    W = env.size
    x = np.arange(W, dtype=np.float64)
    x = x - x.mean()
    denom = (x**2).sum() or 1.0
    slope = float((x * env).sum() / denom)
    return var, slope

def _crest_factor(img: np.ndarray) -> float:
    val = img.astype(np.float64)
    rms = float(np.sqrt(np.mean(val**2) + 1e-12))
    peak = float(val.max())
    return float(peak / (rms + 1e-12))

def _kurtosis_skewness(img: np.ndarray) -> tuple[float, float]:
    x = img.flatten().astype(np.float64)
    mu = x.mean()
    sd = x.std() + 1e-12
    z = (x - mu) / sd
    kurt = float((z**4).mean() - 3.0)
    skew = float((z**3).mean())
    return kurt, skew

def _zcr_like(img: np.ndarray) -> float:
    """Zero-crossing-like measure using column-wise gradient sign changes."""
    g = np.diff(img.mean(axis=0).astype(np.float64))  # (W-1,)
    s = np.sign(g)
    return float(np.mean((s[1:] * s[:-1]) < 0.0))

# ============ Main API ============
def get_feature_names() -> list[str]:
    """
    Keep this list length in sync with config.NUM_PHYSICS_FEATURES
    """
    names = []
    names += [f"band_energy_b{i+1}" for i in range(5)]   # 5
    names += ["centroid", "spread", "rolloff85", "flatness"]  # 4 (9 total)
    names += ["entropy"]  # 1 (10)
    names += ["env_var", "env_slope"]  # 2 (12)
    names += ["crest_factor"]          # 1 (13)
    names += ["kurtosis", "skewness"]  # 2 (15)
    names += ["zcr_like"]              # 1 (16)
    # Add 4 simple band-ratio features to reach 20
    names += ["low_mid_ratio", "mid_high_ratio", "low_high_ratio", "low_plus_high"]
    return names  # total 20

def compute_physics_features_from_image_path(path: str | bytes | "os.PathLike[str]") -> np.ndarray:
    """
    Compute physics-inspired features from a grayscale CWT image.
    Returns np.float32 vector of length len(get_feature_names()).
    """
    img = Image.open(path).convert("L")
    arr = np.array(img, dtype=np.float32) / 255.0  # (H,W), in [0,1]
    # Avoid degenerate all-zero images
    if arr.max() <= 0:
        arr = arr + 1e-6

    feats: list[float] = []
    bands = _band_energies(arr, bands=5)              # 5
    feats += list(bands)

    centroid = _spectral_centroid(arr)
    spread   = _spectral_spread(arr, centroid)
    roll85   = _spectral_rolloff(arr, pct=0.85)
    flat     = _spectral_flatness(arr)
    feats += [centroid, spread, roll85, flat]         # +4 = 9

    ent = _entropy(arr)
    feats += [ent]                                    # +1 = 10

    env_var, env_slope = _temporal_envelope_stats(arr)
    feats += [env_var, env_slope]                     # +2 = 12

    feats += [_crest_factor(arr)]                     # +1 = 13
    kurt, skew = _kurtosis_skewness(arr)
    feats += [kurt, skew]                             # +2 = 15
    feats += [_zcr_like(arr)]                         # +1 = 16

    # simple band ratios/combos (assuming bands[0]=low ... bands[4]=high)
    low  = bands[0] + 1e-12
    mid  = bands[2] + 1e-12
    high = bands[4] + 1e-12
    feats += [low / mid, mid / high, low / high, low + high]  # +4 = 20

    return np.asarray(feats, dtype=np.float32)
