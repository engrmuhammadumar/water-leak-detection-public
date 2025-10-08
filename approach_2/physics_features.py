# physics_features.py
import numpy as np
from scipy.stats import skew, kurtosis
from numpy.fft import rfft

def _safe_entropy(p):
    p = p.astype(np.float64)
    p = p / (p.sum() + 1e-12)
    p = p[p > 0]
    return float(-(p * np.log(p)).sum())

def extract_physics_features(img: np.ndarray) -> np.ndarray:
    """
    Handcrafted physics-inspired features from a 2D CWT scalogram (H, W) float32 in [0,1].
    Returns a fixed-length vector of size 24 (PHYS_FEATURE_DIM).
    """
    img = img.astype(np.float32)
    H, W = img.shape
    flat = img.ravel()

    # 1-4) basic statistics
    mean = float(flat.mean())
    std  = float(flat.std() + 1e-12)
    sk   = float(skew(flat))
    ku   = float(kurtosis(flat))

    # 5) total energy
    energy = float((flat**2).sum())

    # 6-7) spectral centroid & flatness along rows (frequency axis)
    row_energy = img.sum(axis=1) + 1e-12
    f_idx = np.arange(H, dtype=np.float32)
    centroid = float((f_idx * row_energy).sum() / row_energy.sum())
    flatness = float(np.exp(np.log(row_energy).mean()) / row_energy.mean())

    # 8-12) intensity percentiles
    p10, p25, p50, p75, p90 = np.percentile(flat, [10, 25, 50, 75, 90]).astype(np.float32)

    # 13) entropy of normalized intensities
    entropy = _safe_entropy(flat)

    # 14) gradient magnitude mean (edge density proxy)
    gy, gx = np.gradient(img)
    edge_density = float(np.sqrt(gx*gx + gy*gy).mean())

    # 15-16) spatial spread (std of row/col indices weighted by intensity)
    P = img / (img.sum() + 1e-12)
    rr = np.arange(H)[:, None].astype(np.float32)
    cc = np.arange(W)[None, :].astype(np.float32)
    r_mean = float((P * rr).sum())
    c_mean = float((P * cc).sum())
    row_spread = float(np.sqrt(((rr - r_mean)**2 * P).sum()))
    col_spread = float(np.sqrt(((cc - c_mean)**2 * P).sum()))

    # 17-19) energy ratios in low frequency bands (first 10%, 25%, 50% rows)
    b10 = int(max(1, round(0.10 * H)))
    b25 = int(max(1, round(0.25 * H)))
    b50 = int(max(1, round(0.50 * H)))
    total_e = float(row_energy.sum())
    low10 = float(row_energy[:b10].sum() / total_e)
    low25 = float(row_energy[:b25].sum() / total_e)
    low50 = float(row_energy[:b50].sum() / total_e)

    # 20) temporal variance (variance across columns, averaged over rows)
    temporal_var = float(img.var(axis=1).mean())

    # 21) crest factor = max / RMS
    rms = float(np.sqrt((flat**2).mean()) + 1e-12)
    crest = float(flat.max() / rms)

    # 22-23) ridge (peak row index) stats across columns
    ridge_rows = img.argmax(axis=0).astype(np.float32)
    ridge_mean = float(ridge_rows.mean() / max(1, H-1))
    ridge_std  = float(ridge_rows.std()  / max(1, H-1))

    # 24) zero-crossing in center row detrended
    row_c = img[H//2, :] - img[H//2, :].mean()
    zcr = float((np.abs(np.diff(np.sign(row_c))) > 0).mean())

    feats = np.array([
        mean, std, sk, ku,
        energy, centroid, flatness,
        p10, p25, p50, p75, p90,
        entropy, edge_density,
        row_spread, col_spread,
        low10, low25, low50,
        temporal_var, crest,
        ridge_mean, ridge_std,
        zcr
    ], dtype=np.float32)
    assert feats.shape[0] == 24, f"Expected 24 features, got {feats.shape[0]}"
    return feats
