# config.py
from pathlib import Path

# -----------------------------
# Dataset paths (EDIT THESE)
# -----------------------------
# Root that contains per-sensor folders, then class subfolders.
# Example:
#   DATA_ROOT/
#     Accelerometer/Looped/<ClassName>/*.png
#     Dynamic Pressure Sensor/Looped/<ClassName>/*.png
#     Hydrophones/Looped/<ClassName>/*.png
DATA_ROOT = Path(r"E:\Upwork Project\AI_Leak_Detection_Project\images\cwt_log")

# Folder names EXACTLY as they appear on disk:
SENSORS = [
    "Accelerometer",
    "Dynamic Pressure Sensor",
    "Hydrophones",
]

# If your class folders are nested (e.g., "Looped/<Class>"), set this:
SENSOR_MODE_SUBDIR = "Looped"   # set "" if classes are directly under the sensor folder

# EXACT class folder names:
CLASSES = [
    "No-leak",
    "Orifice Leak",
    "Gasket Leak",
    "Longitudinal Crack",
    "Circumferential Crack",
]

# -----------------------------
# Sequence slicing
# -----------------------------
SEQ_LEN = 48           # number of temporal slices along spectrogram width
SEQ_HW  = (64, 64)     # (H, W) of each slice fed to CNN

# -----------------------------
# Physics-feature fusion
# -----------------------------
USE_PHYSICS_FEATURES = True
PHYS_FEATURE_DIM     = 24        # MUST match physics_features.py output
PHYS_PROJ_DIM        = 32        # projection dim before LSTM

# -----------------------------
# Model parameters
# -----------------------------
CNN_OUT_DIM   = 128
LSTM_HIDDEN   = 384
LSTM_LAYERS   = 1
BIDIRECTIONAL = True
DROPOUT       = 0.2
LABEL_SMOOTH  = 0.03
USE_ATTENTION = True   # temporal attention over LSTM outputs

# -----------------------------
# Training hyperparameters
# -----------------------------
BATCH_SIZE    = 32
EPOCHS        = 50
WEIGHT_DECAY  = 1e-4
SEED          = 42
SPLITS        = (0.70, 0.15, 0.15)  # train/val/test

# -----------------------------
# Augmentation (mild, robust)
# -----------------------------
AUGMENT     = True
TIME_MASKS  = 1
TIME_W      = 0.05
FREQ_MASKS  = 1
FREQ_H      = 0.05

# -----------------------------
# Checkpoints & logging
# -----------------------------
CKPT_DIR  = Path("./checkpoints"); CKPT_DIR.mkdir(parents=True, exist_ok=True)
BEST_CKPT = CKPT_DIR / "cnn_lstm_cwt_phys.pt"
OUT_DIR   = Path("./outputs"); OUT_DIR.mkdir(parents=True, exist_ok=True)

# -----------------------------
# System / CUDA tuning
# -----------------------------
DEVICE          = "cuda"  # "cuda" or "cpu"
NUM_WORKERS     = 2       # 0 if you see Windows/DataLoader issues
PIN_MEMORY      = True
AMP             = True    # mixed precision on GPU
CUDNN_BENCHMARK = True
