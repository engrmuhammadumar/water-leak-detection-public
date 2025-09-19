# config.py
import os
from pathlib import Path

# === Paths (env-overridable) ===
# Set DATA_ROOT via env var if possible; fallback to local 'data'
DATA_ROOT = Path(r"E:\Upwork Project\AI_Leak_Detection_Project\images\cwt_log")
SENSOR_MODE_SUBDIR = "Looped"  # or "" if not present


# Sensors & optional mode subfolder inside each sensor dir
SENSORS = [
    "Accelerometer",
    "Dynamic Pressure Sensor",
    "Hydrophones",
]
SENSOR_MODE_SUBDIR = os.environ.get("SENSOR_MODE_SUBDIR", "Looped").strip() or ""

# === Classes (canonical order) ===
CLASSES = [
    "No-leak",
    "Orifice Leak",
    "Gasket Leak",
    "Longitudinal Crack",
    "Circumferential Crack",
]

# === Sequence slicing (turn a spectrogram into a T-length sequence of crops) ===
SEQ_LEN = int(os.environ.get("SEQ_LEN", 48))       # temporal slices along width
IMG_SEG_SIZE = int(os.environ.get("IMG_SEG_SIZE", 64))  # per-slice H==W

# === Training ===
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", 32))
EPOCHS = int(os.environ.get("EPOCHS", 50))
LR = float(os.environ.get("LR", 3e-3))             # OneCycle peak LR
WEIGHT_DECAY = float(os.environ.get("WEIGHT_DECAY", 1e-4))
DROPOUT = float(os.environ.get("DROPOUT", 0.2))
LSTM_HIDDEN = int(os.environ.get("LSTM_HIDDEN", 384))
LSTM_BIDIR = os.environ.get("LSTM_BIDIR", "1") not in {"0", "false", "False"}
SEED = int(os.environ.get("SEED", 42))

# === Model / loss / optim extras ===
FEAT_DIM = int(os.environ.get("FEAT_DIM", 512))
LABEL_SMOOTH = float(os.environ.get("LABEL_SMOOTH", 0.03))
USE_ATTENTION_POOL = os.environ.get("USE_ATTENTION_POOL", "1") not in {"0", "false", "False"}
USE_AMP = os.environ.get("USE_AMP", "1") not in {"0", "false", "False"}
MAX_GRAD_NORM = float(os.environ.get("MAX_GRAD_NORM", 1.0))

# === SpecAugment (train only) — gentle ===
AUG_ENABLE = os.environ.get("AUG_ENABLE", "1") not in {"0", "false", "False"}
TIME_MASKS = int(os.environ.get("TIME_MASKS", 1))
TIME_MASK_WIDTH = float(os.environ.get("TIME_MASK_WIDTH", 0.05))
FREQ_MASKS = int(os.environ.get("FREQ_MASKS", 1))
FREQ_MASK_HEIGHT = float(os.environ.get("FREQ_MASK_HEIGHT", 0.05))

# === Splits ===
TRAIN_PCT = float(os.environ.get("TRAIN_PCT", 0.70))
VAL_PCT   = float(os.environ.get("VAL_PCT",   0.15))
TEST_PCT  = float(os.environ.get("TEST_PCT",  0.15))

# === Saving ===
CKPT_DIR = Path(os.environ.get("CKPT_DIR", "checkpoints")).resolve()
CKPT_DIR.mkdir(parents=True, exist_ok=True)
CKPT_PATH = str(CKPT_DIR / "cnn_lstm_cwt.pt")
HISTORY_PATH = str(CKPT_DIR / "history.json")

# === TTA (eval) ===
# circular width shifts for test-time augmentation
TTA_SHIFTS = tuple(int(x) for x in os.environ.get("TTA_SHIFTS", "0,2,4,6,8").split(","))

# === Physics-based feature branch ===
# If True, we will compute physics-inspired features on each time-slice and fuse with learned features
USE_PHYSICS = os.environ.get("USE_PHYSICS", "1") not in {"0", "false", "False"}
# Dimension of physics feature vector per slice (set to match physics_features.py)
PHYSICS_FEAT_DIM = int(os.environ.get("PHYSICS_FEAT_DIM", 16))
# Fusion mode: "concat" (default) or "gated" (optional later)
FUSION_MODE = os.environ.get("FUSION_MODE", "concat")

# === Baselines ===
# Options: "cnn_lstm", "cnn_lstm_physics_fusion", "ann_baseline"
MODEL_NAME = os.environ.get("MODEL_NAME", "cnn_lstm_physics_fusion")

# === Simulink / Digital Twin I/O (for integration) ===
# We’ll support two simple bridges: CSV folder watch OR UDP socket.
SIM_BRIDGE_MODE = os.environ.get("SIM_BRIDGE_MODE", "csv")  # "csv" or "udp"
SIM_CSV_INBOX = Path(os.environ.get("SIM_CSV_INBOX", "sim_inbox")).resolve()
SIM_CSV_OUTBOX = Path(os.environ.get("SIM_CSV_OUTBOX", "sim_outbox")).resolve()
SIM_CSV_INBOX.mkdir(exist_ok=True, parents=True)
SIM_CSV_OUTBOX.mkdir(exist_ok=True, parents=True)

# For UDP (if you choose to use it)
SIM_UDP_HOST = os.environ.get("SIM_UDP_HOST", "127.0.0.1")
SIM_UDP_PORT = int(os.environ.get("SIM_UDP_PORT", 51000))

# Canonical CSV column names to exchange with Simulink when sending raw sensor signal snapshots
# (these are examples; you can adapt to your actual signal naming)
SIM_COLUMNS = [
    "t",                 # time (s)
    "accel",             # accelerometer (arb units)
    "pressure",          # dynamic pressure
    "hydrophone",        # hydrophone/vibration/acoustic
]

# === Paper / figures ===
FIGURE_DIR = Path(os.environ.get("FIGURE_DIR", "figures")).resolve()
FIGURE_DIR.mkdir(exist_ok=True, parents=True)
