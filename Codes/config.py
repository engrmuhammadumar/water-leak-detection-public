# config.py
from pathlib import Path

# === Paths ===
DATA_ROOT = Path(r"E:\Upwork Project\AI_Leak_Detection_Project\images\cwt_log")
SENSORS = ["Accelerometer", "Dynamic Pressure Sensor", "Hydrophones"]
SENSOR_MODE_SUBDIR = "Looped"

CLASSES = [
    "No-leak",
    "Orifice Leak",
    "Gasket Leak",
    "Longitudinal Crack",
    "Circumferential Crack",
]

# === Sequence slicing ===
SEQ_LEN = 48          # more temporal granularity than 32
IMG_SEG_SIZE = 64

# === Training ===
BATCH_SIZE = 32
EPOCHS = 50
LR = 3e-3             # peak LR for OneCycle
WEIGHT_DECAY = 1e-4
DROPOUT = 0.2
LSTM_HIDDEN = 384     # larger than 256
LSTM_BIDIR = True
SEED = 42

# === Model/loss/optim extras ===
FEAT_DIM = 512                # wider CNN projection
LABEL_SMOOTH = 0.03
USE_ATTENTION_POOL = True     # temporal attention over LSTM outputs
USE_AMP = True                # AMP on GPU (ignored on CPU)
MAX_GRAD_NORM = 1.0           # gradient clipping

# === SpecAugment (train only) — gentle ===
AUG_ENABLE = True
TIME_MASKS = 1
TIME_MASK_WIDTH = 0.05
FREQ_MASKS = 1
FREQ_MASK_HEIGHT = 0.05

# === Splits ===
TRAIN_PCT = 0.70
VAL_PCT   = 0.15
TEST_PCT  = 0.15

# === Saving ===
CKPT_PATH = "cnn_lstm_cwt.pt"
