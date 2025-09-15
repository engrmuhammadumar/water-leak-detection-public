# config.py
from pathlib import Path

# ====== PATHS ======
# Root that contains per-sensor folders with CWT images by class.
# Example:
#   DATA_ROOT/
#     Accelerometer/Looped/<ClassName>/*.png
#     Dynamic Pressure Sensor/Looped/<ClassName>/*.png
#     Hydrophones/Looped/<ClassName>/*.png
DATA_ROOT = Path(r"E:\Upwork Project\AI_Leak_Detection_Project\images\cwt_log")  # <-- set to your CWT root

SENSORS = [
    "Accelerometer",
    "Dynamic Pressure Sensor",
    "Hydrophones",
]

# If your CWT images are one level deeper under each sensor (e.g., "Looped"),
# set this to that subfolder name; otherwise set to "".
SENSOR_MODE_SUBDIR = "Looped"

# ====== CLASSES (canonical order) ======
CLASSES = [
    "No-leak",
    "Orifice Leak",
    "Gasket Leak",
    "Longitudinal Crack",
    "Circumferential Crack",
]

# ====== SEQUENCE SLICING ======
# Turn each (H,W) CWT image into a sequence of T slices along time (width).
SEQ_LEN      = 48
IMG_SIZE     = 64   # each slice resized to (IMG_SIZE x IMG_SIZE) for the CNN

# ====== TRAINING ======
BATCH_SIZE   = 32
EPOCHS       = 50
LR           = 3e-3
WEIGHT_DECAY = 1e-4
DROPOUT      = 0.2
LSTM_HIDDEN  = 384
LSTM_BIDIR   = True
FEAT_DIM     = 512
LABEL_SMOOTH = 0.03
MAX_GRAD_NORM = 1.0
SEED         = 42

# Mixed precision (on GPU only)
USE_AMP      = True

# ====== AUGMENTATION (gentle; turn ON after baseline is stable) ======
AUG_ENABLE        = False
TIME_MASKS        = 1
TIME_MASK_WIDTH   = 0.05   # fraction of width per mask
FREQ_MASKS        = 1
FREQ_MASK_HEIGHT  = 0.05   # fraction of height per mask

# ====== SPLITS ======
TRAIN_PCT = 0.70
VAL_PCT   = 0.15
TEST_PCT  = 0.15

# ====== SAVING / EVAL ======
CKPT_PATH = "cnn_lstm_cwt.pt"
TTA_SHIFTS = (0, 2, 4, 6, 8)

# ====== PHYSICS FEATURES FUSION ======
# Turn on fusion model that concatenates physics-inspired features with LSTM features.
USE_PHYSICS_FEATURES = True
NUM_PHYSICS_FEATURES = 20           # must match physics_features.get_feature_names()
PHYSICS_MLP_HIDDEN   = 64
PHYSICS_DROP         = 0.1
