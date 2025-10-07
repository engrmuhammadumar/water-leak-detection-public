# 02_eval_cnn_lstm_2d.py
from pathlib import Path
import time
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import pandas as pd

from common_2d_dataset import make_loader, CLASSES
from cnn_lstm_2d import CNNLSTM2D

DATA_ROOT = Path(r"E:\Upwork Project\AI_Leak_Detection_Project\data\csv_multisensor_2d")
OUT_DIR   = DATA_ROOT / "outputs_2d_cnnlstm"
CKPT_GLOB = list(OUT_DIR.glob("cnn_lstm2d_*.pt"))

def load_latest_ckpt():
    if not CKPT_GLOB:
        raise FileNotFoundError(f"No checkpoints in {OUT_DIR}")
    return max(CKPT_GLOB, key=lambda p: p.stat().st_mtime)

@torch.no_grad()
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_loader = make_loader(DATA_ROOT/"manifest_test.csv", is_train=False, batch_size=16, num_workers=0)

    ckpt_path = load_latest_ckpt()
    print("Using checkpoint:", ckpt_path)

    model = CNNLSTM2D(num_classes=len(CLASSES), frame_feat=256, lstm_hidden=256, lstm_layers=1, p_drop=0.0).to(device)
    ck = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(ck["model_state"])
    model.eval()

    all_y, all_p = [], []
    t_cwt_model = []

    for i,(x,y,_) in enumerate(tqdm(test_loader, desc="Eval", ncols=90)):
        # (NOTE) dataset already includes CWT; we time only forward pass here.
        # If you want full latency (CWT + forward), time inside DataLoader by
        # enabling precompute or by measuring per-item in dataset. For now:
        t0 = time.time()
        logits = model(x.to(device))
        torch.cuda.synchronize() if device.type=="cuda" else None
        t1 = time.time()
        t_cwt_model.append((t1-t0)*1000.0)  # ms per batch

        pred = logits.argmax(dim=1).cpu().numpy()
        all_p.append(pred)
        all_y.append(y.numpy())

        # only time a few batches
        if i >= 9: pass

    y_true = np.concatenate(all_y)
    y_pred = np.concatenate(all_p)

    acc = (y_true == y_pred).mean()
    print(f"Test accuracy: {acc:.4f} ({(y_true==y_pred).sum()}/{len(y_true)})")

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(CLASSES))))
    fig = plt.figure(figsize=(6,5))
    plt.imshow(cm, interpolation='nearest')
    plt.title('Confusion Matrix (2D CNN-LSTM)')
    plt.colorbar()
    tick_marks = np.arange(len(CLASSES))
    plt.xticks(tick_marks, [c[:12] for c in CLASSES], rotation=45, ha='right')
    plt.yticks(tick_marks, [c[:12] for c in CLASSES])
    plt.xlabel('Predicted'); plt.ylabel('True')
    plt.tight_layout()
    cm_path = OUT_DIR/"confusion_cnnlstm2d.png"
    fig.savefig(cm_path, dpi=150)
    plt.close(fig)
    print("Saved confusion matrix:", cm_path)

    # Per-class metrics
    report = classification_report(y_true, y_pred, target_names=CLASSES, output_dict=True, zero_division=0)
    df = pd.DataFrame(report).transpose()
    df_path = OUT_DIR/"per_class_cnnlstm2d.csv"
    df.to_csv(df_path, index=True)
    print("Saved per-class metrics CSV:", df_path)

    # Latency summary (model-only)
    if t_cwt_model:
        per_batch_ms = np.array(t_cwt_model)
        per_item_ms = per_batch_ms / test_loader.batch_size
        print(f"Model forward (CPU) ~ mean {per_item_ms.mean():.2f} ms / sample "
              f"(p95 {np.percentile(per_item_ms,95):.2f} ms).")
        print("Note: this excludes CWT time (dataset precomputed each item). "
              "If needed, we can time full pipeline by moving timing into the dataset.")

if __name__ == "__main__":
    main()
