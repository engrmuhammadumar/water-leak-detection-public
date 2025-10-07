# 02_infer_eval_seq.py
from pathlib import Path
import torch
import torch.nn as nn
from torchvision import models
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import json

DATA_ROOT = Path(r"E:\Upwork Project\AI_Leak_Detection_Project\images\cwt_log")
OUT_DIR = DATA_ROOT / "outputs"

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

def load_classes_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

# --- dataset that slices images into stripes (same as train) ---
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class SeqStripeDataset(Dataset):
    def __init__(self, csv_path: Path, classes_json: Path, num_segments=8, img_size=224):
        import pandas as pd
        self.df = pd.read_csv(csv_path)
        self.classes = load_classes_json(classes_json)
        self.num_segments = num_segments
        self.img_size = img_size

        if "target" in self.df.columns:
            self.targets = self.df["target"].tolist()
        else:
            self.targets = [self.classes[str(lbl)] for lbl in self.df["label"].tolist()]
        self.paths = self.df["filepath"].tolist()

        self.tf = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])

    def _stripe_bounds(self, w: int):
        seg_w = w / self.num_segments
        bounds = []
        for s in range(self.num_segments):
            x0 = int(round(s * seg_w))
            x1 = int(round((s + 1) * seg_w))
            if x1 <= x0:
                x1 = min(w, x0 + 1)
            bounds.append((x0, min(x1, w)))
        bounds[-1] = (bounds[-1][0], w)
        return bounds

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        p = self.paths[idx]
        y = torch.tensor(self.targets[idx], dtype=torch.long)
        img = Image.open(p).convert("RGB")
        W, H = img.size
        xs = []
        for (x0, x1) in self._stripe_bounds(W):
            stripe = img.crop((x0, 0, x1, H))
            xs.append(self.tf(stripe))
        x = torch.stack(xs, dim=0)  # (S, 3, H, W)
        return x, y, p

def make_loader_seq(csv_path: Path, classes_json: Path, batch_size=16, img_size=224, num_segments=8):
    ds = SeqStripeDataset(csv_path, classes_json, num_segments=num_segments, img_size=img_size)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available()
    )

# --- model def (must match training) ---
class CNNEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        r = models.resnet18(weights=None)
        self.feature_extractor = nn.Sequential(
            r.conv1, r.bn1, r.relu, r.maxpool,
            r.layer1, r.layer2, r.layer3, r.layer4,
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.out_dim = 512

    def forward(self, x):
        f = self.feature_extractor(x)
        return f.view(f.size(0), -1)

class CNNLSTM(nn.Module):
    def __init__(self, num_classes: int, hidden=256, bidirectional=True):
        super().__init__()
        self.cnn = CNNEncoder()
        self.hidden = hidden
        self.bidirectional = bidirectional
        self.lstm = nn.LSTM(
            input_size=self.cnn.out_dim,
            hidden_size=self.hidden,
            num_layers=1,
            batch_first=True,
            bidirectional=self.bidirectional
        )
        d = self.hidden * (2 if bidirectional else 1)
        self.head = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, num_classes))

    def forward(self, x_seq):
        B, S, C, H, W = x_seq.shape
        x = x_seq.view(B * S, C, H, W)
        feats = self.cnn(x).view(B, S, -1)
        out, _ = self.lstm(feats)
        last = out[:, -1, :]
        return self.head(last)

def find_latest_cnnlstm(out_dir: Path):
    cand = sorted(out_dir.glob("cnn_lstm_*.pt"))
    if not cand:
        raise SystemExit("No CNN-LSTM checkpoints found in outputs/")
    return cand[-1]

if __name__ == "__main__":
    import numpy as np

    ckpt_path = find_latest_cnnlstm(OUT_DIR)
    print("Loading checkpoint:", ckpt_path)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    classes = ckpt["classes"]
    img_size = ckpt.get("img_size", 224)
    num_segments = ckpt.get("num_segments", 8)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    num_classes = len(classes)
    model = CNNLSTM(num_classes=num_classes).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    test_csv = DATA_ROOT / "manifest_test.csv"
    classes_json = DATA_ROOT / "classes.json"
    loader = make_loader_seq(
        test_csv,
        classes_json,
        batch_size=16,
        img_size=img_size,
        num_segments=num_segments
    )

    total, correct = 0, 0
    cm = [[0 for _ in range(num_classes)] for _ in range(num_classes)]

    with torch.no_grad():
        for x_seq, y, paths in tqdm(loader, ncols=70):
            x_seq, y = x_seq.to(device), y.to(device)
            logits = model(x_seq)
            preds = logits.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)
            for t, p in zip(y.tolist(), preds.tolist()):
                cm[t][p] += 1

    acc = correct / total if total else 0.0
    print(f"Test accuracy: {acc:.4f} ({correct}/{total})")

    # Confusion matrix fig
    idx_to_class = {v: k for k, v in classes.items()}
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(cm)
    ax.set_title("Confusion Matrix — CNN+LSTM")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_xticks(range(num_classes))
    ax.set_xticklabels([idx_to_class[i] for i in range(num_classes)], rotation=45, ha="right")
    ax.set_yticks(range(num_classes))
    ax.set_yticklabels([idx_to_class[i] for i in range(num_classes)])
    for i in range(num_classes):
        for j in range(num_classes):
            ax.text(j, i, str(cm[i][j]), ha="center", va="center")
    fig.tight_layout()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig_path = OUT_DIR / "confusion_matrix_cnnlstm.png"
    fig.savefig(fig_path, dpi=150)
    print("Saved confusion matrix:", fig_path)

    # Per-class metrics
    cm = np.array(cm)
    tp = cm.diagonal()
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp
    prec = tp / (tp + fp + 1e-9)
    rec  = tp / (tp + fn + 1e-9)
    f1   = 2 * prec * rec / (prec + rec + 1e-9)

    rows = []
    for i in range(num_classes):
        rows.append({
            "class": idx_to_class[i],
            "precision": float(prec[i]),
            "recall": float(rec[i]),
            "f1": float(f1[i]),
            "support": int(cm[i].sum()),
        })
    df = pd.DataFrame(rows).sort_values("class")
    print("\nPer-class metrics:")
    print(df.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    out_csv = OUT_DIR / "per_class_metrics_cnnlstm.csv"
    df.to_csv(out_csv, index=False)
    print(df.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
