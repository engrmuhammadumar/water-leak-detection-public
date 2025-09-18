# batch_infer.py
import argparse, csv, glob, os
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from config import CLASSES, BEST_CKPT, DEVICE, SEQ_LEN, SEQ_HW
from models import HybridCNNLSTM
from utils import normalize_img, slice_time_axis
from physics_features import extract_physics_features

def load_gray(p): return Image.open(p).convert("L")

def build_seq_and_phys(p, img_mean, img_std, phys_mean, phys_std):
    arr = np.array(load_gray(p), dtype=np.float32) / 255.0
    t = torch.from_numpy(arr[None, ...])
    t = normalize_img(t, mean=img_mean, std=img_std)
    seq = slice_time_axis(t, SEQ_LEN, SEQ_HW)              # (T,1,h,w)
    seq_np = seq.squeeze(1).numpy()                        # (T,h,w)
    feats = np.stack([extract_physics_features(seq_np[k]) for k in range(seq_np.shape[0])], axis=0)
    feats = (feats - phys_mean) / phys_std
    return seq, torch.from_numpy(feats).float()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", required=True)
    ap.add_argument("--out",  default="predictions.csv")
    ap.add_argument("--ckpt", default=str(BEST_CKPT))
    args = ap.parse_args()

    device = "cuda" if (DEVICE=="cuda" and torch.cuda.is_available()) else "cpu"
    ckpt = torch.load(args.ckpt, map_location=device)
    meta = ckpt["meta"]
    img_mean = float(meta["img_norm"]["mean"]); img_std = float(meta["img_norm"]["std"])
    phys_mean = np.array(meta["phys_norm"]["mean"], dtype=np.float32)
    phys_std  = np.array(meta["phys_norm"]["std"],  dtype=np.float32)

    model = HybridCNNLSTM(num_classes=len(CLASSES)).to(device)
    model.load_state_dict(ckpt["model"]); model.eval()

    files = sorted(glob.glob(args.glob, recursive=True))
    if not files:
        print(f"No files matched: {args.glob}"); return

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["path", "pred_idx", "pred_class", "pred_prob"] + [f"p_{c}" for c in CLASSES])
        with torch.no_grad():
            for p in files:
                seq, phys = build_seq_and_phys(p, img_mean, img_std, phys_mean, phys_std)
                logits = model(seq.unsqueeze(0).to(device), phys.unsqueeze(0).to(device))
                probs = F.softmax(logits, dim=1).cpu().numpy()[0]
                j = int(probs.argmax()); w.writerow([p, j, CLASSES[j], f"{probs[j]:.4f}"] + [f"{x:.4f}" for x in probs])

    print(f"Saved: {args.out}")

if __name__ == "__main__":
    main()
