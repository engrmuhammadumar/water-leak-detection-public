# eval.py
from __future__ import annotations
import argparse
import torch
import torch.nn as nn

from config import CKPT_PATH, CLASSES, SEQ_LEN, TTA_SHIFTS
from data import make_dataloaders
from models import get_model
from utils import accuracy, confusion_matrix, slice_time_axis

def _tta_logits(model: nn.Module, seq_batch: torch.Tensor, tta_shifts=TTA_SHIFTS):
    """Test-time augmentation via circular time shifts."""
    device = next(model.parameters()).device
    B, T, C, Hs, Ws = seq_batch.shape
    wide = seq_batch.permute(0, 2, 3, 1, 4).contiguous().view(B, C, Hs, T * Ws)

    outs = []
    for s in tta_shifts:
        # roll along width (time axis after stacking)
        shifted = torch.roll(wide, shifts=s, dims=3)
        seqs = []
        for i in range(B):
            seq = slice_time_axis(shifted[i], SEQ_LEN, (Hs, Ws))  # (T,1,Hs,Ws)
            seqs.append(seq.unsqueeze(0))
        seqs = torch.cat(seqs, dim=0).to(device)
        with torch.no_grad():
            logits = model(seqs)
        outs.append(logits)
    return torch.stack(outs, dim=0).mean(dim=0)

@torch.no_grad()
def _run_split(model, loader, criterion, device, use_tta: bool):
    model.eval()
    total_loss = 0.0
    total_acc  = 0.0
    n_batches  = 0
    all_logits, all_targets = [], []

    for seq, target in loader:
        seq, target = seq.to(device), target.to(device)
        logits = _tta_logits(model, seq) if use_tta else model(seq)
        loss = criterion(logits, target)

        total_loss += float(loss.item())
        total_acc  += accuracy(logits, target)
        n_batches  += 1
        all_logits.append(logits.cpu())
        all_targets.append(target.cpu())

    avg_loss = total_loss / max(n_batches, 1)
    avg_acc  = total_acc  / max(n_batches, 1)
    return avg_loss, avg_acc, torch.cat(all_logits), torch.cat(all_targets)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, default=CKPT_PATH)
    ap.add_argument("--model", type=str, default=None,
                    help="cnn_lstm | cnn_lstm_physics_fusion | ann_baseline (override)")
    ap.add_argument("--tta", type=int, default=1, help="1=enable TTA, 0=off")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device} | TTA: {bool(args.tta)}")

    train_loader, val_loader, test_loader, _, (norm_mean, norm_std) = make_dataloaders()
    print(f"Norm: mean={norm_mean:.4f}, std={norm_std:.4f}")

    # FIXED: correct kwarg is map_location
    ckpt = torch.load(args.ckpt, map_location=device)
    model_name = args.model or ckpt.get("model_name", "cnn_lstm")
    print(f"Checkpoint model_name: {ckpt.get('model_name','?')} | Using: {model_name}")

    model = get_model(model_name, num_classes=len(CLASSES), in_ch=1).to(device)
    model.load_state_dict(ckpt["model"])

    criterion = nn.CrossEntropyLoss()

    tr_loss, tr_acc, _, _ = _run_split(model, train_loader, criterion, device, bool(args.tta))
    va_loss, va_acc, _, _ = _run_split(model, val_loader,   criterion, device, bool(args.tta))
    te_loss, te_acc, te_logits, te_targets = _run_split(model, test_loader,  criterion, device, bool(args.tta))

    print(f"\nTrain: loss={tr_loss:.4f} acc={tr_acc:.4f}")
    print(f"Val:   loss={va_loss:.4f} acc={va_acc:.4f}")
    print(f"Test:  loss={te_loss:.4f} acc={te_acc:.4f}")

    cm = confusion_matrix(te_logits, te_targets, num_classes=len(CLASSES))
    print("\nTest Confusion Matrix (rows=true, cols=pred):")
    header = " " * 18 + " ".join(f"{c[:12]:>12}" for c in CLASSES)
    print(header)
    for i, row in enumerate(cm):
        print(f"{CLASSES[i][:16]:>16} | " + " ".join(f"{int(x):12d}" for x in row))

if __name__ == "__main__":
    main()
