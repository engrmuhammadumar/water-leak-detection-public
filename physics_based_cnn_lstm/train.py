# train.py
from __future__ import annotations
import argparse, json, os, time
from collections import defaultdict

import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import autocast, GradScaler

from config import (
    EPOCHS, LR, WEIGHT_DECAY, CKPT_PATH, CLASSES, SEED,
    LABEL_SMOOTH, USE_AMP, MAX_GRAD_NORM, HISTORY_PATH, MODEL_NAME
)
from data import make_dataloaders
from models import get_model
from utils import set_seed, accuracy, confusion_matrix, measure_latency


class EMA:
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {k: p.detach().clone() for k, p in model.state_dict().items()
                       if getattr(p, "dtype", None) is not None and p.dtype.is_floating_point}

    @torch.no_grad()
    def update(self, model):
        for k, p in model.state_dict().items():
            if k in self.shadow and getattr(p, "dtype", None) is not None and p.dtype.is_floating_point:
                self.shadow[k].mul_(self.decay).add_(p.detach(), alpha=1 - self.decay)

    @torch.no_grad()
    def copy_to(self, model):
        msd = model.state_dict()
        for k, v in self.shadow.items():
            if k in msd:
                msd[k].copy_(v)


def epoch_loop(model, loader, criterion, optimizer=None, device="cpu",
               scaler=None, scheduler=None, ema: EMA | None = None, train=True):
    model.train(train)
    total_loss, total_acc, n_batches = 0.0, 0.0, 0
    all_logits, all_targets = [], []

    for seq, target in loader:
        seq, target = seq.to(device), target.to(device)

        if train:
            optimizer.zero_grad(set_to_none=True)
            with autocast(device_type="cuda", enabled=USE_AMP and device.startswith("cuda")):
                logits = model(seq)
                loss = criterion(logits, target)

            if USE_AMP and device.startswith("cuda"):
                scaler.scale(loss).backward()
                if MAX_GRAD_NORM is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
                scaler.step(optimizer); scaler.update()
            else:
                loss.backward()
                if MAX_GRAD_NORM is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
                optimizer.step()
            if scheduler is not None:
                scheduler.step()
            if ema is not None:
                ema.update(model)
        else:
            with torch.no_grad():
                logits = model(seq)
                loss = criterion(logits, target)

        total_loss += float(loss.item())
        total_acc  += accuracy(logits.detach(), target)
        n_batches  += 1
        all_logits.append(logits.detach().cpu())
        all_targets.append(target.detach().cpu())

    avg_loss = total_loss / max(1, n_batches)
    avg_acc  = total_acc  / max(1, n_batches)
    return avg_loss, avg_acc, torch.cat(all_logits), torch.cat(all_targets)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--lr", type=float, default=LR)
    ap.add_argument("--wd", type=float, default=WEIGHT_DECAY)
    ap.add_argument("--save", type=str, default=CKPT_PATH)
    ap.add_argument("--model", type=str, default=MODEL_NAME,
                    help="cnn_lstm | cnn_lstm_physics_fusion | ann_baseline")
    args = ap.parse_args()

    set_seed(SEED)
    train_loader, val_loader, test_loader, (train_idx, _, _), (norm_mean, norm_std) = make_dataloaders()
    print(f"Using dataset normalization: mean={norm_mean:.4f}, std={norm_std:.4f}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}  |  Model: {args.model}")

    # class weights (inverse frequency on TRAIN split)
    class_counts = defaultdict(int)
    for ci, _ in train_idx:
        class_counts[ci] += 1
    total = sum(class_counts.values())
    weights = [total / max(1, class_counts[c]) for c in range(len(CLASSES))]
    class_weights = torch.tensor(weights, dtype=torch.float32, device=device)

    # model + loss + optim
    model = get_model(args.model, num_classes=len(CLASSES), in_ch=1).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTH, weight=class_weights)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)

    steps_per_epoch = max(1, len(train_loader))
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=args.lr,
        epochs=args.epochs,
        steps_per_epoch=steps_per_epoch,
        pct_start=0.1,
        anneal_strategy="cos",
        div_factor=25.0,
        final_div_factor=1e4,
    )

    scaler = GradScaler(enabled=USE_AMP and device.startswith("cuda"))
    ema = EMA(model, decay=0.999)

    # training history for plots
    history = {
        "train_loss": [], "train_acc": [],
        "val_loss":   [], "val_acc":   [],
        "epochs":     [],
        "model":      args.model,
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    best_val = 0.0
    for ep in range(1, args.epochs + 1):
        tr_loss, tr_acc, _, _ = epoch_loop(model, train_loader, criterion,
                                           optimizer, device, scaler, scheduler, ema, train=True)
        va_loss, va_acc, _, _ = epoch_loop(model, val_loader, criterion,
                                           optimizer=None, device=device, scaler=None, scheduler=None, ema=None, train=False)

        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["val_loss"].append(va_loss)
        history["val_acc"].append(va_acc)
        history["epochs"].append(ep)

        if va_acc > best_val:
            best_val = va_acc
            # Save EMA-smoothed weights
            ema_model = get_model(args.model, num_classes=len(CLASSES), in_ch=1).to(device)
            ema.copy_to(ema_model)
            torch.save({
                "model":          ema_model.state_dict(),
                "classes":        CLASSES,
                "norm":           {"mean": float(norm_mean), "std": float(norm_std)},
                "class_weights":  weights,
                "model_name":     args.model,
                "epoch":          ep,
                "best_val_acc":   float(best_val),
            }, args.save)

        print(f"[{ep:03d}/{args.epochs}] "
              f"train_loss={tr_loss:.4f} train_acc={tr_acc:.4f} | "
              f"val_loss={va_loss:.4f} val_acc={va_acc:.4f} | best_val={best_val:.4f}")

        # Persist history every epoch to avoid losing logs on crash
        try:
            os.makedirs(os.path.dirname(HISTORY_PATH) or ".", exist_ok=True)
            with open(HISTORY_PATH, "w", encoding="utf-8") as f:
                json.dump(history, f, indent=2)
        except Exception as e:
            print(f"[WARN] Could not write history to {HISTORY_PATH}: {e}")

    # ---- Test with best EMA checkpoint ----
    ckpt = torch.load(args.save, map_location=device)
    if ckpt.get("model_name", args.model) != args.model:
        print(f"[INFO] Loading checkpoint trained as model={ckpt.get('model_name','?')}")
    model = get_model(ckpt.get("model_name", args.model), num_classes=len(CLASSES), in_ch=1).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    te_loss, te_acc, te_logits, te_targets = epoch_loop(
        model, test_loader, criterion, optimizer=None, device=device, train=False
    )
    print(f"\nTest: loss={te_loss:.4f} acc={te_acc:.4f}")

    cm = confusion_matrix(te_logits, te_targets, num_classes=len(CLASSES))
    print("\nConfusion Matrix (rows=true, cols=pred):")
    header = " " * 18 + " ".join(f"{c[:12]:>12}" for c in CLASSES)
    print(header)
    for i, row in enumerate(cm):
        print(f"{CLASSES[i][:16]:>16} | " + " ".join(f"{x:12d}" for x in row))

    # Latency (forward-only, single sample)
    test_ds = test_loader.dataset
    if len(test_ds) > 0:
        sample_seq, _ = test_ds[0]
        sample_seq = sample_seq.unsqueeze(0)  # (1,T,1,H,W)
        ms = measure_latency(model, sample_seq, device=device, runs=50)
        print(f"\nForward latency: {ms:.2f} ms / sample on {device}")


if __name__ == "__main__":
    main()
