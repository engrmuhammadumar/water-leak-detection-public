import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

from config import *
from data import make_dataloaders
from models import CNNLSTM, CNNLSTM_Fusion
from utils import set_seed, measure_latency

# -----------------------------
# Train + Eval Loop
# -----------------------------
def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, correct, count = 0.0, 0, 0

    for batch in loader:
        if USE_PHYSICS_FEATURES:
            seq, phys, labels = batch
            seq, phys, labels = seq.to(device), phys.to(device), labels.to(device)
            outputs = model(seq, phys)
        else:
            seq, labels = batch
            seq, labels = seq.to(device), labels.to(device)
            outputs = model(seq)

        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * labels.size(0)
        _, preds = outputs.max(1)
        correct += (preds == labels).sum().item()
        count += labels.size(0)

    return total_loss / count, correct / count


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, count = 0.0, 0, 0
    all_labels, all_preds = [], []

    with torch.no_grad():
        for batch in loader:
            if USE_PHYSICS_FEATURES:
                seq, phys, labels = batch
                seq, phys, labels = seq.to(device), phys.to(device), labels.to(device)
                outputs = model(seq, phys)
            else:
                seq, labels = batch
                seq, labels = seq.to(device), labels.to(device)
                outputs = model(seq)

            loss = criterion(outputs, labels)
            total_loss += loss.item() * labels.size(0)
            _, preds = outputs.max(1)
            correct += (preds == labels).sum().item()
            count += labels.size(0)

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    avg_loss = total_loss / count
    avg_acc = correct / count
    return avg_loss, avg_acc, all_labels, all_preds


# -----------------------------
# Main Training Script
# -----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=3e-3)
    args = parser.parse_args()

    set_seed(SEED)

    # Data
    train_loader, val_loader, test_loader, (train_idx, val_idx, test_idx), (norm_mean, norm_std) = make_dataloaders()
    print(f"Using dataset normalization: mean={norm_mean:.4f}, std={norm_std:.4f}")

    # Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if USE_PHYSICS_FEATURES:
        model = CNNLSTM_Fusion(num_classes=len(CLASSES))
    else:
        model = CNNLSTM(num_classes=len(CLASSES))
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    best_val, best_state = 0.0, None
    train_losses, val_losses, train_accs, val_accs = [], [], [], []

    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion, device)

        train_losses.append(tr_loss)
        train_accs.append(tr_acc)
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        if val_acc > best_val:
            best_val = val_acc
            best_state = model.state_dict()

        print(f"[{epoch:03d}/{args.epochs}] "
              f"train_loss={tr_loss:.4f} train_acc={tr_acc:.4f} | "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} | best_val={best_val:.4f}")

    # Save best model
    torch.save(best_state, "cnn_lstm_cwt.pt")

    # Final test evaluation
    model.load_state_dict(best_state)
    test_loss, test_acc, y_true, y_pred = evaluate(model, test_loader, criterion, device)
    print(f"\nTest: loss={test_loss:.4f} acc={test_acc:.4f}\n")

    print(classification_report(y_true, y_pred, target_names=CLASSES))
    print("Confusion Matrix:\n", confusion_matrix(y_true, y_pred))

    # Latency test
    test_ds = test_loader.dataset
    if len(test_ds) > 0:
        sample = test_ds[0]
        if USE_PHYSICS_FEATURES:
            seq, phys, _ = sample
            seq, phys = seq.unsqueeze(0).to(device), phys.unsqueeze(0).to(device)
            ms = measure_latency(lambda x: model(x, phys), seq, device=device, runs=50)
        else:
            seq, _ = sample
            seq = seq.unsqueeze(0).to(device)
            ms = measure_latency(model, seq, device=device, runs=50)
        print(f"\nForward latency: {ms:.2f} ms / sample")


if __name__ == "__main__":
    main()
