# sanity_overfit.py
import torch, torch.nn as nn, torch.optim as optim
from data import make_dataloaders
from models import CNNLSTM, CNNLSTM_Fusion
from config import CLASSES, USE_PHYSICS_FEATURES, NUM_PHYSICS_FEATURES

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

train_loader, _, _, _, _ = make_dataloaders()

# shrink dataset to small subset to ensure overfitting is possible
subset = 200
train_loader.dataset.indices = train_loader.dataset.indices[:subset]

if USE_PHYSICS_FEATURES:
    model = CNNLSTM_Fusion(num_classes=len(CLASSES), physics_dim=NUM_PHYSICS_FEATURES).to(device)
else:
    model = CNNLSTM(num_classes=len(CLASSES)).to(device)

opt = optim.AdamW(model.parameters(), lr=1e-3)
crit = nn.CrossEntropyLoss()

for ep in range(1, 11):
    model.train()
    tot, correct, total = 0, 0, 0
    for batch in train_loader:
        if USE_PHYSICS_FEATURES:
            x, p, y = batch; x, p, y = x.to(device), p.to(device), y.to(device)
            out = model(x, p)
        else:
            x, y = batch; x, y = x.to(device), y.to(device)
            out = model(x)
        loss = crit(out, y)
        opt.zero_grad(); loss.backward(); opt.step()
        tot += loss.item()
        correct += (out.argmax(1) == y).sum().item()
        total += y.size(0)
    print(f"Epoch {ep} | loss={tot/len(train_loader):.4f}, acc={correct/total:.4f}")
