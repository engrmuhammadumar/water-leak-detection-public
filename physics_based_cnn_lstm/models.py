# models.py
from __future__ import annotations
import torch
import torch.nn as nn
from config import (
    DROPOUT, LSTM_HIDDEN, LSTM_BIDIR, CLASSES, FEAT_DIM,
    USE_PHYSICS_FEATURES, NUM_PHYSICS_FEATURES, PHYSICS_MLP_HIDDEN, PHYSICS_DROP
)

class CNNFeatureExtractor(nn.Module):
    """(B,in_ch,H,W) -> (B,feat_dim)"""
    def __init__(self, in_ch: int = 1, feat_dim: int = FEAT_DIM, dropout: float = DROPOUT):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1),    nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2),  # 1/2

            nn.Conv2d(32, 64, 3, padding=1),    nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),    nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2),  # 1/4

            nn.Conv2d(64, 128, 3, padding=1),   nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1),  nn.BatchNorm2d(128), nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(128, feat_dim),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class TemporalAttentionPool(nn.Module):
    def __init__(self, hid: int):
        super().__init__()
        self.W = nn.Linear(hid, hid, bias=True)
        self.v = nn.Linear(hid, 1,  bias=False)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s = torch.tanh(self.W(x))   # (B,T,H)
        e = self.v(s).squeeze(-1)   # (B,T)
        a = torch.softmax(e, dim=1) # (B,T)
        return (x * a.unsqueeze(-1)).sum(dim=1)

class CNNLSTM(nn.Module):
    """
    CNN over slices + LSTM over time + attention (or mean) pooling.
    Input:  (B, T, C, H, W) [C=1]
    Output: (B, num_classes)
    """
    def __init__(self, num_classes=len(CLASSES), feat_dim=FEAT_DIM, hidden=LSTM_HIDDEN,
                 bidir=LSTM_BIDIR, dropout=DROPOUT, in_ch=1, use_attention=True):
        super().__init__()
        self.cnn = CNNFeatureExtractor(in_ch=in_ch, feat_dim=feat_dim, dropout=dropout)
        self.lstm = nn.LSTM(input_size=feat_dim, hidden_size=hidden, num_layers=1,
                            batch_first=True, bidirectional=bidir)
        lstm_out_dim = hidden * (2 if bidir else 1)
        self.attn = TemporalAttentionPool(lstm_out_dim) if use_attention else None
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(lstm_out_dim, num_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C, H, W = x.shape
        x = x.view(B*T, C, H, W)
        feats = self.cnn(x)             # (B*T, feat_dim)
        feats = feats.view(B, T, -1)    # (B, T, feat_dim)
        lstm_out, _ = self.lstm(feats)  # (B, T, H*)
        pooled = self.attn(lstm_out) if self.attn is not None else lstm_out.mean(dim=1)
        logits = self.head(pooled)
        return logits

class CNNLSTM_Fusion(nn.Module):
    """
    Fusion model: image-sequence branch + physics MLP branch.
    Inputs:
      - seq: (B,T,1,H,W)
      - physics_feats: (B,F)
    """
    def __init__(self, num_classes=len(CLASSES), feat_dim=FEAT_DIM, hidden=LSTM_HIDDEN,
                 bidir=LSTM_BIDIR, dropout=DROPOUT, in_ch=1, physics_dim=NUM_PHYSICS_FEATURES,
                 physics_hidden=PHYSICS_MLP_HIDDEN, physics_drop=PHYSICS_DROP, use_attention=True):
        super().__init__()
        self.seq_branch = CNNLSTM(num_classes=num_classes, feat_dim=feat_dim, hidden=hidden,
                                  bidir=bidir, dropout=dropout, in_ch=in_ch, use_attention=use_attention)
        # Replace classifier head by identity to get pooled feature
        self.seq_branch.head = nn.Identity()
        lstm_out_dim = hidden * (2 if bidir else 1)

        self.phys_mlp = nn.Sequential(
            nn.LayerNorm(physics_dim),
            nn.Linear(physics_dim, physics_hidden),
            nn.ReLU(),
            nn.Dropout(physics_drop),
            nn.Linear(physics_hidden, physics_hidden),
            nn.ReLU(),
        )

        self.fuse = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(lstm_out_dim + physics_hidden, lstm_out_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(lstm_out_dim, num_classes),
        )

    def forward(self, seq: torch.Tensor, physics_feats: torch.Tensor) -> torch.Tensor:
        # pooled sequence features
        B, T, C, H, W = seq.shape
        x = seq.view(B*T, C, H, W)
        feats = self.seq_branch.cnn(x).view(B, T, -1)
        lstm_out, _ = self.seq_branch.lstm(feats)
        pooled = (self.seq_branch.attn(lstm_out) if isinstance(self.seq_branch.attn, TemporalAttentionPool)
                  else lstm_out.mean(dim=1))   # (B, H*)

        phys = self.phys_mlp(physics_feats)    # (B, H_phys)
        z = torch.cat([pooled, phys], dim=1)
        return self.fuse(z)
