# models.py
from __future__ import annotations
from typing import Optional
import torch
import torch.nn as nn

from config import (
    DROPOUT, LSTM_HIDDEN, LSTM_BIDIR, CLASSES, FEAT_DIM,
    USE_ATTENTION_POOL, PHYSICS_FEAT_DIM, FUSION_MODE, MODEL_NAME
)
# Physics features module
from physics_features import compute_physics_features_batch, feature_dim as physics_feature_dim


# --------------------------
# Shared building blocks
# --------------------------
class CNNFeatureExtractor(nn.Module):
    """(B,in_ch,H,W) -> (B,feat_dim)"""
    def __init__(self, in_ch: int = 1, feat_dim: int = FEAT_DIM, dropout: float = DROPOUT):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1),    nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),    nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),    nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2),

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
    """Additive attention over time. Input: (B,T,H) -> (B,H)"""
    def __init__(self, hid: int):
        super().__init__()
        self.W = nn.Linear(hid, hid, bias=True)
        self.v = nn.Linear(hid, 1,  bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s = torch.tanh(self.W(x))          # (B,T,H)
        e = self.v(s).squeeze(-1)          # (B,T)
        a = torch.softmax(e, dim=1)        # (B,T)
        return (x * a.unsqueeze(-1)).sum(dim=1)  # (B,H)


# --------------------------
# 1) Original CNN+LSTM
# --------------------------
class CNNLSTM(nn.Module):
    """
    CNN over slices + LSTM over time (+ optional attention) + linear head.
    Input:  (B, T, C, H, W) [C=1]
    Output: (B, num_classes)
    """
    def __init__(self, num_classes=len(CLASSES), feat_dim=FEAT_DIM, hidden=LSTM_HIDDEN,
                 bidir=LSTM_BIDIR, dropout=DROPOUT, in_ch=1):
        super().__init__()
        self.cnn = CNNFeatureExtractor(in_ch=in_ch, feat_dim=feat_dim, dropout=dropout)
        self.lstm = nn.LSTM(input_size=feat_dim, hidden_size=hidden, num_layers=1,
                            batch_first=True, bidirectional=bidir)
        lstm_out_dim = hidden * (2 if bidir else 1)
        self.attn = TemporalAttentionPool(lstm_out_dim) if USE_ATTENTION_POOL else None
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


# --------------------------
# 2) CNN + Physics Features Fusion + LSTM
# --------------------------
class CNNLSTMPhysicsFusion(nn.Module):
    """
    Fuse learned slice features with physics-inspired features, then LSTM.

    Pipeline:
      - CNNFeatureExtractor on each (1,H,W) slice -> F_cnn
      - physics_features on each slice -> F_phys (D_phys)
      - Fusion along feature dim (concat; other modes possible)
      - LSTM over time (+ optional attention)
      - Head -> logits
    """
    def __init__(self, num_classes=len(CLASSES), feat_dim=FEAT_DIM, hidden=LSTM_HIDDEN,
                 bidir=LSTM_BIDIR, dropout=DROPOUT, in_ch=1, fusion_mode: str = FUSION_MODE):
        super().__init__()
        # Sanity: match configured physics dim to implementation
        impl_dim = physics_feature_dim()
        assert impl_dim == PHYSICS_FEAT_DIM, \
            f"physics_features({impl_dim}) != config.PHYSICS_FEAT_DIM({PHYSICS_FEAT_DIM})"

        self.fusion_mode = fusion_mode.lower().strip()
        if self.fusion_mode not in {"concat"}:
            raise ValueError(f"Unsupported FUSION_MODE={fusion_mode}. Use 'concat' (default).")

        self.cnn = CNNFeatureExtractor(in_ch=in_ch, feat_dim=feat_dim, dropout=dropout)
        fused_in = feat_dim + PHYSICS_FEAT_DIM

        self.lstm = nn.LSTM(input_size=fused_in, hidden_size=hidden, num_layers=1,
                            batch_first=True, bidirectional=bidir)
        lstm_out_dim = hidden * (2 if bidir else 1)
        self.attn = TemporalAttentionPool(lstm_out_dim) if USE_ATTENTION_POOL else None
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(lstm_out_dim, num_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, T, 1, H, W)
        """
        B, T, C, H, W = x.shape

        # CNN slice features
        x_2d = x.view(B * T, C, H, W)          # (B*T,1,H,W)
        cnn_feats = self.cnn(x_2d).view(B, T, -1)  # (B,T,feat_dim)

        # Physics features (B, T, D_phys)
        phys_feats = compute_physics_features_batch(x)  # (B,T,PHYSICS_FEAT_DIM)

        if self.fusion_mode == "concat":
            feats = torch.cat([cnn_feats, phys_feats], dim=2)  # (B,T,feat_dim+phys)

        # Temporal modeling
        lstm_out, _ = self.lstm(feats)  # (B,T,H*)
        pooled = self.attn(lstm_out) if self.attn is not None else lstm_out.mean(dim=1)
        logits = self.head(pooled)
        return logits


# --------------------------
# 3) ANN Baseline on Physics Features
# --------------------------
class ANNBaseline(nn.Module):
    """
    Simple ANN baseline that consumes physics features only.

    Strategy:
      - Compute physics features per slice -> (B,T,D)
      - Temporal mean pooling -> (B,D)
      - 2-layer MLP -> logits

    This mirrors the "ANN" flavor in many classical papers and gives a fair
    comparison against the hybrid deep model.
    """
    def __init__(self, num_classes=len(CLASSES), dropout=DROPOUT):
        super().__init__()
        D = PHYSICS_FEAT_DIM
        self.fc = nn.Sequential(
            nn.LayerNorm(D),
            nn.Linear(D, 128), nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, T, 1, H, W)
        """
        # Compute physics features and mean-pool across time
        phys = compute_physics_features_batch(x)   # (B,T,D)
        pooled = phys.mean(dim=1)                  # (B,D)
        return self.fc(pooled)


# --------------------------
# Factory helper
# --------------------------
def get_model(name: Optional[str] = None,
              num_classes: int = len(CLASSES),
              in_ch: int = 1) -> nn.Module:
    """
    Create a model by name (defaults to config.MODEL_NAME).
    Valid: "cnn_lstm", "cnn_lstm_physics_fusion", "ann_baseline"
    """
    if name is None:
        name = MODEL_NAME
    key = name.lower().strip()

    if key == "cnn_lstm":
        return CNNLSTM(num_classes=num_classes, in_ch=in_ch)
    if key == "cnn_lstm_physics_fusion":
        return CNNLSTMPhysicsFusion(num_classes=num_classes, in_ch=in_ch)
    if key == "ann_baseline":
        return ANNBaseline(num_classes=num_classes)

    raise ValueError(f"Unknown model name: {name!r}")


__all__ = [
    "CNNFeatureExtractor",
    "TemporalAttentionPool",
    "CNNLSTM",
    "CNNLSTMPhysicsFusion",
    "ANNBaseline",
    "get_model",
]
