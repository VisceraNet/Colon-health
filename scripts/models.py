# models.py

import torch
import torch.nn as nn
import timm


class EFFResNetViT(nn.Module):
    """
    Hybrid CNN–Transformer model for fuzzy medical image severity estimation.
    - EfficientNet + ResNet for spatial feature extraction
    - Transformer encoder for global reasoning
    - Binary head: Remission vs Active (safety-critical)
    - Ranking head: Ordinal severity score (ListNet-compatible)
    """

    def __init__(self):
        super().__init__()

        # =========================
        # CNN BACKBONES (spatial)
        # =========================

        # EfficientNet (lighter than B4 if you want to swap later)
        self.eff = timm.create_model(
            "efficientnet_b4",   # you can change to b2/b3 if needed
            pretrained=True,
            features_only=True
        )
        eff_dim = self.eff.feature_info[-1]["num_chs"]

        # ResNet-50
        self.res = timm.create_model(
            "resnet50",
            pretrained=True,
            features_only=True
        )
        res_dim = self.res.feature_info[-1]["num_chs"]

        fused_dim = eff_dim + res_dim

        # =========================
        # FEATURE FUSION
        # =========================

        # 1×1 conv to align channels → transformer dimension
        self.fusion = nn.Conv2d(
            fused_dim,
            768,
            kernel_size=1,
            bias=False
        )

        # =========================
        # TRANSFORMER ENCODER
        # =========================

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=768,
            nhead=12,
            dim_feedforward=3072,
            dropout=0.2,
            batch_first=True
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=3
        )

        # =========================
        # TASK HEADS (NOT strict classification)
        # =========================

        # Shared normalization
        self.norm = nn.LayerNorm(768)

        # Binary safety head (Remission vs Active)
        self.binary_head = nn.Linear(768, 1)

        # Ordinal / ranking head (continuous severity)
        self.rank_head = nn.Linear(768, 1)

    # =========================
    # FORWARD
    # =========================
    def forward(self, x):
        """
        Returns:
            dict with:
              - binary_logits: (B, 1)
              - severity_score: (B,)
        """

        # CNN feature maps
        eff_feat = self.eff(x)[-1]   # (B, Ce, H, W)
        res_feat = self.res(x)[-1]   # (B, Cr, H, W)

        # Fuse spatial features
        fused = torch.cat([eff_feat, res_feat], dim=1)  # (B, C, H, W)
        fused = self.fusion(fused)                       # (B, 768, H, W)

        # Convert spatial grid → tokens
        tokens = fused.flatten(2).transpose(1, 2)       # (B, N, 768)

        # Transformer reasoning
        tokens = self.transformer(tokens)

        # Global aggregation (robust for noisy labels)
        pooled = tokens.mean(dim=1)                      # (B, 768)
        pooled = self.norm(pooled)

        # Heads
        binary_logits = self.binary_head(pooled)         # (B, 1)
        severity_score = self.rank_head(pooled).squeeze(1)  # (B,)

        return {
            "binary_logits": binary_logits,
            "severity_score": severity_score
        }
