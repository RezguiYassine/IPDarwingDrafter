"""Hatch-region segmentation model — lightweight U-Net with frozen ResNet18 encoder.

Design choices (agreed):
  • Frozen ResNet18 encoder (ImageNet pretrained) — avoids overfitting on ~50 figures.
    Unfreeze via model.unfreeze_encoder() after initial convergence if needed.
  • Grayscale input (1 ch): first ResNet conv weights are averaged across the 3 RGB
    channels so the pretrained initialization is preserved.
  • Decoder: 4 bilinear-upsample + skip-concat + DoubleConv blocks → pixel logits.
  • Output: (B, 1, H, W) raw logits; apply sigmoid for probabilities.

Usage:
    from tools.hatch_model import HatchUNet
    model = HatchUNet(freeze_encoder=True)
    logits = model(patch)   # patch: (B, 1, 512, 512) float32 in [0, 1]
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tvm


class _DoubleConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class _DecoderBlock(nn.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.conv = _DoubleConv(in_ch + skip_ch, out_ch)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=True)
        return self.conv(torch.cat([x, skip], dim=1))


class HatchUNet(nn.Module):
    """U-Net with ResNet18 encoder for hatch-region pixel segmentation.

    Input : (B, 1, H, W)  grayscale patch, float32, values in [0, 1]
    Output: (B, 1, H, W)  raw logits (apply sigmoid for probabilities)
    """

    def __init__(self, freeze_encoder: bool = True):
        super().__init__()
        bb = tvm.resnet18(weights=tvm.ResNet18_Weights.DEFAULT)

        # ── Adapt first conv for grayscale input ──────────────────────────────
        # Average the 3 RGB channels → 1 input channel, preserving learned weights.
        orig_w = bb.conv1.weight.data          # (64, 3, 7, 7)
        new_conv = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        new_conv.weight.data = orig_w.mean(dim=1, keepdim=True)
        bb.conv1 = new_conv

        # ── Encoder stages ────────────────────────────────────────────────────
        # e0: stride 2  → H/2  × W/2  × 64
        self.enc0 = nn.Sequential(bb.conv1, bb.bn1, bb.relu)
        # e1: stride 4  → H/4  × W/4  × 64   (maxpool + layer1)
        self.enc1 = nn.Sequential(bb.maxpool, bb.layer1)
        # e2: stride 8  → H/8  × W/8  × 128
        self.enc2 = bb.layer2
        # e3: stride 16 → H/16 × W/16 × 256
        self.enc3 = bb.layer3
        # e4: stride 32 → H/32 × W/32 × 512  (bottleneck)
        self.enc4 = bb.layer4

        if freeze_encoder:
            for p in [self.enc0, self.enc1, self.enc2, self.enc3, self.enc4]:
                for param in p.parameters():
                    param.requires_grad_(False)

        # ── Decoder ───────────────────────────────────────────────────────────
        self.dec4 = _DecoderBlock(512, 256, 256)   # /32 → /16
        self.dec3 = _DecoderBlock(256, 128, 128)   # /16 → /8
        self.dec2 = _DecoderBlock(128, 64, 64)     # /8  → /4
        self.dec1 = _DecoderBlock(64, 64, 32)      # /4  → /2
        self.dec0 = nn.Sequential(                  # /2  → /1
            _DoubleConv(32, 16),
        )
        self.head = nn.Conv2d(16, 1, kernel_size=1)

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e0 = self.enc0(x)           # (B, 64,  H/2,  W/2)
        e1 = self.enc1(e0)          # (B, 64,  H/4,  W/4)
        e2 = self.enc2(e1)          # (B, 128, H/8,  W/8)
        e3 = self.enc3(e2)          # (B, 256, H/16, W/16)
        e4 = self.enc4(e3)          # (B, 512, H/32, W/32)

        d = self.dec4(e4, e3)       # (B, 256, H/16, W/16)
        d = self.dec3(d, e2)        # (B, 128, H/8,  W/8)
        d = self.dec2(d, e1)        # (B, 64,  H/4,  W/4)
        d = self.dec1(d, e0)        # (B, 32,  H/2,  W/2)
        d = F.interpolate(d, scale_factor=2, mode="bilinear", align_corners=True)
        d = self.dec0(d)            # (B, 16,  H,    W)
        return self.head(d)         # (B, 1,   H,    W)  logits

    # ── Helpers ───────────────────────────────────────────────────────────────

    def unfreeze_encoder(self, blocks: int = 2) -> None:
        """Unfreeze the last `blocks` ResNet stages for fine-tuning."""
        stages = [self.enc4, self.enc3, self.enc2, self.enc1, self.enc0]
        for stage in stages[:blocks]:
            for p in stage.parameters():
                p.requires_grad_(True)

    def decoder_parameters(self):
        for m in [self.dec4, self.dec3, self.dec2, self.dec1, self.dec0, self.head]:
            yield from m.parameters()

    def encoder_parameters(self):
        for m in [self.enc0, self.enc1, self.enc2, self.enc3, self.enc4]:
            yield from m.parameters()
