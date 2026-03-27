"""
MODNet — Matting Objective Decomposition Network
Produces fine-grained alpha transparency for hair, fur, glass, etc.

Uses a MobileNetV2 backbone with three branches:
  S-branch (Semantic), D-branch (Detail), F-branch (Fusion)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from bg_models.modnet.arch.mobilenetv2 import MobileNetV2
from bg_models.modnet.arch.wrapper import Conv2dIBNormRelu, SEBlock, GaussianBlurLayer


class MODNet(nn.Module):
    """
    MODNet architecture for portrait/alpha matting.
    Three-branch design: Semantic (S), Detail (D), Fusion (F).
    """

    def __init__(self, in_channels=3, hr_channels=32, backbone_pretrained=False):
        super().__init__()

        self.in_channels = in_channels
        self.hr_channels = hr_channels

        # ── Backbone ─────────────────────────────────────────
        self.backbone = MobileNetV2()

        # ── S-Branch (Semantic Estimation) ───────────────────
        self.se_block = SEBlock(self.backbone.last_channel, reduction=4)
        self.s_head = nn.Sequential(
            nn.Conv2d(self.backbone.last_channel, 1, kernel_size=1),
            nn.Sigmoid(),
        )

        # ── D-Branch (Detail Estimation) ─────────────────────
        self.d_conv1 = Conv2dIBNormRelu(in_channels + 1, hr_channels, 3, padding=1)
        self.d_conv2 = Conv2dIBNormRelu(hr_channels + self.backbone.last_channel, hr_channels, 3, padding=1)
        self.d_conv3 = Conv2dIBNormRelu(hr_channels, hr_channels, 3, padding=1)
        self.d_conv4 = Conv2dIBNormRelu(hr_channels, hr_channels, 3, padding=1)
        self.d_conv5 = nn.Sequential(
            nn.Conv2d(hr_channels, 1, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )

        # ── F-Branch (Fusion) ────────────────────────────────
        self.f_conv1 = Conv2dIBNormRelu(in_channels + 2 + hr_channels, hr_channels, 3, padding=1)
        self.f_conv2 = Conv2dIBNormRelu(hr_channels, hr_channels, 3, padding=1)
        self.f_conv3 = nn.Sequential(
            nn.Conv2d(hr_channels, 1, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )

        # Gaussian blur for S-branch supervision
        self.blur = GaussianBlurLayer(1, kernel_size=3)

    def forward(self, x, inference=False):
        """
        Forward pass.
        
        Args:
            x: Input tensor [B, 3, H, W] normalized to [-1, 1]
            inference: If True, only return the fused alpha matte
            
        Returns:
            If inference=True: alpha matte [B, 1, H, W]
            If inference=False: (semantic, detail, matte) tuple
        """
        # ── Backbone features ────────────────────────────────
        enc_features = self.backbone(x)
        enc2x = enc_features[0]    # stride 2
        enc4x = enc_features[1]    # stride 4
        enc32x = enc_features[-1]  # stride 32 (deepest)

        # ── S-Branch ─────────────────────────────────────────
        s = self.se_block(enc32x)
        s = self.s_head(s)             # [B, 1, H/32, W/32]

        # Upsample semantic prediction to full resolution
        s_full = F.interpolate(s, scale_factor=32, mode="bilinear", align_corners=False)

        if inference:
            # ── D-Branch ─────────────────────────────────────
            d = torch.cat([x, s_full], dim=1)  # [B, 4, H, W]
            d = self.d_conv1(d)

            # Incorporate deep features
            enc32x_up = F.interpolate(enc32x, size=d.shape[2:], mode="bilinear", align_corners=False)
            d = torch.cat([d, enc32x_up], dim=1)
            d = self.d_conv2(d)
            d = self.d_conv3(d)
            d_feat = self.d_conv4(d)
            d_out = self.d_conv5(d_feat)  # [B, 1, H, W]

            # ── F-Branch ─────────────────────────────────────
            f = torch.cat([x, s_full, d_out, d_feat], dim=1)
            f = self.f_conv1(f)
            f = self.f_conv2(f)
            matte = self.f_conv3(f)  # [B, 1, H, W]

            return matte

        # Training: return all three outputs
        # D-Branch
        d = torch.cat([x, s_full.detach()], dim=1)
        d = self.d_conv1(d)
        enc32x_up = F.interpolate(enc32x, size=d.shape[2:], mode="bilinear", align_corners=False)
        d = torch.cat([d, enc32x_up], dim=1)
        d = self.d_conv2(d)
        d = self.d_conv3(d)
        d_feat = self.d_conv4(d)
        d_out = self.d_conv5(d_feat)

        # F-Branch
        f = torch.cat([x, s_full.detach(), d_out.detach(), d_feat.detach()], dim=1)
        f = self.f_conv1(f)
        f = self.f_conv2(f)
        matte = self.f_conv3(f)

        return s, d_out, matte
