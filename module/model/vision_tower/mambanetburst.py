"""
MambaNetBurst VisionTower for mmTraffic.

Replaces NetMamba as the traffic perception module.
Input:  (B, 1, 1600) float32 in [0,1]  (BGTD npy format, same as NetMamba)
Output: (B, 400, 256) features  +  (B, num_classes) logits

Pipeline:
  float [0,1] → int64 [0,255]          (ByteEmbedding format)
  → MambaNetBurst.forward_sequence()   (B, 1600, 256)
  → AdaptiveAvgPool1d(400)             (B, 400, 256)
"""

import os
import sys
from collections import OrderedDict

import torch
import torch.nn as nn

from . import register_vision_tower
from .base import VisionTower

# ---------------------------------------------------------------------------
# Add MambaNetBurst src to path
# ---------------------------------------------------------------------------
_MAMBANETBURST_SRC = "/root/autodl-tmp/MambaNetBurst/src"
if _MAMBANETBURST_SRC not in sys.path:
    sys.path.insert(0, _MAMBANETBURST_SRC)

from models_net_mamba2_mmtraffic import mambanetburst_classifier  # noqa: E402


@register_vision_tower('mambanetburst')
class MambaNetBurstVisionTower(VisionTower):
    """
    Traffic perception module built on MambaNetBurst (Mamba-2, tokenizer-free).

    Differences from NetmambaVisionTower:
    - Input stays as float [0,1]; conversion to int64 happens inside forward()
    - forward_sequence() is used instead of the full forward() to get all hidden states
    - 1600 byte-position features are pooled to 400 via AdaptiveAvgPool1d
    - vision_hidden_size = 256 (d_model), not 512
    """

    def __init__(self, cfg):
        super().__init__(cfg)
        device = getattr(cfg, "device", None)
        dtype = getattr(cfg, "dtype", None)
        if dtype is None:
            dtype = torch.float32

        # num_classes placeholder; head not used in mmTraffic inference path
        self._vision_tower = mambanetburst_classifier(
            num_classes=7,
            byte_length=1600,
            device=device,
            dtype=dtype,
        )
        self._image_processor = None

        # Pool 1600 byte positions → 400 tokens to match NetMamba's output length
        self._pool = nn.AdaptiveAvgPool1d(400)

    def _load_model(self, vision_tower_name, **kwargs):
        pretrained_vision_tower_path = kwargs.pop(
            'pretrained_vision_tower_path', None
        )

        if pretrained_vision_tower_path is None:
            print("[MambaNetBurst] No pretrained checkpoint provided, using random init.")
            return

        ckpt_path = pretrained_vision_tower_path
        if os.path.isdir(ckpt_path):
            ckpt_path = os.path.join(ckpt_path, "pytorch_model.bin")

        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"[MambaNetBurst] Checkpoint not found: {ckpt_path}")

        checkpoint = torch.load(ckpt_path, map_location="cpu")
        print("[MambaNetBurst] Loading checkpoint from:", ckpt_path)

        # Extract model state dict
        if isinstance(checkpoint, dict) and "model" in checkpoint:
            checkpoint_model = checkpoint["model"]
        else:
            checkpoint_model = checkpoint

        # Remove DDP 'module.' prefix if present
        new_state_dict = OrderedDict()
        for k, v in checkpoint_model.items():
            name = k[7:] if k.startswith('module.') else k
            new_state_dict[name] = v

        # Handle class-count mismatch in head (checkpoint may have different nb_classes)
        if "head.weight" in new_state_dict:
            pretrained_nc = new_state_dict["head.weight"].shape[0]
            current_nc = self._vision_tower.head.weight.shape[0]
            if pretrained_nc != current_nc:
                print(f"[MambaNetBurst] Resizing head: {current_nc} → {pretrained_nc}")
                in_feat = self._vision_tower.head.in_features
                self._vision_tower.head = nn.Linear(in_feat, pretrained_nc).to(
                    device=self._vision_tower.head.weight.device,
                    dtype=self._vision_tower.head.weight.dtype,
                )

        msg = self._vision_tower.load_state_dict(new_state_dict, strict=False)
        print("[MambaNetBurst] Load result:", msg)

    def forward(self, x, **kwargs):
        """
        Args:
            x: (B, 1, 1600) float32 in [0, 1]  — BGTD npy format

        Returns:
            features: (B, 400, 256)
            logits:   (B, num_classes)
        """
        device = x.device
        self.to(device)

        # (B, 1, 1600) → (B, 1600) int64 in [0, 255]
        x_int = (x.squeeze(1) * 255.0).clamp(0, 255).long()   # (B, 1600)

        # Full hidden states for all byte positions: (B, 1600, 256)
        features = self._vision_tower.forward_sequence(x_int)   # (B, 1600, 256)

        # AdaptiveAvgPool1d operates on (B, C, L); transpose accordingly
        # (B, 1600, 256) → (B, 256, 1600) → pool → (B, 256, 400) → (B, 400, 256)
        features = self._pool(
            features.transpose(1, 2)
        ).transpose(1, 2)                                        # (B, 400, 256)

        # Classification logits from CLS token
        logits = self._vision_tower(x_int)                       # (B, num_classes)

        return features, logits
