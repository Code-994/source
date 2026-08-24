"""
MambaNetBurst — mmTraffic 


Architecture overview :
    raw bytes (0-255)
        → ByteEmbedding          (nn.Embedding, vocab=256)
        → EmbedProjection        (2-layer MLP, optional)
        → cat([seq, CLS_token])  (CLS appended to END)
        → + pos_embed            (learnable, covers seq+CLS)
        → Mamba-2 blocks × depth (residual pre-norm)
        → x[:, -1, :]            (CLS output representation)
        → Linear head            (num_classes)
"""

import math
import os
import sys
from functools import partial
from typing import Optional

import torch
import torch.nn as nn
from timm.models.layers import trunc_normal_

# models_mamba2.py 
_SRC_DIR = os.path.dirname(os.path.abspath(__file__))
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from models_mamba2 import create_block2, RMSNorm, rms_norm_fn
from models_mamba1 import create_block1


# --------------------------------------------------------------------------- #
# Weight initialisation helpers (mirrors models_net_mamba.py style)
# --------------------------------------------------------------------------- #

def _init_weights(
    module,
    n_layer: int,
    initializer_range: float = 0.02,
    rescale_prenorm_residual: bool = True,
    n_residuals_per_layer: int = 1,
):
    """GPT-2-style init: zero biases, rescale residual projections by 1/√N."""
    if isinstance(module, nn.Linear):
        if module.bias is not None:
            if not getattr(module.bias, "_no_reinit", False):
                nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, std=initializer_range)

    if rescale_prenorm_residual:
        # Scale out_proj / fc2 by 1/√(n_residuals_per_layer * n_layer)
        for name, p in module.named_parameters():
            if name in ("out_proj.weight", "fc2.weight"):
                nn.init.kaiming_uniform_(p, a=math.sqrt(5))
                with torch.no_grad():
                    p /= math.sqrt(n_residuals_per_layer * n_layer)


def _segm_init_weights(m):
    if isinstance(m, nn.Linear):
        trunc_normal_(m.weight, std=0.02)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Embedding):
        trunc_normal_(m.weight, std=0.02)
    elif isinstance(m, (nn.LayerNorm, nn.GroupNorm, nn.BatchNorm2d)):
        nn.init.zeros_(m.bias)
        nn.init.ones_(m.weight)


# --------------------------------------------------------------------------- #
# Input pipeline modules
# --------------------------------------------------------------------------- #

class ByteEmbedding(nn.Module):
    """
    Tokenizer-free byte embedding.

    Maps raw byte values {0, …, 255} → R^{d_model} via a learnable lookup table.
    Avoids any tokenisation step; works on arbitrary packet content (Section III-B).
    """

    def __init__(self, d_model: int = 256):
        super().__init__()
        self.embed = nn.Embedding(256, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L)  — integer byte values in [0, 255]
        return self.embed(x)    # (B, L, d_model)


class EmbedProjection(nn.Module):
    """
    Lightweight 2-layer MLP applied per-byte after embedding (Section III-C):

        ẽ_i = W_2 · φ(W_1 · e_i)

    φ = GELU.  Maps d_model → d_mlp → d_model.
    Provides non-linear per-byte feature lifting before the Mamba backbone.
    """

    def __init__(self, d_model: int = 256, d_mlp: int = 512):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_mlp)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(d_mlp, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


# --------------------------------------------------------------------------- #
# Main model
# --------------------------------------------------------------------------- #

class MambaNetBurst(nn.Module):
    """
    MambaNetBurst — tokenizer-free, pretraining-free byte-level classifier.

    Input :  (B, byte_length)  — raw integers in [0, 255]
    Output:  (B, num_classes)  — class logits

    Key design decisions from the paper:
    • CLS token is appended to the END of the byte sequence (not prepended).
    • All positions (bytes + CLS) share learnable positional embeddings.
    • Mamba-2 uses SSD instead of selective scan → faster backward pass.
    • No self-supervised pre-training is required.

    Default config (Section IV-B, Table IV best row):
        d_model=256, depth=4, d_state=16, headdim=64, d_mlp=512, d_conv=4

    mmTraffic 新增：forward_sequence() 返回全序列隐藏状态供 LLM connector 使用。
    """

    def __init__(
        self,
        byte_length: int = 1600,        # 5 packets × 320 bytes
        num_classes: int = 1000,
        # ── model dims ──────────────────────────────────────────────────── #
        d_model: int = 256,
        depth: int = 4,
        # ── Mamba-2 SSM hyper-params ─────────────────────────────────────  #
        d_state: int = 16,              # paper: 16  (Mamba-2 default is 128)
        d_conv: int = 4,
        expand: int = 2,
        headdim: int = 64,             # nheads = expand*d_model/headdim
        ngroups: int = 1,
        chunk_size: int = 256,
        # ── MLP projection ───────────────────────────────────────────────  #
        d_mlp: int = 512,
        use_embed_proj: bool = True,
        # ── regularisation ──────────────────────────────────────────────   #
        drop_rate: float = 0.1,
        # ── implementation ──────────────────────────────────────────────   #
        rms_norm: bool = True,
        residual_in_fp32: bool = True,
        fused_add_norm: bool = True,
        block_fn=None,              # create_block2（默认）或 create_block1
        device=None,
        dtype=None,
        **kwargs,
    ):
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        if block_fn is None:
            block_fn = create_block2

        self.byte_length = byte_length
        self.d_model = d_model
        self.num_features = d_model
        self.num_classes = num_classes
        self.use_embed_proj = use_embed_proj
        self.fused_add_norm = fused_add_norm
        self.residual_in_fp32 = residual_in_fp32

        # ------------------------------------------------------------------ #
        # Stage 1 — input pipeline
        # ------------------------------------------------------------------ #

        # Byte vocabulary: 256 symbols, each → R^{d_model}
        self.byte_embed = ByteEmbedding(d_model)

        # Optional per-byte MLP (Section III-C); ablation "No emb proj" drops it
        if use_embed_proj:
            self.embed_proj = EmbedProjection(d_model, d_mlp)

        # CLS token (appended to END of sequence, Section III-D)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))

        # Learnable positional embeddings for all byte positions + CLS
        self.pos_embed = nn.Parameter(torch.zeros(1, byte_length + 1, d_model))

        self.pos_drop = nn.Dropout(p=drop_rate)

        # ------------------------------------------------------------------ #
        # Stage 2 — Mamba-2 backbone
        # ------------------------------------------------------------------ #

        _mamba2_kwargs = dict(headdim=headdim, ngroups=ngroups, chunk_size=chunk_size)
        self.blocks = nn.ModuleList([
            block_fn(
                d_model=d_model,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
                **(  _mamba2_kwargs if block_fn is create_block2 else {}),
                rms_norm=rms_norm,
                residual_in_fp32=residual_in_fp32,
                fused_add_norm=fused_add_norm,
                layer_idx=i,
                **factory_kwargs,
            )
            for i in range(depth)
        ])

        # Final norm (applied to the last residual stream before classification)
        norm_cls = RMSNorm if rms_norm else nn.LayerNorm
        self.norm_f = norm_cls(d_model, eps=1e-5)

        # ------------------------------------------------------------------ #
        # Stage 3 — classifier head
        # ------------------------------------------------------------------ #
        self.head = nn.Linear(d_model, num_classes) if num_classes > 0 else nn.Identity()

        # ------------------------------------------------------------------ #
        # Initialisation
        # ------------------------------------------------------------------ #
        self._init_model_weights(depth)

    # ---------------------------------------------------------------------- #
    # Initialisation helpers
    # ---------------------------------------------------------------------- #

    def _init_model_weights(self, depth: int):
        trunc_normal_(self.cls_token, std=0.02)
        trunc_normal_(self.pos_embed, std=0.02)
        self.byte_embed.apply(_segm_init_weights)
        if self.use_embed_proj:
            self.embed_proj.apply(_segm_init_weights)
        self.head.apply(_segm_init_weights)
        self.apply(partial(_init_weights, n_layer=depth))

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"pos_embed", "cls_token"}

    # ---------------------------------------------------------------------- #
    # Forward
    # ---------------------------------------------------------------------- #

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode raw bytes into a CLS-token representation.

        Args:
            x: (B, L) integer byte values, L = byte_length

        Returns:
            (B, d_model) — output at the CLS position
        """
        B = x.shape[0]

        # ── Byte embedding ───────────────────────────────────────────────── #
        x = self.byte_embed(x)                         # (B, L, d_model)

        # ── Optional MLP projection ──────────────────────────────────────── #
        if self.use_embed_proj:
            x = self.embed_proj(x)                     # (B, L, d_model)

        # ── Append CLS token to END (Section III-D) ──────────────────────── #
        cls_tokens = self.cls_token.expand(B, -1, -1)  # (B, 1, d_model)
        x = torch.cat([x, cls_tokens], dim=1)          # (B, L+1, d_model)

        # ── Add positional embeddings ─────────────────────────────────────── #
        x = x + self.pos_embed                         # (B, L+1, d_model)
        x = self.pos_drop(x)

        # ── Mamba-2 backbone ─────────────────────────────────────────────── #
        residual = None
        hidden_states = x
        for blk in self.blocks:
            hidden_states, residual = blk(hidden_states, residual)

        # ── Final norm (fuse residual stream) ────────────────────────────── #
        if self.fused_add_norm and rms_norm_fn is not None:
            # Triton fused Add+Norm for the last residual
            x = rms_norm_fn(
                hidden_states,
                self.norm_f.weight,
                getattr(self.norm_f, "bias", None),
                eps=self.norm_f.eps,
                residual=residual,
                prenorm=False,
                residual_in_fp32=self.residual_in_fp32,
            )
        else:
            # Unfused fallback
            if residual is not None:
                hidden_states = hidden_states + residual
            x = self.norm_f(hidden_states)

        # ── CLS representation (last position) ───────────────────────────── #
        return x[:, -1, :]                             # (B, d_model)

    # ------------------------------------------------------------------ #
    # [mmTraffic 新增] forward_sequence
    # 与 forward_features 共享相同的前向路径，但返回所有字节位置的隐藏状态，
    # 供 MambaNetBurstVisionTower 做 AdaptiveAvgPool 后传给 LLM connector。
    # ------------------------------------------------------------------ #
    def forward_sequence(self, x: torch.Tensor) -> torch.Tensor:
        """
        返回所有字节位置经过完整 Mamba-2 backbone 后的隐藏状态（不含末尾 CLS）。

        Args:
            x: (B, L) int64，原始字节值 [0, 255]，L = byte_length

        Returns:
            (B, L, d_model) — 每个字节位置的上下文化隐藏状态
        """
        B = x.shape[0]

        # ── Byte embedding ───────────────────────────────────────────────── #
        x = self.byte_embed(x)                         # (B, L, d_model)

        # ── Optional MLP projection ──────────────────────────────────────── #
        if self.use_embed_proj:
            x = self.embed_proj(x)                     # (B, L, d_model)

        # ── Append CLS token to END (Section III-D) ──────────────────────── #
        cls_tokens = self.cls_token.expand(B, -1, -1)  # (B, 1, d_model)
        x = torch.cat([x, cls_tokens], dim=1)          # (B, L+1, d_model)

        # ── Add positional embeddings ─────────────────────────────────────── #
        x = x + self.pos_embed                         # (B, L+1, d_model)
        x = self.pos_drop(x)

        # ── Mamba-2 backbone ─────────────────────────────────────────────── #
        residual = None
        hidden_states = x
        for blk in self.blocks:
            hidden_states, residual = blk(hidden_states, residual)

        # ── Final norm (fuse residual stream) ────────────────────────────── #
        if self.fused_add_norm and rms_norm_fn is not None:
            x = rms_norm_fn(
                hidden_states,
                self.norm_f.weight,
                getattr(self.norm_f, "bias", None),
                eps=self.norm_f.eps,
                residual=residual,
                prenorm=False,
                residual_in_fp32=self.residual_in_fp32,
            )
        else:
            if residual is not None:
                hidden_states = hidden_states + residual
            x = self.norm_f(hidden_states)

        # 去掉末尾 CLS，只返回字节位置
        return x[:, :-1, :]                            # (B, L, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, L) raw byte integers in [0, 255], L = byte_length

        Returns:
            (B, num_classes) logits
        """
        return self.head(self.forward_features(x))


# --------------------------------------------------------------------------- #
# Factory functions
# --------------------------------------------------------------------------- #

def mambanetburst_classifier(**kwargs) -> MambaNetBurst:
    """
    Default MambaNetBurst (paper Table IV best configuration):
        d_model=256, depth=4, d_state=16, headdim=64, d_mlp=512, d_conv=4
    ~2.5 M parameters.
    """
    return MambaNetBurst(
        d_model=256,
        depth=4,
        d_state=16,
        d_conv=4,
        expand=2,
        headdim=64,
        ngroups=1,
        chunk_size=256,
        d_mlp=512,
        use_embed_proj=True,
        **kwargs,
    )


def mambanetburst_compact(**kwargs) -> MambaNetBurst:
    """
    Compact variant (paper Table IV 'Compact(64/64/2)'):
        d_model=64, depth=2, headdim=16, d_mlp=64
    ~0.2 M parameters.
    """
    return MambaNetBurst(
        d_model=64,
        depth=2,
        d_state=16,
        d_conv=4,
        expand=2,
        headdim=16,         # nheads = 2*64/16 = 8
        ngroups=1,
        chunk_size=256,
        d_mlp=64,
        use_embed_proj=True,
        **kwargs,
    )


def mambanetburst_medium(**kwargs) -> MambaNetBurst:
    """
    Intermediate variant: d_model=128, depth=4, headdim=32.
    ~0.7 M parameters.
    """
    return MambaNetBurst(
        d_model=128,
        depth=4,
        d_state=16,
        d_conv=4,
        expand=2,
        headdim=32,         # nheads = 2*128/32 = 8
        ngroups=1,
        chunk_size=256,
        d_mlp=256,
        use_embed_proj=True,
        **kwargs,
    )

    """
    Mamba-1 对比模型（速度基准用）。

    与 mambanetburst_classifier 参数对齐（d_model=256, depth=4, d_state=16），
    仅将 Mamba-2 SSD 核替换为 Mamba-1 Selective Scan。
    """

def mambanetburst_mamba1_classifier(**kwargs) -> MambaNetBurst:

    return MambaNetBurst(
        d_model=256,
        depth=4,
        d_state=16,
        d_conv=4,
        expand=2,
        d_mlp=512,
        use_embed_proj=True,
        block_fn=create_block1,
        **kwargs,
    )
