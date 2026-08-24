"""
Mamba-2 building blocks for MambaNetBurst.
MambaNetBurst/src/models_mamba2.py
"""

from __future__ import annotations

import os
import sys
from functools import partial
from typing import Optional

import torch
import torch.nn as nn

# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------

_SRC_DIR = os.path.dirname(os.path.abspath(__file__))
_MAMBA_DIR = os.path.normpath(os.path.join(_SRC_DIR, "..", "mamba"))
if _MAMBA_DIR not in sys.path:
    sys.path.insert(0, _MAMBA_DIR)


# ---------------------------------------------------------------------------
# mamba_ssm 导入核心模块
# ---------------------------------------------------------------------------

# Mamba2Simple: 
from mamba_ssm.modules.mamba2_simple import Mamba2Simple

# 融合 Add+Norm Triton 算子（mamba ≥2.0 位于 ops.triton.layer_norm）
try:
    from mamba_ssm.ops.triton.layer_norm import (
        RMSNorm,
        layer_norm_fn,
        rms_norm_fn,
    )
except ImportError:
    try:
        from mamba_ssm.ops.triton.layernorm import (
            RMSNorm,
            layer_norm_fn,
            rms_norm_fn,
        )
    except ImportError:
        RMSNorm = None
        layer_norm_fn = None
        rms_norm_fn = None

# 纯 PyTorch RMSNorm 回退（Triton 不可用时）
if RMSNorm is None:
    class RMSNorm(nn.Module):
        def __init__(self, d: int, eps: float = 1e-5):
            super().__init__()
            self.eps = eps
            self.weight = nn.Parameter(torch.ones(d))

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            norm = x.float().pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
            return (x.float() * norm * self.weight).to(x.dtype)


# ---------------------------------------------------------------------------
# Block2 — 预归一化 Mamba-2 残差块
# ---------------------------------------------------------------------------

class Block2(nn.Module):
    """
    预归一化残差包装器，内含单个 Mamba2Simple 混合器。

    计算图（prenorm 约定，与 mamba repo 的 block.py 保持一致）：
        residual_n   = hidden_states_{n-1} + residual_{n-1}   (首层 residual=None)
        normed       = Norm(residual_n)
        hidden_n     = Mamba2Simple(normed)

    当 fused_add_norm=True 时，Add+Norm 由单个 Triton kernel 完成，减少显存带宽。

    返回 (hidden_states, residual)，由调用者传给下一层或最终归一化。
    """

    def __init__(
        self,
        d_model: int,
        mixer_cls,
        norm_cls=nn.LayerNorm,
        fused_add_norm: bool = False,
        residual_in_fp32: bool = False,
    ):
        super().__init__()
        self.fused_add_norm = fused_add_norm
        self.residual_in_fp32 = residual_in_fp32

        self.norm = norm_cls(d_model)
        self.mixer = mixer_cls(d_model)

        if fused_add_norm:
            assert layer_norm_fn is not None, (
                "fused_add_norm=True 需要 Triton layer_norm 算子，"
                "请先编译 mamba CUDA 扩展：pip install -e ../mamba/"
            )
            assert isinstance(self.norm, (nn.LayerNorm, RMSNorm)), (
                "fused_add_norm 仅支持 LayerNorm 和 RMSNorm。"
            )

    def forward(
        self,
        hidden_states: torch.Tensor,
        residual: Optional[torch.Tensor] = None,
        inference_params=None,
    ):
        """
        Args:
            hidden_states : (B, L, d_model)
            residual      : (B, L, d_model) | None
            inference_params : 传给 Mamba2Simple，用于增量推理缓存

        Returns:
            hidden_states : (B, L, d_model)
            residual      : (B, L, d_model)
        """
        if not self.fused_add_norm:
            # 标准 unfused 路径
            residual = (hidden_states + residual) if residual is not None else hidden_states
            hidden_states = self.norm(residual.to(dtype=self.norm.weight.dtype))
            if self.residual_in_fp32:
                residual = residual.to(torch.float32)
        else:
            # Triton 融合 Add+Norm
            hidden_states, residual = layer_norm_fn(
                hidden_states,
                self.norm.weight,
                getattr(self.norm, "bias", None),   # RMSNorm 无 bias
                residual=residual,
                prenorm=True,
                residual_in_fp32=self.residual_in_fp32,
                eps=self.norm.eps,
                is_rms_norm=isinstance(self.norm, RMSNorm),
            )

        hidden_states = self.mixer(hidden_states)
        return hidden_states, residual

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return self.mixer.allocate_inference_cache(
            batch_size, max_seqlen, dtype=dtype, **kwargs
        )


# ---------------------------------------------------------------------------
# create_block2 — 工厂函数
# ---------------------------------------------------------------------------

def create_block2(
    d_model: int,
    d_state: int = 16,         
    d_conv: int = 4,
    expand: int = 2,
    headdim: int = 64,          
    ngroups: int = 1,
    chunk_size: int = 256,
    norm_epsilon: float = 1e-5,
    rms_norm: bool = True,
    residual_in_fp32: bool = True,
    fused_add_norm: bool = True,
    layer_idx: Optional[int] = None,
    device=None,
    dtype=None,
) -> Block2:
    """
    构建单个 Block2（Mamba2Simple 混合器 + 预归一化残差）。

    """
    factory_kwargs = {"device": device, "dtype": dtype}

    mixer_cls = partial(
        Mamba2Simple,
        d_state=d_state,
        d_conv=d_conv,
        expand=expand,
        headdim=headdim,
        ngroups=ngroups,
        chunk_size=chunk_size,
        bias=False,
        conv_bias=True,
        layer_idx=layer_idx,
        **factory_kwargs,
    )

    norm_cls = partial(
        RMSNorm if rms_norm else nn.LayerNorm,
        eps=norm_epsilon,
    )

    return Block2(
        d_model=d_model,
        mixer_cls=mixer_cls,
        norm_cls=norm_cls,
        fused_add_norm=fused_add_norm,
        residual_in_fp32=residual_in_fp32,
    )
