"""
Mamba-1 building blocks for MambaNetBurst speed comparison.

"""

from __future__ import annotations

from functools import partial
from typing import Optional

import torch
import torch.nn as nn

from mamba_ssm.modules.mamba_simple import Mamba

from models_mamba2 import RMSNorm, layer_norm_fn


# ---------------------------------------------------------------------------
# Block1 — 预归一化 Mamba-1 残差块
# ---------------------------------------------------------------------------

class Block1(nn.Module):
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
            assert layer_norm_fn is not None
            assert isinstance(self.norm, (nn.LayerNorm, RMSNorm))

    def forward(
        self,
        hidden_states: torch.Tensor,
        residual: Optional[torch.Tensor] = None,
        inference_params=None,
    ):
        if not self.fused_add_norm:
            residual = (hidden_states + residual) if residual is not None else hidden_states
            hidden_states = self.norm(residual.to(dtype=self.norm.weight.dtype))
            if self.residual_in_fp32:
                residual = residual.to(torch.float32)
        else:
            hidden_states, residual = layer_norm_fn(
                hidden_states,
                self.norm.weight,
                getattr(self.norm, "bias", None),
                residual=residual,
                prenorm=True,
                residual_in_fp32=self.residual_in_fp32,
                eps=self.norm.eps,
                is_rms_norm=isinstance(self.norm, RMSNorm),
            )

        hidden_states = self.mixer(hidden_states, inference_params=inference_params)
        return hidden_states, residual

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return self.mixer.allocate_inference_cache(
            batch_size, max_seqlen, dtype=dtype, **kwargs
        )


# ---------------------------------------------------------------------------
# create_block1 
# ---------------------------------------------------------------------------

def create_block1(
    d_model: int,
    d_state: int = 16,
    d_conv: int = 4,
    expand: int = 2,
    norm_epsilon: float = 1e-5,
    rms_norm: bool = True,
    residual_in_fp32: bool = True,
    fused_add_norm: bool = True,
    layer_idx: Optional[int] = None,
    device=None,
    dtype=None,
) -> Block1:
    factory_kwargs = {"device": device, "dtype": dtype}

    mixer_cls = partial(
        Mamba,
        d_state=d_state,
        d_conv=d_conv,
        expand=expand,
        bias=False,
        conv_bias=True,
        layer_idx=layer_idx,
        **factory_kwargs,
    )

    norm_cls = partial(
        RMSNorm if rms_norm else nn.LayerNorm,
        eps=norm_epsilon,
    )

    return Block1(
        d_model=d_model,
        mixer_cls=mixer_cls,
        norm_cls=norm_cls,
        fused_add_norm=fused_add_norm,
        residual_in_fp32=residual_in_fp32,
    )
