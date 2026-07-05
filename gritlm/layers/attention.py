# ===----------------------------------------------------------------------=== #
# GritLM attention layer — Mistral-style GQA with sliding window on ALL layers.
#
# Key difference from plain Mistral in MAX:
#   MAX's existing Mistral uses the older Graph API (AttentionWithRope).
#   We use the ModuleV3 fused kernel path (rope_split_store_ragged +
#   flash_attention_ragged) with MHAMaskVariant.SLIDING_WINDOW_CAUSAL_MASK.
#
# GritLM sliding window = 4096 on every layer (no alternating pattern).
# ===----------------------------------------------------------------------=== #
"""GritLM attention: GQA + sliding window on every layer."""

from __future__ import annotations

import math

from max.driver import CPU
from max.dtype import DType
from max.experimental import functional as F
from max.experimental.nn import Module
from max.experimental.nn.common_layers.functional_kernels import (
    flash_attention_ragged,
    rope_split_store_ragged,
)
from max.experimental.nn.linear import Linear
from max.experimental.tensor import Tensor
from max.nn.attention import MHAMaskVariant
from max.nn.kv_cache import KVCacheParams, PagedCacheValues


class GritLMAttention(Module[..., Tensor]):
    """GQA attention with sliding window causal mask.

    GritLM applies SWA with window=4096 on every one of its 32 layers.
    flash_attention_ragged is called with SLIDING_WINDOW_CAUSAL_MASK and
    local_window_size=sliding_window.
    """

    def __init__(
        self,
        *,
        rope,
        num_attention_heads: int,
        num_key_value_heads: int,
        hidden_size: int,
        kv_params: KVCacheParams,
        layer_idx: int,
        sliding_window: int = 4096,
        scale: float | None = None,
        has_bias: bool = False,
        clip_qkv: float | None = None,
    ) -> None:
        super().__init__()
        self.rope = rope
        self.n_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.hidden_size = hidden_size
        self.kv_params = kv_params
        self.layer_idx = layer_idx
        self.sliding_window = sliding_window
        self.has_bias = has_bias
        self.clip_qkv = clip_qkv
        self.scale = (
            scale if scale is not None
            else math.sqrt(1.0 / self.kv_params.head_dim)
        )

        q_dim  = self.kv_params.head_dim * num_attention_heads
        kv_dim = self.kv_params.head_dim * num_key_value_heads
        self.q_weight_dim = q_dim

        self.q_proj = Linear(in_dim=hidden_size, out_dim=q_dim,  bias=has_bias)
        self.k_proj = Linear(in_dim=hidden_size, out_dim=kv_dim, bias=has_bias)
        self.v_proj = Linear(in_dim=hidden_size, out_dim=kv_dim, bias=has_bias)
        self.o_proj = Linear(in_dim=q_dim, out_dim=hidden_size,  bias=False)

    @property
    def wqkv(self) -> Tensor:
        wq: Tensor = self.q_proj.weight
        wk: Tensor = self.k_proj.weight
        wv: Tensor = self.v_proj.weight
        if self.clip_qkv:
            wq = F.clip(wq, -self.clip_qkv, self.clip_qkv)
            wk = F.clip(wk, -self.clip_qkv, self.clip_qkv)
            wv = F.clip(wv, -self.clip_qkv, self.clip_qkv)
        return F.concat([wq, wk, wv], axis=0)

    @property
    def wqkv_bias(self) -> Tensor | None:
        if not self.has_bias:
            return None
        assert self.q_proj.bias is not None
        assert self.k_proj.bias is not None
        assert self.v_proj.bias is not None
        return F.concat(
            [self.q_proj.bias, self.k_proj.bias, self.v_proj.bias], axis=0
        )

    def forward(
        self,
        x: Tensor,
        kv_collection: PagedCacheValues,
        **kwargs,
    ) -> Tensor:
        total_seq_len = x.shape[0]
        layer_idx = F.constant(self.layer_idx, DType.uint32, device=CPU())

        qkv = x @ self.wqkv.T
        if self.wqkv_bias is not None:
            qkv = qkv + self.wqkv_bias

        freqs_cis = F.cast(self.rope.freqs_cis, qkv.dtype).to(qkv.device)

        # Apply RoPE + split QKV + store K/V into paged cache
        xq = rope_split_store_ragged(
            kv_params=self.kv_params,
            qkv=qkv,
            input_row_offsets=kwargs["input_row_offsets"],
            freqs_cis=freqs_cis,
            kv_collection=kv_collection,
            layer_idx=layer_idx,
            n_heads=self.n_heads,
            interleaved=self.rope.interleaved,
        )
        xq = xq.reshape((-1, self.n_heads, self.kv_params.head_dim))

        # Sliding window causal attention — all GritLM layers use SWA
        attn_out = flash_attention_ragged(
            self.kv_params,
            input=xq,
            kv_collection=kv_collection,
            layer_idx=layer_idx,
            input_row_offsets=kwargs["input_row_offsets"],
            mask_variant=MHAMaskVariant.SLIDING_WINDOW_CAUSAL_MASK,
            scale=self.scale,
            local_window_size=self.sliding_window,
        )
        attn_out = F.reshape(attn_out, shape=[total_seq_len, self.q_weight_dim])
        return self.o_proj(attn_out)
