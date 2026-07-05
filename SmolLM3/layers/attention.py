# ===----------------------------------------------------------------------=== #
# SmolLM3 attention layer with conditional RoPE (NoPE support).
#
# MAX's pipeline uses two fused kernels:
#   rope_split_store_ragged  — splits flat QKV, applies RoPE to Q/K, stores K/V
#   flash_attention_ragged   — computes attention from roped Q + paged KV cache
#
# freqs_cis shape (from RotaryEmbedding.freqs_cis cached property):
#   [max_seq_len * 2, head_dim]   (flat: cos/sin pairs interleaved per half-dim)
#
# NoPE identity: cos=1.0, sin=0.0 throughout → rotation is a mathematical no-op.
# We build identity freqs_cis matching the exact flat layout:
#   half = head_dim // 2
#   flat = concat([ones([S, half]), zeros([S, half])], axis=-1)  — non-interleaved
#   OR interleaved: stack([ones, zeros], axis=-1).reshape([S, head_dim])
# ===----------------------------------------------------------------------=== #
"""SmolLM3 attention: fused RoPE+KV-store with NoPE identity for every 4th layer."""

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


class SmolLM3Attention(Module[..., Tensor]):
    """GQA attention with optional RoPE per layer.

    RoPE layers  (use_rope=True):  real freqs_cis passed to rope_split_store_ragged.
    NoPE layers  (use_rope=False): identity freqs_cis (cos=1, sin=0) so rotation
                                   is a no-op while KV store still happens.
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
        use_rope: bool = True,
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
        self.use_rope = use_rope
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

    def _make_identity_freqs_cis(self, freqs_cis: Tensor) -> Tensor:
        """Build identity freqs_cis matching the shape of the real one.

        freqs_cis flat layout: [max_seq_len*2, head_dim]
        Non-interleaved packing: first head_dim//2 values are cos, next are sin.
        Interleaved packing: cos/sin alternate per position.

        Identity rotation: cos=1, sin=0 everywhere.

        We construct this by:
          ones  = same shape, all 1.0  (cos terms)
          zeros = same shape, all 0.0  (sin terms)
        Then pack them in the same layout as real freqs_cis:
          non-interleaved: concat([ones[:, :H//2], zeros[:, :H//2]], axis=-1)
          interleaved:     stack([ones_half, zeros_half], axis=-1).reshape(shape)
        """
        shape = freqs_cis.shape          # [S, head_dim]
        seq_dim = shape[0]
        head_dim = shape[1]
        half = head_dim // 2

        cos_half = F.constant(1.0, dtype=freqs_cis.dtype, device=freqs_cis.device)
        sin_half = F.constant(0.0, dtype=freqs_cis.dtype, device=freqs_cis.device)

        # Broadcast scalars to [S, half] slabs
        cos_slab = F.broadcast_to(
            F.reshape(cos_half, [1, 1]), [seq_dim, half]
        )
        sin_slab = F.broadcast_to(
            F.reshape(sin_half, [1, 1]), [seq_dim, half]
        )

        if self.rope.interleaved:
            # interleaved: [S, half, 2] → [S, head_dim]
            stacked = F.stack([cos_slab, sin_slab], axis=-1)
            return F.reshape(stacked, [seq_dim, head_dim])
        else:
            # non-interleaved: [cos_half | sin_half] along last axis
            return F.concat([cos_slab, sin_slab], axis=-1)

    def forward(
        self,
        x: Tensor,
        kv_collection: PagedCacheValues,
        **kwargs,
    ) -> Tensor:
        total_seq_len = x.shape[0]
        layer_idx = F.constant(self.layer_idx, DType.uint32, device=CPU())

        # Single fused QKV matmul
        qkv = x @ self.wqkv.T
        if self.wqkv_bias is not None:
            qkv = qkv + self.wqkv_bias

        # freqs_cis: real for RoPE layers, identity for NoPE layers
        freqs_cis = F.cast(self.rope.freqs_cis, qkv.dtype).to(qkv.device)
        if not self.use_rope:
            freqs_cis = self._make_identity_freqs_cis(freqs_cis)

        # Fused: apply RoPE (or identity) + split + KV store → returns roped Q
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

        # Flash attention (causal, paged KV cache)
        attn_out = flash_attention_ragged(
            self.kv_params,
            input=xq,
            kv_collection=kv_collection,
            layer_idx=layer_idx,
            input_row_offsets=kwargs["input_row_offsets"],
            mask_variant=MHAMaskVariant.CAUSAL_MASK,
            scale=self.scale,
        )
        attn_out = F.reshape(attn_out, shape=[total_seq_len, self.q_weight_dim])
        return self.o_proj(attn_out)