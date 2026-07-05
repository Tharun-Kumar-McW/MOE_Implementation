# ===----------------------------------------------------------------------=== #
# Copyright (c) 2026, Modular Inc. All rights reserved.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions:
# https://llvm.org/LICENSE.txt
# ===----------------------------------------------------------------------=== #

"""SmolLM3 attention layer.

Identical to Llama attention except some layers skip RoPE entirely,
controlled by the no_rope_layers config flag.
"""

from __future__ import annotations

from max.dtype import DType
from max.graph import DeviceRef, TensorValue, ops
from max.nn.kv_cache import KVCacheParams, PagedCacheValues
from max.nn.layer import Module
from max.nn.linear import ColumnParallelLinear, Linear
from max.nn.rotary_embedding import Llama3RotaryEmbedding


class SmolLM3Attention(Module):
    """GQA attention for SmolLM3 with optional per-layer RoPE skip.

    When use_rope=False (every 4th layer in SmolLM3), Q and K are used
    directly without positional encoding — pure content-based attention.
    """

    def __init__(
        self,
        rope: Llama3RotaryEmbedding,
        num_attention_heads: int,
        num_key_value_heads: int,
        hidden_size: int,
        kv_params: KVCacheParams,
        layer_idx: int,
        dtype: DType,
        devices: list[DeviceRef],
        use_rope: bool = True,
        attention_bias: bool = False,
    ) -> None:
        super().__init__()

        self.num_heads = num_attention_heads
        self.num_kv_heads = num_key_value_heads
        self.hidden_size = hidden_size
        self.head_dim = hidden_size // num_attention_heads
        self.kv_params = kv_params
        self.layer_idx = layer_idx
        # use_rope=False means this layer is a "no-rope" layer
        self.use_rope = use_rope
        self.rope = rope

        self.q_proj = ColumnParallelLinear(
            hidden_size,
            num_attention_heads * self.head_dim,
            dtype=dtype,
            devices=devices,
            bias=attention_bias,
        )
        self.k_proj = ColumnParallelLinear(
            hidden_size,
            num_key_value_heads * self.head_dim,
            dtype=dtype,
            devices=devices,
            bias=attention_bias,
        )
        self.v_proj = ColumnParallelLinear(
            hidden_size,
            num_key_value_heads * self.head_dim,
            dtype=dtype,
            devices=devices,
            bias=attention_bias,
        )
        self.o_proj = Linear(
            num_attention_heads * self.head_dim,
            hidden_size,
            dtype=dtype,
            device=devices[0],
            bias=attention_bias,
        )

    def __call__(
        self,
        x: TensorValue,
        kv_collection: PagedCacheValues,
        input_row_offsets: TensorValue,
        **kwargs,
    ) -> TensorValue:
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # Only apply RoPE if this layer uses it (3 out of every 4 layers)
        if self.use_rope:
            q, k = self.rope(q, k, input_row_offsets)

        attn_out = ops.paged_attention(
            q,
            k,
            v,
            kv_collection,
            input_row_offsets=input_row_offsets,
            layer_idx=self.layer_idx,
        )

        return self.o_proj(attn_out)
