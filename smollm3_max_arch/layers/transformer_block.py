# ===----------------------------------------------------------------------=== #
# Copyright (c) 2026, Modular Inc. All rights reserved.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions:
# https://llvm.org/LICENSE.txt
# ===----------------------------------------------------------------------=== #

"""SmolLM3 transformer block.

Standard Llama-style sequential block:
    x = x + Attention(RMSNorm(x))
    x = x + MLP(RMSNorm(x))

No parallel attention+MLP (unlike Cohere), no extra post-norms (unlike Gemma3).
"""

from __future__ import annotations

from max.graph import BufferValue, DeviceRef, ShardingStrategy, TensorValue
from max.nn.comm.allreduce import Allreduce
from max.nn.kv_cache import PagedCacheValues
from max.nn.layer import Module
from max.nn.transformer.distributed_transformer import (
    ShardableCallable,
    forward_sharded_layers,
)

from .attention import SmolLM3Attention


class SmolLM3TransformerBlock(Module):
    """Standard Llama-style transformer block for SmolLM3."""

    def __init__(
        self,
        attention: SmolLM3Attention,
        mlp: ShardableCallable,
        input_layernorm: ShardableCallable,
        post_attention_layernorm: ShardableCallable,
        devices: list[DeviceRef],
    ) -> None:
        super().__init__()

        self.self_attn = attention
        self.self_attn.sharding_strategy = ShardingStrategy.tensor_parallel(
            len(devices)
        )
        self.self_attn_shards = attention.shard(devices)

        self.mlp = mlp
        self.mlp.sharding_strategy = ShardingStrategy.tensor_parallel(
            len(devices)
        )
        self.mlp_shards = mlp.shard(devices)

        self.input_layernorm = input_layernorm
        self.input_layernorm.sharding_strategy = ShardingStrategy.replicate(
            len(devices)
        )
        self.input_layernorm_shards = input_layernorm.shard(devices)

        self.post_attention_layernorm = post_attention_layernorm
        self.post_attention_layernorm.sharding_strategy = (
            ShardingStrategy.replicate(len(devices))
        )
        self.post_attention_layernorm_shards = (
            post_attention_layernorm.shard(devices)
        )

        self.devices = devices
        self.allreduce = Allreduce(num_accelerators=len(devices))

    def __call__(
        self,
        layer_idx: TensorValue,
        xs: list[TensorValue],
        signal_buffers: list[BufferValue],
        kv_collections: list[PagedCacheValues],
        input_row_offsets: list[TensorValue],
        **kwargs,
    ) -> list[TensorValue]:
        # Attention sub-layer: x = x + Attention(RMSNorm(x))
        residual = xs
        normed = forward_sharded_layers(self.input_layernorm_shards, xs)
        attn_out = [
            shard(
                normed[i],
                kv_collections[i],
                input_row_offsets=input_row_offsets[i],
                **kwargs,
            )
            for i, shard in enumerate(self.self_attn_shards)
        ]
        attn_out = self.allreduce(attn_out, signal_buffers)
        xs = [residual[i] + attn_out[i] for i in range(len(xs))]

        # MLP sub-layer: x = x + MLP(RMSNorm(x))
        residual = xs
        normed = forward_sharded_layers(self.post_attention_layernorm_shards, xs)
        mlp_out = forward_sharded_layers(self.mlp_shards, normed)
        mlp_out = self.allreduce(mlp_out, signal_buffers)
        return [residual[i] + mlp_out[i] for i in range(len(xs))]
