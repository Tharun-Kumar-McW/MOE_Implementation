# ===----------------------------------------------------------------------=== #
# Copyright (c) 2026, Modular Inc. All rights reserved.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions:
# https://llvm.org/LICENSE.txt
# ===----------------------------------------------------------------------=== #

"""SmolLM3 compute graph — Llama3 architecture with selective RoPE layers."""

from __future__ import annotations

import functools

from max.dtype import DType
from max.graph import TensorValue, ops
from max.nn.embedding import VocabParallelEmbedding
from max.nn.kv_cache import PagedCacheValues
from max.nn.layer import LayerList, Module
from max.nn.linear import MLP, ColumnParallelLinear
from max.nn.norm import RMSNorm
from max.nn.rotary_embedding import Llama3RotaryEmbedding

from .layers.attention import SmolLM3Attention
from .layers.transformer_block import SmolLM3TransformerBlock
from .model_config import SmolLM3Config


class SmolLM3Model(Module):
    """SmolLM3 — Llama3-style model with selective no-RoPE layers."""

    def __init__(self, config: SmolLM3Config) -> None:
        super().__init__()

        head_dim = config.hidden_size // config.num_attention_heads

        rope = Llama3RotaryEmbedding(
            dim=config.hidden_size,
            n_heads=config.num_attention_heads,
            theta=config.rope_theta,
            max_seq_len=config.max_position_embeddings,
            head_dim=head_dim,
            interleaved=False,
            scaling_params=None,
        )

        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            config.dtype,
            config.devices,
        )

        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)

        self.lm_head = ColumnParallelLinear(
            config.hidden_size,
            config.vocab_size,
            dtype=config.dtype,
            devices=config.devices,
            tied_weight=(
                self.embed_tokens.weight if config.tie_word_embeddings else None
            ),
        )

        create_norm = functools.partial(
            RMSNorm, config.hidden_size, config.rms_norm_eps
        )

        self.layers = LayerList([
            SmolLM3TransformerBlock(
                attention=SmolLM3Attention(
                    rope=rope,
                    num_attention_heads=config.num_attention_heads,
                    num_key_value_heads=config.num_key_value_heads,
                    hidden_size=config.hidden_size,
                    kv_params=config.kv_params,
                    layer_idx=i,
                    dtype=config.dtype,
                    devices=config.devices,
                    use_rope=config.uses_rope_at_layer(i),
                    attention_bias=config.attention_bias,
                ),
                mlp=MLP(
                    dtype=config.dtype,
                    quantization_encoding=None,
                    hidden_dim=config.hidden_size,
                    feed_forward_length=config.intermediate_size,
                    devices=config.devices,
                    activation_function=config.hidden_act,
                ),
                input_layernorm=create_norm(),
                post_attention_layernorm=create_norm(),
                devices=config.devices,
            )
            for i in range(config.num_hidden_layers)
        ])

        self.kv_params = config.kv_params
        self.return_logits = config.return_logits
        self.devices = config.devices

    def __call__(
        self,
        tokens: TensorValue,
        kv_collection: PagedCacheValues,
        return_n_logits: TensorValue,
        input_row_offsets: TensorValue,
    ) -> tuple[TensorValue, ...]:
        # Single-device call signature matching DeepseekV2/Llama single-GPU pattern
        h = self.embed_tokens(tokens, [])

        for idx, layer in enumerate(self.layers):
            layer_idx_tensor = ops.constant(
                idx, DType.uint32, device=self.devices[0]
            )
            h = layer(
                layer_idx=layer_idx_tensor,
                x=h[0] if isinstance(h, list) else h,
                kv_collection=kv_collection,
                input_row_offsets=input_row_offsets,
            )

        h_out = h[0] if isinstance(h, list) else h
        h_out = self.norm(h_out)

        logits = self.lm_head(h_out)
        if isinstance(logits, list):
            logits = logits[0]

        return (logits,)