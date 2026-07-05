# ===----------------------------------------------------------------------=== #
# SmolLM3 ModuleV3 graph implementation.
#
# Key differences from plain Llama3:
#   1. Per-layer RoPE toggle (no_rope_layers mask from config)
#   2. Every 4th layer is a NoPE layer (uses SmolLM3Attention with use_rope=False)
#   3. tie_word_embeddings=True (lm_head weight = embed_tokens weight transposed)
# ===----------------------------------------------------------------------=== #
"""Implements the SmolLM3 model using the ModuleV3 API."""

from __future__ import annotations

import functools
from collections.abc import Callable

from max.dtype import DType
from max.experimental import functional as F
from max.experimental.nn import Module
from max.experimental.nn.common_layers.mlp import MLP
from max.experimental.nn.embedding import Embedding
from max.experimental.nn.linear import Linear
from max.experimental.nn.norm import RMSNorm
from max.experimental.nn.sequential import ModuleList
from max.experimental.tensor import Tensor
from max.graph import TensorValue, ops
from max.nn.kv_cache import KVCacheParamInterface, PagedCacheValues
from max.nn.transformer import ReturnHiddenStates, ReturnLogits

from max.pipelines.architectures.llama3_modulev3.layers.rotary_embedding import (
    Llama3RotaryEmbedding,
)

from .layers.attention import SmolLM3Attention
from .layers.transformer_block import SmolLM3TransformerBlock
from .model_config import SmolLM3Config


class SmolLM3TextModel(
    Module[[Tensor, PagedCacheValues, Tensor, Tensor], tuple[Tensor, ...]]
):
    """SmolLM3 decoder-only transformer.

    36 layers, GQA (16 Q heads / 4 KV heads), SwiGLU MLP, RMSNorm.
    Every 4th layer (indices 3, 7, 11, …) uses NoPE (no positional encoding).
    """

    def __init__(self, config: SmolLM3Config) -> None:
        super().__init__()
        self.devices = config.devices

        if config.rms_norm_eps is None:
            raise ValueError("rms_norm_eps cannot be None for SmolLM3.")

        create_norm: Callable[..., Module[[Tensor], Tensor]] = functools.partial(
            RMSNorm, config.hidden_size, eps=config.rms_norm_eps
        )

        # Shared RoPE embedding (used only by RoPE layers; NoPE layers ignore it).
        rope = Llama3RotaryEmbedding(
            dim=config.hidden_size,
            n_heads=config.num_attention_heads,
            theta=config.rope_theta,
            max_seq_len=config.max_seq_len,
            device=config.devices[0].to_device(),
            head_dim=SmolLM3Config.get_head_dim_from_config(config),
            interleaved=config.interleaved_rope_weights,
            scaling_params=None,  # rope_scaling: null in SmolLM3 config.json
        )

        self.embed_tokens = Embedding(
            config.vocab_size,
            dim=config.hidden_size,
        )
        self.norm = create_norm()

        self.tie_word_embeddings = config.tie_word_embeddings
        if config.tie_word_embeddings:
            self.lm_head = None
        else:
            self.lm_head = Linear(
                in_dim=config.hidden_size,
                out_dim=config.vocab_size,
                bias=False,
            )

        # Build 36 decoder layers with per-layer RoPE / NoPE selection.
        layers = []
        no_rope_layers = config.no_rope_layers
        for i in range(config.num_hidden_layers):
            # no_rope_layers[i] == 1 → use RoPE; == 0 → NoPE
            use_rope: bool = bool(no_rope_layers[i]) if no_rope_layers else True

            attention = SmolLM3Attention(
                rope=rope,
                num_attention_heads=config.num_attention_heads,
                num_key_value_heads=config.num_key_value_heads,
                hidden_size=config.hidden_size,
                kv_params=config.kv_params,
                layer_idx=i,
                use_rope=use_rope,
                scale=config.attention_multiplier,
                has_bias=config.attention_bias,
                clip_qkv=config.clip_qkv,
            )

            mlp = MLP(
                hidden_dim=config.hidden_size,
                feed_forward_length=config.intermediate_size,
            )

            layers.append(
                SmolLM3TransformerBlock(
                    attention=attention,
                    mlp=mlp,
                    input_layernorm=create_norm(),
                    post_attention_layernorm=create_norm(),
                    residual_multiplier=config.residual_multiplier,
                )
            )

        self.layers = ModuleList(layers)
        self.dim = config.hidden_size
        self.n_heads = config.num_attention_heads
        self.kv_params = config.kv_params
        self.return_logits = config.return_logits
        self.return_hidden_states = config.return_hidden_states
        self.embedding_multiplier = config.embedding_multiplier
        self.logits_scaling = config.logits_scaling

    def _compute_logits(self, h: Tensor) -> Tensor:
        """Compute logits; handles tied embedding weights."""
        if self.tie_word_embeddings:
            # lm_head is the transpose of embed_tokens — share weight tensor.
            return F.cast(h @ self.embed_tokens.weight.T, DType.float32)
        assert self.lm_head is not None
        return F.cast(self.lm_head(h), DType.float32)

    def forward(
        self,
        tokens: Tensor,
        kv_collection: PagedCacheValues,
        return_n_logits: Tensor,
        input_row_offsets: Tensor,
    ) -> tuple[Tensor, ...]:
        h = self.embed_tokens(tokens)

        if self.embedding_multiplier != 1.0:
            h = h * F.constant(
                self.embedding_multiplier, h.dtype, device=h.device
            )

        # Run all 36 decoder layers.
        for idx, layer in enumerate(self.layers):
            layer_idx_tensor = F.constant(idx, DType.uint32, device=h.device)
            h = layer(
                layer_idx_tensor,
                h,
                kv_collection,
                input_row_offsets=input_row_offsets,
            )

        # Last-token logits (always produced).
        last_h = F.gather(h, input_row_offsets[1:] - 1, axis=0)
        last_logits = self._compute_logits(self.norm(last_h))
        logits = None
        offsets = None

        if self.return_logits == ReturnLogits.VARIABLE:
            return_n_logits_range = ops.range(
                return_n_logits[0],
                0,
                -1,
                out_dim="return_n_logits_range",
                device=h.device,
                dtype=DType.int64,
            )
            offsets = (
                F.unsqueeze(input_row_offsets[1:], -1) - return_n_logits_range
            )
            last_indices = F.reshape(offsets, shape=(-1,))
            last_tokens = F.gather(h, last_indices, axis=0)
            logits = self._compute_logits(self.norm(last_tokens))
            offsets = ops.range(
                0,
                TensorValue(last_indices.shape[0]) + return_n_logits[0],
                return_n_logits[0],
                out_dim="logit_offsets",
                device=h.device,
                dtype=DType.int64,
            )
        elif self.return_logits == ReturnLogits.ALL:
            logits = self._compute_logits(self.norm(h))
            offsets = input_row_offsets

        if self.logits_scaling != 1.0:
            last_logits = last_logits / self.logits_scaling
            if logits is not None:
                logits = logits / self.logits_scaling

        ret_val: tuple[Tensor, ...] = (last_logits,)
        if offsets is not None:
            assert logits is not None
            ret_val += (logits, offsets)

        if self.return_hidden_states == ReturnHiddenStates.LAST:
            ret_val += (last_h,)
        elif self.return_hidden_states == ReturnHiddenStates.ALL_NORMALIZED:
            ret_val += (self.norm(h),)

        return ret_val


class SmolLM3(Module[..., tuple[Tensor, ...]]):
    """Top-level SmolLM3 model.

    Unflattens variadic KV cache args and delegates to SmolLM3TextModel.
    Mirrors the Llama3 wrapper pattern so the pipeline machinery is identical.
    """

    def __init__(
        self,
        config: SmolLM3Config,
        kv_params: KVCacheParamInterface,
    ) -> None:
        super().__init__()
        self.language_model = SmolLM3TextModel(config)
        self.config = config
        self.kv_params = kv_params

    def forward(
        self,
        tokens: Tensor,
        return_n_logits: Tensor,
        input_row_offsets: Tensor,
        *variadic_args: Tensor,
    ) -> tuple[Tensor, ...]:
        kv_inputs = iter(x._graph_value for x in variadic_args)
        kv_collections = (
            self.kv_params.get_symbolic_inputs().unflatten(kv_inputs).inputs
        )
        return self.language_model(
            tokens, kv_collections[0], return_n_logits, input_row_offsets
        )
