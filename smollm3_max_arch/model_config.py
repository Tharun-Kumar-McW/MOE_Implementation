# ===----------------------------------------------------------------------=== #
# Copyright (c) 2026, Modular Inc. All rights reserved.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions:
# https://llvm.org/LICENSE.txt
# ===----------------------------------------------------------------------=== #

from __future__ import annotations

from dataclasses import dataclass, field

from max.graph.weights import WeightData, WeightsFormat, weights_format
from max.dtype import DType
from max.graph import DeviceRef
from max.nn.kv_cache import KVCacheParams
from max.nn.transformer import ReturnLogits
from max.pipelines.lib import KVCacheConfig, MAXModelConfig, PipelineConfig
from max.pipelines.lib.config.config_enums import supported_encoding_dtype
from max.pipelines.lib.interfaces.arch_config import ArchConfigWithKVCache
from transformers import AutoConfig
from typing_extensions import Self, override


@dataclass(kw_only=True)
class SmolLM3Config(ArchConfigWithKVCache):
    """MAX Engine configuration for SmolLM3.

    Identical to LlamaConfig with one addition:
    - no_rope_layers: per-layer flag, 1=skip RoPE, 0=apply RoPE
      Pattern repeats every 4 layers: [1,1,1,0, 1,1,1,0, ...]
    """

    # Standard Llama-style parameters
    vocab_size: int
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    hidden_act: str
    max_position_embeddings: int
    rms_norm_eps: float
    rope_theta: float
    attention_bias: bool
    mlp_bias: bool
    tie_word_embeddings: bool

    # SmolLM3-specific: which layers skip RoPE (1=no rope, 0=use rope)
    no_rope_layers: list[int] = field(default_factory=list)

    # MAX-specific
    dtype: DType
    devices: list[DeviceRef]
    kv_params: KVCacheParams
    return_logits: ReturnLogits = ReturnLogits.LAST_TOKEN

    def get_kv_params(self) -> KVCacheParams:
        return self.kv_params

    def get_max_seq_len(self) -> int:
        return self.max_position_embeddings

    def uses_rope_at_layer(self, layer_idx: int) -> bool:
        """Returns True if this layer should apply RoPE.

        SmolLM3 skips RoPE when no_rope_layers[i] == 1.
        Every 4th layer (indices 3, 7, 11, ...) uses RoPE.
        """
        if not self.no_rope_layers or layer_idx >= len(self.no_rope_layers):
            return True
        return self.no_rope_layers[layer_idx] == 0

    @staticmethod
    def construct_kv_params(
        huggingface_config: AutoConfig,
        pipeline_config: PipelineConfig,
        devices: list[DeviceRef],
        kv_cache_config: KVCacheConfig,
        cache_dtype: DType,
    ) -> KVCacheParams:
        head_dim = (
            huggingface_config.hidden_size
            // huggingface_config.num_attention_heads
        )
        return kv_cache_config.to_params(
            dtype=cache_dtype,
            n_kv_heads=huggingface_config.num_key_value_heads,
            head_dim=head_dim,
            num_layers=huggingface_config.num_hidden_layers,
            devices=devices,
            data_parallel_degree=pipeline_config.model.data_parallel_degree,
        )

    @staticmethod
    def calculate_max_seq_len(
        pipeline_config: PipelineConfig, huggingface_config: AutoConfig
    ) -> int:
        max_seq_len = pipeline_config.model.max_length
        if max_seq_len:
            return max_seq_len
        return huggingface_config.max_position_embeddings

    @override
    @classmethod
    def initialize(
        cls,
        pipeline_config: PipelineConfig,
        model_config: MAXModelConfig | None = None,
    ) -> Self:
        model_config = model_config or pipeline_config.model
        huggingface_config = model_config.huggingface_config
        if huggingface_config is None:
            raise ValueError(
                f"HuggingFace config required for '{model_config.model_path}'"
            )

        quantization_encoding = pipeline_config.model.quantization_encoding
        if quantization_encoding is None:
            raise ValueError("quantization_encoding must not be None")

        dtype = supported_encoding_dtype(quantization_encoding)
        cache_dtype = pipeline_config.model.kv_cache.cache_dtype
        kv_cache_config = pipeline_config.model.kv_cache

        device_refs = [
            DeviceRef(spec.device_type, spec.id)
            for spec in pipeline_config.model.device_specs
        ]

        # Map activation names
        hidden_act = _ACTIVATION_MAP.get(
            huggingface_config.hidden_act,
            huggingface_config.hidden_act,
        )

        # Extract no_rope_layers
        no_rope_layers = list(
            getattr(huggingface_config, "no_rope_layers", [])
        )

        # SmolLM3 nests rope_theta inside rope_parameters dict
        rope_params = getattr(huggingface_config, "rope_parameters", None)
        if isinstance(rope_params, dict):
            rope_theta = float(rope_params.get("rope_theta", 500000.0))
        else:
            rope_theta = float(getattr(huggingface_config, "rope_theta", 500000.0))

        return cls(
            vocab_size=huggingface_config.vocab_size,
            hidden_size=huggingface_config.hidden_size,
            intermediate_size=huggingface_config.intermediate_size,
            num_hidden_layers=huggingface_config.num_hidden_layers,
            num_attention_heads=huggingface_config.num_attention_heads,
            num_key_value_heads=huggingface_config.num_key_value_heads,
            hidden_act=hidden_act,
            max_position_embeddings=SmolLM3Config.calculate_max_seq_len(
                pipeline_config, huggingface_config
            ),
            rms_norm_eps=huggingface_config.rms_norm_eps,
            rope_theta=rope_theta,
            attention_bias=huggingface_config.attention_bias,
            mlp_bias=getattr(huggingface_config, "mlp_bias", False),
            tie_word_embeddings=getattr(
                huggingface_config, "tie_word_embeddings", True
            ),
            no_rope_layers=no_rope_layers,
            dtype=dtype,
            devices=device_refs,
            kv_params=SmolLM3Config.construct_kv_params(
                huggingface_config=huggingface_config,
                pipeline_config=pipeline_config,
                devices=device_refs,
                kv_cache_config=kv_cache_config,
                cache_dtype=cache_dtype,
            ),
        )

    def finalize(
        self,
        huggingface_config: AutoConfig,
        state_dict: dict[str, WeightData],
        return_logits: ReturnLogits,
    ) -> None:
        self.tie_word_embeddings = (
            getattr(huggingface_config, "tie_word_embeddings", True)
            or "lm_head.weight" not in state_dict
        )
        self.return_logits = return_logits


_ACTIVATION_MAP = {
    "silu": "silu",
    "swish": "silu",
    "gelu_pytorch_tanh": "gelu_tanh",
}