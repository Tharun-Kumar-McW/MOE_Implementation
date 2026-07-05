# ===----------------------------------------------------------------------=== #
# SmolLM3 model configuration for MAX pipeline (ModuleV3 API).
# Extends Llama3Config with SmolLM3-specific fields:
#   - no_rope_layers: per-layer RoPE toggle (0 = NoPE, 1 = RoPE)
#   - no_rope_layer_interval: interval at which RoPE is skipped (default 4)
# ===----------------------------------------------------------------------=== #
"""Config for SmolLM3 models (ModuleV3)."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Literal

from max.dtype import DType
from max.graph import DeviceRef
from max.graph.weights import WeightData, WeightsFormat, weights_format
from max.nn.kv_cache import KVCacheParams
from max.nn.transformer import ReturnHiddenStates, ReturnLogits
from max.pipelines.lib import (
    KVCacheConfig,
    MAXModelConfig,
    PipelineConfig,
    upper_bounded_default,
)
from max.pipelines.lib.config.config_enums import supported_encoding_dtype
from max.pipelines.lib.interfaces.arch_config import ArchConfigWithKVCache
from max.pipelines.lib.pipeline_variants.utils import get_rope_theta
from transformers import AutoConfig
from typing_extensions import Self, override

# Reuse Llama3's RoPE embedding — SmolLM3 uses the same vanilla RoPE
# (rope_scaling: null in config.json) with a high theta of 5_000_000.
from max.pipelines.architectures.llama3_modulev3.layers.rotary_embedding import (
    Llama3RotaryEmbedding,
)


@dataclass(kw_only=True)
class SmolLM3Config(ArchConfigWithKVCache):
    """Model configuration for SmolLM3 graph construction/execution.

    Identical to Llama3Config plus:
      - ``no_rope_layers``: list[int] of length num_hidden_layers.
            1 = apply RoPE on this layer, 0 = NoPE (skip RoPE).
      - ``no_rope_layer_interval``: convenience int; every Nth layer is NoPE.
    """

    hidden_size: int
    num_attention_heads: int
    num_key_value_heads: int
    num_hidden_layers: int
    rope_theta: float
    max_seq_len: int
    intermediate_size: int
    interleaved_rope_weights: bool
    vocab_size: int
    dtype: DType
    kv_params: KVCacheParams
    devices: list[DeviceRef]

    # SmolLM3-specific: per-layer RoPE mask
    # Length == num_hidden_layers; 1 = RoPE, 0 = NoPE
    no_rope_layers: list[int] = field(default_factory=list)
    no_rope_layer_interval: int = 4

    # Standard optional fields (match Llama3Config)
    return_logits: ReturnLogits = ReturnLogits.LAST_TOKEN
    return_hidden_states: ReturnHiddenStates = ReturnHiddenStates.NONE
    norm_method: Literal["rms_norm"] = "rms_norm"
    rms_norm_eps: float | None = None
    attention_bias: bool = False      # attention_bias: false in config.json
    tie_word_embeddings: bool = True   # tie_word_embeddings: true
    stacked_mlp: bool = False
    stacked_qkv: bool = False
    attention_multiplier: float = 0.0  # computed in initialize()
    embedding_multiplier: float = 1.0
    residual_multiplier: float = 1.0
    clip_qkv: float | None = None
    logits_scaling: float = 1.0

    # ------------------------------------------------------------------ #
    # ArchConfigWithKVCache interface
    # ------------------------------------------------------------------ #

    def get_kv_params(self) -> KVCacheParams:
        return self.kv_params

    def get_max_seq_len(self) -> int:
        return self.max_seq_len

    # ------------------------------------------------------------------ #
    # Helper statics (mirrors Llama3Config API so callers are compatible)
    # ------------------------------------------------------------------ #

    @staticmethod
    def get_head_dim(huggingface_config: AutoConfig) -> int:
        if hasattr(huggingface_config, "head_dim"):
            return huggingface_config.head_dim
        return (
            huggingface_config.hidden_size
            // huggingface_config.num_attention_heads
        )

    @staticmethod
    def get_head_dim_from_config(config: SmolLM3Config) -> int:
        return config.kv_params.head_dim

    @staticmethod
    def get_num_layers(huggingface_config: AutoConfig) -> int:
        return huggingface_config.num_hidden_layers

    @staticmethod
    def calculate_attention_multiplier(huggingface_config: AutoConfig) -> float:
        return getattr(
            huggingface_config,
            "attention_multiplier",
            math.sqrt(1.0 / float(SmolLM3Config.get_head_dim(huggingface_config))),
        )

    @staticmethod
    def construct_kv_params(
        huggingface_config: AutoConfig,
        pipeline_config: PipelineConfig,
        devices: list[DeviceRef],
        kv_cache_config: KVCacheConfig,
        cache_dtype: DType,
    ) -> KVCacheParams:
        return kv_cache_config.to_params(
            dtype=cache_dtype,
            n_kv_heads=huggingface_config.num_key_value_heads,
            head_dim=SmolLM3Config.get_head_dim(huggingface_config),
            num_layers=SmolLM3Config.get_num_layers(huggingface_config),
            devices=devices,
            data_parallel_degree=pipeline_config.model.data_parallel_degree,
        )

    @staticmethod
    def calculate_max_seq_len(
        pipeline_config: PipelineConfig,
        huggingface_config: AutoConfig,
    ) -> int:
        try:
            return upper_bounded_default(
                upper_bound=huggingface_config.max_position_embeddings,
                default=pipeline_config.model.max_length,
            )
        except ValueError as e:
            raise ValueError(
                "Unable to infer max_length for SmolLM3, the provided "
                f"max_length ({pipeline_config.model.max_length}) exceeds the "
                f"model's max_position_embeddings "
                f"({huggingface_config.max_position_embeddings})."
            ) from e

    # ------------------------------------------------------------------ #
    # Factory
    # ------------------------------------------------------------------ #

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
                f"HuggingFace config is required for '{model_config.model_path}', "
                "but config could not be loaded. "
                "Please ensure the model repository contains a valid config.json."
            )

        kv_cache_config = model_config.kv_cache
        quantization_encoding = model_config.quantization_encoding
        if quantization_encoding is None:
            raise ValueError("quantization_encoding must not be None")
        dtype = supported_encoding_dtype(quantization_encoding)
        cache_dtype = model_config.kv_cache.cache_dtype

        _weights_format = weights_format(model_config.weight_path)
        interleaved_rope_weights = (
            _weights_format == WeightsFormat.gguf
            and model_config.rope_type == "normal"
        )

        device_refs = [
            DeviceRef(spec.device_type, spec.id)
            for spec in model_config.device_specs
        ]

        # SmolLM3-specific: read the per-layer NoPE mask from HF config.
        # Falls back to computing from interval if not present.
        num_layers = huggingface_config.num_hidden_layers
        no_rope_layer_interval: int = getattr(
            huggingface_config, "no_rope_layer_interval", 4
        )
        no_rope_layers: list[int] = list(
            getattr(huggingface_config, "no_rope_layers", [])
        )
        if not no_rope_layers:
            # Reconstruct: 0 every `interval`-th layer (0-indexed), else 1.
            no_rope_layers = [
                0 if (i + 1) % no_rope_layer_interval == 0 else 1
                for i in range(num_layers)
            ]

        attention_multiplier = SmolLM3Config.calculate_attention_multiplier(
            huggingface_config
        )

        return cls(
            hidden_size=huggingface_config.hidden_size,
            num_attention_heads=huggingface_config.num_attention_heads,
            num_key_value_heads=huggingface_config.num_key_value_heads,
            num_hidden_layers=num_layers,
            rope_theta=get_rope_theta(huggingface_config),
            intermediate_size=huggingface_config.intermediate_size,
            interleaved_rope_weights=interleaved_rope_weights,
            vocab_size=huggingface_config.vocab_size,
            dtype=dtype,
            max_seq_len=SmolLM3Config.calculate_max_seq_len(
                pipeline_config, huggingface_config=huggingface_config
            ),
            kv_params=SmolLM3Config.construct_kv_params(
                huggingface_config=huggingface_config,
                pipeline_config=pipeline_config,
                devices=device_refs,
                kv_cache_config=kv_cache_config,
                cache_dtype=cache_dtype,
            ),
            attention_multiplier=attention_multiplier,
            devices=device_refs,
            no_rope_layers=no_rope_layers,
            no_rope_layer_interval=no_rope_layer_interval,
            tie_word_embeddings=getattr(
                huggingface_config, "tie_word_embeddings", True
            ),
            clip_qkv=getattr(huggingface_config, "clip_qkv", None),
            logits_scaling=getattr(huggingface_config, "logits_scaling", 1.0),
        )

    def finalize(
        self,
        huggingface_config: AutoConfig,
        state_dict: dict[str, WeightData],
        return_logits: ReturnLogits,
        return_hidden_states: ReturnHiddenStates = ReturnHiddenStates.NONE,
        norm_method: Literal["rms_norm"] = "rms_norm",
        attention_bias: bool = False,
    ) -> None:
        """Populate fields that require the weight state dict to determine."""

        def _strip_prefix(s: str, prefix: str) -> str:
            return s.removeprefix(prefix)

        has_lm_prefix = any(k.startswith("language_model.") for k in state_dict)
        has_model_prefix = any(k.startswith("model.") for k in state_dict)

        if has_lm_prefix:
            normalized: dict[str, WeightData] = {
                _strip_prefix(k, "language_model."): v
                for k, v in state_dict.items()
                if k.startswith("language_model.")
            }
        elif has_model_prefix:
            normalized = {
                _strip_prefix(k, "model."): v
                for k, v in state_dict.items()
                if k.startswith("model.")
            }
        else:
            normalized = dict(state_dict)

        tie_word_embeddings = (
            getattr(huggingface_config, "tie_word_embeddings", True)
            or "lm_head.weight" not in normalized
        )

        rms_norm_eps = huggingface_config.rms_norm_eps if norm_method == "rms_norm" else None

        self.norm_method = norm_method
        self.rms_norm_eps = rms_norm_eps
        self.tie_word_embeddings = tie_word_embeddings
        self.stacked_mlp = "layers.0.mlp.gate_up_proj.weight" in normalized
        self.stacked_qkv = "layers.0.self_attn.qkv_proj.weight" in normalized
        self.attention_bias = attention_bias
        self.return_logits = return_logits
        self.return_hidden_states = return_hidden_states
