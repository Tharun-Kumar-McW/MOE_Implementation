"""Config for KeyLM-75M.

KeyLM-75M config.json fields:
  architectures          : ["KeyLM75M"]
  model_type             : keylm75m
  vocab_size             : 12020
  hidden_size            : 512
  head_dim               : 64
  num_attention_heads    : 8
  num_key_value_heads    : 2   (GQA)
  intermediate_size      : 1280
  num_hidden_layers      : 24
  max_position_embeddings: 2048
  rope_theta             : 10000.0
  rms_norm_eps           : 1e-06
  hidden_act             : silu
  tie_word_embeddings    : false

Architecture is identical to LLaMA (RMSNorm, SwiGLU, RoPE, no bias).
We subclass Llama3Config and patch model_type so the parent's
initialize_from_config does not reject the unknown type.
"""

from __future__ import annotations

from dataclasses import dataclass

from max.pipelines.architectures.llama3.model_config import Llama3Config
from max.pipelines.lib import MAXModelConfig, PipelineConfig
from transformers import AutoConfig
from typing_extensions import Self, override


@dataclass(kw_only=True)
class KeyLMConfig(Llama3Config):
    """Llama3Config adapted for KeyLM-75M (model_type='keylm75m')."""

    model_type: str = "keylm75m"

    @override
    @classmethod
    def initialize_from_config(
        cls,
        pipeline_config: PipelineConfig,
        huggingface_config: AutoConfig,
        model_config: MAXModelConfig | None = None,
    ) -> Self:
        # Patch model_type to "llama" so Llama3Config's field mapping works
        # without raising an unknown-type error.
        original_model_type = getattr(huggingface_config, "model_type", None)
        huggingface_config.model_type = "llama"

        config = super().initialize_from_config(
            pipeline_config, huggingface_config, model_config
        )

        # Restore the real model_type
        huggingface_config.model_type = original_model_type
        config.model_type = "keylm75m"

        return config