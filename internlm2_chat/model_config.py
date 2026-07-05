from __future__ import annotations
from dataclasses import dataclass
from max.pipelines.architectures.llama3.model_config import Llama3Config
from max.pipelines.lib import MAXModelConfig, PipelineConfig
from transformers import AutoConfig
from typing_extensions import Self, override

PADDED_VOCAB_SIZE = 92550  # must match weight_adapters.py


@dataclass(kw_only=True)
class InternLM2Config(Llama3Config):

    @override
    @classmethod
    def initialize_from_config(
        cls,
        pipeline_config: PipelineConfig,
        huggingface_config: AutoConfig,
        model_config: MAXModelConfig | None = None,
    ) -> Self:
        # Force padded vocab BEFORE super() reads it
        huggingface_config.vocab_size = PADDED_VOCAB_SIZE
        huggingface_config.tie_word_embeddings = False

        config = super().initialize_from_config(
            pipeline_config, huggingface_config, model_config
        )

        # Force again after super() in case it reset
        config.vocab_size = PADDED_VOCAB_SIZE
        config.rope_theta = float(
            getattr(huggingface_config, "rope_theta", 1_000_000.0)
        )
        return config
