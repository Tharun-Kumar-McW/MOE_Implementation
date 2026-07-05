"""Config for XiaomiMiMo/MiMo-7B-Base.

MiMoConfig (configuration_mimo.py) is:
    class MiMoConfig(Qwen2Config):
        model_type = "mimo"
        def __init__(self, *args, num_nextn_predict_layers=0, **kwargs):
            self.num_nextn_predict_layers = num_nextn_predict_layers
            super().__init__(*args, **kwargs)

So all standard fields (hidden_size, num_attention_heads, rms_norm_eps,
attention_bias, rope_theta, etc.) come directly from Qwen2Config with
the exact same attribute names that Llama3Config expects.

The only thing blocking Llama3Config.initialize_from_config is
model_type="mimo" — it expects "llama" or "qwen2" etc.
We patch it temporarily so the parent reads all fields correctly.

config.json values:
  hidden_size            : 4096
  intermediate_size      : 11008
  num_hidden_layers      : 36
  num_attention_heads    : 32
  num_key_value_heads    : 8
  head_dim               : 128
  vocab_size             : 151680
  rope_theta             : 640000
  rms_norm_eps           : 1e-05
  attention_bias         : true   ← QKV projections have bias
  use_sliding_window     : false
  num_nextn_predict_layers: 1     ← MTP head, ignored in MAX
"""

from __future__ import annotations

from dataclasses import dataclass

from max.pipelines.architectures.llama3.model_config import Llama3Config
from max.pipelines.lib import MAXModelConfig, PipelineConfig
from transformers import AutoConfig
from typing_extensions import Self, override


@dataclass(kw_only=True)
class MiMoConfig(Llama3Config):
    """Llama3Config adapted for MiMo-7B (model_type='mimo')."""

    model_type: str = "mimo"

    @override
    @classmethod
    def initialize_from_config(
        cls,
        pipeline_config: PipelineConfig,
        huggingface_config: AutoConfig,   # this is a MiMoConfig / Qwen2Config instance
        model_config: MAXModelConfig | None = None,
    ) -> Self:
        # MiMoConfig inherits Qwen2Config which has all the same field names
        # as LlamaConfig (hidden_size, num_attention_heads, rms_norm_eps, etc.)
        # The only thing blocking the parent is model_type="mimo".
        # Patch it to "llama" so Llama3Config's field mapping runs cleanly.
        original_model_type = getattr(huggingface_config, "model_type", None)
        huggingface_config.model_type = "llama"

        config = super().initialize_from_config(
            pipeline_config, huggingface_config, model_config
        )

        # Restore
        huggingface_config.model_type = original_model_type
        config.model_type = "mimo"

        return config