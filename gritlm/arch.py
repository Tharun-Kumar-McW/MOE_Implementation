# ===----------------------------------------------------------------------=== #
# GritLM architecture registration for MAX pipeline.
# ===----------------------------------------------------------------------=== #
"""GritLM architecture descriptor."""

from max.graph.weights import WeightsFormat
from max.interfaces import PipelineTask
from max.pipelines.core import TextContext
from max.pipelines.lib import SupportedArchitecture, TextTokenizer

from . import weight_adapters
from .model import GritLMModel
from .model_config import GritLMConfig

gritlm_arch = SupportedArchitecture(
    # Must match the 'architectures' field in GritLM's config.json
    name="GritLM",
    example_repo_ids=["GritLM/GritLM-7B", "GritLM/GritLM-8x7B"],
    default_encoding="bfloat16",
    supported_encodings={"float32", "bfloat16"},
    pipeline_model=GritLMModel,
    tokenizer=TextTokenizer,
    context_type=TextContext,
    # GritLM uses standard Mistral RoPE (non-interleaved safetensors)
    rope_type="normal",
    default_weights_format=WeightsFormat.safetensors,
    multi_gpu_supported=False,
    weight_adapters={
        WeightsFormat.safetensors: weight_adapters.convert_safetensor_state_dict,
    },
    task=PipelineTask.TEXT_GENERATION,
    config=GritLMConfig,
)
