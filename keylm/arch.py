"""KeyLM-75M architecture registration."""

from max.graph.weights import WeightsFormat
from max.interfaces import PipelineTask
from max.pipelines.core import TextContext
from max.pipelines.lib import SupportedArchitecture, TextTokenizer

from .model import KeyLMModel
from .model_config import KeyLMConfig
from . import weight_adapters

keylm_arch = SupportedArchitecture(
    name="KeyLM75M",                  # matches architectures[0] in config.json
    task=PipelineTask.TEXT_GENERATION,
    example_repo_ids=[
        "Eclipse-Senpai/KeyLM-75M",
    ],
    default_encoding="bfloat16",
    supported_encodings={
        "bfloat16",
        "float32",
    },
    default_weights_format=WeightsFormat.safetensors,
    pipeline_model=KeyLMModel,
    tokenizer=TextTokenizer,
    context_type=TextContext,
    rope_type="normal",
    weight_adapters={
        WeightsFormat.safetensors: weight_adapters.convert_safetensor_state_dict,
    },
    config=KeyLMConfig,
    multi_gpu_supported=False,
)