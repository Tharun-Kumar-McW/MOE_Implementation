"""MiMo-7B architecture registration."""

from max.graph.weights import WeightsFormat
from max.interfaces import PipelineTask
from max.pipelines.core import TextContext
from max.pipelines.lib import SupportedArchitecture, TextTokenizer

from .model import MiMoModel
from .model_config import MiMoConfig
from . import weight_adapters

mimo_arch = SupportedArchitecture(
    name="MiMoForCausalLM",
    task=PipelineTask.TEXT_GENERATION,
    example_repo_ids=[
        "XiaomiMiMo/MiMo-7B-Base",
        "XiaomiMiMo/MiMo-7B-RL",
    ],
    default_encoding="float32",
    supported_encodings={
        "float32",
        "bfloat16",
        "q4_k",
        "q4_0",
        "q6_k",
        "float8_e4m3fn",
        "gptq",
    },
    default_weights_format=WeightsFormat.safetensors,
    pipeline_model=MiMoModel,
    tokenizer=TextTokenizer,
    context_type=TextContext,
    rope_type="normal",
    weight_adapters={
        WeightsFormat.safetensors: weight_adapters.convert_safetensor_state_dict,
    },
    config=MiMoConfig,
    multi_gpu_supported=False,
)