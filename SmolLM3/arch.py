# ===----------------------------------------------------------------------=== #
# SmolLM3 architecture registration.
#
# Registers SmolLM3ForCausalLM with the MAX pipeline so it is auto-selected
# when loading any SmolLM3 checkpoint via AutoModelForCausalLM.
# ===----------------------------------------------------------------------=== #
"""SmolLM3 architecture descriptor for the MAX pipeline."""

from max.graph.weights import WeightsFormat
from max.interfaces import PipelineTask
from max.pipelines.core import TextContext
from max.pipelines.lib import SupportedArchitecture, TextTokenizer

from . import weight_adapters
from .model import SmolLM3Model
from .model_config import SmolLM3Config

smollm3_arch = SupportedArchitecture(
    # Must match the `architectures` field in SmolLM3's config.json.
    name="SmolLM3ForCausalLM",
    example_repo_ids=[
        "HuggingFaceTB/SmolLM3-3B",
        "HuggingFaceTB/SmolLM3-3B-Instruct",
    ],
    # SmolLM3 config.json specifies torch_dtype: bfloat16.
    default_encoding="bfloat16",
    supported_encodings={
        "float32",
        "bfloat16",
    },
    pipeline_model=SmolLM3Model,
    tokenizer=TextTokenizer,
    context_type=TextContext,
    # SmolLM3 uses vanilla RoPE (rope_scaling: null, high theta = 5e6).
    rope_type="normal",
    default_weights_format=WeightsFormat.safetensors,
    # SmolLM3-3B is a single-GPU model (pretraining_tp hint is for training only).
    multi_gpu_supported=False,
    weight_adapters={
        WeightsFormat.safetensors: weight_adapters.convert_safetensor_state_dict,
        WeightsFormat.gguf: weight_adapters.convert_gguf_state_dict,
    },
    task=PipelineTask.TEXT_GENERATION,
    config=SmolLM3Config,
)
