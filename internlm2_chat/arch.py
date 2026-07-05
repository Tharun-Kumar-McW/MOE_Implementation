from max.graph.weights import WeightsFormat
from max.interfaces import PipelineTask
from max.pipelines.core import TextContext
from max.pipelines.lib import SupportedArchitecture
from max.pipelines.architectures.llama3.model import Llama3Model
from .model import InternLM2Model
from .model_config import InternLM2Config
from .tokenizer import InternLM2Tokenizer
from . import weight_adapters

internlm2_arch = SupportedArchitecture(
    name="InternLM2ForCausalLM",
    task=PipelineTask.TEXT_GENERATION,
    example_repo_ids=["internlm/internlm2-chat-1_8b"],
    default_encoding="float32",
    supported_encodings={"float32"},
    default_weights_format=WeightsFormat.safetensors,
    pipeline_model=InternLM2Model,
    tokenizer=InternLM2Tokenizer,
    context_type=TextContext,
    rope_type="normal",
    config=InternLM2Config,
    multi_gpu_supported=False,
    weight_adapters={
        WeightsFormat.safetensors: weight_adapters.convert_safetensor_state_dict,
    },
)
