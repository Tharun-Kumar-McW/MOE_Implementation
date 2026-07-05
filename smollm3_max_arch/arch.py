# ===----------------------------------------------------------------------=== #
# Copyright (c) 2026, Modular Inc. All rights reserved.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions:
# https://llvm.org/LICENSE.txt
# ===----------------------------------------------------------------------=== #

from max.graph.weights import WeightsFormat
from max.interfaces import PipelineTask
from max.pipelines.core import TextContext
from max.pipelines.lib import SupportedArchitecture, TextTokenizer

from . import weight_adapters
from .model import SmolLM3PipelineModel
from .model_config import SmolLM3Config

smollm3_arch = SupportedArchitecture(
    name="SmolLM3ForCausalLM",
    example_repo_ids=[
        "HuggingFaceTB/SmolLM3-3B",
    ],
    default_encoding="bfloat16",
    supported_encodings={
        "bfloat16",
        "float32",
    },
    pipeline_model=SmolLM3PipelineModel,
    task=PipelineTask.TEXT_GENERATION,
    tokenizer=TextTokenizer,
    context_type=TextContext,
    default_weights_format=WeightsFormat.safetensors,
    multi_gpu_supported=False,
    rope_type="normal",
    weight_adapters={
        WeightsFormat.safetensors: weight_adapters.convert_safetensor_state_dict,
    },
    config=SmolLM3Config,
)