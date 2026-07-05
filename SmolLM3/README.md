# SmolLM3 Package README

## Overview

This directory contains a MAX custom architecture implementation for the SmolLM3 model.
It provides the complete integration needed to register SmolLM3 with MAX, configure the model,
compile it for inference, and execute it with KV cache support.

The package files are:

- `__init__.py`
- `arch.py`
- `model_config.py`
- `model.py`
- `smollm3.py`
- `weight_adapters.py`
- `layers/__init__.py`
- `layers/attention.py`
- `layers/transformer_block.py`

Each section below explains the file purpose and the step-by-step process that code implements.

---

## Folder Structure

```
SmolLM3/
  __init__.py
  arch.py
  model_config.py
  model.py
  smollm3.py
  weight_adapters.py
  layers/
    __init__.py
    attention.py
    transformer_block.py
```

---

## File Purpose Summary

- `__init__.py`: package entry point, exports the MAX architecture list and public symbols.
- `arch.py`: registers the SmolLM3 architecture with MAX and specifies supported weights/encodings.
- `model_config.py`: maps HuggingFace configuration into MAX pipeline configuration.
- `model.py`: compiles and executes the model in the MAX runtime.
- `smollm3.py`: implements the transformer model body, token embedding, and logits computation.
- `weight_adapters.py`: adapts checkpoint weights from HuggingFace/GGUF naming and dtype into MAX internal naming.
- `layers/__init__.py`: exports the custom layer classes.
- `layers/attention.py`: implements fused attention with optional RoPE and NoPE modes.
- `layers/transformer_block.py`: implements the decoder transformer block structure.

---

## `__init__.py`

### Source

```python
"""SmolLM3 transformer architecture for text generation.

Decoder-only transformer (LLaMA-style) with:
  - 36 layers, hidden_size=2048, intermediate_size=11008
  - GQA: 16 Q heads, 4 KV heads (head_dim=128)
  - SwiGLU MLP (mlp_bias: false)
  - RMSNorm (rms_norm_eps=1e-6)
  - RoPE theta=5_000_000 with per-layer NoPE: every 4th layer skips RoPE
  - Tied input/output embeddings (tie_word_embeddings: true)
  - max_position_embeddings: 65536

MAX custom architecture registration
--------------------------------------
MAX's --custom-architectures loader does:

    sys.path.append(os.path.dirname(module_spec))
    module = importlib.import_module(os.path.basename(module_spec))

It then requires the imported module to expose:

    ARCHITECTURES: list[SupportedArchitecture]

This module satisfies that contract via the ARCHITECTURES list below.

CLI usage
---------
Place this package at e.g. ``/path/to/architectures/smollm3/`` then run::

    max serve \\
        --model-path HuggingFaceTB/SmolLM3-3B \\
        --custom-architectures /path/to/architectures/smollm3 \\
        --devices cpu \\
        --max-batch-size 1 \\
        --max-length 4096 \\
        --quantization-encoding bfloat16

Note: the path must point to the *package directory* (containing this
__init__.py). MAX splits the path as::

    module_path = os.path.dirname("/path/to/architectures/smollm3")
    # → "/path/to/architectures"
    module_name = os.path.basename("/path/to/architectures/smollm3")
    # → "smollm3"

So the directory name on disk must exactly match the Python module name
(all lowercase, no hyphens).
"""

from .arch import smollm3_arch
from .model import SmolLM3Inputs, SmolLM3Model
from .model_config import SmolLM3Config

# !! Required by MAX's custom architecture loader !!
# max/pipelines/lib/config/config.py checks for this list and registers
# each entry into PIPELINE_REGISTRY.
ARCHITECTURES = [smollm3_arch]

__all__ = [
    "ARCHITECTURES",
    "SmolLM3Config",
    "SmolLM3Inputs",
    "SmolLM3Model",
    "smollm3_arch",
]
```

### Explanation

This package entry point is the bridge between MAX and the SmolLM3 custom architecture.
It imports the architecture descriptor, pipeline model, and config class, then exposes them via the required `ARCHITECTURES` list.

Process details:

1. When MAX loads custom architecture code, it imports this module.
2. The loader expects `ARCHITECTURES`, so this file provides one entry: `smollm3_arch`.
3. `__all__` declares the public API for external imports.
4. This file contains no model logic itself; it only wires package exports for MAX discovery.

---

## `arch.py`

### Source

```python
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
```

### Explanation

This file defines the architecture descriptor MAX uses to recognize SmolLM3 and select the correct runtime components.

Process details:

1. `SupportedArchitecture(...)` creates the registration object.
2. `name` must match the architecture identifier stored inside the model’s HuggingFace config.
3. `example_repo_ids` helps MAX users locate compatible repositories.
4. `default_encoding` and `supported_encodings` declare the expected and acceptable compute dtypes.
5. `pipeline_model` points to the runtime wrapper class that handles compilation and execution.
6. `tokenizer` and `context_type` choose MAX’s text pipeline elements.
7. `rope_type="normal"` signals that this model uses standard rotary embeddings rather than scaled variants.
8. `default_weights_format` defaults the loader to safetensors, while `weight_adapters` also supports GGUF.
9. `multi_gpu_supported=False` documents that this implementation is not built for model parallel or data parallel execution beyond a single device.
10. `task=TEXT_GENERATION` ensures the pipeline behaves as a text generation model.
11. `config=SmolLM3Config` binds the architecture to the custom configuration class used by the runtime.

---

## `model_config.py`

### Source

```python
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

    no_rope_layers: list[int] = field(default_factory=list)
    no_rope_layer_interval: int = 4

    return_logits: ReturnLogits = ReturnLogits.LAST_TOKEN
    return_hidden_states: ReturnHiddenStates = ReturnHiddenStates.NONE
    norm_method: Literal["rms_norm"] = "rms_norm"
    rms_norm_eps: float | None = None
    attention_bias: bool = False
    tie_word_embeddings: bool = True
    stacked_mlp: bool = False
    stacked_qkv: bool = False
    attention_multiplier: float = 0.0
    embedding_multiplier: float = 1.0
    residual_multiplier: float = 1.0
    clip_qkv: float | None = None
    logits_scaling: float = 1.0

    def get_kv_params(self) -> KVCacheParams:
        return self.kv_params

    def get_max_seq_len(self) -> int:
        return self.max_seq_len

    @staticmethod
    def get_head_dim(huggingface_config: AutoConfig) -> int:
        if hasattr(huggingface_config, "head_dim"):
            return huggingface_config.head_dim
        return (
            huggingface_config.hidden_size
            // huggingface_config.num_attention_heads
        )

    @staticmethod
    def get_head_dim_from_config(config: "SmolLM3Config") -> int:
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

    @override
    @classmethod
    def initialize(
        cls,
        pipeline_config: PipelineConfig,
        model_config: MAXModelConfig | None = None,
    ) -> "SmolLM3Config":
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

        num_layers = huggingface_config.num_hidden_layers
        no_rope_layer_interval: int = getattr(
            huggingface_config, "no_rope_layer_interval", 4
        )
        no_rope_layers: list[int] = list(
            getattr(huggingface_config, "no_rope_layers", [])
        )
        if not no_rope_layers:
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
        )
```

### Explanation

This file is the configuration factory for SmolLM3. It prepares every runtime hyperparameter and MAX-specific control value needed to construct the graph.

Detailed process:

1. The dataclass fields store both model hyperparameters from the HuggingFace config and additional MAX runtime settings.
   - `hidden_size`, `num_attention_heads`, `num_key_value_heads`, `intermediate_size`, and `num_hidden_layers` come from the source model.
   - `max_seq_len` is the maximum sequence length enforced at runtime.
   - `dtype` is the compute dtype derived from MAX’s pipeline quantization settings.
   - `kv_params` contains the KV cache layout, head dim, number of layers, and device placement.
   - `devices` holds the target devices used to build the graph.

2. `no_rope_layers` and `no_rope_layer_interval` support SmolLM3’s per-layer RoPE/NoPE behavior.
   - `no_rope_layers` is a binary mask of length `num_hidden_layers`.
   - `1` means apply RoPE, `0` means skip RoPE and use identity rotation.
   - `no_rope_layer_interval` is a convenience field used to build the mask when the HuggingFace config does not explicitly define the mask.

3. Static helpers compute model internals from the HuggingFace config.
   - `get_head_dim()` returns the Q/K/V head dimension from either `head_dim` or by dividing `hidden_size` by `num_attention_heads`.
   - `get_head_dim_from_config()` reads the already-constructed `kv_params.head_dim`.
   - `get_num_layers()` returns `num_hidden_layers` from the config.
   - `calculate_attention_multiplier()` uses an explicit `attention_multiplier` if present, otherwise defaults to `sqrt(1/head_dim)`.

4. `construct_kv_params()` turns MAX pipeline KV cache settings into a runtime `KVCacheParams` object.
   - It passes `cache_dtype` as the cache storage dtype.
   - It uses the model’s `num_key_value_heads` and the computed head dimension.
   - It sets the number of transformer layers and device layout.
   - It also forwards MAX data parallel degree so KV cache input shapes are correct.

5. `calculate_max_seq_len()` validates the runtime max length.
   - It returns the smaller of the pipeline’s requested `max_length` and the model’s `max_position_embeddings`.
   - If the pipeline requests a longer sequence than the model supports, it raises a clear error.

6. `initialize()` is the main constructor used during model compilation.
   - It loads the HuggingFace config from `pipeline_config.model.huggingface_config` and fails early if missing.
   - It ensures `quantization_encoding` is set, then converts that encoding into an actual MAX `DType`.
   - It reads `cache_dtype` from `model_config.kv_cache`.
   - It detects whether the loaded checkpoint format is GGUF and whether the model uses normal RoPE so it can choose interleaved RoPE weights correctly.
   - It converts MAX device specs into `DeviceRef` objects for graph construction.
   - It builds the `no_rope_layers` array from either the explicit config field or the `no_rope_layer_interval` fallback.
   - It computes `attention_multiplier` from external config or default formula.
   - It constructs and returns the final `SmolLM3Config` dataclass instance with every derived value set.

7. This config class is the central place where the HuggingFace checkpoint metadata and MAX runtime policy are merged into one object that the model builder can consume.

---

## `model.py`

### Source

```python
# ===----------------------------------------------------------------------=== #
# SmolLM3 pipeline model (PipelineModelWithKVCache wrapper).
# Handles compilation, input preparation, and output unpacking.
# ===----------------------------------------------------------------------=== #
"""SmolLM3 pipeline model using the ModuleV3 API."""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Literal, cast

import numpy as np
from max.driver import Buffer, Device
from max.dtype import DType
from max.engine import InferenceSession
from max.experimental import functional as F
from max.experimental.tensor import default_dtype
from max.graph import DeviceRef, TensorType
from max.graph.weights import Weights, WeightsAdapter
from max.nn.kv_cache import KVCacheInputs, KVCacheParams
from max.nn.transformer import ReturnHiddenStates, ReturnLogits
from max.pipelines.core import TextContext
from max.pipelines.lib import (
    CompilationTimer,
    KVCacheConfig,
    ModelInputs,
    ModelOutputs,
    PipelineConfig,
    PipelineModelWithKVCache,
)
from max.pipelines.lib.log_probabilities import LogProbabilitiesMixin
from transformers import AutoConfig

from .smollm3 import SmolLM3
from .model_config import SmolLM3Config

logger = logging.getLogger("max.pipelines")


@dataclass
class SmolLM3Inputs(ModelInputs):
    """Typed input bundle for the SmolLM3 pipeline model."""

    tokens: Buffer
    input_row_offsets: Buffer
    return_n_logits: Buffer

    @property
    def buffers(self) -> tuple[Buffer, ...]:
        if isinstance(self.input_row_offsets, np.ndarray):
            input_row_offsets = Buffer.from_numpy(self.input_row_offsets).to(
                self.tokens.device
            )
        else:
            input_row_offsets = self.input_row_offsets
        return (
            self.tokens,
            self.return_n_logits,
            input_row_offsets,
            *(
                self.kv_cache_inputs.flatten()
                if self.kv_cache_inputs is not None
                else ()
            ),
        )


class SmolLM3Model(LogProbabilitiesMixin, PipelineModelWithKVCache[TextContext]):
    """SmolLM3 pipeline model using the ModuleV3 API."""

    config_class: type[SmolLM3Config] = SmolLM3Config
    norm_method: Literal["rms_norm"] = "rms_norm"
    attention_bias: bool = False

    def __init__(
        self,
        pipeline_config: PipelineConfig,
        session: InferenceSession,
        devices: list[Device],
        kv_cache_config: KVCacheConfig,
        weights: Weights,
        adapter: WeightsAdapter | None = None,
        return_logits: ReturnLogits = ReturnLogits.LAST_TOKEN,
        return_hidden_states: ReturnHiddenStates = ReturnHiddenStates.NONE,
    ) -> None:
        super().__init__(
            pipeline_config,
            session,
            devices,
            kv_cache_config,
            weights,
            adapter,
            return_logits,
            return_hidden_states,
        )
        self.model = self.load_model()

    @staticmethod
    def calculate_max_seq_len(
        pipeline_config: PipelineConfig,
        huggingface_config: AutoConfig,
    ) -> int:
        return SmolLM3Config.calculate_max_seq_len(
            pipeline_config, huggingface_config
        )

    @classmethod
    def get_kv_params(
        cls,
        huggingface_config: AutoConfig,
        pipeline_config: PipelineConfig,
        devices: list[DeviceRef],
        kv_cache_config: KVCacheConfig,
        cache_dtype: DType,
    ) -> KVCacheParams:
        return SmolLM3Config.construct_kv_params(
            huggingface_config,
            pipeline_config,
            devices,
            kv_cache_config,
            cache_dtype,
        )

    def load_model(self) -> Callable[..., Any]:
        assert self.pipeline_config.runtime.max_batch_size, (
            "Expected max_batch_size to be set"
        )
        self._input_row_offsets_prealloc = Buffer.from_numpy(
            np.arange(
                self.pipeline_config.runtime.max_batch_size + 1, dtype=np.uint32
            )
        ).to(self.devices[0])

        with CompilationTimer("smollm3") as timer:
            device0 = self.devices[0]
            device_ref = DeviceRef(device0.label, device0.id)

            tokens_type = TensorType(
                DType.int64, shape=["total_seq_len"], device=device_ref
            )
            input_row_offsets_type = TensorType(
                DType.uint32,
                shape=["input_row_offsets_len"],
                device=device0,
            )
            return_n_logits_type = TensorType(
                DType.int64,
                shape=["return_n_logits"],
                device=DeviceRef.CPU(),
            )

            huggingface_config = self.huggingface_config
            if self.adapter:
                state_dict = self.adapter(
                    dict(self.weights.items()),
                    huggingface_config=huggingface_config,
                    pipeline_config=self.pipeline_config,
                )
            else:
                state_dict = {
                    key: value.data() for key, value in self.weights.items()
                }

            model_config = self.config_class.initialize(self.pipeline_config)
            model_config.finalize(
                huggingface_config=huggingface_config,
                state_dict=state_dict,
                norm_method=self.norm_method,
                attention_bias=self.attention_bias,
                return_logits=self.return_logits,
                return_hidden_states=self.return_hidden_states,
            )

            with F.lazy(), default_dtype(model_config.dtype):
                nn_model = SmolLM3(model_config, self.kv_params)
                nn_model.to(self.devices[0])

            kv_inputs = self.kv_params.get_symbolic_inputs()
            flattened_kv_types = kv_inputs.flatten()

            timer.mark_build_complete()
            compiled_model = nn_model.compile(
                tokens_type,
                return_n_logits_type,
                input_row_offsets_type,
                *flattened_kv_types,
                weights=state_dict,
            )

        return compiled_model

    def execute(self, model_inputs: ModelInputs) -> ModelOutputs:
        model_inputs = cast(SmolLM3Inputs, model_inputs)
        model_outputs = self.model(*model_inputs.buffers)

        has_offsets = self.return_logits in (
            ReturnLogits.VARIABLE,
            ReturnLogits.ALL,
        )
        has_hidden_states = self.return_hidden_states != ReturnHiddenStates.NONE

        if has_offsets and has_hidden_states:
            assert len(model_outputs) == 4
            return ModelOutputs(
                logits=cast(Buffer, model_outputs[1].driver_tensor),
                next_token_logits=cast(Buffer, model_outputs[0].driver_tensor),
                logit_offsets=cast(Buffer, model_outputs[2].driver_tensor),
                hidden_states=cast(Buffer, model_outputs[3].driver_tensor),
            )
        elif has_offsets:
            assert len(model_outputs) == 3
            return ModelOutputs(
                logits=cast(Buffer, model_outputs[1].driver_tensor),
                next_token_logits=cast(Buffer, model_outputs[0].driver_tensor),
                logit_offsets=cast(Buffer, model_outputs[2].driver_tensor),
            )
        elif has_hidden_states:
            assert len(model_outputs) == 2
            return ModelOutputs(
                logits=cast(Buffer, model_outputs[0].driver_tensor),
                next_token_logits=cast(Buffer, model_outputs[0].driver_tensor),
                hidden_states=cast(Buffer, model_outputs[1].driver_tensor),
            )
        else:
            assert len(model_outputs) == 1
            return ModelOutputs(
                logits=cast(Buffer, model_outputs[0].driver_tensor),
                next_token_logits=cast(Buffer, model_outputs[0].driver_tensor),
            )

    def prepare_initial_token_inputs(
        self,
        replica_batches: Sequence[Sequence[TextContext]],
        kv_cache_inputs: KVCacheInputs[Buffer, Buffer] | None = None,
        return_n_logits: int = 1,
    ) -> ModelInputs:
        if len(replica_batches) > 1:
            raise ValueError("SmolLM3Model does not support DP>1")

        context_batch = replica_batches[0]
        assert kv_cache_inputs is not None

        input_row_offsets = np.cumsum(
            [0] + [ctx.tokens.active_length for ctx in context_batch],
            dtype=np.uint32,
        )
        tokens = np.concatenate([ctx.tokens.active for ctx in context_batch])

        return SmolLM3Inputs(
            tokens=Buffer.from_numpy(tokens).to(self.devices[0]),
            input_row_offsets=Buffer.from_numpy(input_row_offsets).to(
                self.devices[0]
            ),
            return_n_logits=Buffer.from_numpy(
                np.array([return_n_logits], dtype=np.int64)
            ),
            kv_cache_inputs=kv_cache_inputs,
        )

    def prepare_next_token_inputs(
        self,
        next_tokens: Buffer,
        prev_model_inputs: ModelInputs,
    ) -> ModelInputs:
        prev_model_inputs = cast(SmolLM3Inputs, prev_model_inputs)
        row_offsets_size = prev_model_inputs.input_row_offsets.shape[0]

        next_row_offsets = self._input_row_offsets_prealloc[
            :row_offsets_size
        ].to(self.devices[0])

        return SmolLM3Inputs(
            tokens=next_tokens,
            input_row_offsets=next_row_offsets,
            return_n_logits=prev_model_inputs.return_n_logits,
            kv_cache_inputs=prev_model_inputs.kv_cache_inputs,
        )
```

### Explanation

This file is the runtime wrapper that connects MAX’s pipeline interface to the SmolLM3 ModuleV3 computation graph.

Process details:

1. `SmolLM3Inputs` defines the exact buffers the model expects at runtime.
   - `tokens` is the flattened token sequence.
   - `input_row_offsets` marks sequence boundaries inside the flattened token tensor.
   - `return_n_logits` controls the number of logits returned for variable-logit requests.
   - The `buffers` property also appends flattened KV cache tensors when present.

2. `SmolLM3Model` inherits from `PipelineModelWithKVCache`, so it implements the standard MAX pipeline lifecycle.

3. In `load_model()`:
   - It validates `max_batch_size` and creates a preallocated row-offset buffer for incremental decoding.
   - It constructs `TensorType` metadata for tokens, offsets, and logits request sizes.
   - It adapts the loaded checkpoint weights using the adapter if one is configured.
   - It initializes `SmolLM3Config` from `pipeline_config` and finalizes it with the actual state dict.
   - It enters a lazy graph-building context and creates the `SmolLM3` model on the target device.
   - It converts the KV cache parameters into symbolic input types and compiles the model with weights.

4. In `execute()`:
   - It calls the compiled model with the prepared buffers.
   - It inspects the requested `return_logits` and `return_hidden_states` modes.
   - It extracts and wraps the output tensors into `ModelOutputs` in the correct order.
   - This preserves compatibility with MAX’s expectation for logits, offsets, and hidden states.

5. In `prepare_initial_token_inputs()`:
   - It accepts a batch of `TextContext` objects and flattens them into a single token tensor.
   - It creates cumulative row offsets so the model can know where each example begins and ends.
   - It returns a `SmolLM3Inputs` object ready for the compiled graph.

6. In `prepare_next_token_inputs()`:
   - It uses the stored row-offset preallocation to build a stable offsets buffer for next-token decoding.
   - It preserves the previous KV cache inputs so decoding can continue without recomputing past keys/values.

---

## `smollm3.py`

### Source

```python
# ===----------------------------------------------------------------------=== #
# SmolLM3 ModuleV3 graph implementation.
#
# Key differences from plain Llama3:
#   1. Per-layer RoPE toggle (no_rope_layers mask from config)
#   2. Every 4th layer is a NoPE layer (uses SmolLM3Attention with use_rope=False)
#   3. tie_word_embeddings=True (lm_head weight = embed_tokens weight transposed)
# ===----------------------------------------------------------------------=== #
"""Implements the SmolLM3 model using the ModuleV3 API."""

from __future__ import annotations

import functools
from collections.abc import Callable

from max.dtype import DType
from max.experimental import functional as F
from max.experimental.nn import Module
from max.experimental.nn.common_layers.mlp import MLP
from max.experimental.nn.embedding import Embedding
from max.experimental.nn.linear import Linear
from max.experimental.nn.norm import RMSNorm
from max.experimental.nn.sequential import ModuleList
from max.experimental.tensor import Tensor
from max.graph import TensorValue, ops
from max.nn.kv_cache import KVCacheParamInterface, PagedCacheValues
from max.nn.transformer import ReturnHiddenStates, ReturnLogits

from max.pipelines.architectures.llama3_modulev3.layers.rotary_embedding import (
    Llama3RotaryEmbedding,
)

from .layers.attention import SmolLM3Attention
from .layers.transformer_block import SmolLM3TransformerBlock
from .model_config import SmolLM3Config


class SmolLM3TextModel(
    Module[[Tensor, PagedCacheValues, Tensor, Tensor], tuple[Tensor, ...]]
):
    """SmolLM3 decoder-only transformer.

    36 layers, GQA (16 Q heads / 4 KV heads), SwiGLU MLP, RMSNorm.
    Every 4th layer (indices 3, 7, 11, …) uses NoPE (no positional encoding).
    """

    def __init__(self, config: SmolLM3Config) -> None:
        super().__init__()
        self.devices = config.devices

        if config.rms_norm_eps is None:
            raise ValueError("rms_norm_eps cannot be None for SmolLM3.")

        create_norm: Callable[..., Module[[Tensor], Tensor]] = functools.partial(
            RMSNorm, config.hidden_size, eps=config.rms_norm_eps
        )

        rope = Llama3RotaryEmbedding(
            dim=config.hidden_size,
            n_heads=config.num_attention_heads,
            theta=config.rope_theta,
            max_seq_len=config.max_seq_len,
            device=config.devices[0].to_device(),
            head_dim=SmolLM3Config.get_head_dim_from_config(config),
            interleaved=config.interleaved_rope_weights,
            scaling_params=None,
        )

        self.embed_tokens = Embedding(
            config.vocab_size,
            dim=config.hidden_size,
        )
        self.norm = create_norm()

        self.tie_word_embeddings = config.tie_word_embeddings
        if config.tie_word_embeddings:
            self.lm_head = None
        else:
            self.lm_head = Linear(
                in_dim=config.hidden_size,
                out_dim=config.vocab_size,
                bias=False,
            )

        layers = []
        no_rope_layers = config.no_rope_layers
        for i in range(config.num_hidden_layers):
            use_rope: bool = bool(no_rope_layers[i]) if no_rope_layers else True

            attention = SmolLM3Attention(
                rope=rope,
                num_attention_heads=config.num_attention_heads,
                num_key_value_heads=config.num_key_value_heads,
                hidden_size=config.hidden_size,
                kv_params=config.kv_params,
                layer_idx=i,
                use_rope=use_rope,
                scale=config.attention_multiplier,
                has_bias=config.attention_bias,
                clip_qkv=config.clip_qkv,
            )

            mlp = MLP(
                hidden_dim=config.hidden_size,
                feed_forward_length=config.intermediate_size,
            )

            layers.append(
                SmolLM3TransformerBlock(
                    attention=attention,
                    mlp=mlp,
                    input_layernorm=create_norm(),
                    post_attention_layernorm=create_norm(),
                    residual_multiplier=config.residual_multiplier,
                )
            )

        self.layers = ModuleList(layers)
        self.dim = config.hidden_size
        self.n_heads = config.num_attention_heads
        self.kv_params = config.kv_params
        self.return_logits = config.return_logits
        self.return_hidden_states = config.return_hidden_states
        self.embedding_multiplier = config.embedding_multiplier
        self.logits_scaling = config.logits_scaling

    def _compute_logits(self, h: Tensor) -> Tensor:
        if self.tie_word_embeddings:
            return F.cast(h @ self.embed_tokens.weight.T, DType.float32)
        assert self.lm_head is not None
        return F.cast(self.lm_head(h), DType.float32)

    def forward(
        self,
        tokens: Tensor,
        kv_collection: PagedCacheValues,
        return_n_logits: Tensor,
        input_row_offsets: Tensor,
    ) -> tuple[Tensor, ...]:
        h = self.embed_tokens(tokens)

        if self.embedding_multiplier != 1.0:
            h = h * F.constant(
                self.embedding_multiplier, h.dtype, device=h.device
            )

        for idx, layer in enumerate(self.layers):
            layer_idx_tensor = F.constant(idx, DType.uint32, device=h.device)
            h = layer(
                layer_idx_tensor,
                h,
                kv_collection,
                input_row_offsets=input_row_offsets,
            )

        last_h = F.gather(h, input_row_offsets[1:] - 1, axis=0)
        last_logits = self._compute_logits(self.norm(last_h))
        logits = None
        offsets = None

        if self.return_logits == ReturnLogits.VARIABLE:
            return_n_logits_range = ops.range(
                return_n_logits[0],
                0,
                -1,
                out_dim="return_n_logits_range",
                device=h.device,
                dtype=DType.int64,
            )
            offsets = (
                F.unsqueeze(input_row_offsets[1:], -1) - return_n_logits_range
            )
            last_indices = F.reshape(offsets, shape=(-1,))
            last_tokens = F.gather(h, last_indices, axis=0)
            logits = self._compute_logits(self.norm(last_tokens))
            offsets = ops.range(
                0,
                TensorValue(last_indices.shape[0]) + return_n_logits[0],
                return_n_logits[0],
                out_dim="logit_offsets",
                device=h.device,
                dtype=DType.int64,
            )
        elif self.return_logits == ReturnLogits.ALL:
            logits = self._compute_logits(self.norm(h))
            offsets = input_row_offsets

        if self.logits_scaling != 1.0:
            last_logits = last_logits / self.logits_scaling
            if logits is not None:
                logits = logits / self.logits_scaling

        ret_val: tuple[Tensor, ...] = (last_logits,)
        if offsets is not None:
            assert logits is not None
            ret_val += (logits, offsets)

        if self.return_hidden_states == ReturnHiddenStates.LAST:
            ret_val += (last_h,)
        elif self.return_hidden_states == ReturnHiddenStates.ALL_NORMALIZED:
            ret_val += (self.norm(h),)

        return ret_val


class SmolLM3(Module[..., tuple[Tensor, ...]]):
    def __init__(self, config: SmolLM3Config, kv_params: KVCacheParamInterface) -> None:
        super().__init__()
        self.language_model = SmolLM3TextModel(config)
        self.config = config
        self.kv_params = kv_params

    def forward(
        self,
        tokens: Tensor,
        return_n_logits: Tensor,
        input_row_offsets: Tensor,
        *variadic_args: Tensor,
    ) -> tuple[Tensor, ...]:
        kv_inputs = iter(x._graph_value for x in variadic_args)
        kv_collections = (
            self.kv_params.get_symbolic_inputs().unflatten(kv_inputs).inputs
        )
        return self.language_model(
            tokens, kv_collections[0], return_n_logits, input_row_offsets
        )
```

### Explanation

This file implements the SmolLM3 computation graph itself, including the embedding layer, transformer stack, and output logic.

Process details:

1. `SmolLM3TextModel.__init__()` constructs the model body:
   - It creates a shared RoPE object from the runtime config.
   - It creates the token embedding layer and RMS normalization module.
   - It chooses either tied embeddings or a separate linear LM head based on `tie_word_embeddings`.
   - It builds `num_hidden_layers` transformer blocks, passing a per-layer `use_rope` flag into `SmolLM3Attention`.
   - Every layer gets its own attention and MLP modules, but all RoPE layers reuse the same rotary embedding object.

2. `SmolLM3TextModel._compute_logits()` handles two logits modes:
   - If `tie_word_embeddings` is true, it multiplies the hidden state by the transposed embedding weight.
   - Otherwise it uses a dedicated linear head.
   - It always casts logits to `float32` for stable output.

3. `SmolLM3TextModel.forward()` executes the sequence:
   - Embed the input tokens.
   - Optionally scale the embeddings by `embedding_multiplier`.
   - Run each transformer layer sequentially, passing KV cache and row offsets.
   - Gather the last hidden state for each sequence end from the flattened token batch.
   - Compute `last_logits` from the normalized final hidden state.
   - If `ReturnLogits.VARIABLE` or `ReturnLogits.ALL` is active, compute full logits and offsets accordingly.
   - Apply `logits_scaling` if configured.
   - Append hidden states to the output tuple if requested.

4. `SmolLM3` is the top-level ModuleV3 wrapper:
   - It accepts the flattened list of KV cache tensors as variadic arguments.
   - It reconstructs the symbolic KV collection expected by the text model.
   - It forwards tokens, return count, offsets, and KV cache into `SmolLM3TextModel`.

This separation keeps the outer pipeline interface simple while still supporting SmolLM3’s complex per-layer RoPE and KV cache behavior.

---

## `weight_adapters.py`

### Source

```python
# ===----------------------------------------------------------------------=== #
# SmolLM3 weight adapters.
#
# SmolLM3 safetensors use the standard HuggingFace layout:
#   model.embed_tokens.weight
#   model.layers.N.self_attn.{q,k,v,o}_proj.weight
#   model.layers.N.mlp.{gate,up,down}_proj.weight
#   model.layers.N.{input,post_attention}_layernorm.weight
#   model.norm.weight
#   lm_head.weight  ← absent when tie_word_embeddings=True
#
# MAX validates that each WeightData dtype matches the graph parameter dtype
# exactly (no implicit promotion). When --quantization-encoding float32 is
# passed but the checkpoint is bfloat16, we must cast here.
# ===----------------------------------------------------------------------=== #
"""Weight name adapters + optional dtype cast for SmolLM3."""

from __future__ import annotations

from max.dtype import DType
from max.graph.weights import WeightData, Weights
from transformers import AutoConfig

SMOLLM3_SAFETENSOR_MAPPING: dict[str, str] = {
    "model.": "language_model.",
    "lm_head": "language_model.lm_head",
}

SMOLLM3_GGUF_MAPPING: dict[str, str] = {
    "token_embd": "language_model.embed_tokens",
    "blk": "language_model.layers",
    "ffn_up": "mlp.up_proj",
    "ffn_down": "mlp.down_proj",
    "ffn_gate": "mlp.gate_proj",
    "ffn_norm": "post_attention_layernorm",
    "attn_norm": "input_layernorm",
    "attn_q": "self_attn.q_proj",
    "attn_v": "self_attn.v_proj",
    "attn_k": "self_attn.k_proj",
    "attn_output": "self_attn.o_proj",
    "output.weight": "language_model.lm_head.weight",
    "output_norm": "language_model.norm",
}


def _target_dtype(pipeline_config) -> DType | None:
    try:
        from max.pipelines.lib.config.config_enums import supported_encoding_dtype
        enc = pipeline_config.model.quantization_encoding
        if enc is not None:
            return supported_encoding_dtype(enc)
    except Exception:
        pass
    return None


def convert_safetensor_state_dict(
    state_dict: dict[str, Weights],
    huggingface_config: AutoConfig | None = None,
    pipeline_config=None,
    **kwargs,
) -> dict[str, WeightData]:
    target_dtype = _target_dtype(pipeline_config) if pipeline_config else None

    new_state_dict: dict[str, WeightData] = {}
    for weight_name, value in state_dict.items():
        max_name: str = weight_name
        for before, after in SMOLLM3_SAFETENSOR_MAPPING.items():
            max_name = max_name.replace(before, after)
        wd: WeightData = value.data()
        if target_dtype is not None and wd.dtype != target_dtype:
            wd = wd.astype(target_dtype)
        new_state_dict[max_name] = wd
    return new_state_dict


def convert_gguf_state_dict(
    state_dict: dict[str, Weights],
    huggingface_config: AutoConfig | None = None,
    pipeline_config=None,
    **unused_kwargs,
) -> dict[str, WeightData]:
    target_dtype = _target_dtype(pipeline_config) if pipeline_config else None

    new_state_dict: dict[str, WeightData] = {}
    for gguf_name, value in state_dict.items():
        max_name = gguf_name
        for before, after in SMOLLM3_GGUF_MAPPING.items():
            max_name = max_name.replace(before, after)
        wd: WeightData = value.data()
        if target_dtype is not None and wd.dtype != target_dtype:
            wd = wd.astype(target_dtype)
        new_state_dict[max_name] = wd

    new_state_dict.pop("rope_freqs.weight", None)
    return new_state_dict
```

### Explanation

This file handles the translation of external checkpoint weights into the internal naming and dtype expectations of the SmolLM3 MAX graph.

Process details:

1. `SMOLLM3_SAFETENSOR_MAPPING` maps HuggingFace safetensor prefixes to the MAX graph prefix.
   - For example, `model.` becomes `language_model.`.
   - It also rewrites the optional `lm_head` name when tied embeddings are used.

2. `SMOLLM3_GGUF_MAPPING` converts GGUF naming conventions into equivalent MAX parameter names.
   - It rewrites block prefixes and projection layer names to match the graph structure.

3. `_target_dtype()` reads `pipeline_config.model.quantization_encoding` and converts it into a MAX `DType`.
   - This lets the adapter cast checkpoint tensors to the actual runtime compute dtype.

4. `convert_safetensor_state_dict()` iterates over every weight.
   - It renames the key using the safetensor mapping.
   - It reads the underlying `WeightData` tensor.
   - If the checkpoint dtype differs from the requested runtime dtype, it casts the tensor.
   - Finally, it returns a new dictionary keyed by MAX graph names.

5. `convert_gguf_state_dict()` does the same for GGUF weights.
   - It also drops `rope_freqs.weight` because runtime RoPE values are computed from the model config instead of loaded statically.

This adapter layer is critical because MAX requires exact weight names and exact dtypes; mismatches would prevent graph compilation or execution.

---

## `layers/__init__.py`

### Source

```python
"""SmolLM3 custom layers."""

from .attention import SmolLM3Attention
from .transformer_block import SmolLM3TransformerBlock

__all__ = [
    "SmolLM3Attention",
    "SmolLM3TransformerBlock",
]
```

### Explanation

This file is a thin package initializer for the `layers` submodule.
It exports the two custom layer classes so importing them from `SmolLM3.layers` is straightforward.

---

## `layers/attention.py`

### Source

```python
# ===----------------------------------------------------------------------=== #
# SmolLM3 attention layer with conditional RoPE (NoPE support).
#
# MAX's pipeline uses two fused kernels:
#   rope_split_store_ragged  — splits flat QKV, applies RoPE to Q/K, stores K/V
#   flash_attention_ragged   — computes attention from roped Q + paged KV cache
#
# freqs_cis shape (from RotaryEmbedding.freqs_cis cached property):
#   [max_seq_len * 2, head_dim]   (flat: cos/sin pairs interleaved per half-dim)
#
# NoPE identity: cos=1.0, sin=0.0 throughout → rotation is a mathematical no-op.
# We build identity freqs_cis matching the exact flat layout:
#   half = head_dim // 2
#   flat = concat([ones([S, half]), zeros([S, half])], axis=-1)  — non-interleaved
#   OR interleaved: stack([ones, zeros], axis=-1).reshape([S, head_dim])
# ===----------------------------------------------------------------------=== #
"""SmolLM3 attention: fused RoPE+KV-store with NoPE identity for every 4th layer."""

from __future__ import annotations

import math

from max.driver import CPU
from max.dtype import DType
from max.experimental import functional as F
from max.experimental.nn import Module
from max.experimental.nn.common_layers.functional_kernels import (
    flash_attention_ragged,
    rope_split_store_ragged,
)
from max.experimental.nn.linear import Linear
from max.experimental.tensor import Tensor
from max.nn.attention import MHAMaskVariant
from max.nn.kv_cache import KVCacheParams, PagedCacheValues


class SmolLM3Attention(Module[..., Tensor]):
    """GQA attention with optional RoPE per layer.

    RoPE layers  (use_rope=True):  real freqs_cis passed to rope_split_store_ragged.
    NoPE layers  (use_rope=False): identity freqs_cis (cos=1, sin=0) so rotation
                                   is a no-op while KV store still happens.
    """

    def __init__(
        self,
        *,
        rope,
        num_attention_heads: int,
        num_key_value_heads: int,
        hidden_size: int,
        kv_params: KVCacheParams,
        layer_idx: int,
        use_rope: bool = True,
        scale: float | None = None,
        has_bias: bool = False,
        clip_qkv: float | None = None,
    ) -> None:
        super().__init__()
        self.rope = rope
        self.n_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.hidden_size = hidden_size
        self.kv_params = kv_params
        self.layer_idx = layer_idx
        self.use_rope = use_rope
        self.has_bias = has_bias
        self.clip_qkv = clip_qkv
        self.scale = (
            scale if scale is not None
            else math.sqrt(1.0 / self.kv_params.head_dim)
        )

        q_dim  = self.kv_params.head_dim * num_attention_heads
        kv_dim = self.kv_params.head_dim * num_key_value_heads
        self.q_weight_dim = q_dim

        self.q_proj = Linear(in_dim=hidden_size, out_dim=q_dim,  bias=has_bias)
        self.k_proj = Linear(in_dim=hidden_size, out_dim=kv_dim, bias=has_bias)
        self.v_proj = Linear(in_dim=hidden_size, out_dim=kv_dim, bias=has_bias)
        self.o_proj = Linear(in_dim=q_dim, out_dim=hidden_size,  bias=False)

    @property
    def wqkv(self) -> Tensor:
        wq: Tensor = self.q_proj.weight
        wk: Tensor = self.k_proj.weight
        wv: Tensor = self.v_proj.weight
        if self.clip_qkv:
            wq = F.clip(wq, -self.clip_qkv, self.clip_qkv)
            wk = F.clip(wk, -self.clip_qkv, self.clip_qkv)
            wv = F.clip(wv, -self.clip_qkv, self.clip_qkv)
        return F.concat([wq, wk, wv], axis=0)

    @property
    def wqkv_bias(self) -> Tensor | None:
        if not self.has_bias:
            return None
        assert self.q_proj.bias is not None
        assert self.k_proj.bias is not None
        assert self.v_proj.bias is not None
        return F.concat(
            [self.q_proj.bias, self.k_proj.bias, self.v_proj.bias], axis=0
        )

    def _make_identity_freqs_cis(self, freqs_cis: Tensor) -> Tensor:
        shape = freqs_cis.shape
        seq_dim = shape[0]
        head_dim = shape[1]
        half = head_dim // 2

        cos_half = F.constant(1.0, dtype=freqs_cis.dtype, device=freqs_cis.device)
        sin_half = F.constant(0.0, dtype=freqs_cis.dtype, device=freqs_cis.device)

        cos_slab = F.broadcast_to(
            F.reshape(cos_half, [1, 1]), [seq_dim, half]
        )
        sin_slab = F.broadcast_to(
            F.reshape(sin_half, [1, 1]), [seq_dim, half]
        )

        if self.rope.interleaved:
            stacked = F.stack([cos_slab, sin_slab], axis=-1)
            return F.reshape(stacked, [seq_dim, head_dim])
        else:
            return F.concat([cos_slab, sin_slab], axis=-1)

    def forward(
        self,
        x: Tensor,
        kv_collection: PagedCacheValues,
        **kwargs,
    ) -> Tensor:
        total_seq_len = x.shape[0]
        layer_idx = F.constant(self.layer_idx, DType.uint32, device=CPU())

        qkv = x @ self.wqkv.T
        if self.wqkv_bias is not None:
            qkv = qkv + self.wqkv_bias

        freqs_cis = F.cast(self.rope.freqs_cis, qkv.dtype).to(qkv.device)
        if not self.use_rope:
            freqs_cis = self._make_identity_freqs_cis(freqs_cis)

        xq = rope_split_store_ragged(
            kv_params=self.kv_params,
            qkv=qkv,
            input_row_offsets=kwargs["input_row_offsets"],
            freqs_cis=freqs_cis,
            kv_collection=kv_collection,
            layer_idx=layer_idx,
            n_heads=self.n_heads,
            interleaved=self.rope.interleaved,
        )
        xq = xq.reshape((-1, self.n_heads, self.kv_params.head_dim))

        attn_out = flash_attention_ragged(
            self.kv_params,
            input=xq,
            kv_collection=kv_collection,
            layer_idx=layer_idx,
            input_row_offsets=kwargs["input_row_offsets"],
            mask_variant=MHAMaskVariant.CAUSAL_MASK,
            scale=self.scale,
        )
        attn_out = F.reshape(attn_out, shape=[total_seq_len, self.q_weight_dim])
        return self.o_proj(attn_out)
```

### Explanation

This file contains the core attention mechanics used by SmolLM3.

Process details:

1. `SmolLM3Attention.__init__()` sets up the layer with:
   - separate projections for Q, K, V, and output.
   - head counts for GQA (more query heads than key/value heads).
   - an optional `clip_qkv` range.
   - a scale factor for the attention scores.

2. `wqkv` concatenates q, k, and v projection weights into one fused tensor.
   - If `clip_qkv` is configured, the weight values are clipped before concatenation.

3. `wqkv_bias` concatenates the biases only when attention bias is enabled.

4. `_make_identity_freqs_cis()` constructs a dummy rotary frequency tensor for NoPE layers.
   - The dummy tensor has cos=1 and sin=0 in the same packed layout as the real tensor.
   - This means the rotation step becomes a no-op without changing the rest of the attention pipeline.

5. `forward()` executes the fused attention flow:
   - It computes a single fused QKV matmul using the concatenated weights.
   - It applies the optional bias.
   - It selects real RoPE frequencies for RoPE layers or identity frequencies for NoPE layers.
   - It calls `rope_split_store_ragged()` to apply the rotation, split Q/K/V, and store keys/values in the paged KV cache.
   - It reshapes the returned query tensor and runs `flash_attention_ragged()` with causal masking.
   - It projects the attention output back to hidden_size with `o_proj`.

This module is the place where SmolLM3’s per-layer RoPE/NoPE behavior is realized in the graph.

---

## `layers/transformer_block.py`

### Source

```python
# ===----------------------------------------------------------------------=== #
# SmolLM3 transformer block.
# Identical structure to LlamaTransformerBlock but references
# SmolLM3Attention so the NoPE/RoPE toggle is handled at the attention level.
# ===----------------------------------------------------------------------=== #
"""SmolLM3 transformer block."""

from __future__ import annotations

from max.experimental import functional as F
from max.experimental.nn import Module
from max.experimental.tensor import Tensor
from max.nn.kv_cache import PagedCacheValues

from .attention import SmolLM3Attention


class SmolLM3TransformerBlock(Module[..., Tensor]):
    """Pre-norm decoder block: Attention → residual → MLP → residual.

    The ``residual_multiplier`` field is kept for forward-compat with
    any future SmolLM3 variant that uses scaled residuals.
    """

    def __init__(
        self,
        attention: SmolLM3Attention,
        mlp: Module[[Tensor], Tensor],
        input_layernorm: Module[[Tensor], Tensor],
        post_attention_layernorm: Module[[Tensor], Tensor],
        residual_multiplier: float = 1.0,
    ) -> None:
        super().__init__()
        self.self_attn = attention
        self.mlp = mlp
        self.input_layernorm = input_layernorm
        self.post_attention_layernorm = post_attention_layernorm
        self.residual_multiplier = residual_multiplier

    def forward(
        self,
        layer_idx: Tensor,
        x: Tensor,
        kv_collection: PagedCacheValues,
        input_row_offsets: Tensor,
        **kwargs,
    ) -> Tensor:
        attn_out = self.self_attn(
            self.input_layernorm(x),
            kv_collection,
            input_row_offsets=input_row_offsets,
            **kwargs,
        )

        if self.residual_multiplier != 1.0:
            m = F.constant(self.residual_multiplier, x.dtype, device=x.device)
            attn_out = attn_out * m

        h = x + attn_out

        mlp_out = self.mlp(self.post_attention_layernorm(h))

        if self.residual_multiplier != 1.0:
            m = F.constant(self.residual_multiplier, h.dtype, device=h.device)
            mlp_out = mlp_out * m

        return h + mlp_out
```

### Explanation

This file defines the transformer block structure used inside the SmolLM3 stack.

Process details:

1. Input normalization is applied before attention (`input_layernorm`).
2. The normalized tensor is fed into the attention layer.
3. If `residual_multiplier` differs from 1.0, the attention output is scaled.
4. The attention output is added back to the original input to produce the first residual connection.
5. The result is normalized again and passed through the MLP.
6. The MLP output is optionally scaled and added back to the residual state.
7. The final tensor is returned as the block output.

This pattern preserves the standard decoder transformer architecture while letting SmolLM3Attention handle the RoPE / NoPE decision.

---

## Notes on the `layers` folder

- `layers/__init__.py` exports the custom classes.
- `layers/attention.py` contains the core fused attention implementation.
- `layers/transformer_block.py` composes attention and the feed-forward network.

---

## Summary

This `SmolLM3` folder is a complete MAX custom architecture package that:

1. exports the architecture for MAX discovery,
2. defines the runtime config and KV cache parameters,
3. compiles the ModuleV3 model with weights,
4. builds a SmolLM3 transformer with per-layer RoPE/NoPE support,
5. adapts checkpoint weights from HuggingFace or GGUF formats.

If you want, I can also add a short usage example showing how to load this package in MAX.
- `weight_adapters` maps supported format adapters.
- `task` declares text generation.
- `config` ties the architecture to the config class.

---

## `model_config.py`

### Source

```python
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

    no_rope_layers: list[int] = field(default_factory=list)
    no_rope_layer_interval: int = 4

    return_logits: ReturnLogits = ReturnLogits.LAST_TOKEN
    return_hidden_states: ReturnHiddenStates = ReturnHiddenStates.NONE
    norm_method: Literal["rms_norm"] = "rms_norm"
    rms_norm_eps: float | None = None
    attention_bias: bool = False
    tie_word_embeddings: bool = True
    stacked_mlp: bool = False
    stacked_qkv: bool = False
    attention_multiplier: float = 0.0
    embedding_multiplier: float = 1.0
    residual_multiplier: float = 1.0
    clip_qkv: float | None = None
    logits_scaling: float = 1.0

    def get_kv_params(self) -> KVCacheParams:
        return self.kv_params

    def get_max_seq_len(self) -> int:
        return self.max_seq_len

    @staticmethod
    def get_head_dim(huggingface_config: AutoConfig) -> int:
        if hasattr(huggingface_config, "head_dim"):
            return huggingface_config.head_dim
        return (
            huggingface_config.hidden_size
            // huggingface_config.num_attention_heads
        )

    @staticmethod
    def get_head_dim_from_config(config: "SmolLM3Config") -> int:
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

    @override
    @classmethod
    def initialize(
        cls,
        pipeline_config: PipelineConfig,
        model_config: MAXModelConfig | None = None,
    ) -> "SmolLM3Config":
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

        num_layers = huggingface_config.num_hidden_layers
        no_rope_layer_interval: int = getattr(
            huggingface_config, "no_rope_layer_interval", 4
        )
        no_rope_layers: list[int] = list(
            getattr(huggingface_config, "no_rope_layers", [])
        )
        if not no_rope_layers:
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
        )
```

### Explanation

- `SmolLM3Config` stores the model hyperparameters and MAX runtime settings.
- `no_rope_layers` and `no_rope_layer_interval` implement per-layer RoPE/NoPE control.
- The static methods compute dimensions and attention scaling from HuggingFace config.
- `construct_kv_params()` builds MAX KV cache parameters from pipeline settings.
- `initialize()` validates config, reads requested dtype, constructs devices, and builds the final config.

---

## `model.py`

### Source

```python
# ===----------------------------------------------------------------------=== #
# SmolLM3 pipeline model (PipelineModelWithKVCache wrapper).
# Handles compilation, input preparation, and output unpacking.
# ===----------------------------------------------------------------------=== #
"""SmolLM3 pipeline model using the ModuleV3 API."""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Literal, cast

import numpy as np
from max.driver import Buffer, Device
from max.dtype import DType
from max.engine import InferenceSession
from max.experimental import functional as F
from max.experimental.tensor import default_dtype
from max.graph import DeviceRef, TensorType
from max.graph.weights import Weights, WeightsAdapter
from max.nn.kv_cache import KVCacheInputs, KVCacheParams
from max.nn.transformer import ReturnHiddenStates, ReturnLogits
from max.pipelines.core import TextContext
from max.pipelines.lib import (
    CompilationTimer,
    KVCacheConfig,
    ModelInputs,
    ModelOutputs,
    PipelineConfig,
    PipelineModelWithKVCache,
)
from max.pipelines.lib.log_probabilities import LogProbabilitiesMixin
from transformers import AutoConfig

from .smollm3 import SmolLM3
from .model_config import SmolLM3Config

logger = logging.getLogger("max.pipelines")


@dataclass
class SmolLM3Inputs(ModelInputs):
    """Typed input bundle for the SmolLM3 pipeline model."""

    tokens: Buffer
    input_row_offsets: Buffer
    return_n_logits: Buffer

    @property
    def buffers(self) -> tuple[Buffer, ...]:
        if isinstance(self.input_row_offsets, np.ndarray):
            input_row_offsets = Buffer.from_numpy(self.input_row_offsets).to(
                self.tokens.device
            )
        else:
            input_row_offsets = self.input_row_offsets
        return (
            self.tokens,
            self.return_n_logits,
            input_row_offsets,
            *(
                self.kv_cache_inputs.flatten()
                if self.kv_cache_inputs is not None
                else ()
            ),
        )


class SmolLM3Model(LogProbabilitiesMixin, PipelineModelWithKVCache[TextContext]):
    """SmolLM3 pipeline model using the ModuleV3 API."""

    config_class: type[SmolLM3Config] = SmolLM3Config
    norm_method: Literal["rms_norm"] = "rms_norm"
    attention_bias: bool = False

    def __init__(
        self,
        pipeline_config: PipelineConfig,
        session: InferenceSession,
        devices: list[Device],
        kv_cache_config: KVCacheConfig,
        weights: Weights,
        adapter: WeightsAdapter | None = None,
        return_logits: ReturnLogits = ReturnLogits.LAST_TOKEN,
        return_hidden_states: ReturnHiddenStates = ReturnHiddenStates.NONE,
    ) -> None:
        super().__init__(
            pipeline_config,
            session,
            devices,
            kv_cache_config,
            weights,
            adapter,
            return_logits,
            return_hidden_states,
        )
        self.model = self.load_model()

    @staticmethod
    def calculate_max_seq_len(
        pipeline_config: PipelineConfig,
        huggingface_config: AutoConfig,
    ) -> int:
        return SmolLM3Config.calculate_max_seq_len(
            pipeline_config, huggingface_config
        )

    @classmethod
    def get_kv_params(
        cls,
        huggingface_config: AutoConfig,
        pipeline_config: PipelineConfig,
        devices: list[DeviceRef],
        kv_cache_config: KVCacheConfig,
        cache_dtype: DType,
    ) -> KVCacheParams:
        return SmolLM3Config.construct_kv_params(
            huggingface_config,
            pipeline_config,
            devices,
            kv_cache_config,
            cache_dtype,
        )

    def load_model(self) -> Callable[..., Any]:
        assert self.pipeline_config.runtime.max_batch_size, (
            "Expected max_batch_size to be set"
        )
        self._input_row_offsets_prealloc = Buffer.from_numpy(
            np.arange(
                self.pipeline_config.runtime.max_batch_size + 1, dtype=np.uint32
            )
        ).to(self.devices[0])

        with CompilationTimer("smollm3") as timer:
            device0 = self.devices[0]
            device_ref = DeviceRef(device0.label, device0.id)

            tokens_type = TensorType(
                DType.int64, shape=["total_seq_len"], device=device_ref
            )
            input_row_offsets_type = TensorType(
                DType.uint32,
                shape=["input_row_offsets_len"],
                device=device0,
            )
            return_n_logits_type = TensorType(
                DType.int64,
                shape=["return_n_logits"],
                device=DeviceRef.CPU(),
            )

            huggingface_config = self.huggingface_config
            if self.adapter:
                state_dict = self.adapter(
                    dict(self.weights.items()),
                    huggingface_config=huggingface_config,
                    pipeline_config=self.pipeline_config,
                )
            else:
                state_dict = {
                    key: value.data() for key, value in self.weights.items()
                }

            model_config = self.config_class.initialize(self.pipeline_config)
            model_config.finalize(
                huggingface_config=huggingface_config,
                state_dict=state_dict,
                norm_method=self.norm_method,
                attention_bias=self.attention_bias,
                return_logits=self.return_logits,
                return_hidden_states=self.return_hidden_states,
            )

            with F.lazy(), default_dtype(model_config.dtype):
                nn_model = SmolLM3(model_config, self.kv_params)
                nn_model.to(self.devices[0])

            kv_inputs = self.kv_params.get_symbolic_inputs()
            flattened_kv_types = kv_inputs.flatten()

            timer.mark_build_complete()
            compiled_model = nn_model.compile(
                tokens_type,
                return_n_logits_type,
                input_row_offsets_type,
                *flattened_kv_types,
                weights=state_dict,
            )

        return compiled_model

    def execute(self, model_inputs: ModelInputs) -> ModelOutputs:
        model_inputs = cast(SmolLM3Inputs, model_inputs)
        model_outputs = self.model(*model_inputs.buffers)

        has_offsets = self.return_logits in (
            ReturnLogits.VARIABLE,
            ReturnLogits.ALL,
        )
        has_hidden_states = self.return_hidden_states != ReturnHiddenStates.NONE

        if has_offsets and has_hidden_states:
            assert len(model_outputs) == 4
            return ModelOutputs(
                logits=cast(Buffer, model_outputs[1].driver_tensor),
                next_token_logits=cast(Buffer, model_outputs[0].driver_tensor),
                logit_offsets=cast(Buffer, model_outputs[2].driver_tensor),
                hidden_states=cast(Buffer, model_outputs[3].driver_tensor),
            )
        elif has_offsets:
            assert len(model_outputs) == 3
            return ModelOutputs(
                logits=cast(Buffer, model_outputs[1].driver_tensor),
                next_token_logits=cast(Buffer, model_outputs[0].driver_tensor),
                logit_offsets=cast(Buffer, model_outputs[2].driver_tensor),
            )
        elif has_hidden_states:
            assert len(model_outputs) == 2
            return ModelOutputs(
                logits=cast(Buffer, model_outputs[0].driver_tensor),
                next_token_logits=cast(Buffer, model_outputs[0].driver_tensor),
                hidden_states=cast(Buffer, model_outputs[1].driver_tensor),
            )
        else:
            assert len(model_outputs) == 1
            return ModelOutputs(
                logits=cast(Buffer, model_outputs[0].driver_tensor),
                next_token_logits=cast(Buffer, model_outputs[0].driver_tensor),
            )

    def prepare_initial_token_inputs(
        self,
        replica_batches: Sequence[Sequence[TextContext]],
        kv_cache_inputs: KVCacheInputs[Buffer, Buffer] | None = None,
        return_n_logits: int = 1,
    ) -> ModelInputs:
        if len(replica_batches) > 1:
            raise ValueError("SmolLM3Model does not support DP>1")

        context_batch = replica_batches[0]
        assert kv_cache_inputs is not None

        input_row_offsets = np.cumsum(
            [0] + [ctx.tokens.active_length for ctx in context_batch],
            dtype=np.uint32,
        )
        tokens = np.concatenate([ctx.tokens.active for ctx in context_batch])

        return SmolLM3Inputs(
            tokens=Buffer.from_numpy(tokens).to(self.devices[0]),
            input_row_offsets=Buffer.from_numpy(input_row_offsets).to(
                self.devices[0]
            ),
            return_n_logits=Buffer.from_numpy(
                np.array([return_n_logits], dtype=np.int64)
            ),
            kv_cache_inputs=kv_cache_inputs,
        )

    def prepare_next_token_inputs(
        self,
        next_tokens: Buffer,
        prev_model_inputs: ModelInputs,
    ) -> ModelInputs:
        prev_model_inputs = cast(SmolLM3Inputs, prev_model_inputs)
        row_offsets_size = prev_model_inputs.input_row_offsets.shape[0]

        next_row_offsets = self._input_row_offsets_prealloc[
            :row_offsets_size
        ].to(self.devices[0])

        return SmolLM3Inputs(
            tokens=next_tokens,
            input_row_offsets=next_row_offsets,
            return_n_logits=prev_model_inputs.return_n_logits,
            kv_cache_inputs=prev_model_inputs.kv_cache_inputs,
        )
```

### Explanation

- `SmolLM3Inputs` packages all model input buffers and flattens KV cache inputs.
- `load_model()` creates tensor metadata, adapts weights, finalizes config, and compiles the model.
- `execute()` transforms raw model outputs into `ModelOutputs` for the pipeline.
- `prepare_initial_token_inputs()` builds concatenated token and offset buffers.
- `prepare_next_token_inputs()` reuses preallocated row offsets for next-token decoding.

---

## `smollm3.py`

### Source

```python
# ===----------------------------------------------------------------------=== #
# SmolLM3 ModuleV3 graph implementation.
#
# Key differences from plain Llama3:
#   1. Per-layer RoPE toggle (no_rope_layers mask from config)
#   2. Every 4th layer is a NoPE layer (uses SmolLM3Attention with use_rope=False)
#   3. tie_word_embeddings=True (lm_head weight = embed_tokens weight transposed)
# ===----------------------------------------------------------------------=== #
"""Implements the SmolLM3 model using the ModuleV3 API."""

from __future__ import annotations

import functools
from collections.abc import Callable

from max.dtype import DType
from max.experimental import functional as F
from max.experimental.nn import Module
from max.experimental.nn.common_layers.mlp import MLP
from max.experimental.nn.embedding import Embedding
from max.experimental.nn.linear import Linear
from max.experimental.nn.norm import RMSNorm
from max.experimental.nn.sequential import ModuleList
from max.experimental.tensor import Tensor
from max.graph import TensorValue, ops
from max.nn.kv_cache import KVCacheParamInterface, PagedCacheValues
from max.nn.transformer import ReturnHiddenStates, ReturnLogits

from max.pipelines.architectures.llama3_modulev3.layers.rotary_embedding import (
    Llama3RotaryEmbedding,
)

from .layers.attention import SmolLM3Attention
from .layers.transformer_block import SmolLM3TransformerBlock
from .model_config import SmolLM3Config


class SmolLM3TextModel(
    Module[[Tensor, PagedCacheValues, Tensor, Tensor], tuple[Tensor, ...]]
):
    """SmolLM3 decoder-only transformer.

    36 layers, GQA (16 Q heads / 4 KV heads), SwiGLU MLP, RMSNorm.
    Every 4th layer (indices 3, 7, 11, …) uses NoPE (no positional encoding).
    """

    def __init__(self, config: SmolLM3Config) -> None:
        super().__init__()
        self.devices = config.devices

        if config.rms_norm_eps is None:
            raise ValueError("rms_norm_eps cannot be None for SmolLM3.")

        create_norm: Callable[..., Module[[Tensor], Tensor]] = functools.partial(
            RMSNorm, config.hidden_size, eps=config.rms_norm_eps
        )

        rope = Llama3RotaryEmbedding(
            dim=config.hidden_size,
            n_heads=config.num_attention_heads,
            theta=config.rope_theta,
            max_seq_len=config.max_seq_len,
            device=config.devices[0].to_device(),
            head_dim=SmolLM3Config.get_head_dim_from_config(config),
            interleaved=config.interleaved_rope_weights,
            scaling_params=None,
        )

        self.embed_tokens = Embedding(
            config.vocab_size,
            dim=config.hidden_size,
        )
        self.norm = create_norm()

        self.tie_word_embeddings = config.tie_word_embeddings
        if config.tie_word_embeddings:
            self.lm_head = None
        else:
            self.lm_head = Linear(
                in_dim=config.hidden_size,
                out_dim=config.vocab_size,
                bias=False,
            )

        layers = []
        no_rope_layers = config.no_rope_layers
        for i in range(config.num_hidden_layers):
            use_rope: bool = bool(no_rope_layers[i]) if no_rope_layers else True

            attention = SmolLM3Attention(
                rope=rope,
                num_attention_heads=config.num_attention_heads,
                num_key_value_heads=config.num_key_value_heads,
                hidden_size=config.hidden_size,
                kv_params=config.kv_params,
                layer_idx=i,
                use_rope=use_rope,
                scale=config.attention_multiplier,
                has_bias=config.attention_bias,
                clip_qkv=config.clip_qkv,
            )

            mlp = MLP(
                hidden_dim=config.hidden_size,
                feed_forward_length=config.intermediate_size,
            )

            layers.append(
                SmolLM3TransformerBlock(
                    attention=attention,
                    mlp=mlp,
                    input_layernorm=create_norm(),
                    post_attention_layernorm=create_norm(),
                    residual_multiplier=config.residual_multiplier,
                )
            )

        self.layers = ModuleList(layers)
        self.dim = config.hidden_size
        self.n_heads = config.num_attention_heads
        self.kv_params = config.kv_params
        self.return_logits = config.return_logits
        self.return_hidden_states = config.return_hidden_states
        self.embedding_multiplier = config.embedding_multiplier
        self.logits_scaling = config.logits_scaling

    def _compute_logits(self, h: Tensor) -> Tensor:
        if self.tie_word_embeddings:
            return F.cast(h @ self.embed_tokens.weight.T, DType.float32)
        assert self.lm_head is not None
        return F.cast(self.lm_head(h), DType.float32)

    def forward(
        self,
        tokens: Tensor,
        kv_collection: PagedCacheValues,
        return_n_logits: Tensor,
        input_row_offsets: Tensor,
    ) -> tuple[Tensor, ...]:
        h = self.embed_tokens(tokens)

        if self.embedding_multiplier != 1.0:
            h = h * F.constant(
                self.embedding_multiplier, h.dtype, device=h.device
            )

        for idx, layer in enumerate(self.layers):
            layer_idx_tensor = F.constant(idx, DType.uint32, device=h.device)
            h = layer(
                layer_idx_tensor,
                h,
                kv_collection,
                input_row_offsets=input_row_offsets,
            )

        last_h = F.gather(h, input_row_offsets[1:] - 1, axis=0)
        last_logits = self._compute_logits(self.norm(last_h))
        logits = None
        offsets = None

        if self.return_logits == ReturnLogits.VARIABLE:
            return_n_logits_range = ops.range(
                return_n_logits[0],
                0,
                -1,
                out_dim="return_n_logits_range",
                device=h.device,
                dtype=DType.int64,
            )
            offsets = (
                F.unsqueeze(input_row_offsets[1:], -1) - return_n_logits_range
            )
            last_indices = F.reshape(offsets, shape=(-1,))
            last_tokens = F.gather(h, last_indices, axis=0)
            logits = self._compute_logits(self.norm(last_tokens))
            offsets = ops.range(
                0,
                TensorValue(last_indices.shape[0]) + return_n_logits[0],
                return_n_logits[0],
                out_dim="logit_offsets",
                device=h.device,
                dtype=DType.int64,
            )
        elif self.return_logits == ReturnLogits.ALL:
            logits = self._compute_logits(self.norm(h))
            offsets = input_row_offsets

        if self.logits_scaling != 1.0:
            last_logits = last_logits / self.logits_scaling
            if logits is not None:
                logits = logits / self.logits_scaling

        ret_val: tuple[Tensor, ...] = (last_logits,)
        if offsets is not None:
            assert logits is not None
            ret_val += (logits, offsets)

        if self.return_hidden_states == ReturnHiddenStates.LAST:
            ret_val += (last_h,)
        elif self.return_hidden_states == ReturnHiddenStates.ALL_NORMALIZED:
            ret_val += (self.norm(h),)

        return ret_val


class SmolLM3(Module[..., tuple[Tensor, ...]]):
    def __init__(self, config: SmolLM3Config, kv_params: KVCacheParamInterface) -> None:
        super().__init__()
        self.language_model = SmolLM3TextModel(config)
        self.config = config
        self.kv_params = kv_params

    def forward(
        self,
        tokens: Tensor,
        return_n_logits: Tensor,
        input_row_offsets: Tensor,
        *variadic_args: Tensor,
    ) -> tuple[Tensor, ...]:
        kv_inputs = iter(x._graph_value for x in variadic_args)
        kv_collections = (
            self.kv_params.get_symbolic_inputs().unflatten(kv_inputs).inputs
        )
        return self.language_model(
            tokens, kv_collections[0], return_n_logits, input_row_offsets
        )
```

### Explanation

- `SmolLM3TextModel` builds embeddings, layer normalization, and transformer layers.
- The RoPE object is shared while certain layers may use NoPE.
- `tie_word_embeddings` controls whether logits are computed via the embedding matrix.
- `forward()` stacks transformer layers and computes output logits.
- `SmolLM3` wraps the text model and unpacks variadic KV cache tensors.

---

## `weight_adapters.py`

### Source

```python
# ===----------------------------------------------------------------------=== #
# SmolLM3 weight adapters.
#
# SmolLM3 safetensors use the standard HuggingFace layout:
#   model.embed_tokens.weight
#   model.layers.N.self_attn.{q,k,v,o}_proj.weight
#   model.layers.N.mlp.{gate,up,down}_proj.weight
#   model.layers.N.{input,post_attention}_layernorm.weight
#   model.norm.weight
#   lm_head.weight  ← absent when tie_word_embeddings=True
#
# MAX validates that each WeightData dtype matches the graph parameter dtype
# exactly (no implicit promotion). When --quantization-encoding float32 is
# passed but the checkpoint is bfloat16, we must cast here.
# ===----------------------------------------------------------------------=== #
"""Weight name adapters + optional dtype cast for SmolLM3."""

from __future__ import annotations

from max.dtype import DType
from max.graph.weights import WeightData, Weights
from transformers import AutoConfig

SMOLLM3_SAFETENSOR_MAPPING: dict[str, str] = {
    "model.": "language_model.",
    "lm_head": "language_model.lm_head",
}

SMOLLM3_GGUF_MAPPING: dict[str, str] = {
    "token_embd": "language_model.embed_tokens",
    "blk": "language_model.layers",
    "ffn_up": "mlp.up_proj",
    "ffn_down": "mlp.down_proj",
    "ffn_gate": "mlp.gate_proj",
    "ffn_norm": "post_attention_layernorm",
    "attn_norm": "input_layernorm",
    "attn_q": "self_attn.q_proj",
    "attn_v": "self_attn.v_proj",
    "attn_k": "self_attn.k_proj",
    "attn_output": "self_attn.o_proj",
    "output.weight": "language_model.lm_head.weight",
    "output_norm": "language_model.norm",
}


def _target_dtype(pipeline_config) -> DType | None:
    try:
        from max.pipelines.lib.config.config_enums import supported_encoding_dtype
        enc = pipeline_config.model.quantization_encoding
        if enc is not None:
            return supported_encoding_dtype(enc)
    except Exception:
        pass
    return None


def convert_safetensor_state_dict(
    state_dict: dict[str, Weights],
    huggingface_config: AutoConfig | None = None,
    pipeline_config=None,
    **kwargs,
) -> dict[str, WeightData]:
    target_dtype = _target_dtype(pipeline_config) if pipeline_config else None

    new_state_dict: dict[str, WeightData] = {}
    for weight_name, value in state_dict.items():
        max_name: str = weight_name
        for before, after in SMOLLM3_SAFETENSOR_MAPPING.items():
            max_name = max_name.replace(before, after)
        wd: WeightData = value.data()
        if target_dtype is not None and wd.dtype != target_dtype:
            wd = wd.astype(target_dtype)
        new_state_dict[max_name] = wd
    return new_state_dict


def convert_gguf_state_dict(
    state_dict: dict[str, Weights],
    huggingface_config: AutoConfig | None = None,
    pipeline_config=None,
    **unused_kwargs,
) -> dict[str, WeightData]:
    target_dtype = _target_dtype(pipeline_config) if pipeline_config else None

    new_state_dict: dict[str, WeightData] = {}
    for gguf_name, value in state_dict.items():
        max_name = gguf_name
        for before, after in SMOLLM3_GGUF_MAPPING.items():
            max_name = max_name.replace(before, after)
        wd: WeightData = value.data()
        if target_dtype is not None and wd.dtype != target_dtype:
            wd = wd.astype(target_dtype)
        new_state_dict[max_name] = wd

    new_state_dict.pop("rope_freqs.weight", None)
    return new_state_dict
```

### Explanation

- `SMOLLM3_SAFETENSOR_MAPPING` renames standard HuggingFace safetensor keys.
- `SMOLLM3_GGUF_MAPPING` maps GGUF block names to MAX parameter names.
- `_target_dtype()` extracts the dtype requested by the pipeline.
- Each converter renames keys and casts weights when needed.
- `rope_freqs.weight` is removed for GGUF because the runtime uses runtime RoPE values.

---

## `layers/__init__.py`

### Source

```python
"""SmolLM3 custom layers."""

from .attention import SmolLM3Attention
from .transformer_block import SmolLM3TransformerBlock

__all__ = [
    "SmolLM3Attention",
    "SmolLM3TransformerBlock",
]
```

### Explanation

- This file exports the layer classes from the `layers` package.
- It provides a simple import surface for package consumers.

---

## `layers/attention.py`

### Source

```python
# ===----------------------------------------------------------------------=== #
# SmolLM3 attention layer with conditional RoPE (NoPE support).
#
# MAX's pipeline uses two fused kernels:
#   rope_split_store_ragged  — splits flat QKV, applies RoPE to Q/K, stores K/V
#   flash_attention_ragged   — computes attention from roped Q + paged KV cache
#
# freqs_cis shape (from RotaryEmbedding.freqs_cis cached property):
#   [max_seq_len * 2, head_dim]   (flat: cos/sin pairs interleaved per half-dim)
#
# NoPE identity: cos=1.0, sin=0.0 throughout → rotation is a mathematical no-op.
# We build identity freqs_cis matching the exact flat layout:
#   half = head_dim // 2
#   flat = concat([ones([S, half]), zeros([S, half])], axis=-1)  — non-interleaved
#   OR interleaved: stack([ones, zeros], axis=-1).reshape([S, head_dim])
# ===----------------------------------------------------------------------=== #
"""SmolLM3 attention: fused RoPE+KV-store with NoPE identity for every 4th layer."""

from __future__ import annotations

import math

from max.driver import CPU
from max.dtype import DType
from max.experimental import functional as F
from max.experimental.nn import Module
from max.experimental.nn.common_layers.functional_kernels import (
    flash_attention_ragged,
    rope_split_store_ragged,
)
from max.experimental.nn.linear import Linear
from max.experimental.tensor import Tensor
from max.nn.attention import MHAMaskVariant
from max.nn.kv_cache import KVCacheParams, PagedCacheValues


class SmolLM3Attention(Module[..., Tensor]):
    """GQA attention with optional RoPE per layer.

    RoPE layers  (use_rope=True):  real freqs_cis passed to rope_split_store_ragged.
    NoPE layers  (use_rope=False): identity freqs_cis (cos=1, sin=0) so rotation
                                   is a no-op while KV store still happens.
    """

    def __init__(
        self,
        *,
        rope,
        num_attention_heads: int,
        num_key_value_heads: int,
        hidden_size: int,
        kv_params: KVCacheParams,
        layer_idx: int,
        use_rope: bool = True,
        scale: float | None = None,
        has_bias: bool = False,
        clip_qkv: float | None = None,
    ) -> None:
        super().__init__()
        self.rope = rope
        self.n_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.hidden_size = hidden_size
        self.kv_params = kv_params
        self.layer_idx = layer_idx
        self.use_rope = use_rope
        self.has_bias = has_bias
        self.clip_qkv = clip_qkv
        self.scale = (
            scale if scale is not None
            else math.sqrt(1.0 / self.kv_params.head_dim)
        )

        q_dim  = self.kv_params.head_dim * num_attention_heads
        kv_dim = self.kv_params.head_dim * num_key_value_heads
        self.q_weight_dim = q_dim

        self.q_proj = Linear(in_dim=hidden_size, out_dim=q_dim,  bias=has_bias)
        self.k_proj = Linear(in_dim=hidden_size, out_dim=kv_dim, bias=has_bias)
        self.v_proj = Linear(in_dim=hidden_size, out_dim=kv_dim, bias=has_bias)
        self.o_proj = Linear(in_dim=q_dim, out_dim=hidden_size,  bias=False)

    @property
    def wqkv(self) -> Tensor:
        wq: Tensor = self.q_proj.weight
        wk: Tensor = self.k_proj.weight
        wv: Tensor = self.v_proj.weight
        if self.clip_qkv:
            wq = F.clip(wq, -self.clip_qkv, self.clip_qkv)
            wk = F.clip(wk, -self.clip_qkv, self.clip_qkv)
            wv = F.clip(wv, -self.clip_qkv, self.clip_qkv)
        return F.concat([wq, wk, wv], axis=0)

    @property
    def wqkv_bias(self) -> Tensor | None:
        if not self.has_bias:
            return None
        assert self.q_proj.bias is not None
        assert self.k_proj.bias is not None
        assert self.v_proj.bias is not None
        return F.concat(
            [self.q_proj.bias, self.k_proj.bias, self.v_proj.bias], axis=0
        )

    def _make_identity_freqs_cis(self, freqs_cis: Tensor) -> Tensor:
        shape = freqs_cis.shape
        seq_dim = shape[0]
        head_dim = shape[1]
        half = head_dim // 2

        cos_half = F.constant(1.0, dtype=freqs_cis.dtype, device=freqs_cis.device)
        sin_half = F.constant(0.0, dtype=freqs_cis.dtype, device=freqs_cis.device)

        cos_slab = F.broadcast_to(
            F.reshape(cos_half, [1, 1]), [seq_dim, half]
        )
        sin_slab = F.broadcast_to(
            F.reshape(sin_half, [1, 1]), [seq_dim, half]
        )

        if self.rope.interleaved:
            stacked = F.stack([cos_slab, sin_slab], axis=-1)
            return F.reshape(stacked, [seq_dim, head_dim])
        else:
            return F.concat([cos_slab, sin_slab], axis=-1)

    def forward(
        self,
        x: Tensor,
        kv_collection: PagedCacheValues,
        **kwargs,
    ) -> Tensor:
        total_seq_len = x.shape[0]
        layer_idx = F.constant(self.layer_idx, DType.uint32, device=CPU())

        qkv = x @ self.wqkv.T
        if self.wqkv_bias is not None:
            qkv = qkv + self.wqkv_bias

        freqs_cis = F.cast(self.rope.freqs_cis, qkv.dtype).to(qkv.device)
        if not self.use_rope:
            freqs_cis = self._make_identity_freqs_cis(freqs_cis)

        xq = rope_split_store_ragged(
            kv_params=self.kv_params,
            qkv=qkv,
            input_row_offsets=kwargs["input_row_offsets"],
            freqs_cis=freqs_cis,
            kv_collection=kv_collection,
            layer_idx=layer_idx,
            n_heads=self.n_heads,
            interleaved=self.rope.interleaved,
        )
        xq = xq.reshape((-1, self.n_heads, self.kv_params.head_dim))

        attn_out = flash_attention_ragged(
            self.kv_params,
            input=xq,
            kv_collection=kv_collection,
            layer_idx=layer_idx,
            input_row_offsets=kwargs["input_row_offsets"],
            mask_variant=MHAMaskVariant.CAUSAL_MASK,
            scale=self.scale,
        )
        attn_out = F.reshape(attn_out, shape=[total_seq_len, self.q_weight_dim])
        return self.o_proj(attn_out)
```

### Explanation

- `SmolLM3Attention` computes fused QKV projection and attention.
- `use_rope` chooses between true RoPE and identity RoPE for NoPE layers.
- `_make_identity_freqs_cis()` constructs the identity frequency tensor when NoPE is active.
- `rope_split_store_ragged()` applies rotation and stores KV cache.
- `flash_attention_ragged()` performs causal attention on the stored KV cache.

---

## `layers/transformer_block.py`

### Source

```python
# ===----------------------------------------------------------------------=== #
# SmolLM3 transformer block.
# Identical structure to LlamaTransformerBlock but references
# SmolLM3Attention so the NoPE/RoPE toggle is handled at the attention level.
# ===----------------------------------------------------------------------=== #
"""SmolLM3 transformer block."""

from __future__ import annotations

from max.experimental import functional as F
from max.experimental.nn import Module
from max.experimental.tensor import Tensor
from max.nn.kv_cache import PagedCacheValues

from .attention import SmolLM3Attention


class SmolLM3TransformerBlock(Module[..., Tensor]):
    """Pre-norm decoder block: Attention → residual → MLP → residual.

    The ``residual_multiplier`` field is kept for forward-compat with
    any future SmolLM3 variant that uses scaled residuals.
    """

    def __init__(
        self,
        attention: SmolLM3Attention,
        mlp: Module[[Tensor], Tensor],
        input_layernorm: Module[[Tensor], Tensor],
        post_attention_layernorm: Module[[Tensor], Tensor],
        residual_multiplier: float = 1.0,
    ) -> None:
        super().__init__()
        self.self_attn = attention
        self.mlp = mlp
        self.input_layernorm = input_layernorm
        self.post_attention_layernorm = post_attention_layernorm
        self.residual_multiplier = residual_multiplier

    def forward(
        self,
        layer_idx: Tensor,
        x: Tensor,
        kv_collection: PagedCacheValues,
        input_row_offsets: Tensor,
        **kwargs,
    ) -> Tensor:
        attn_out = self.self_attn(
            self.input_layernorm(x),
            kv_collection,
            input_row_offsets=input_row_offsets,
            **kwargs,
        )

        if self.residual_multiplier != 1.0:
            m = F.constant(self.residual_multiplier, x.dtype, device=x.device)
            attn_out = attn_out * m

        h = x + attn_out

        mlp_out = self.mlp(self.post_attention_layernorm(h))

        if self.residual_multiplier != 1.0:
            m = F.constant(self.residual_multiplier, h.dtype, device=h.device)
            mlp_out = mlp_out * m

        return h + mlp_out
```

### Explanation

- The transformer block performs pre-normalization before attention.
- It adds the attention output back to the input with a residual connection.
- The MLP is applied to the normalized residual state.
- A second residual connection adds the MLP output to the block input.
- `residual_multiplier` can scale residual contributions.

---

## Notes on the `layers` folder

- `layers/__init__.py` exports the custom classes.
- `layers/attention.py` contains the core fused attention implementation.
- `layers/transformer_block.py` composes attention and the feed-forward network.

---

## Summary

This `SmolLM3` folder is a complete MAX custom architecture package that:

1. exports the architecture for MAX discovery,
2. defines the runtime config and KV cache parameters,
3. compiles the ModuleV3 model with weights,
4. builds a SmolLM3 transformer with per-layer RoPE/NoPE support,
5. adapts checkpoint weights from HuggingFace or GGUF formats.

If you want, I can also add a short usage example showing how to load this package in MAX.
