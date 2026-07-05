# ===----------------------------------------------------------------------=== #
# Copyright (c) 2026, Modular Inc. All rights reserved.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions:
# https://llvm.org/LICENSE.txt
# ===----------------------------------------------------------------------=== #

"""SmolLM3 PipelineModel — follows exact DeepseekV2 pattern."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from max.driver import Buffer, Device
from max.dtype import DType
from max.engine import InferenceSession, Model
from max.graph import BufferType, DeviceRef, Graph, TensorType, Value
from max.graph.weights import Weights, WeightsAdapter
from max.nn.kv_cache import KVCacheInputs, KVCacheParams, PagedCacheValues
from max.nn.kv_cache.cache_params import KVCacheParamInterface
from max.nn.transformer import ReturnLogits, ReturnHiddenStates
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
from max.nn.comm import Signals
from transformers import AutoConfig

from .smollm3 import SmolLM3Model
from .model_config import SmolLM3Config

logger = logging.getLogger("max.pipelines")


@dataclass
class SmolLM3Inputs(ModelInputs):
    tokens: Buffer
    input_row_offsets: Buffer
    signal_buffers: list[Buffer]
    return_n_logits: Buffer


class SmolLM3PipelineModel(
    LogProbabilitiesMixin,
    PipelineModelWithKVCache[TextContext],
):
    """MAX PipelineModel for SmolLM3."""

    model: Model

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
        self.model = self.load_model(session)

    @classmethod
    def get_kv_params(
        cls,
        huggingface_config: AutoConfig,
        pipeline_config: PipelineConfig,
        devices: list[DeviceRef],
        kv_cache_config: KVCacheConfig,
        cache_dtype: DType,
    ) -> KVCacheParamInterface:
        return SmolLM3Config.construct_kv_params(
            huggingface_config=huggingface_config,
            pipeline_config=pipeline_config,
            devices=devices,
            kv_cache_config=kv_cache_config,
            cache_dtype=cache_dtype,
        )

    @classmethod
    def calculate_max_seq_len(
        cls, pipeline_config: PipelineConfig, huggingface_config: AutoConfig
    ) -> int:
        max_seq_len = pipeline_config.model.max_length
        if max_seq_len:
            return max_seq_len
        return huggingface_config.max_position_embeddings

    def graph_inputs(self) -> tuple[TensorType | BufferType, ...]:
        device_ref = DeviceRef.from_device(self.devices[0])

        tokens_type = TensorType(
            DType.int64, shape=["total_seq_len"], device=device_ref
        )
        input_row_offsets_type = TensorType(
            DType.uint32, shape=["input_row_offsets_len"], device=device_ref
        )
        return_n_logits_type = TensorType(
            DType.int64, shape=["return_n_logits"], device=device_ref
        )

        if len(self.devices) > 1:
            signals = Signals(
                devices=(DeviceRef(d.label, d.id) for d in self.devices)
            )
            return (
                tokens_type,
                input_row_offsets_type,
                return_n_logits_type,
                *signals.input_types(),
                *self.kv_params.get_symbolic_inputs().flatten(),
            )
        else:
            return (
                tokens_type,
                input_row_offsets_type,
                return_n_logits_type,
                *self.kv_params.get_symbolic_inputs().flatten(),
            )

    def _unflatten_kv_inputs(
        self,
        kv_inputs_flat: Sequence[Value[Any]],
        kv_params: KVCacheParamInterface | None = None,
    ) -> list[PagedCacheValues]:
        kv_params = kv_params or self.get_kv_params(
            huggingface_config=self.huggingface_config,
            pipeline_config=self.pipeline_config,
            devices=[DeviceRef.from_device(d) for d in self.devices],
            kv_cache_config=self.kv_cache_config,
            cache_dtype=self.pipeline_config.model.kv_cache.cache_dtype,
        )
        return list(
            kv_params.get_symbolic_inputs()
            .unflatten(iter(kv_inputs_flat))
            .inputs
        )

    def _build_graph(self) -> Graph:
        max_batch_size = self.pipeline_config.runtime.max_batch_size
        assert max_batch_size, "Expected max_batch_size to be set"

        self._input_row_offsets_prealloc = Buffer.from_numpy(
            np.arange(max_batch_size + 1, dtype=np.uint32)
        ).to(self.devices[0])

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

        model_config = SmolLM3Config.initialize(self.pipeline_config)
        model_config.finalize(
            huggingface_config=huggingface_config,
            state_dict=state_dict,
            return_logits=self.return_logits,
        )

        graph_inputs = self.graph_inputs()

        # Single device case (CPU or single GPU)
        nn_model = SmolLM3Model(model_config)
        nn_model.load_state_dict(state_dict, weight_alignment=1)
        self.state_dict = nn_model.state_dict()

        with Graph("smollm3", input_types=[*graph_inputs]) as graph:
            tokens, input_row_offsets, return_n_logits, *kv_cache_inputs = (
                graph.inputs
            )
            kv_collections = self._unflatten_kv_inputs(kv_cache_inputs)
            outputs = nn_model(
                tokens.tensor,
                kv_collections[0],
                return_n_logits.tensor,
                input_row_offsets.tensor,
            )
            graph.output(*outputs)
            return graph

    def load_model(self, session: InferenceSession) -> Model:
        with CompilationTimer("model") as timer:
            graph = self._build_graph()
            timer.mark_build_complete()
            model = session.load(graph, weights_registry=self.state_dict)
        return model

    def execute(self, model_inputs: ModelInputs) -> ModelOutputs:
        assert isinstance(model_inputs, SmolLM3Inputs)
        assert model_inputs.kv_cache_inputs is not None

        model_outputs = self.model.execute(
            model_inputs.tokens,
            model_inputs.input_row_offsets,
            model_inputs.return_n_logits,
            *model_inputs.signal_buffers,
            *model_inputs.kv_cache_inputs.flatten(),
        )

        if len(model_outputs) == 3:
            assert isinstance(model_outputs[0], Buffer)
            assert isinstance(model_outputs[1], Buffer)
            assert isinstance(model_outputs[2], Buffer)
            return ModelOutputs(
                next_token_logits=model_outputs[0],
                logits=model_outputs[1],
                logit_offsets=model_outputs[2],
            )
        else:
            assert isinstance(model_outputs[0], Buffer)
            return ModelOutputs(
                next_token_logits=model_outputs[0],
                logits=model_outputs[0],
            )

    def prepare_initial_token_inputs(
        self,
        replica_batches: Sequence[Sequence[TextContext]],
        kv_cache_inputs: KVCacheInputs | None = None,
        return_n_logits: int = 1,
    ) -> SmolLM3Inputs:
        if len(replica_batches) > 1:
            raise ValueError("Model does not support DP>1")

        context_batch = replica_batches[0]
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
            signal_buffers=self.signal_buffers,
            kv_cache_inputs=kv_cache_inputs,
            return_n_logits=Buffer.from_numpy(
                np.array([return_n_logits], dtype=np.int64)
            ).to(self.devices[0]),
        )

    def prepare_next_token_inputs(
        self,
        next_tokens: Buffer,
        prev_model_inputs: ModelInputs,
    ) -> SmolLM3Inputs:
        assert isinstance(prev_model_inputs, SmolLM3Inputs)
        row_offsets_size = prev_model_inputs.input_row_offsets.shape[0]
        next_row_offsets = self._input_row_offsets_prealloc[:row_offsets_size]
        return SmolLM3Inputs(
            tokens=next_tokens,
            input_row_offsets=next_row_offsets,
            signal_buffers=self.signal_buffers,
            kv_cache_inputs=prev_model_inputs.kv_cache_inputs,
            return_n_logits=prev_model_inputs.return_n_logits,
        )