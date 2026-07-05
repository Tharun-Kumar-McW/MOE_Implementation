"""MiMo-7B pipeline model.

MiMoForCausalLM (modeling_mimo.py) inherits Qwen2ForCausalLM directly:
  - self.model   = MiMoModel (Qwen2Model + mtp_layers ModuleList)
  - self.lm_head = nn.Linear (standard)
  - self.mtp_layers are appended AFTER the main model, completely separate

The main 36 transformer layers are 100% Qwen2/LLaMA weight layout.
We use MiMoTransformer (not Llama3 graph) because attention_bias=True
means QKV projections carry learned bias terms that Llama3's graph
does not allocate — loading them with strict=True would error, and
strict=False would silently produce wrong outputs.

MTP layer weights (model.layers.36.*, model.enorm.*, etc.) are
dropped by the weight adapter before reaching here.
"""

from __future__ import annotations

from max.graph import Graph
from max.graph.weights import Weights, WeightsAdapter
from max.nn.transformer import ReturnHiddenStates, ReturnLogits
from max.pipelines.architectures.llama3.model import Llama3Model
from max.pipelines.lib.utils import parse_state_dict_from_weights

from .model_config import MiMoConfig
from .mimo_transformer import MiMoTransformer


class MiMoModel(Llama3Model):
    """MiMo-7B pipeline model — Qwen2 graph with attention bias, MTP skipped."""

    def _build_graph(
        self,
        weights: Weights,
        adapter: WeightsAdapter | None = None,
    ) -> Graph:
        state_dict = parse_state_dict_from_weights(
            self.pipeline_config, weights, adapter
        )

        model_config = MiMoConfig.initialize(self.pipeline_config)
        model_config.finalize(
            huggingface_config=self.huggingface_config,
            state_dict=state_dict,
            return_logits=self.return_logits,
            return_hidden_states=self.return_hidden_states,
        )

        # Use MiMoTransformer which has has_bias=True on attention
        single_model = MiMoTransformer(model_config)

        single_model.load_state_dict(
            state_dict,
            override_quantization_encoding=True,
            weight_alignment=1,
            strict=False,   # MTP weights already stripped by adapter; strict=False
                            # handles any remaining mismatches gracefully
        )
        self.state_dict = single_model.state_dict()

        with Graph(
            "mimo",
            input_types=single_model.input_types(
                self.kv_params, lora_manager=None
            ),
        ) as graph:
            (
                tokens,
                input_row_offsets,
                return_n_logits,
                *kv_cache_inputs,
            ) = graph.inputs
            kv_collections = self._unflatten_kv_inputs(kv_cache_inputs)
            outputs = single_model(
                tokens.tensor,
                kv_collections[0],
                return_n_logits.tensor,
                input_row_offsets.tensor,
            )
            graph.output(*outputs)
            return graph