from __future__ import annotations
from max.graph import Graph
from max.graph.weights import Weights, WeightsAdapter
from max.nn.transformer import ReturnHiddenStates
from max.pipelines.architectures.llama3.model import Llama3Model
from max.pipelines.lib.utils import parse_state_dict_from_weights
from .model_config import InternLM2Config
from .internlm2 import InternLM2


class InternLM2Model(Llama3Model):

    def _build_graph(
        self,
        weights: Weights,
        adapter: WeightsAdapter | None = None,
    ) -> Graph:
        state_dict = parse_state_dict_from_weights(
            self.pipeline_config, weights, adapter
        )

        print("[InternLM2] === ADAPTED KEYS ===")
        for k in sorted(state_dict.keys())[:20]:
            print(f"  {k}")

        model_config = InternLM2Config.initialize(self.pipeline_config)
        model_config.finalize(
            huggingface_config=self.huggingface_config,
            state_dict=state_dict,
            return_logits=self.return_logits,
            return_hidden_states=self.return_hidden_states,
        )

        single_model = InternLM2(model_config)

        print("[InternLM2] === MODEL EXPECTED KEYS ===")
        for k in sorted(single_model.state_dict().keys())[:20]:
            print(f"  {k}")

        single_model.load_state_dict(
            state_dict,
            override_quantization_encoding=True,
            weight_alignment=1,
            strict=False,
        )
        self.state_dict = single_model.state_dict()

        with Graph(
            "internlm2",
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
