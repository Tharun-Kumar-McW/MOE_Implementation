"""GritLM transformer block — standard pre-norm decoder block."""

from __future__ import annotations

from max.experimental import functional as F
from max.experimental.nn import Module
from max.experimental.tensor import Tensor
from max.nn.kv_cache import PagedCacheValues

from .attention import GritLMAttention


class GritLMTransformerBlock(Module[..., Tensor]):
    """Pre-norm decoder block: Attention → residual → MLP → residual."""

    def __init__(
        self,
        attention: GritLMAttention,
        mlp: Module[[Tensor], Tensor],
        input_layernorm: Module[[Tensor], Tensor],
        post_attention_layernorm: Module[[Tensor], Tensor],
    ) -> None:
        super().__init__()
        self.self_attn = attention
        self.mlp = mlp
        self.input_layernorm = input_layernorm
        self.post_attention_layernorm = post_attention_layernorm

    def forward(
        self,
        layer_idx: Tensor,
        x: Tensor,
        kv_collection: PagedCacheValues,
        input_row_offsets: Tensor,
        **kwargs,
    ) -> Tensor:
        h = x + self.self_attn(
            self.input_layernorm(x),
            kv_collection,
            input_row_offsets=input_row_offsets,
            **kwargs,
        )
        return h + self.mlp(self.post_attention_layernorm(h))
