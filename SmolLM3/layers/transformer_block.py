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
