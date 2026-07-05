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

# HuggingFace safetensors → MAX internal naming
SMOLLM3_SAFETENSOR_MAPPING: dict[str, str] = {
    "model.": "language_model.",
    "lm_head": "language_model.lm_head",
}

# GGUF block names → MAX internal naming
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
    """Extract the requested compute dtype from the pipeline config."""
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
    """Remap HuggingFace safetensor weight names to MAX's internal convention.

    Also casts weights to the pipeline's requested dtype when the checkpoint
    dtype differs (e.g. bfloat16 checkpoint + float32 quantization-encoding).
    """
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
    """Remap GGUF weight names to MAX's internal convention for SmolLM3."""
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