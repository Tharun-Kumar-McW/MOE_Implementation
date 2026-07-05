# ===----------------------------------------------------------------------=== #
# GritLM weight adapters.
#
# GritLM safetensors layout (standard HF naming, no trust_remote_code needed
# for weights — only for the modeling class):
#
#   model.embed_tokens.weight
#   model.layers.N.self_attn.{q,k,v,o}_proj.weight
#   model.layers.N.mlp.{gate,up,down}_proj.weight
#   model.layers.N.{input,post_attention}_layernorm.weight
#   model.norm.weight
#   lm_head.weight                    (tie_word_embeddings: false)
#
# MAX internal convention: no "model." prefix — weights live directly under
# "layers.N.*", "embed_tokens.*", "norm.*", "lm_head.*".
#
# GritLM also ships a "gritlm_pooling_head.weight" for the embedding path;
# we drop it since MAX only serves the CausalLM path.
# ===----------------------------------------------------------------------=== #
"""Weight adapters for GritLM safetensors."""

from __future__ import annotations

from max.dtype import DType
from max.graph.weights import WeightData, Weights
from transformers import AutoConfig

GRITLM_SAFETENSOR_MAPPING: dict[str, str] = {
    "model.": "",   # strip "model." prefix — matches MAX Mistral convention
}

# Weights to discard — GritLM pooling head, not used in CausalLM serving
_DROP_PREFIXES = (
    "gritlm_pooling",
    "pooling",
)


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
    """Remap GritLM safetensor weight names to MAX internal convention.

    Strips the 'model.' prefix and drops the pooling head weights that
    are only used for the embedding (non-CausalLM) path.
    Also casts dtype when checkpoint (bfloat16) differs from requested encoding.
    """
    target_dtype = _target_dtype(pipeline_config) if pipeline_config else None

    new_state_dict: dict[str, WeightData] = {}
    for name, value in state_dict.items():
        # Drop pooling head — not used in CausalLM serving
        if any(name.startswith(p) for p in _DROP_PREFIXES):
            continue

        max_name = name
        for before, after in GRITLM_SAFETENSOR_MAPPING.items():
            max_name = max_name.replace(before, after)

        wd: WeightData = value.data()
        if target_dtype is not None and wd.dtype != target_dtype:
            wd = wd.astype(target_dtype)
        new_state_dict[max_name] = wd

    return new_state_dict
