from __future__ import annotations
import numpy as np
from max.dtype import DType
from max.graph.weights import WeightData, Weights

PADDED_VOCAB_SIZE = 92550

# num_attention_heads=16, num_key_value_heads=8, head_dim=128
NUM_Q_HEADS  = 16
NUM_KV_HEADS = 8
HEAD_DIM     = 128
Q_DIM  = NUM_Q_HEADS  * HEAD_DIM   # 2048
KV_DIM = NUM_KV_HEADS * HEAD_DIM   # 1024

VOCAB_KEYS = {"embed_tokens.weight", "lm_head.weight"}

_SIMPLE_RENAMES = {
    "model.tok_embeddings.weight": "embed_tokens.weight",
    "model.output.weight":         "lm_head.weight",
    "output.weight":               "lm_head.weight",
    "model.norm.weight":           "norm.weight",
}

_LAYER_RENAMES = {
    "attention_norm.weight":  "input_layernorm.weight",
    "ffn_norm.weight":        "post_attention_layernorm.weight",
    "attention.wo.weight":    "self_attn.o_proj.weight",
    "self_attn.wo.weight":    "self_attn.o_proj.weight",
    "feed_forward.w1.weight": "mlp.gate_proj.weight",
    "feed_forward.w2.weight": "mlp.down_proj.weight",
    "feed_forward.w3.weight": "mlp.up_proj.weight",
}


def convert_safetensor_state_dict(
    state_dict: dict[str, Weights],
    huggingface_config=None,
    pipeline_config=None,
) -> dict[str, WeightData]:
    new_sd: dict[str, WeightData] = {}

    for orig_key, value in state_dict.items():
        raw: WeightData = value.data()

        # bf16 -> f32
        if raw.dtype == DType.bfloat16:
            raw = raw.astype(DType.float32)

        # Handle wqkv: split into q_proj, k_proj, v_proj
        if orig_key.startswith("model.layers.") and orig_key.endswith(("attention.wqkv.weight", "self_attn.wqkv.weight")):
            parts = orig_key.split(".", 3)
            layer_n = parts[2]
            arr = np.from_dlpack(raw.data)
            if arr.ndim != 2:
                raise ValueError(
                    f"Expected 2D wqkv weight for {orig_key}, got shape {arr.shape}"
                )
            if arr.shape == (Q_DIM + 2 * KV_DIM, Q_DIM):
                pass
            elif arr.shape == (Q_DIM, Q_DIM + 2 * KV_DIM):
                arr = arr.T
            else:
                raise ValueError(
                    f"Unexpected wqkv weight shape for {orig_key}: {arr.shape}"
                )

            q = arr[:Q_DIM]
            k = arr[Q_DIM:Q_DIM+KV_DIM]
            v = arr[Q_DIM+KV_DIM:]
            prefix = f"layers.{layer_n}.self_attn"
            new_sd[f"{prefix}.q_proj.weight"] = WeightData.from_numpy(q, f"{prefix}.q_proj.weight")
            new_sd[f"{prefix}.k_proj.weight"] = WeightData.from_numpy(k, f"{prefix}.k_proj.weight")
            new_sd[f"{prefix}.v_proj.weight"] = WeightData.from_numpy(v, f"{prefix}.v_proj.weight")
            continue

        # Simple top-level renames
        if orig_key in _SIMPLE_RENAMES:
            new_key = _SIMPLE_RENAMES[orig_key]
        # Per-layer renames
        elif orig_key.startswith("model.layers."):
            parts = orig_key.split(".", 3)
            suffix = parts[3] if len(parts) == 4 else None
            if suffix and suffix in _LAYER_RENAMES:
                new_key = f"layers.{parts[2]}.{_LAYER_RENAMES[suffix]}"
            else:
                new_key = orig_key
        else:
            new_key = orig_key

        # Pad vocab tensors
        if new_key in VOCAB_KEYS:
            arr = np.from_dlpack(raw.data)
            if arr.ndim != 2:
                raise ValueError(
                    f"Expected 2D vocab weight for {new_key}, got shape {arr.shape}"
                )
            if arr.shape[0] < PADDED_VOCAB_SIZE:
                pad = np.zeros((PADDED_VOCAB_SIZE - arr.shape[0], arr.shape[1]), dtype=arr.dtype)
                arr = np.concatenate([arr, pad], axis=0)
                raw = WeightData.from_numpy(arr, new_key)
            elif arr.shape[1] < PADDED_VOCAB_SIZE:
                arr = arr.T
                if arr.shape[0] < PADDED_VOCAB_SIZE:
                    pad = np.zeros((PADDED_VOCAB_SIZE - arr.shape[0], arr.shape[1]), dtype=arr.dtype)
                    arr = np.concatenate([arr, pad], axis=0)
                raw = WeightData.from_numpy(arr, new_key)

        new_sd[new_key] = raw

    return new_sd
