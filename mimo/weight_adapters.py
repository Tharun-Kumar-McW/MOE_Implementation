"""Weight adapters for XiaomiMiMo/MiMo-7B-Base.

From modeling_mimo.py, the checkpoint has these key groups:

GROUP 1 — Main model (keep these):
  model.embed_tokens.weight
  model.layers.{0..35}.input_layernorm.weight
  model.layers.{0..35}.self_attn.q_proj.{weight,bias}   ← bias due to attention_bias=True
  model.layers.{0..35}.self_attn.k_proj.{weight,bias}
  model.layers.{0..35}.self_attn.v_proj.{weight,bias}
  model.layers.{0..35}.self_attn.o_proj.weight
  model.layers.{0..35}.mlp.gate_proj.weight
  model.layers.{0..35}.mlp.up_proj.weight
  model.layers.{0..35}.mlp.down_proj.weight
  model.layers.{0..35}.post_attention_layernorm.weight
  model.norm.weight
  lm_head.weight

GROUP 2 — MTP head (skip these — from MiMoMTPLayers):
  model.mtp_layers.0.token_layernorm.weight
  model.mtp_layers.0.hidden_layernorm.weight
  model.mtp_layers.0.input_proj.weight
  model.mtp_layers.0.input_layernorm.weight
  model.mtp_layers.0.self_attn.q_proj.{weight,bias}
  model.mtp_layers.0.self_attn.k_proj.{weight,bias}
  model.mtp_layers.0.self_attn.v_proj.{weight,bias}
  model.mtp_layers.0.self_attn.o_proj.weight
  model.mtp_layers.0.mlp.{gate_proj,up_proj,down_proj}.weight
  model.mtp_layers.0.post_attention_layernorm.weight
  model.mtp_layers.0.final_layernorm.weight

The main model keys are already in LLaMA-style format — no renaming needed.
We only need to drop the MTP layer keys.
"""

from __future__ import annotations

import torch


def convert_safetensor_state_dict(
    state_dict: dict[str, torch.Tensor],
    huggingface_config,
) -> dict[str, torch.Tensor]:
    """Drop MTP head weights, pass all main model weights through unchanged."""

    new_sd: dict[str, torch.Tensor] = {}

    for key, tensor in state_dict.items():
        # Drop anything belonging to the MTP head
        # MiMoMTPLayers are stored under model.mtp_layers.*
        if "mtp_layers" in key:
            continue

        new_sd[key] = tensor

    return new_sd