"""Weight adapters for KeyLM-75M.

KeyLM-75M is a pure LlamaForCausalLM model. Its SafeTensors checkpoint uses
exactly the same key structure as Llama3:

  model.embed_tokens.weight
  model.layers.{i}.input_layernorm.weight
  model.layers.{i}.self_attn.q_proj.weight
  model.layers.{i}.self_attn.k_proj.weight
  model.layers.{i}.self_attn.v_proj.weight
  model.layers.{i}.self_attn.o_proj.weight
  model.layers.{i}.mlp.gate_proj.weight
  model.layers.{i}.mlp.up_proj.weight
  model.layers.{i}.mlp.down_proj.weight
  model.layers.{i}.post_attention_layernorm.weight
  model.norm.weight
  lm_head.weight

No renaming needed — Llama3's adapter works as-is.
"""

from max.pipelines.architectures.llama3.weight_adapters import (
    convert_safetensor_state_dict,
)

__all__ = ["convert_safetensor_state_dict"]