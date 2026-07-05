"""KeyLM-75M pipeline model.

KeyLM-75M (Eclipse-Senpai/KeyLM-75M) is a 75M parameter causal LM with a
standard LlamaForCausalLM architecture:
  - GQA: 8 attention heads, 2 KV heads
  - RoPE positional encoding (full, no partial factor)
  - Pre-RMSNorm (no bias)
  - SwiGLU (SiLU) activation
  - No QKV bias, no sliding window

It is architecturally identical to Llama3 at the graph level, so Llama3Model
handles it completely with no overrides required.
"""

from max.pipelines.architectures.llama3.model import Llama3Model


class KeyLMModel(Llama3Model):
    """KeyLM-75M pipeline model — pure LLaMA, no graph overrides needed."""
    pass