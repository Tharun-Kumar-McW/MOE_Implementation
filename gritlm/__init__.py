"""GritLM architecture for MAX text generation.

GritLM-7B (Generative Representational Instruction Tuning) — Mistral-7B backbone
fine-tuned for both generation and dense retrieval. This implementation covers
the CausalLM (text generation) path only; the embedding/pooling path is not
supported in MAX serving.

Architecture (from config.json):
  - 32 layers, hidden_size=4096, intermediate_size=14336
  - GQA: 32 Q heads, 8 KV heads (head_dim=128)
  - SwiGLU MLP (no bias)
  - RMSNorm (rms_norm_eps=1e-5)
  - RoPE theta=10000 (standard Mistral, no scaling)
  - Sliding window attention: window=4096 on ALL layers
  - Separate lm_head (tie_word_embeddings: false)
  - vocab_size: 32000 (Mistral/Llama-1 tokenizer)
  - max_position_embeddings: 32768

CLI usage:
    max serve \\
        --model-path GritLM/GritLM-7B \\
        --custom-architectures /path/to/architectures/gritlm \\
        --devices cpu \\
        --max-batch-size 1 \\
        --max-length 4096 \\
        --quantization-encoding bfloat16 \\
        --trust-remote-code
"""

from .arch import gritlm_arch
from .model import GritLMInputs, GritLMModel
from .model_config import GritLMConfig

# Required by MAX's custom architecture loader
ARCHITECTURES = [gritlm_arch]

__all__ = [
    "ARCHITECTURES",
    "GritLMConfig",
    "GritLMInputs",
    "GritLMModel",
    "gritlm_arch",
]
