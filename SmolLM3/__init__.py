"""SmolLM3 transformer architecture for text generation.

Decoder-only transformer (LLaMA-style) with:
  - 36 layers, hidden_size=2048, intermediate_size=11008
  - GQA: 16 Q heads, 4 KV heads (head_dim=128)
  - SwiGLU MLP (mlp_bias: false)
  - RMSNorm (rms_norm_eps=1e-6)
  - RoPE theta=5_000_000 with per-layer NoPE: every 4th layer skips RoPE
  - Tied input/output embeddings (tie_word_embeddings: true)
  - max_position_embeddings: 65536

MAX custom architecture registration
--------------------------------------
MAX's --custom-architectures loader does:

    sys.path.append(os.path.dirname(module_spec))
    module = importlib.import_module(os.path.basename(module_spec))

It then requires the imported module to expose:

    ARCHITECTURES: list[SupportedArchitecture]

This module satisfies that contract via the ARCHITECTURES list below.

CLI usage
---------
Place this package at e.g. ``/path/to/architectures/smollm3/`` then run::

    max serve \\
        --model-path HuggingFaceTB/SmolLM3-3B \\
        --custom-architectures /path/to/architectures/smollm3 \\
        --devices cpu \\
        --max-batch-size 1 \\
        --max-length 4096 \\
        --quantization-encoding bfloat16

Note: the path must point to the *package directory* (containing this
__init__.py). MAX splits the path as::

    module_path = os.path.dirname("/path/to/architectures/smollm3")
    # → "/path/to/architectures"
    module_name = os.path.basename("/path/to/architectures/smollm3")
    # → "smollm3"

So the directory name on disk must exactly match the Python module name
(all lowercase, no hyphens).
"""

from .arch import smollm3_arch
from .model import SmolLM3Inputs, SmolLM3Model
from .model_config import SmolLM3Config

# !! Required by MAX's custom architecture loader !!
# max/pipelines/lib/config/config.py checks for this list and registers
# each entry into PIPELINE_REGISTRY.
ARCHITECTURES = [smollm3_arch]

__all__ = [
    "ARCHITECTURES",
    "SmolLM3Config",
    "SmolLM3Inputs",
    "SmolLM3Model",
    "smollm3_arch",
]