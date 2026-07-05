"""MiMo-7B transformer compute graph.

MiMo-7B is Qwen2-based with these properties vs vanilla Llama3:
  1. attention_bias=True  — Q/K/V projections have learned bias terms
  2. RMSNorm (no bias)    — same as LLaMA, unlike StableLM which uses LayerNorm
  3. Full RoPE            — all head dims get position encoding (no partial factor)
  4. 36 layers            — loaded as-is, MTP layer (index 36) skipped via weight adapter

The only meaningful difference from Llama3's graph is has_bias=True on
AttentionWithRope. We build this explicitly rather than using the Llama3
graph so attention bias weights are actually allocated and loaded.
"""

from __future__ import annotations

import functools
from collections.abc import Callable

from max.dtype import DType
from max.graph import BufferType, DeviceRef, TensorType
from max.nn.attention import AttentionWithRope
from max.nn.embedding import Embedding
from max.nn.kv_cache import KVCacheParamInterface
from max.nn.linear import MLP, Linear
from max.nn.norm import RMSNorm
from max.nn.rotary_embedding import Llama3RotaryEmbedding
from max.nn.transformer import Transformer, TransformerBlock
from max.pipelines.lib.lora import LoRAManager

from max.pipelines.architectures.llama3.model_config import Llama3Config


class MiMoTransformer(Transformer):
    """
    MiMo-7B transformer graph.

    Identical to Llama3 except AttentionWithRope has has_bias=True to
    match Qwen2ForCausalLM's attention_bias=True in the config.json.
    """

    def __init__(self, config: Llama3Config) -> None:
        assert len(config.devices) == 1
        self.config = config

        # Full RoPE — all head dims get position encoding (no partial factor)
        full_head_dim = config.hidden_size // config.num_attention_heads

        rope = Llama3RotaryEmbedding(
            dim=config.hidden_size,
            n_heads=config.num_attention_heads,
            theta=config.rope_theta,
            max_seq_len=config.max_seq_len,
            head_dim=full_head_dim,             # full head dim, no partial RoPE
            interleaved=config.interleaved_rope_weights,
            scaling_params=config.rope_scaling_params,
        )

        # RMSNorm — same as LLaMA, no bias on the norm layer
        eps = config.rms_norm_eps if config.rms_norm_eps is not None else 1e-5
        create_norm: Callable[[], RMSNorm] = functools.partial(
            RMSNorm,
            dims=config.hidden_size,
            dtype=config.norm_dtype or config.dtype,
            eps=eps,
        )

        linear_cls: Callable[..., Linear] = functools.partial(
            Linear, quant_config=config.quant_config
        )

        # AttentionWithRope with QKV bias — this is the key difference from Llama3.
        # Qwen2/MiMo config has attention_bias=True which adds learned bias to
        # Q, K, V projection matrices.
        attention_cls: Callable[..., AttentionWithRope] = functools.partial(
            AttentionWithRope,
            stacked_qkv=config.stacked_qkv,
            scale=config.attention_multiplier,
            clip_qkv=config.clip_qkv,
            has_bias=True,                      # ← attention_bias=True in config.json
            quant_config=config.quant_config,
        )

        mlp_cls = functools.partial(MLP, quant_config=config.quant_config)

        layers = [
            TransformerBlock(
                attention=attention_cls(
                    num_attention_heads=config.num_attention_heads,
                    num_key_value_heads=config.num_key_value_heads,
                    hidden_size=config.hidden_size,
                    kv_params=config.kv_params,
                    dtype=config.dtype,
                    rope=rope,
                    linear_cls=linear_cls,
                    devices=config.devices,
                ),
                mlp=mlp_cls(
                    config.dtype,
                    config.model_quantization_encoding,
                    config.hidden_size,
                    config.intermediate_size,
                    config.devices,
                    linear_cls,
                ),
                attention_norm=create_norm(),
                mlp_norm=create_norm(),
                residual_multiplier=config.residual_multiplier,
            )
            for _ in range(config.num_hidden_layers)
        ]

        embedding_layer = Embedding(
            config.vocab_size,
            config.hidden_size,
            config.dtype,
            config.devices[0],
        )
        output = Linear(
            config.hidden_size,
            config.vocab_size,
            config.dtype,
            config.devices[0],
        )

        if config.tie_word_embeddings:
            output.set_shared_weight("weight", embedding_layer.weight)

        super().__init__(
            dim=config.hidden_size,
            n_heads=config.num_attention_heads,
            layers=layers,
            norm=create_norm(),
            output=output,
            embedding=embedding_layer,
            kv_params=config.kv_params,
            rope=rope,
            return_logits=config.return_logits,
            return_hidden_states=config.return_hidden_states,
            embedding_multiplier=config.embedding_multiplier,
            logits_scaling=config.logits_scaling,
        )

    def input_types(
        self,
        kv_params: KVCacheParamInterface,
        lora_manager: LoRAManager | None,
        needs_hidden_state_input: bool = False,
    ) -> tuple[TensorType | BufferType, ...]:
        device_ref = self.config.devices[0]
        return_n_logits_type = TensorType(
            DType.int64, shape=["return_n_logits"], device=DeviceRef.CPU()
        )
        kv_inputs = kv_params.get_symbolic_inputs()
        tokens_type = TensorType(
            DType.int64, shape=["total_seq_len"], device=device_ref
        )
        input_row_offsets_type = TensorType(
            DType.uint32, shape=["input_row_offsets_len"], device=device_ref
        )
        return (
            tokens_type,
            input_row_offsets_type,
            return_n_logits_type,
            *kv_inputs.flatten(),
        )