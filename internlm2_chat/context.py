from __future__ import annotations
from typing import TYPE_CHECKING, Any
from max.pipelines.core import TextContext
from transformers import AutoTokenizer

if TYPE_CHECKING:
    from max.pipelines.lib import PipelineConfig


class InternLM2Context(TextContext):
    """Custom context that applies InternLM2 chat template."""

    def __init__(self, pipeline_config: PipelineConfig) -> None:
        super().__init__(pipeline_config)
        # Load the tokenizer to get chat template
        self.hf_tokenizer = AutoTokenizer.from_pretrained(
            pipeline_config.model_path,
            trust_remote_code=True,
        )

    def format_prompt(self, messages: list[dict[str, str]] | str, **kwargs) -> str:
        """Format messages or text using InternLM2 chat template."""
        # Handle both string and message list inputs
        if isinstance(messages, str):
            text = messages
            messages = [{"role": "user", "content": text}]
        
        # Apply the chat template from HuggingFace config
        if hasattr(self.hf_tokenizer, 'apply_chat_template'):
            formatted = self.hf_tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            # Fallback to manual template
            formatted = "<|im_start|>user\n"
            for msg in messages:
                if msg.get("role") == "user":
                    formatted += msg.get("content", "")
            formatted += "<|im_end|>\n<|im_start|>assistant\n"
        
        return formatted

    def encode_text(self, text: str) -> list[int]:
        """Encode text using the proper chat template."""
        formatted_text = self.format_prompt(text)
        tokens = self.hf_tokenizer.encode(formatted_text, add_special_tokens=False)
        return tokens

    def decode_tokens(self, tokens: list[int]) -> str:
        """Decode tokens to text."""
        return self.hf_tokenizer.decode(tokens, skip_special_tokens=False)


