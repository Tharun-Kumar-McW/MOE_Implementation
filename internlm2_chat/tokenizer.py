from __future__ import annotations
from typing import TYPE_CHECKING, Any
from max.pipelines.lib import TextTokenizer

if TYPE_CHECKING:
    from max.pipelines.lib import PipelineConfig


class InternLM2Tokenizer(TextTokenizer):
    """Custom tokenizer that enables trust_remote_code for InternLM2."""

    def __init__(
        self,
        model_path: str,
        pipeline_config: PipelineConfig,
        *,
        revision: str | None = None,
        max_length: int | None = None,
        trust_remote_code: bool = True,
        enable_llama_whitespace_fix: bool = False,
        chat_template: str | None = None,
        **unused_kwargs,
    ) -> None:
        super().__init__(
            model_path=model_path,
            pipeline_config=pipeline_config,
            revision=revision,
            max_length=max_length,
            trust_remote_code=trust_remote_code,
            enable_llama_whitespace_fix=enable_llama_whitespace_fix,
            chat_template=chat_template,
            **unused_kwargs,
        )

    @property
    def hf_tokenizer(self):
        return self.delegate

