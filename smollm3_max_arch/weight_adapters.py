# ===----------------------------------------------------------------------=== #
# Copyright (c) 2026, Modular Inc. All rights reserved.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions:
# https://llvm.org/LICENSE.txt
# ===----------------------------------------------------------------------=== #

from __future__ import annotations

from max.graph.weights import WeightData, Weights

# SmolLM3 uses the same weight naming convention as Llama3 — no remapping needed.
# model.embed_tokens, model.layers.N.*, model.norm, lm_head all match directly.


def convert_safetensor_state_dict(
    state_dict: dict[str, Weights], **unused_kwargs
) -> dict[str, WeightData]:
    """Pass-through adapter: SmolLM3 weight names match Llama3 MAX slots."""
    return {name: value.data() for name, value in state_dict.items()}
