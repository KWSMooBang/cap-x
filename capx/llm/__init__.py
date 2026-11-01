"""LLM client module for querying language models.

This module provides utilities for querying various LLM providers
(OpenAI, Claude, open-source models, OpenRouter) with support for
streaming, ensemble queries, and backward-compatible aliases.
"""

from capx.llm.client import (
    CLAUDE_MODELS,
    DEFAULT_SYNTHESIS_MODEL,
    ENSEMBLE_CONFIGS,
    GPT_MODELS,
    OPENROUTER_MODELS,
    OPENROUTER_SERVER_URL,
    OSS_MODELS,
    TEXT_ONLY_MODELS,
    VLM_MODELS,
    ModelQueryArgs,
    TextOnlyModelError,
    _completions_to_responses_convert_prompt,
    collapse_text_image_inputs,
    is_openrouter_model,
    is_text_only_model,
    parse_ensemble_candidates,
    query_model,
    query_model_ensemble,
    query_model_streaming,
    query_single_model_ensemble,
)

__all__ = [
    "CLAUDE_MODELS",
    "DEFAULT_SYNTHESIS_MODEL",
    "ENSEMBLE_CONFIGS",
    "GPT_MODELS",
    "OPENROUTER_MODELS",
    "OPENROUTER_SERVER_URL",
    "OSS_MODELS",
    "TEXT_ONLY_MODELS",
    "VLM_MODELS",
    "ModelQueryArgs",
    "TextOnlyModelError",
    "_completions_to_responses_convert_prompt",
    "collapse_text_image_inputs",
    "is_openrouter_model",
    "is_text_only_model",
    "parse_ensemble_candidates",
    "query_model",
    "query_model_ensemble",
    "query_model_streaming",
    "query_single_model_ensemble",
]
