"""
llm_config.py — LLM provider configuration.

Reads settings from environment variables (or a .env file) via pydantic-settings
and exposes a single ``build_model()`` factory that returns the correct
pydantic-ai Model for the chosen provider.

Supported providers
-------------------
- ``llama_cpp``  — llama-cpp-python direct (no server needed).
                   Loads the GGUF file in-process via Llama().
- ``openai``     — OpenAI Chat Completions API
- ``anthropic``  — Anthropic Claude API
- ``ollama``     — Ollama local server (OpenAI-compatible)
- ``groq``       — Groq Cloud API

Environment variables
---------------------
All variables are prefixed with ``AI_AGENT_`` and nested sections use ``__``
as the delimiter (pydantic-settings convention).

Examples::

    AI_AGENT_PROVIDER=llama_cpp
    AI_AGENT_LLAMA_CPP__MODEL_PATH=C:/path/to/model.gguf
    AI_AGENT_LLAMA_CPP__CHAT_FORMAT=chatml-function-calling
    AI_AGENT_LLAMA_CPP__N_GPU_LAYERS=-1

    AI_AGENT_PROVIDER=openai
    AI_AGENT_OPENAI__API_KEY=sk-...
    AI_AGENT_OPENAI__MODEL=gpt-4o-mini

    AI_AGENT_PROVIDER=anthropic
    AI_AGENT_ANTHROPIC__API_KEY=sk-ant-...
    AI_AGENT_ANTHROPIC__MODEL=claude-3-5-haiku-20241022

    AI_AGENT_PROVIDER=ollama
    AI_AGENT_OLLAMA__BASE_URL=http://localhost:11434/v1
    AI_AGENT_OLLAMA__MODEL=llama3.2

    AI_AGENT_PROVIDER=groq
    AI_AGENT_GROQ__API_KEY=gsk_...
    AI_AGENT_GROQ__MODEL=llama-3.1-8b-instant
"""

from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

if TYPE_CHECKING:
    from pydantic_ai.models import Model


# ---------------------------------------------------------------------------
# Provider enum
# ---------------------------------------------------------------------------


class LLMProvider(str, Enum):
    """Supported LLM backends."""

    LLAMA_CPP = "llama_cpp"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    OLLAMA = "ollama"
    GROQ = "groq"


# ---------------------------------------------------------------------------
# Per-provider settings (nested pydantic models)
# ---------------------------------------------------------------------------


class LlamaCppSettings(BaseModel):
    """Settings for the llama-cpp-python direct (in-process) backend.

    The GGUF model is loaded into the same process — no HTTP server needed.

    CRITICAL: ``chat_format`` must match the model family for tool calling
    to work.  From the benchmark (ai-models_benchmark_llamacpp.ipynb):

    - ``"chatml-function-calling"`` — LFM2.5, Qwen2.5, Phi-4-mini, most models
    - ``"llama-3"``                 — Llama 3.x models
    - ``None`` / ``""``             — let llama-cpp autodetect from GGUF metadata
                                      (fine for text, unreliable for tools)
    """

    model_path: str = ""
    chat_format: str = "chatml-function-calling"
    n_ctx: int = 4096
    n_gpu_layers: int = -1   # -1 = all layers on GPU; 0 = CPU only
    verbose: bool = False


class OpenAISettings(BaseModel):
    """Settings for the OpenAI Chat Completions API."""

    api_key: str = ""
    model: str = "gpt-4o-mini"
    #: Override to point at any OpenAI-compatible endpoint.
    base_url: str = ""


class AnthropicSettings(BaseModel):
    """Settings for the Anthropic Claude API."""

    api_key: str = ""
    model: str = "claude-3-5-haiku-20241022"


class OllamaSettings(BaseModel):
    """Settings for a local Ollama server (OpenAI-compatible)."""

    base_url: str = "http://localhost:11434/v1"
    model: str = "llama3.2"


class GroqSettings(BaseModel):
    """Settings for the Groq Cloud API."""

    api_key: str = ""
    model: str = "llama-3.1-8b-instant"


# ---------------------------------------------------------------------------
# Root config
# ---------------------------------------------------------------------------


class LLMConfig(BaseSettings):
    """Root configuration for the LLM subsystem.

    Loaded from environment variables (prefix ``AI_AGENT_``) and/or a
    ``.env`` file in the current working directory.

    Usage::

        config = LLMConfig()
        model  = config.build_model()   # pydantic-ai Model, ready to use
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_nested_delimiter="__",
        env_prefix="AI_AGENT_",
        extra="ignore",
    )

    provider: LLMProvider = LLMProvider.LLAMA_CPP

    llama_cpp: LlamaCppSettings = Field(default_factory=LlamaCppSettings)
    openai: OpenAISettings = Field(default_factory=OpenAISettings)
    anthropic: AnthropicSettings = Field(default_factory=AnthropicSettings)
    ollama: OllamaSettings = Field(default_factory=OllamaSettings)
    groq: GroqSettings = Field(default_factory=GroqSettings)

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    def build_model(self) -> "Model":
        """Return a pydantic-ai Model configured for the chosen provider.

        Raises
        ------
        ValueError
            If the selected provider is unknown.
        ImportError
            If provider-specific dependencies are not installed.
        """
        match self.provider:
            case LLMProvider.LLAMA_CPP:
                return self._build_llama_cpp()
            case LLMProvider.OPENAI:
                return self._build_openai()
            case LLMProvider.ANTHROPIC:
                return self._build_anthropic()
            case LLMProvider.OLLAMA:
                return self._build_ollama()
            case LLMProvider.GROQ:
                return self._build_groq()
            case _:
                raise ValueError(f"Unknown LLM provider: {self.provider!r}")

    # ------------------------------------------------------------------
    # Private builders — one per provider
    # ------------------------------------------------------------------

    def _build_llama_cpp(self) -> "Model":
        """Load a GGUF model in-process using llama-cpp-python directly."""
        from llama_cpp import Llama

        from ai_agent.config.llama_cpp_model import LlamaCppDirectModel

        cfg = self.llama_cpp
        if not cfg.model_path:
            raise ValueError(
                "AI_AGENT_LLAMA_CPP__MODEL_PATH is not set. "
                "Point it at your .gguf file, e.g.:\n"
                "  AI_AGENT_LLAMA_CPP__MODEL_PATH=C:/path/to/model.gguf"
            )

        import logging
        logging.getLogger(__name__).info(
            "[config] loading llama-cpp model: %s (chat_format=%r, n_gpu_layers=%d)",
            cfg.model_path, cfg.chat_format or "autodetect", cfg.n_gpu_layers,
        )

        llm = Llama(
            model_path=cfg.model_path,
            n_ctx=cfg.n_ctx,
            n_gpu_layers=cfg.n_gpu_layers,
            chat_format=cfg.chat_format or None,
            verbose=cfg.verbose,
        )

        import os
        display_name = os.path.basename(cfg.model_path)
        return LlamaCppDirectModel(llm, display_name=display_name)

    def _build_openai(self) -> "Model":
        """OpenAI Chat Completions API (or any compatible endpoint)."""
        from pydantic_ai.models.openai import OpenAIChatModel
        from pydantic_ai.providers.openai import OpenAIProvider

        cfg = self.openai
        provider = OpenAIProvider(
            api_key=cfg.api_key or None,
            base_url=cfg.base_url or None,
        )
        return OpenAIChatModel(cfg.model, provider=provider)

    def _build_anthropic(self) -> "Model":
        """Anthropic Claude API."""
        from pydantic_ai.models.anthropic import AnthropicModel
        from pydantic_ai.providers.anthropic import AnthropicProvider

        cfg = self.anthropic
        provider = AnthropicProvider(api_key=cfg.api_key or None)
        return AnthropicModel(cfg.model, provider=provider)

    def _build_ollama(self) -> "Model":
        """Ollama local server (OpenAI-compatible endpoint)."""
        from pydantic_ai.models.openai import OpenAIChatModel
        from pydantic_ai.providers.openai import OpenAIProvider

        cfg = self.ollama
        provider = OpenAIProvider(base_url=cfg.base_url, api_key="ollama")
        return OpenAIChatModel(cfg.model, provider=provider)

    def _build_groq(self) -> "Model":
        """Groq Cloud API."""
        from pydantic_ai.models.groq import GroqModel
        from pydantic_ai.providers.groq import GroqProvider

        cfg = self.groq
        provider = GroqProvider(api_key=cfg.api_key or None)
        return GroqModel(cfg.model, provider=provider)
