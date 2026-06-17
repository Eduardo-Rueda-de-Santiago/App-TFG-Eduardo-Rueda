"""
model_manager.py — Loads and holds all three Llama models in memory.

Each model is configured with its own chat_format, context size, and GPU
offload settings.  All three coexist in the same process simultaneously.

Usage::

    manager = ModelManager()
    manager.load_all()

    # Access models individually:
    brain_llm        = manager.brain
    tool_caller_llm  = manager.tool_caller
    communicator_llm = manager.communicator

    # Release VRAM / RAM when done:
    manager.unload_all()
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field

from llama_cpp import Llama

from ai_agent.config import (
    BRAIN_CHAT_FORMAT,
    BRAIN_MODEL_PATH,
    COMMUNICATOR_CHAT_FORMAT,
    COMMUNICATOR_MODEL_PATH,
    N_CTX,
    N_GPU_LAYERS,
    TOOL_CALLER_CHAT_FORMAT,
    TOOL_CALLER_MODEL_PATH,
)


@dataclass
class ModelManager:
    """
    Holds three Llama model instances and provides a clean loading interface.

    Attributes
    ----------
    brain:
        The routing / intent-classification model (Brain).
    tool_caller:
        The function-calling model (Tool Caller).
    communicator:
        The conversational response model (Communicator / Nova).
    """

    brain: Llama | None = field(default=None, init=False)
    tool_caller: Llama | None = field(default=None, init=False)
    communicator: Llama | None = field(default=None, init=False)

    # -------------------------------------------------------------------------
    # Private helpers
    # -------------------------------------------------------------------------

    @staticmethod
    def _load_one(
        label: str,
        path: str,
        chat_format: str | None,
        n_gpu_layers: int = N_GPU_LAYERS,
        n_ctx: int = N_CTX,
    ) -> Llama:
        """
        Load a single GGUF model and return the Llama instance.

        Parameters
        ----------
        label:
            Human-readable name for progress output (e.g. "Brain").
        path:
            Absolute path to the ``.gguf`` model file.
        chat_format:
            Chat format string for llama-cpp-python (e.g. ``"chatml"``).
            Pass ``None`` to let llama-cpp auto-detect from the GGUF metadata.
        n_gpu_layers:
            Number of transformer layers to offload to GPU.
            ``-1`` offloads all layers; ``0`` runs entirely on CPU.
        n_ctx:
            Context window size in tokens.
        """
        print(f"  Loading [{label}] from {path} …", flush=True)
        try:
            llm = Llama(
                model_path=path,
                n_gpu_layers=n_gpu_layers,
                n_ctx=n_ctx,
                chat_format=chat_format,
                verbose=False,
            )
        except Exception as exc:
            print(f"  ✗ Failed to load [{label}]: {exc}", file=sys.stderr)
            raise
        print(f"  ✓ [{label}] loaded  (ctx={n_ctx}, gpu_layers={n_gpu_layers})")
        return llm

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    def load_all(self) -> None:
        """
        Load all three models into memory.

        Should be called once at application startup.  Raises on the first
        model that fails to load; models loaded before the failure remain in
        memory until ``unload_all()`` is called.
        """
        print("=" * 60)
        print("Loading models …")
        print("=" * 60)

        self.brain = self._load_one(
            label="Brain",
            path=BRAIN_MODEL_PATH,
            chat_format=BRAIN_CHAT_FORMAT,
        )
        self.tool_caller = self._load_one(
            label="Tool Caller",
            path=TOOL_CALLER_MODEL_PATH,
            chat_format=TOOL_CALLER_CHAT_FORMAT,
        )
        self.communicator = self._load_one(
            label="Communicator",
            path=COMMUNICATOR_MODEL_PATH,
            chat_format=COMMUNICATOR_CHAT_FORMAT,
        )

        print("=" * 60)
        print("All models loaded and ready.")
        print("=" * 60)

    def unload_all(self) -> None:
        """
        Release model memory.

        Deletes each Llama instance and sets the attribute to ``None``.
        Safe to call even if ``load_all`` was never called or partially
        completed.
        """
        for attr in ("brain", "tool_caller", "communicator"):
            model = getattr(self, attr, None)
            if model is not None:
                del model
                setattr(self, attr, None)
        print("All models unloaded.")
