"""
input_terminal.py — Terminal (stdin) input implementation.

Reads user prompts from standard input using a blocking ``input()`` call
offloaded to a thread executor so the asyncio event loop remains responsive.

Behaviour
---------
- Prints a prompt prefix (``You ▸ ``) on each iteration.
- Returns ``None`` to signal exit when the user types one of the quit words
  or sends EOF (Ctrl-D / Ctrl-Z).
- Skips empty lines silently.
- Strips leading/trailing whitespace from every prompt.
"""

from __future__ import annotations

import asyncio
import sys

from ai_agent.input.input_generic import InputGeneric

# Words that end the interactive session
_QUIT_WORDS = frozenset({"quit", "exit", "q", "bye"})


class TerminalInput(InputGeneric):
    """
    Reads user prompts from stdin in an async-friendly way.

    ``input()`` is a blocking call; it is run inside a thread executor so
    that the asyncio event loop is not blocked between keystrokes.

    Parameters
    ----------
    prompt_prefix:
        The string displayed before each input line.  Defaults to ``"You ▸ "``.
    """

    def __init__(self, prompt_prefix: str = "You ▸ ") -> None:
        self._prefix = prompt_prefix

    async def get_next_prompt(self) -> str | None:
        """
        Await the next line typed by the user.

        Returns
        -------
        str
            The stripped user input.
        None
            On EOF (Ctrl-D), KeyboardInterrupt, or a recognised quit word.
        """
        loop = asyncio.get_event_loop()

        while True:
            try:
                raw = await loop.run_in_executor(
                    None, lambda: input(self._prefix)
                )
            except (EOFError, KeyboardInterrupt):
                print("\nGoodbye!")
                return None

            text = raw.strip()

            if not text:
                # Empty line — ask again
                continue

            if text.lower() in _QUIT_WORDS:
                print("Goodbye!")
                return None

            return text
