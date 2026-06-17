"""
output_terminal.py — Terminal (stdout) output implementation.

Prints the agent's response to the terminal in a formatted box, matching
the style used in the original demo's ``print_result`` function.
"""

from __future__ import annotations

from ai_agent.output.output_generic import OutputGeneric


class TerminalOutput(OutputGeneric):
    """
    Prints agent responses to stdout with a simple decorative border.

    Parameters
    ----------
    agent_name:
        Display name used in the output header.  Defaults to ``"NOVA"``.
    width:
        Width of the separator lines in characters.
    """

    def __init__(self, agent_name: str = "NOVA", width: int = 60) -> None:
        self._agent_name = agent_name
        self._width = width

    async def emit(self, message: str, *, task_achieved: bool = True) -> None:
        """
        Print *message* to stdout inside a decorative box.

        Parameters
        ----------
        message:
            The agent's natural-language response.
        task_achieved:
            Displayed as a brief status line below the message.
        """
        sep = "=" * self._width
        print(f"\n{sep}")
        print(f"  {self._agent_name} says:")
        print(sep)
        print(f"\n{message}\n")
        achieved_label = "Yes" if task_achieved else "No"
        print(f"  Task achieved: {achieved_label}")
        print(sep)
