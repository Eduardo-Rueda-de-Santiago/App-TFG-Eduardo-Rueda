"""
Memory Game MCP Server

This module implements a Model Context Protocol (MCP) server using the `fastmcp`
package. It exposes tools to interact with the authoritative realtime memory game
backend running on port 4000.

The tools provided enable an AI agent to:
1. Retrieve the current game state (with card value masking to prevent cheating).
2. Flip cards on the board by their ID.
3. Reset/restart the game.
"""

import json
import urllib.request
import urllib.error
from typing import Any, Dict, Optional
from fastmcp import FastMCP

# Initialize the FastMCP server
mcp = FastMCP("Memory Game MCP Server")

# Base URL for the authoritative NestJS game backend
BACKEND_URL = "http://localhost:4000"


def _make_request(
    endpoint: str, method: str = "GET", data: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Helper function to send HTTP requests to the NestJS backend.

    Args:
        endpoint: The path component of the URL (e.g. '/game/state').
        method: The HTTP method (GET, POST).
        data: Optional dictionary to be serialized as JSON payload for POST.

    Returns:
        The response dictionary parsed from the server's JSON response.

    Raises:
        RuntimeError: If the server is unreachable, times out, or returns an error.
    """
    url = f"{BACKEND_URL.rstrip('/')}/{endpoint.lstrip('/')}"
    headers = {"Content-Type": "application/json"}

    req_data = None
    if data is not None:
        req_data = json.dumps(data).encode("utf-8")

    req = urllib.request.Request(url, data=req_data, headers=headers, method=method)

    try:
        with urllib.request.urlopen(req, timeout=5) as response:
            res_data = response.read().decode("utf-8")
            return json.loads(res_data)
    except urllib.error.URLError as e:
        # Inform the user specifically if the backend server is not running
        if (
            isinstance(e.reason, ConnectionRefusedError)
            or "connection" in str(e.reason).lower()
        ):
            raise RuntimeError(
                f"Could not connect to the memory game backend at {BACKEND_URL}. "
                "Please verify that the NestJS backend is running "
                "(e.g., run `npm run dev` in the `web/memory-game` directory)."
            ) from e
        raise RuntimeError(f"HTTP request to {url} failed: {e}") from e
    except Exception as e:
        raise RuntimeError(
            f"An unexpected error occurred during request to {url}: {e}"
        ) from e


@mcp.tool()
def get_game_state(mask_unrevealed: bool = True) -> Dict[str, Any]:
    """
    Retrieve the current game state from the memory game server.

    Args:
        mask_unrevealed: If True, card values that are not currently flipped
            or matched will be replaced with "?" to ensure the AI plays
            honestly by remembering cards. Set to False only if you need
            to inspect the true layout for debugging.

    Returns:
        A dictionary containing:
            - cards: A list of card dictionaries, each containing 'id', 'value',
              'flipped', and 'matched'.
            - moves: Total number of moves made in the game.
            - matches: Number of card pairs successfully matched.
            - isWon: A boolean indicating if the game has been won.
    """
    state = _make_request("/game/state", "GET")

    if mask_unrevealed and "cards" in state:
        masked_cards = []
        for card in state["cards"]:
            masked_card = card.copy()
            if not card.get("flipped") and not card.get("matched"):
                masked_card["value"] = "?"
            masked_cards.append(masked_card)
        state["cards"] = masked_cards

    return state


@mcp.tool()
def flip_card(card_id: int) -> Dict[str, Any]:
    """
    Flip a card by its ID.

    Args:
        card_id: The 1-based ID of the card to flip (typically 1 to 16).

    Returns:
        A dictionary indicating the outcome, e.g.:
            {"success": True} or {"success": False, "message": "..."}
        Note that if two mismatched cards are flipped, the backend will auto-unflip
        them after a brief delay (1.2 seconds).
    """
    return _make_request("/game/flip", "POST", {"cardId": card_id})


@mcp.tool()
def reset_game() -> Dict[str, Any]:
    """
    Reset the memory game to a brand new state, shuffling all cards.

    Returns:
        The initial game state.
    """
    return _make_request("/game/reset", "POST")


@mcp.tool()
def play_again() -> Dict[str, Any]:
    """
    Start a new game session. Useful after victory.

    Returns:
        The brand new initial game state.
    """
    return _make_request("/game/play-again", "POST")


if __name__ == "__main__":
    # Start the FastMCP server when executed directly
    mcp.run()
