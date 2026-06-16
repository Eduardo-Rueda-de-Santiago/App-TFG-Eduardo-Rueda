"use client";

import { useEffect, useState } from "react";
import { useSocket } from "../components/SocketProvider";
import { GameState, Card } from "../types/game";

export const useGameSocket = () => {
  const { socket, connected } = useSocket();
  const [gameState, setGameState] = useState<GameState>({
    cards: [],
    moves: 0,
    matches: 0,
    isWon: false,
  });

  useEffect(() => {
    if (!socket) return;

    // Handle full state synchronization
    const handleGameState = (state: GameState) => {
      setGameState(state);
    };

    // Handle initial state and updates
    socket.on("game_state", handleGameState);
    socket.on("reset_game", handleGameState);

    // Handle individual real-time card flip event (for instant UI response)
    socket.on("flip_card", ({ cardId }: { cardId: number }) => {
      setGameState((prevState) => {
        const updatedCards = prevState.cards.map((card) =>
          card.id === cardId ? { ...card, flipped: true } : card
        );
        return {
          ...prevState,
          cards: updatedCards,
        };
      });
    });

    // Request initial state on mount if connected
    if (connected) {
      socket.emit("get_game_state");
    }

    return () => {
      socket.off("game_state", handleGameState);
      socket.off("reset_game", handleGameState);
      socket.off("flip_card");
    };
  }, [socket, connected]);

  // Method to trigger a flip request via REST API (as authoritative action)
  const requestFlip = async (cardId: number) => {
    try {
      const response = await fetch("http://localhost:4000/game/flip", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ cardId }),
      });
      const data = await response.json();
      return data;
    } catch (error) {
      console.error("Failed to request card flip:", error);
    }
  };

  // Method to trigger a game reset request via REST API
  const requestReset = async () => {
    try {
      const response = await fetch("http://localhost:4000/game/reset", {
        method: "POST",
      });
      const data = await response.json();
      return data;
    } catch (error) {
      console.error("Failed to request game reset:", error);
    }
  };

  // Method to trigger a play again request via REST API
  const requestPlayAgain = async () => {
    try {
      const response = await fetch("http://localhost:4000/game/play-again", {
        method: "POST",
      });
      const data = await response.json();
      return data;
    } catch (error) {
      console.error("Failed to request play again:", error);
    }
  };

  return {
    gameState,
    connected,
    requestFlip,
    requestReset,
    requestPlayAgain,
  };
};
