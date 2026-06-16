"use client";

import React, { useState } from "react";
import { useGameSocket } from "../hooks/useGameSocket";
import { MemoryCard } from "./MemoryCard";
import styles from "./GameBoard.module.css";

export const GameBoard: React.FC = () => {
  const { gameState, connected, requestFlip, requestReset, requestPlayAgain } = useGameSocket();
  const [selectedSimId, setSelectedSimId] = useState<number>(1);
  const [lastApiAction, setLastApiAction] = useState<string>("");
  const [activeCurlTab, setActiveCurlTab] = useState<"flip" | "reset" | "play-again">("flip");

  const handleCardClick = async (cardId: number) => {
    await requestFlip(cardId);
  };

  const handleSimulateApiFlip = async () => {
    setLastApiAction(`POST /game/flip - Payload: { cardId: ${selectedSimId} }`);
    await requestFlip(selectedSimId);
  };

  const handleSimulateReset = async () => {
    setLastApiAction(`POST /game/reset`);
    await requestReset();
  };

  const handleSimulatePlayAgain = async () => {
    setLastApiAction(`POST /game/play-again`);
    await requestPlayAgain();
  };

  const getCurlCommand = () => {
    if (activeCurlTab === "flip") {
      return `curl -X POST -H "Content-Type: application/json" -d '{"cardId": ${selectedSimId}}' http://localhost:4000/game/flip`;
    } else if (activeCurlTab === "reset") {
      return `curl -X POST http://localhost:4000/game/reset`;
    } else {
      return `curl -X POST http://localhost:4000/game/play-again`;
    }
  };

  return (
    <div className={styles.boardWrapper}>
      {/* HEADER SECTION */}
      <header className={styles.header}>
        <div className={styles.titleArea}>
          <h1 className={styles.glowTitle}>Cosmic Memory</h1>
          <p className={styles.subtitle}>Authoritative Realtime State Synchronization</p>
        </div>
        
        <div className={styles.statusBadge}>
          <span className={`${styles.statusDot} ${connected ? styles.online : styles.offline}`}></span>
          <span className={styles.statusText}>
            {connected ? "Socket Server Connected" : "Connecting to Socket..."}
          </span>
        </div>
      </header>

      <div className={styles.mainLayout}>
        {/* GAME STATS & BOARD PANEL */}
        <section className={styles.gameContainer}>
          <div className={styles.statsRow}>
            <div className={styles.statBox}>
              <span className={styles.statLabel}>Moves</span>
              <span className={styles.statVal}>{gameState.moves}</span>
            </div>
            
            <div className={styles.statBox}>
              <span className={styles.statLabel}>Matches</span>
              <span className={styles.statVal}>{gameState.matches} / 8</span>
            </div>

            <button className={styles.resetButton} onClick={requestReset}>
              Reset Game
            </button>
          </div>

          {gameState.isWon && (
            <div className={styles.victoryCard}>
              <h2>Victory! 🎉</h2>
              <p>You completed the memory grid in {gameState.moves} moves!</p>
              <button className={styles.victoryReset} onClick={requestPlayAgain}>Play Again</button>
            </div>
          )}

          <div className={styles.cardGrid}>
            {gameState.cards.map((card) => (
              <MemoryCard
                key={card.id}
                card={card}
                onClick={() => handleCardClick(card.id)}
              />
            ))}
          </div>
        </section>

        {/* DEVELOPER SIMULATOR PANEL */}
        <aside className={styles.devPanel}>
          <div className={styles.panelHeader}>
            <h3>Developer Control Console</h3>
            <p>Simulate external API commands triggering state events</p>
          </div>

          <div className={styles.consoleSection}>
            <label className={styles.consoleLabel}>Trigger Flip via HTTP API</label>
            <div className={styles.flexGroup}>
              <select
                className={styles.selectInput}
                value={selectedSimId}
                onChange={(e) => setSelectedSimId(Number(e.target.value))}
              >
                {gameState.cards.map((card) => (
                  <option key={card.id} value={card.id}>
                    Card {card.id} {card.matched ? "(Matched)" : card.flipped ? "(Flipped)" : ""}
                  </option>
                ))}
              </select>
              
              <button className={styles.simButton} onClick={handleSimulateApiFlip}>
                Trigger POST /flip
              </button>
            </div>
          </div>

          <div className={styles.consoleSection}>
            <label className={styles.consoleLabel}>Trigger Board Actions via API</label>
            <div className={styles.flexGroup}>
              <button className={`${styles.simButton} ${styles.dangerButton}`} onClick={handleSimulateReset}>
                POST /reset
              </button>
              <button className={`${styles.simButton} ${styles.simButtonActive}`} onClick={handleSimulatePlayAgain}>
                POST /play-again
              </button>
            </div>
          </div>

          <div className={styles.consoleSection}>
            <label className={styles.consoleLabel}>Select & Copy curl Command</label>
            
            <div className={styles.tabBar}>
              <button
                className={`${styles.tabButton} ${activeCurlTab === "flip" ? styles.tabActive : ""}`}
                onClick={() => setActiveCurlTab("flip")}
              >
                Flip
              </button>
              <button
                className={`${styles.tabButton} ${activeCurlTab === "reset" ? styles.tabActive : ""}`}
                onClick={() => setActiveCurlTab("reset")}
              >
                Reset
              </button>
              <button
                className={`${styles.tabButton} ${activeCurlTab === "play-again" ? styles.tabActive : ""}`}
                onClick={() => setActiveCurlTab("play-again")}
              >
                Play Again
              </button>
            </div>

            <div className={styles.curlBox}>
              <code>{getCurlCommand()}</code>
              <button 
                className={styles.copyButton}
                onClick={() => {
                  navigator.clipboard.writeText(getCurlCommand());
                  alert("Curl command copied to clipboard!");
                }}
              >
                Copy
              </button>
            </div>
          </div>

          <div className={styles.consoleSection}>
            <label className={styles.consoleLabel}>Server API Activity Log</label>
            <div className={styles.logBox}>
              {lastApiAction ? (
                <div className={styles.logItem}>
                  <span className={styles.logTimestamp}>
                    [{new Date().toLocaleTimeString()}]
                  </span>{" "}
                  <span className={styles.logSuccess}>SUCCESS</span> — {lastApiAction}
                </div>
              ) : (
                <div className={styles.logPlaceholder}>No HTTP actions triggered yet</div>
              )}
            </div>
          </div>
        </aside>
      </div>
    </div>
  );
};
