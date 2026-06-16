"use client";

import React from "react";
import { Card } from "../types/game";
import styles from "./MemoryCard.module.css";

interface MemoryCardProps {
  card: Card;
  onClick: () => void;
}

export const MemoryCard: React.FC<MemoryCardProps> = ({ card, onClick }) => {
  const isFlippedOrMatched = card.flipped || card.matched;

  return (
    <div className={styles.cardContainer} onClick={onClick}>
      <div
        className={`${styles.cardInner} ${
          isFlippedOrMatched ? styles.flipped : ""
        } ${card.matched ? styles.matched : ""}`}
      >
        {/* CARD BACK (Hidden when flipped) */}
        <div className={styles.cardBack}>
          <div className={styles.innerBackPattern}>
            <span>?</span>
          </div>
        </div>

        {/* CARD FRONT (Shown when flipped) */}
        <div className={`${styles.cardFront} ${card.matched ? styles.glowMatched : ""}`}>
          <span className={styles.cardEmoji}>{card.value}</span>
        </div>
      </div>
    </div>
  );
};
