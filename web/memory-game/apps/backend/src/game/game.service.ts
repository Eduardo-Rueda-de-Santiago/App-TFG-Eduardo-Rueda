import { Injectable, Inject, forwardRef } from '@nestjs/common';
import { GameState, Card } from './game.types';
import { GameGateway } from './game.gateway';

@Injectable()
export class GameService {
  private gameState: GameState;
  private isProcessing = false; // Input lock to prevent extra flips during mismatches

  private readonly cardValues = ['🚀', '🛸', '🛰️', '🌌', '🪐', '🌠', '☄️', '🛡️'];

  constructor(
    @Inject(forwardRef(() => GameGateway))
    private readonly gateway: GameGateway,
  ) {
    this.resetGame();
  }

  getGameState(): GameState {
    return this.gameState;
  }

  resetGame(): GameState {
    this.isProcessing = false;
    
    // Create 8 pairs of cards (16 total)
    const pairs = [...this.cardValues, ...this.cardValues];
    
    // Shuffle using Fisher-Yates algorithm
    for (let i = pairs.length - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1));
      [pairs[i], pairs[j]] = [pairs[j], pairs[i]];
    }

    const cards: Card[] = pairs.map((val, idx) => ({
      id: idx + 1,
      value: val,
      flipped: false,
      matched: false,
    }));

    this.gameState = {
      cards,
      moves: 0,
      matches: 0,
      isWon: false,
    };

    if (this.gateway) {
      this.gateway.emitReset();
    }

    return this.gameState;
  }

  flipCard(cardId: number): { success: boolean; message?: string } {
    // If we are currently locked (waiting to auto-unflip a mismatched pair), reject flip
    if (this.isProcessing) {
      return { success: false, message: 'Game is currently processing a match.' };
    }

    const card = this.gameState.cards.find(c => c.id === cardId);
    if (!card) {
      return { success: false, message: 'Card not found.' };
    }

    if (card.flipped || card.matched) {
      return { success: false, message: 'Card is already flipped or matched.' };
    }

    // Flip the card
    card.flipped = true;
    this.gateway.emitFlip(cardId);

    // Find all currently flipped cards that are not yet matched
    const flippedUnmatched = this.gameState.cards.filter(
      c => c.flipped && !c.matched,
    );

    if (flippedUnmatched.length === 2) {
      this.gameState.moves++;
      const [card1, card2] = flippedUnmatched;

      if (card1.value === card2.value) {
        // MATCH DETECTED!
        card1.matched = true;
        card2.matched = true;
        this.gameState.matches++;

        // Check if won
        const allMatched = this.gameState.cards.every(c => c.matched);
        if (allMatched) {
          this.gameState.isWon = true;
        }

        // Broadcast updated state immediately
        this.gateway.broadcastGameState();
      } else {
        // MISMATCH DETECTED!
        // Lock input and unflip after 1.5s delay
        this.isProcessing = true;

        setTimeout(() => {
          card1.flipped = false;
          card2.flipped = false;
          this.isProcessing = false;
          
          // Broadcast updated state with unflipped cards
          this.gateway.broadcastGameState();
        }, 1200); // 1.2s delay for perfect visual pacing
      }
    } else if (flippedUnmatched.length > 2) {
      // Safety reset: if for some reason more than 2 are flipped, force them shut
      this.gameState.cards.forEach(c => {
        if (!c.matched) c.flipped = false;
      });
      card.flipped = true;
      this.gateway.broadcastGameState();
    }

    return { success: true };
  }
}
