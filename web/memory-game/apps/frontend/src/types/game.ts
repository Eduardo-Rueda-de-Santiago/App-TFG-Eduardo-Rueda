export interface Card {
  id: number;
  value: string;
  flipped: boolean;
  matched: boolean;
}

export interface GameState {
  cards: Card[];
  moves: number;
  matches: number;
  isWon: boolean;
}
