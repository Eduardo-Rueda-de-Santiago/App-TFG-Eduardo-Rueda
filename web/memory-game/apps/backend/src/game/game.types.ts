export class Card {
  id!: number;
  value!: string;
  flipped!: boolean;
  matched!: boolean;
}

export class GameState {
  cards!: Card[];
  moves!: number;
  matches!: number;
  isWon!: boolean;
}
