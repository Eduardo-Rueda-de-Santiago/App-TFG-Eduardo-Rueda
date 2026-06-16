import { Controller, Get, Post, Body, HttpCode, HttpStatus } from '@nestjs/common';
import { GameService } from './game.service';
import { GameState } from './game.types';

@Controller('game')
export class GameController {
  constructor(private readonly gameService: GameService) {}

  @Get('state')
  getGameState(): GameState {
    return this.gameService.getGameState();
  }

  @Post('flip')
  @HttpCode(HttpStatus.OK)
  flipCard(@Body() body: { cardId: number }): { success: boolean; message?: string } {
    return this.gameService.flipCard(body.cardId);
  }

  @Post('reset')
  @HttpCode(HttpStatus.OK)
  resetGame(): GameState {
    return this.gameService.resetGame();
  }

  @Post('play-again')
  @HttpCode(HttpStatus.OK)
  playAgain(): GameState {
    return this.gameService.resetGame();
  }
}
