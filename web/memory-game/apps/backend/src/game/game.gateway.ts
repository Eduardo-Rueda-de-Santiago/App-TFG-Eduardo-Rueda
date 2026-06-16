import {
  WebSocketGateway,
  WebSocketServer,
  OnGatewayConnection,
  OnGatewayDisconnect,
} from '@nestjs/websockets';
import { Server, Socket } from 'socket.io';
import { Inject, forwardRef } from '@nestjs/common';
import { GameService } from './game.service';

@WebSocketGateway({
  cors: {
    origin: '*',
  },
})
export class GameGateway implements OnGatewayConnection, OnGatewayDisconnect {
  @WebSocketServer()
  server: Server;

  constructor(
    @Inject(forwardRef(() => GameService))
    private readonly gameService: GameService,
  ) {}

  handleConnection(client: Socket) {
    console.log(`Client connected: ${client.id}`);
    // Send current game state immediately to the newly connected client
    const currentState = this.gameService.getGameState();
    client.emit('game_state', currentState);
  }

  handleDisconnect(client: Socket) {
    console.log(`Client disconnected: ${client.id}`);
  }

  broadcastGameState() {
    if (this.server) {
      const currentState = this.gameService.getGameState();
      this.server.emit('game_state', currentState);
    }
  }

  emitFlip(cardId: number) {
    if (this.server) {
      this.server.emit('flip_card', { cardId });
    }
  }

  emitReset() {
    if (this.server) {
      this.server.emit('reset_game', this.gameService.getGameState());
    }
  }
}
