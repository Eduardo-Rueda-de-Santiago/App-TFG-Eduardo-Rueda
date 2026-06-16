import { SocketProvider } from "../components/SocketProvider";
import { GameBoard } from "../components/GameBoard";

export default function Home() {
  return (
    <SocketProvider>
      <main style={{ minHeight: "100vh", background: "radial-gradient(ellipse at bottom, #0d1224 0%, #030712 100%)" }}>
        <GameBoard />
      </main>
    </SocketProvider>
  );
}
