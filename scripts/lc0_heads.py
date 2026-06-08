"""
lc0_heads.py — Extract raw value-head and policy-head outputs from Lc0.

Value head  → WDL (Win / Draw / Loss) probabilities from the neural network.
            Taken from the last "info depth … wdl W D L" search-info line, which
            reflects the NN evaluation with minimal MCTS influence at low node counts.

Policy head → Per-move prior probabilities P(move) from the neural network.
            Taken from the "info string MOVE (NNN) … (P: X.XX%) …" verbose lines
            that lc0 emits after search when VerboseMoveStats=true.

Uses nodes=10 so we get the full policy prior table while keeping MCTS influence
minimal. The P values printed by lc0 are always the raw NN policy priors.
"""

import queue
import re
import subprocess
import threading
import chess
import chess.pgn
from pathlib import Path

ROOT     = Path(__file__).resolve().parent.parent
LC0      = ROOT / "engines/lc0/build/release/lc0"
LC0_NET  = ROOT / "engines/lc0/weights/t1-256x10-distilled-swa-2432500.pb.gz"
PGN_PATH = Path(__file__).resolve().parent / "last_game.pgn"

TOP_N_POLICY = 10  # how many top moves to show from the policy head
NODES = 10         # nodes per position – enough to populate full policy table

# ---------------------------------------------------------------------------
# Low-level UCI helper (uses a background reader thread to avoid deadlocks)
# ---------------------------------------------------------------------------

class Lc0UCI:
    def __init__(self):
        self.proc = subprocess.Popen(
            [LC0, f"--weights={LC0_NET}"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            bufsize=1,
        )
        self._q: queue.Queue[str | None] = queue.Queue()
        self._reader = threading.Thread(target=self._read_loop, daemon=True)
        self._reader.start()

        self._send("uci")
        self._drain_until("uciok")
        self._send("setoption name VerboseMoveStats value true")
        self._send("setoption name UCI_ShowWDL value true")
        self._send("setoption name Threads value 1")
        self._send("isready")
        self._drain_until("readyok")

    def _read_loop(self):
        for line in self.proc.stdout:
            self._q.put(line.rstrip("\n"))
        self._q.put(None)

    def _send(self, cmd: str):
        self.proc.stdin.write(cmd + "\n")
        self.proc.stdin.flush()

    def _drain_until(self, sentinel: str, timeout: float = 15.0) -> list[str]:
        import time
        deadline = time.monotonic() + timeout
        lines = []
        while time.monotonic() < deadline:
            try:
                line = self._q.get(timeout=0.5)
            except queue.Empty:
                continue
            if line is None:
                break
            lines.append(line)
            if sentinel in line:
                return lines
        return lines

    def query(self, fen: str) -> dict:
        """
        Run lc0 on *fen* for NODES nodes and return:
          {
            'wdl':    (W, D, L),          # value head – floats in [0,1] summing to 1
            'policy': [(move_uci, P)],    # policy head priors – sorted desc
            'q':      float | None,       # root Q (WL) from NN value head
          }
        """
        self._send(f"position fen {fen}")
        self._send(f"go nodes {NODES}")
        raw_lines = self._drain_until("bestmove")

        wdl = None
        q_root = None
        policy: list[tuple[str, float]] = []

        for line in raw_lines:
            # Value head WDL: "info depth N … wdl W D L …"
            # Example: info depth 3 seldepth 5 time 7056 nodes 10 score cp -18 wdl 168 582 250 …
            if "wdl" in line and "info depth" in line:
                m = re.search(r"\bwdl\s+(\d+)\s+(\d+)\s+(\d+)", line)
                if m:
                    w, d, l = int(m.group(1)), int(m.group(2)), int(m.group(3))
                    total = w + d + l
                    if total > 0:
                        wdl = (w / total, d / total, l / total)

            # Root Q (value head W-L for root): "info string node … (WL: X.XXXXX) …"
            if "info string node" in line:
                m = re.search(r"\(WL:\s*([+-]?[\d.]+)\)", line)
                if m:
                    q_root = float(m.group(1))

            # Policy head: "info string MOVE  (NNN ) … (P: X.XX%) …"
            # Example: info string e7e5  (322 ) N:       5 (+ 0) (P: 30.93%) …
            if "info string" in line and "info string node" not in line:
                m = re.match(
                    r"info string\s+([a-h][1-8][a-h][1-8][qrbn]?)\s+\(\s*\d+\s*\).*\(P:\s*([\d.]+)%\)",
                    line,
                )
                if m:
                    policy.append((m.group(1), float(m.group(2)) / 100.0))

        policy.sort(key=lambda x: x[1], reverse=True)
        return {"wdl": wdl, "policy": policy, "q": q_root}

    def close(self):
        self._send("quit")
        self.proc.wait()

    def close(self):
        self._send("quit")
        self.proc.wait()


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

def bar_wdl(w: float, d: float, l: float, width: int = 30) -> str:
    """Compact horizontal bar:  ███W░░░D───L───"""
    wb = round(w * width)
    db = round(d * width)
    lb = width - wb - db
    lb = max(lb, 0)
    return "█" * wb + "░" * db + "─" * lb


def policy_bar(p: float, max_p: float, width: int = 20) -> str:
    filled = round((p / max_p) * width) if max_p > 0 else 0
    return "▓" * filled + "·" * (width - filled)


def print_heads(board: chess.Board, result: dict, label: str):
    wdl    = result["wdl"]
    policy = result["policy"]
    q_root = result.get("q")

    print(f"\n{'─' * 62}")
    print(f"  {label}")
    print(f"  FEN: {board.fen()}")
    print(f"{'─' * 62}")

    # Value head
    if wdl:
        w, d, l = wdl
        q_str = f"  Q(WL) = {q_root:+.4f}" if q_root is not None else ""
        print(f"\n  VALUE HEAD  (NN WDL, White's perspective){q_str}")
        print(f"    Win  {w*100:6.2f}%  Draw  {d*100:6.2f}%  Loss  {l*100:6.2f}%")
        print(f"    {bar_wdl(w, d, l)}  W/D/L")
    else:
        print("\n  VALUE HEAD  — no WDL output (position may be terminal)")

    # Policy head
    legal_ucis = {mv.uci() for mv in board.legal_moves}
    shown = [(mv, p) for mv, p in policy if mv in legal_ucis][:TOP_N_POLICY]
    total_shown = sum(p for _, p in shown)
    rest_p = 1.0 - sum(p for mv, p in policy if mv in legal_ucis)

    print(f"\n  POLICY HEAD  (top {len(shown)} of {len(policy)} moves, raw NN priors)")
    print(f"  {'Move':<8}  {'SAN':<8}  {'Prior':>7}  {'':20}  cumul")

    cumul = 0.0
    max_p = shown[0][1] if shown else 1.0
    for uci, p in shown:
        try:
            san = board.san(chess.Move.from_uci(uci))
        except Exception:
            san = uci
        cumul += p
        print(
            f"  {uci:<8}  {san:<8}  {p*100:6.2f}%  "
            f"{policy_bar(p, max_p)}  {cumul*100:.1f}%"
        )

    if len(policy) > TOP_N_POLICY or rest_p > 0.001:
        print(f"  {'…rest':<8}  {'':8}  {(1.0 - total_shown)*100:6.2f}%")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run():
    with open(PGN_PATH) as f:
        game = chess.pgn.read_game(f)

    moves = list(game.mainline_moves())
    board = game.board()

    print("Connecting to lc0 …")
    engine = Lc0UCI()
    print("Ready.\n")
    print(f"Game: {game.headers.get('White','?')} vs {game.headers.get('Black','?')}")
    print(f"Analysing {len(moves)} positions at nodes={NODES} (raw NN output)\n")

    # Sample every 5th half-move to keep runtime reasonable; always include move 1
    sample_indices = sorted(set([0] + list(range(4, len(moves), 5))))

    for i, move in enumerate(moves):
        if i in sample_indices:
            full  = (i // 2) + 1
            side  = "W" if board.turn == chess.WHITE else "B"
            san   = board.san(move)
            label = f"After {full}{'.' if side == 'W' else '...'}{san}"
            board.push(move)
            if board.is_checkmate() or board.is_game_over():
                print(f"\n  [skipping terminal position at {label}]")
                continue
            result = engine.query(board.fen())
            print_heads(board, result, label)
        else:
            board.push(move)

    engine.close()
    print(f"\n{'─' * 60}")
    print("Done.")


if __name__ == "__main__":
    run()
