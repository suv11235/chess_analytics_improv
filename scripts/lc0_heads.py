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

import json
import queue
import re
import subprocess
import threading
import chess
import chess.pgn
import pandas as pd
from pathlib import Path

ROOT     = Path(__file__).resolve().parent.parent
LC0      = ROOT / "engines/lc0/build/release/lc0"
LC0_NET  = ROOT / "engines/lc0/weights/t3-512x15x16h-distill-swa-2767500.pb.gz"
PGN_PATH = Path(__file__).resolve().parent / "last_game.pgn"
ANALYSIS_DIR = ROOT / "lc0" / "analysis"

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

    def _drain_until(self, sentinel: str, timeout: float = 120.0) -> list[str]:
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

    def query(self, fen: str, nodes: int | None = None) -> dict:
        """
        Run lc0 on *fen* for `nodes` nodes (default: module-level NODES) and return:
          {
            'wdl':      (W, D, L),          # value head – floats in [0,1] summing to 1
            'policy':   [(move_uci, P)],    # policy head priors – sorted desc
            'q':        float | None,       # root Q (WL) from NN value head
            'bestmove': str | None,         # UCI best move chosen by search
          }
        """
        if nodes is None:
            nodes = NODES
        self._send(f"position fen {fen}")
        self._send(f"go nodes {nodes}")
        raw_lines = self._drain_until("bestmove")

        wdl = None
        q_root = None
        bestmove = None
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

            # bestmove line: "bestmove e2e4 ..."
            if line.startswith("bestmove "):
                parts = line.split()
                if len(parts) >= 2:
                    bestmove = parts[1]

        policy.sort(key=lambda x: x[1], reverse=True)
        return {"wdl": wdl, "policy": policy, "q": q_root, "bestmove": bestmove}

    def close(self):
        self._send("quit")
        self.proc.wait()


# ---------------------------------------------------------------------------
# Game analysis
# ---------------------------------------------------------------------------

def analyze_game(pgn_path: str, nodes: int = 10, max_ply: int = None):
    """Analyze each position in a PGN game with lc0's value and policy heads.

    Args:
        pgn_path: Path to a PGN file (reads the first game).
        nodes: Nodes per position (low = closer to raw NN output).
        max_ply: If set, stop analysis after this many half-moves.

    Returns:
        Dict with "metadata" and "analysis" keys.
    """
    with open(pgn_path) as f:
        game = chess.pgn.read_game(f)

    metadata = {key: game.headers.get(key, "") for key in game.headers}

    board = game.board()
    engine = Lc0UCI()
    analysis = []

    for ply_idx, move in enumerate(game.mainline_moves()):
        if max_ply is not None and ply_idx >= max_ply:
            break

        fen = board.fen()
        side = "white" if board.turn == chess.WHITE else "black"
        move_number = board.fullmove_number
        move_uci = move.uci()

        result = engine.query(fen)

        # Store WDL always from White's perspective.
        # lc0's WDL is from the side-to-move's perspective, so flip for Black.
        wdl = result["wdl"]
        if wdl and side == "black":
            wdl = (wdl[2], wdl[1], wdl[0])  # swap W and L

        q = result["q"]
        if q is not None and side == "black":
            q = -q

        # Policy priors are always for the side to move — no flip needed,
        # they're just probabilities over that side's legal moves.
        policy = [{"move": mv, "prior": p} for mv, p in result["policy"]]

        analysis.append({
            "ply": ply_idx + 1,
            "move_number": move_number,
            "side": side,
            "fen": fen,
            "move_played": move_uci,
            "value_head": {
                "win": wdl[0] if wdl else None,
                "draw": wdl[1] if wdl else None,
                "loss": wdl[2] if wdl else None,
                "q": q,
            },
            "policy_head": policy,
        })

        san = board.san(move)
        q_str = f"  Q={q:+.3f}" if q is not None else ""
        print(f"  Analyzed {move_number}{'.' if side == 'white' else '...'}"
              f"{san}{q_str}")

        board.push(move)

    engine.close()
    return {"metadata": metadata, "analysis": analysis}


# ---------------------------------------------------------------------------
# Output: JSON + DataFrames
# ---------------------------------------------------------------------------

def results_to_dataframes(result: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Convert analysis results into two DataFrames.

    Returns:
        value_df: One row per position (ply, move_number, side, fen,
                  move_played, win, draw, loss, q).
        policy_df: One row per candidate move per position (ply, move_number,
                   side, fen, move_played, move, prior).
    """
    value_rows = []
    policy_rows = []

    for entry in result["analysis"]:
        base = {
            "ply": entry["ply"],
            "move_number": entry["move_number"],
            "side": entry["side"],
            "fen": entry["fen"],
            "move_played": entry["move_played"],
        }

        vh = entry["value_head"]
        value_rows.append({**base, **vh})

        for p in entry["policy_head"]:
            policy_rows.append({**base, "move": p["move"], "prior": p["prior"]})

    return pd.DataFrame(value_rows), pd.DataFrame(policy_rows)


def save_results(result: dict, name: str):
    """Save analysis as JSON and Parquet files in lc0/analysis/."""
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

    json_path = ANALYSIS_DIR / f"{name}.json"
    with open(json_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Saved JSON:           {json_path}")

    value_df, policy_df = results_to_dataframes(result)

    value_path = ANALYSIS_DIR / f"{name}_value.parquet"
    value_df.to_parquet(value_path, index=False)
    print(f"Saved value Parquet:  {value_path}")

    policy_path = ANALYSIS_DIR / f"{name}_policy.parquet"
    policy_df.to_parquet(policy_path, index=False)
    print(f"Saved policy Parquet: {policy_path}")

    return value_df, policy_df


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

def bar_wdl(w: float, d: float, l: float, width: int = 30) -> str:
    """Compact horizontal bar:  ███W░░░D───L───"""
    wb = round(w * width)
    db = round(d * width)
    lb = width - wb - db
    lb = max(lb, 0)
    return "\u2588" * wb + "\u2591" * db + "\u2500" * lb


def policy_bar(p: float, max_p: float, width: int = 20) -> str:
    filled = round((p / max_p) * width) if max_p > 0 else 0
    return "\u2593" * filled + "\u00b7" * (width - filled)


def print_heads(board: chess.Board, result: dict, label: str):
    wdl    = result["wdl"]
    policy = result["policy"]
    q_root = result.get("q")

    print(f"\n{'\u2500' * 62}")
    print(f"  {label}")
    print(f"  FEN: {board.fen()}")
    print(f"{'\u2500' * 62}")

    # Value head
    if wdl:
        w, d, l = wdl
        q_str = f"  Q(WL) = {q_root:+.4f}" if q_root is not None else ""
        print(f"\n  VALUE HEAD  (NN WDL, White's perspective){q_str}")
        print(f"    Win  {w*100:6.2f}%  Draw  {d*100:6.2f}%  Loss  {l*100:6.2f}%")
        print(f"    {bar_wdl(w, d, l)}  W/D/L")
    else:
        print("\n  VALUE HEAD  \u2014 no WDL output (position may be terminal)")

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
        print(f"  {'\u2026rest':<8}  {'':8}  {(1.0 - total_shown)*100:6.2f}%")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run():
    with open(PGN_PATH) as f:
        game = chess.pgn.read_game(f)

    moves = list(game.mainline_moves())
    board = game.board()

    print("Connecting to lc0 \u2026")
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
    print(f"\n{'\u2500' * 60}")
    print("Done.")


if __name__ == "__main__":
    import sys

    if "--analyze" in sys.argv:
        # Full game analysis with structured output
        pgn = str(PGN_PATH)
        max_ply = 20  # first 10 moves per side by default

        for i, arg in enumerate(sys.argv):
            if arg == "--pgn" and i + 1 < len(sys.argv):
                pgn = sys.argv[i + 1]
            if arg == "--max-ply" and i + 1 < len(sys.argv):
                max_ply = int(sys.argv[i + 1])

        print(f"Analyzing {pgn} (max_ply={max_ply})...\n")
        result = analyze_game(pgn, max_ply=max_ply)

        # Derive a name from PGN metadata
        meta = result["metadata"]
        white = meta.get("White", "unknown").split(",")[0].lower().replace(" ", "_")
        black = meta.get("Black", "unknown").split(",")[0].lower().replace(" ", "_")
        name = f"{white}_vs_{black}_lc0"

        print()
        value_df, policy_df = save_results(result, name)
        print(f"\nValue head DataFrame: {value_df.shape}")
        print(value_df.to_string())
        print(f"\nPolicy head DataFrame: {policy_df.shape}")
        print(policy_df.head(30).to_string())
    else:
        # Original display-only mode
        run()
