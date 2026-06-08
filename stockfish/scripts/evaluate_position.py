"""Evaluate a chess position with Stockfish and get the top move(s)."""

import subprocess
import os

STOCKFISH_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "engines", "stockfish", "src", "stockfish"
)


def evaluate(fen: str, depth: int = 20, multipv: int = 1):
    """Return Stockfish's evaluation and best move(s) for a FEN position.

    When multipv=1, returns (eval_score, best_move).
    When multipv>1, returns a list of (eval_score, best_move, pv) dicts,
    ranked from best to worst.
    """
    proc = subprocess.Popen(
        [STOCKFISH_PATH],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
    )

    proc.stdin.write("uci\n")
    proc.stdin.write("isready\n")
    if multipv > 1:
        proc.stdin.write(f"setoption name MultiPV value {multipv}\n")
    proc.stdin.write(f"position fen {fen}\n")
    proc.stdin.write(f"go depth {depth}\n")
    proc.stdin.flush()

    # For MultiPV, we collect the last info line for each PV rank.
    # Stockfish outputs one info line per PV at each depth, so by the final
    # depth all slots will hold the deepest evaluation.
    lines_by_pv = {}
    best_move = None

    for line in proc.stdout:
        line = line.strip()
        if line.startswith("info") and "score" in line:
            parts = line.split()
            # Determine which PV rank this line belongs to
            pv_rank = 1
            if "multipv" in parts:
                pv_rank = int(parts[parts.index("multipv") + 1])
            lines_by_pv[pv_rank] = parts
        if line.startswith("bestmove"):
            best_move = line.split()[1]
            break

    proc.stdin.close()
    proc.stdout.close()
    proc.wait()

    def parse_score(parts):
        if "cp" in parts:
            return int(parts[parts.index("cp") + 1]) / 100
        elif "mate" in parts:
            return f"mate in {parts[parts.index('mate') + 1]}"
        return None

    def parse_pv(parts):
        if "pv" in parts:
            return parts[parts.index("pv") + 1:]
        return []

    if multipv == 1:
        score = parse_score(lines_by_pv.get(1, []))
        return score, best_move

    results = []
    for rank in sorted(lines_by_pv.keys()):
        parts = lines_by_pv[rank]
        pv_moves = parse_pv(parts)
        results.append({
            "rank": rank,
            "score": parse_score(parts),
            "move": pv_moves[0] if pv_moves else None,
            "pv": pv_moves,
        })
    return results


if __name__ == "__main__":
    # Italian Game — top 3 moves for Black
    fen = "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 3 3"
    print("Position: Italian Game (1.e4 e5 2.Nf3 Nc6 3.Bc4), Black to move")
    print(f"FEN: {fen}\n")

    results = evaluate(fen, depth=18, multipv=3)
    for r in results:
        score = r["score"]
        score_str = f"{score:+.2f}" if isinstance(score, float) else str(score)
        pv_str = " ".join(r["pv"][:6])  # first 6 moves of the line
        print(f"  #{r['rank']}  {r['move']}  (eval: {score_str})  line: {pv_str}")
