"""
Evaluate every position in last_game.pgn with both Stockfish and lc0.
SF: depth 12 (fast).  lc0: 100ms per position.
"""

import sys
import chess
import chess.engine
import chess.pgn

STOCKFISH = "/Users/suvajitmajumder/chess_projects/engines/stockfish/src/stockfish"
LC0       = "/Users/suvajitmajumder/chess_projects/engines/lc0/build/release/lc0"
LC0_NET   = "/Users/suvajitmajumder/chess_projects/engines/lc0/weights/t1-256x10-distilled-swa-2432500.pb.gz"
PGN_PATH  = "/Users/suvajitmajumder/chess_projects/scripts/last_game.pgn"

SF_DEPTH   = 12
LC0_TIME   = 0.1   # seconds per position


def cp_str(score: chess.engine.PovScore) -> str:
    pov = score.white()
    if pov.is_mate():
        return f"M{pov.mate():+d}"
    return f"{pov.score() / 100:+.2f}"


def bar(cp: float, width: int = 32) -> str:
    """ASCII eval bar centred at 0. Clamped to ±4 cp."""
    mid = width // 2
    clamped = max(-4.0, min(4.0, cp))
    offset = int(clamped / 4.0 * mid)
    chars = list("─" * width)
    chars[mid] = "┼"
    if offset > 0:
        for i in range(mid, mid + offset):
            chars[i] = "█"
    elif offset < 0:
        for i in range(mid + offset, mid):
            chars[i] = "░"
    return "".join(chars)


def run():
    with open(PGN_PATH) as f:
        game = chess.pgn.read_game(f)

    moves = list(game.mainline_moves())
    board = game.board()

    print("Starting engines…", flush=True)
    sf  = chess.engine.SimpleEngine.popen_uci(STOCKFISH)
    lc0 = chess.engine.SimpleEngine.popen_uci([LC0, f"--weights={LC0_NET}"])

    print(f"Analysing {len(moves)} half-moves  "
          f"(SF depth {SF_DEPTH} | lc0 {LC0_TIME}s)\n", flush=True)

    hdr = f"{'Move':<8} {'SAN':<8} {'SF cp':>8} {'lc0 cp':>8}   Black░░░░░░┼██████White"
    print(hdr)
    print("─" * len(hdr), flush=True)

    sf_cps  = []
    lc0_cps = []
    labels  = []

    for i, move in enumerate(moves):
        full = (i // 2) + 1
        side = "W" if board.turn == chess.WHITE else "B"
        san  = board.san(move)
        board.push(move)

        # Skip analysis on terminal positions (checkmate / stalemate)
        if board.is_game_over():
            sf_cp  = 10000.0 if board.is_checkmate() else 0.0
            lc0_cp = sf_cp
            sf_str = "M+0" if board.is_checkmate() else "draw"
            lc0_str = sf_str
            sf_cps.append(sf_cp); lc0_cps.append(lc0_cp); labels.append(f"{full}{'.' if side=='W' else '…'}")
            print(f"{f'{full}{dot}':<8} {san:<8} {sf_str:>8} {lc0_str:>8}   {bar(sf_cp)}", flush=True)
            continue

        sf_info   = sf.analyse(board,  chess.engine.Limit(depth=SF_DEPTH))
        lc0_info  = lc0.analyse(board, chess.engine.Limit(time=LC0_TIME))

        sf_sc  = sf_info["score"]
        lc0_sc = lc0_info["score"]

        sf_cp  = sf_sc.white().score(mate_score=10000) / 100
        lc0_cp = lc0_sc.white().score(mate_score=10000) / 100

        sf_cps.append(sf_cp)
        lc0_cps.append(lc0_cp)

        dot   = "." if side == "W" else "…"
        label = f"{full}{dot}"
        labels.append(label)

        print(f"{label:<8} {san:<8} {cp_str(sf_sc):>8} {cp_str(lc0_sc):>8}   {bar(sf_cp)}",
              flush=True)

    sf.quit()
    lc0.quit()

    # ── Summary ──────────────────────────────────────────────────────────
    print("\n" + "─" * 70)
    print(f"{'':30} {'Stockfish':>12} {'lc0':>12}")
    print("─" * 70)
    print(f"{'Peak White advantage (cp)':30} {max(sf_cps):>+12.2f} {max(lc0_cps):>+12.2f}")
    print(f"{'Peak Black advantage (cp)':30} {min(sf_cps):>+12.2f} {min(lc0_cps):>+12.2f}")

    # Biggest swing (single half-move)
    swings = [(abs(sf_cps[i] - sf_cps[i-1]), i) for i in range(1, len(sf_cps))]
    sw_val, sw_idx = max(swings)
    print(f"{'Biggest swing (SF)':30} {sw_val:>+12.2f}  at {labels[sw_idx]}")

    # Agreement between engines (avg |SF - lc0|)
    diffs = [abs(sf_cps[i] - lc0_cps[i]) for i in range(len(sf_cps))]
    print(f"{'Mean |SF−lc0| disagreement':30} {sum(diffs)/len(diffs):>12.2f}")

    # Save
    out = "/Users/suvajitmajumder/chess_projects/scripts/game_evals.txt"
    with open(out, "w") as f:
        f.write(hdr + "\n")
        for i, (label, san, sf_cp, lc0_cp) in enumerate(
                zip(labels, [board2.san(m) for board2, m in
                    zip(_replay_boards(game), moves)],
                    sf_cps, lc0_cps)):
            f.write(f"{label:<8} {san:<8} {sf_cp:>+8.2f} {lc0_cp:>+8.2f}   {bar(sf_cp)}\n")
    print(f"\n[Saved to {out}]")


def _replay_boards(game):
    b = game.board()
    yield b.copy()
    for m in game.mainline_moves():
        b.push(m)
        yield b.copy()


if __name__ == "__main__":
    run()
