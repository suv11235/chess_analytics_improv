"""
Arbiter script: Stockfish (White) vs lc0 (Black)
Drives both engines over UCI via python-chess and prints the game.
"""

import sys
import chess
import chess.engine
import chess.pgn
import datetime

STOCKFISH = "/Users/suvajitmajumder/chess_projects/engines/stockfish/src/stockfish"
LC0       = "/Users/suvajitmajumder/chess_projects/engines/lc0/build/release/lc0"
LC0_NET   = "/Users/suvajitmajumder/chess_projects/engines/lc0/weights/t1-256x10-distilled-swa-2432500.pb.gz"

# Time per move (seconds)
MOVETIME  = 0.5

def fmt_score(info):
    sc = info.get("score")
    if sc is None:
        return "?"
    pov = sc.white()
    if pov.is_mate():
        return f"M{pov.mate()}"
    cp = pov.score()
    return f"{'+' if cp >= 0 else ''}{cp/100:.2f}"

def run():
    board = chess.Board()
    game  = chess.pgn.Game()
    game.headers["Event"]  = "Stockfish vs lc0"
    game.headers["White"]  = "Stockfish"
    game.headers["Black"]  = "Lc0"
    game.headers["Date"]   = datetime.date.today().isoformat()
    node = game

    print("Starting engines…")
    sf  = chess.engine.SimpleEngine.popen_uci(STOCKFISH)
    lc0 = chess.engine.SimpleEngine.popen_uci(
        [LC0, f"--weights={LC0_NET}"]
    )
    lc0.configure({"Threads": 1})

    print(f"\n{'─'*54}")
    print(f"  Stockfish (White) vs lc0 (Black)  |  {MOVETIME}s/move")
    print(f"{'─'*54}\n")
    print(board, "\n")

    try:
        move_num = 1
        while not board.is_game_over():
            is_white = board.turn == chess.WHITE
            engine   = sf if is_white else lc0
            label    = "Stockfish" if is_white else "lc0      "

            result = engine.play(
                board,
                chess.engine.Limit(time=MOVETIME),
                info=chess.engine.INFO_SCORE | chess.engine.INFO_PV,
            )
            move  = result.move
            score = fmt_score(result.info)
            pv    = " ".join(m.uci() for m in result.info.get("pv", [])[:4])

            prefix = f"{move_num:>3}." if is_white else "   "
            print(f"{prefix} {label}  {board.san(move):<8}  score {score:>7}  pv {pv}")

            board.push(move)
            node = node.add_variation(move)

            if not is_white:
                move_num += 1

    except KeyboardInterrupt:
        print("\n[interrupted]")
    finally:
        sf.quit()
        lc0.quit()

    outcome = board.outcome()
    result_str = board.result()
    game.headers["Result"] = result_str

    print(f"\n{'─'*54}")
    if outcome:
        if outcome.winner == chess.WHITE:
            print("  Result: Stockfish (White) wins")
        elif outcome.winner == chess.BLACK:
            print("  Result: lc0 (Black) wins")
        else:
            print(f"  Result: Draw ({outcome.termination.name})")
    print(f"  {result_str}")
    print(f"{'─'*54}\n")

    # Print PGN
    exporter = chess.pgn.StringExporter(headers=True, variations=False)
    pgn_str  = game.accept(exporter)
    print(pgn_str)

    # Save PGN
    pgn_path = "/Users/suvajitmajumder/chess_projects/scripts/last_game.pgn"
    with open(pgn_path, "w") as f:
        f.write(pgn_str + "\n")
    print(f"\n[PGN saved to {pgn_path}]")

if __name__ == "__main__":
    run()
