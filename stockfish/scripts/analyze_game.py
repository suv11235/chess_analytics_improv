"""Analyze a PGN game move-by-move with Stockfish."""

import chess
import chess.pgn
import json
import os
import pandas as pd
from evaluate_position import evaluate

ANALYSIS_DIR = os.path.join(os.path.dirname(__file__), "..", "analysis")


def analyze_game(pgn_path: str, depth: int = 18, multipv: int = 3, max_ply: int = None):
    """Analyze each position in a PGN game and return structured results.

    Args:
        pgn_path: Path to a PGN file (reads the first game).
        depth: Stockfish search depth per position.
        multipv: Number of top moves to report per position.
        max_ply: If set, stop analysis after this many half-moves.

    Returns:
        Dict with "metadata" and "analysis" keys.
    """
    with open(pgn_path) as f:
        game = chess.pgn.read_game(f)

    # Extract metadata from PGN headers
    metadata = {key: game.headers.get(key, "") for key in game.headers}

    # Replay moves and analyze each position
    board = game.board()
    analysis = []

    for ply_idx, move in enumerate(game.mainline_moves()):
        if max_ply is not None and ply_idx >= max_ply:
            break

        fen = board.fen()
        side = "white" if board.turn == chess.WHITE else "black"
        move_number = board.fullmove_number

        # Get the move in UCI notation before pushing
        move_uci = move.uci()

        # Evaluate the position
        results = evaluate(fen, depth=depth, multipv=multipv)

        # Flip scores for Black's moves so everything is from White's perspective
        if side == "black":
            for r in results:
                if isinstance(r["score"], (int, float)):
                    r["score"] = -r["score"]
                elif isinstance(r["score"], str) and r["score"].startswith("mate in "):
                    mate_val = int(r["score"].split()[-1])
                    r["score"] = f"mate in {-mate_val}"

        analysis.append({
            "ply": ply_idx + 1,
            "move_number": move_number,
            "side": side,
            "fen": fen,
            "move_played": move_uci,
            "top_moves": results,
        })

        print(f"  Analyzed move {move_number}{'.' if side == 'white' else '...'} "
              f"{board.san(move)}")

        board.push(move)

    return {"metadata": metadata, "analysis": analysis}


def result_to_dataframe(result: dict) -> pd.DataFrame:
    """Flatten analysis results into a DataFrame with one row per candidate move.

    Columns: ply, move_number, side, fen, move_played, rank, move, score, pv
    """
    rows = []
    for entry in result["analysis"]:
        for top in entry["top_moves"]:
            rows.append({
                "ply": entry["ply"],
                "move_number": entry["move_number"],
                "side": entry["side"],
                "fen": entry["fen"],
                "move_played": entry["move_played"],
                "rank": top["rank"],
                "move": top["move"],
                "score": top["score"],
                "pv": " ".join(top["pv"]),
            })
    return pd.DataFrame(rows)


def save_results(result: dict, name: str):
    """Save analysis as both JSON and a Parquet DataFrame in stockfish/analysis/."""
    os.makedirs(ANALYSIS_DIR, exist_ok=True)

    json_path = os.path.join(ANALYSIS_DIR, f"{name}.json")
    with open(json_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Saved JSON:      {json_path}")

    df = result_to_dataframe(result)
    parquet_path = os.path.join(ANALYSIS_DIR, f"{name}.parquet")
    df.to_parquet(parquet_path, index=False)
    print(f"Saved Parquet:   {parquet_path}")

    return df


if __name__ == "__main__":
    pgn_path = "/Users/davidpepper/Dropbox/David/Personal/Chess/Neimann/Hans/SinqTest.pgn"

    print("Analyzing first 10 moves per side (20 plies)...\n")
    result = analyze_game(pgn_path, depth=18, multipv=3, max_ply=20)

    print()
    df = save_results(result, "mamedyarov_so_sinquefield_2022")

    print(f"\nDataFrame shape: {df.shape}")
    print(f"\n{df.to_string()}")
