#!/usr/bin/env python3
"""
surprise_moves.py — Find positions where the objectively best move is
"hard to find": deep search says it's best, but lc0's policy head
(a proxy for human intuition) would not predict it.

Analyses 10 games from the Sinquefield Cup 2022.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import chess
import chess.pgn
import pandas as pd

# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PGN = Path("/Users/davidpepper/Dropbox/David/Personal/Chess/Neimann/Hans/sinqcup22.pgn")
ANALYSIS_DIR = ROOT / "analysis"

# Engine wrappers (same directory)
sys.path.insert(0, str(Path(__file__).resolve().parent))
from stockfish_eval import StockfishEvaluator
from lc0_heads import Lc0UCI


# ---------------------------------------------------------------------------
# Surprise metric
# ---------------------------------------------------------------------------

def compute_surprise_metrics(
    policy: list[tuple[str, float]], move_uci: str
) -> dict:
    """Given lc0's sorted policy list and a move, return surprise metrics.

    Returns:
        {
            'policy_prior': float,   # raw NN prior for the move
            'policy_rank':  int,     # 1-based rank in the policy list
            'surprise':     float,   # log2(top_prior / move_prior); 0 if move is top
        }
    """
    if not policy:
        return {"policy_prior": 0.0, "policy_rank": None, "surprise": None}

    top_prior = policy[0][1]  # policy is already sorted desc

    move_prior = 0.0
    move_rank = None
    for rank, (mv, p) in enumerate(policy, start=1):
        if mv == move_uci:
            move_prior = p
            move_rank = rank
            break

    if move_rank is None or move_prior <= 0:
        # Move not found in policy list — maximally surprising
        return {
            "policy_prior": 0.0,
            "policy_rank": len(policy) + 1,
            "surprise": None,  # undefined (would be +inf)
        }

    surprise = math.log2(top_prior / move_prior) if move_prior > 0 else None
    return {
        "policy_prior": move_prior,
        "policy_rank": move_rank,
        "surprise": surprise,
    }


# ---------------------------------------------------------------------------
# Main analysis loop
# ---------------------------------------------------------------------------

def analyze_games(
    pgn_path: Path,
    num_games: int = 10,
    sf_depth: int = 18,
    lc0_nodes: int = 100,
    start_game: int = 0,
    skip_ply: int = 20,
) -> pd.DataFrame:
    """Analyze positions from multiple games with both engines.

    For each position: run Stockfish (depth) and lc0 (nodes), compute
    surprise metrics for SF's best move, lc0's best move, and the move
    actually played.

    Args:
        start_game: 0-based index of the first game to analyze (skip earlier games).
        skip_ply: Skip the first N half-moves (opening book moves). Default 20
                  (10 moves per side).

    Returns a DataFrame with one row per position.
    """
    sf = StockfishEvaluator()
    lc0 = Lc0UCI()

    rows: list[dict] = []

    with open(pgn_path) as f:
        # Skip games before start_game
        for _ in range(start_game):
            skipped = chess.pgn.read_game(f)
            if skipped is None:
                break

        for game_idx in range(num_games):
            game = chess.pgn.read_game(f)
            if game is None:
                print(f"  Only {game_idx} games found after offset, stopping.")
                break

            abs_idx = start_game + game_idx
            white = game.headers.get("White", "?")
            black = game.headers.get("Black", "?")
            game_id = f"{abs_idx + 1}_{white}_vs_{black}"
            print(f"\n{'=' * 60}")
            print(f"Game {abs_idx + 1}: {white} vs {black}")
            print(f"{'=' * 60}")

            board = game.board()
            moves = list(game.mainline_moves())

            for ply_idx, move in enumerate(moves):
                if ply_idx < skip_ply:
                    board.push(move)
                    continue

                fen = board.fen()
                side = "white" if board.turn == chess.WHITE else "black"
                move_number = board.fullmove_number
                move_uci = move.uci()
                move_san = board.san(move)

                # --- Stockfish ---
                sf_eval = sf.evaluate(fen=fen, depth=sf_depth)
                sf_best = sf_eval.pv[0] if sf_eval.pv else None
                sf_score = sf_eval.score_cp
                if sf_score is None and sf_eval.mate_in is not None:
                    sf_score = 10000 * (1 if sf_eval.mate_in > 0 else -1)

                # --- lc0 ---
                lc0_result = lc0.query(fen, nodes=lc0_nodes)
                lc0_policy = lc0_result["policy"]
                lc0_best = lc0_result["bestmove"]
                lc0_q = lc0_result["q"]

                # --- Surprise metrics ---
                sf_metrics = compute_surprise_metrics(lc0_policy, sf_best) if sf_best else {
                    "policy_prior": None, "policy_rank": None, "surprise": None
                }
                lc0_metrics = compute_surprise_metrics(lc0_policy, lc0_best) if lc0_best else {
                    "policy_prior": None, "policy_rank": None, "surprise": None
                }
                played_metrics = compute_surprise_metrics(lc0_policy, move_uci)

                row = {
                    "game_id": game_id,
                    "ply": ply_idx + 1,
                    "move_number": move_number,
                    "side": side,
                    "fen": fen,
                    "move_played": move_uci,
                    "move_played_san": move_san,
                    # Stockfish
                    "sf_best_move": sf_best,
                    "sf_score": sf_score,
                    "sf_policy_prior": sf_metrics["policy_prior"],
                    "sf_policy_rank": sf_metrics["policy_rank"],
                    "sf_surprise": sf_metrics["surprise"],
                    # lc0
                    "lc0_best_move": lc0_best,
                    "lc0_q": lc0_q,
                    "lc0_policy_prior": lc0_metrics["policy_prior"],
                    "lc0_policy_rank": lc0_metrics["policy_rank"],
                    "lc0_surprise": lc0_metrics["surprise"],
                    # Move played
                    "move_played_policy_prior": played_metrics["policy_prior"],
                    "move_played_policy_rank": played_metrics["policy_rank"],
                    "move_played_surprise": played_metrics["surprise"],
                }
                rows.append(row)

                # Progress
                dot = "." if side == "white" else "..."
                sf_str = f"SF={sf_score:+d}" if sf_score is not None else "SF=?"
                q_str = f"Q={lc0_q:+.3f}" if lc0_q is not None else "Q=?"
                sf_surp = f"sf_s={sf_metrics['surprise']:.1f}" if sf_metrics["surprise"] is not None else "sf_s=?"
                print(f"  {move_number}{dot}{move_san:<7} {sf_str:>10}  {q_str:>10}  {sf_surp}")

                board.push(move)

    sf.close()
    lc0.close()

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def save_results(df: pd.DataFrame, name: str) -> None:
    """Save DataFrame to analysis/ as parquet and JSON."""
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

    parquet_path = ANALYSIS_DIR / f"{name}.parquet"
    df.to_parquet(parquet_path, index=False)
    print(f"\nSaved Parquet: {parquet_path}")

    json_path = ANALYSIS_DIR / f"{name}.json"
    df.to_json(json_path, orient="records", indent=2)
    print(f"Saved JSON:    {json_path}")


def print_summary(df: pd.DataFrame) -> None:
    """Print top 20 most surprising positions and summary stats."""
    # Ensure numeric columns are actually numeric (can be object after concat)
    numeric_cols = [
        "sf_policy_prior", "sf_policy_rank", "sf_surprise",
        "lc0_policy_prior", "lc0_policy_rank", "lc0_surprise",
        "move_played_policy_prior", "move_played_policy_rank", "move_played_surprise",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    print(f"\n{'=' * 70}")
    print("SUMMARY")
    print(f"{'=' * 70}")
    print(f"Total positions analyzed: {len(df)}")

    # --- Top 20 by SF surprise ---
    sf_valid = df.dropna(subset=["sf_surprise"])
    if not sf_valid.empty:
        print(f"\n{'─' * 70}")
        print("Top 20 positions where Stockfish's best move is hardest to find")
        print(f"{'─' * 70}")
        top_sf = sf_valid.nlargest(20, "sf_surprise")
        for _, r in top_sf.iterrows():
            dot = "." if r["side"] == "white" else "..."
            print(
                f"  {r['game_id'][:30]:<30}  "
                f"{r['move_number']}{dot}{r['move_played_san']:<7}  "
                f"SF best={r['sf_best_move']:<6}  "
                f"rank={int(r['sf_policy_rank']):>3}  "
                f"prior={r['sf_policy_prior']:.3f}  "
                f"surprise={r['sf_surprise']:.2f} bits"
            )

    # --- Top 20 by move-played surprise ---
    played_valid = df.dropna(subset=["move_played_surprise"])
    if not played_valid.empty:
        print(f"\n{'─' * 70}")
        print("Top 20 positions where the move actually played is hardest to find")
        print(f"{'─' * 70}")
        top_played = played_valid.nlargest(20, "move_played_surprise")
        for _, r in top_played.iterrows():
            dot = "." if r["side"] == "white" else "..."
            print(
                f"  {r['game_id'][:30]:<30}  "
                f"{r['move_number']}{dot}{r['move_played_san']:<7}  "
                f"rank={int(r['move_played_policy_rank']):>3}  "
                f"prior={r['move_played_policy_prior']:.3f}  "
                f"surprise={r['move_played_surprise']:.2f} bits"
            )

    # --- Summary stats ---
    print(f"\n{'─' * 70}")
    print("Summary statistics")
    print(f"{'─' * 70}")
    for label, col in [
        ("SF best move surprise", "sf_surprise"),
        ("lc0 best move surprise", "lc0_surprise"),
        ("Move played surprise", "move_played_surprise"),
    ]:
        valid = df[col].dropna()
        if not valid.empty:
            print(f"  {label}:")
            print(f"    mean={valid.mean():.3f}  median={valid.median():.3f}  "
                  f"max={valid.max():.3f}  std={valid.std():.3f}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def merge_results() -> int:
    """Merge per-game parquet files into one combined result."""
    parts = sorted(ANALYSIS_DIR.glob("surprise_sinqcup22_game*.parquet"))
    if not parts:
        print("No per-game parquet files found to merge.", file=sys.stderr)
        return 1

    dfs = [pd.read_parquet(p) for p in parts]
    df = pd.concat(dfs, ignore_index=True)
    print(f"Merged {len(parts)} files, {len(df)} total positions.")

    save_results(df, "surprise_sinqcup22")
    print_summary(df)

    # Clean up per-game files
    for p in parts:
        p.unlink()
        json_p = p.with_suffix(".json")
        if json_p.exists():
            json_p.unlink()
    print(f"Cleaned up {len(parts)} per-game files.")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Find positions where the objectively best move is hard to find."
    )
    parser.add_argument(
        "--pgn", type=Path, default=DEFAULT_PGN,
        help="Path to PGN file (default: sinqcup22.pgn)",
    )
    parser.add_argument(
        "--games", type=int, default=10,
        help="Number of games to analyze (default: 10)",
    )
    parser.add_argument(
        "--start-game", type=int, default=0,
        help="0-based index of first game to analyze (for parallel runs)",
    )
    parser.add_argument(
        "--sf-depth", type=int, default=18,
        help="Stockfish search depth (default: 18)",
    )
    parser.add_argument(
        "--lc0-nodes", type=int, default=100,
        help="lc0 nodes per position (default: 100)",
    )
    parser.add_argument(
        "--skip-ply", type=int, default=20,
        help="Skip first N half-moves per game as opening book (default: 20 = 10 moves/side)",
    )
    parser.add_argument(
        "--merge", action="store_true",
        help="Merge per-game parquet files instead of running analysis",
    )
    args = parser.parse_args(argv)

    if args.merge:
        return merge_results()

    if not args.pgn.is_file():
        print(f"error: PGN file not found: {args.pgn}", file=sys.stderr)
        return 1

    print(f"PGN:        {args.pgn}")
    print(f"Games:      {args.games}")
    print(f"Start game: {args.start_game}")
    print(f"SF depth:   {args.sf_depth}")
    print(f"lc0 nodes:  {args.lc0_nodes}")
    print(f"Skip ply:   {args.skip_ply}")

    df = analyze_games(
        pgn_path=args.pgn,
        num_games=args.games,
        sf_depth=args.sf_depth,
        lc0_nodes=args.lc0_nodes,
        start_game=args.start_game,
        skip_ply=args.skip_ply,
    )

    suffix = f"_game{args.start_game}" if args.start_game > 0 or args.games == 1 else ""
    save_results(df, f"surprise_sinqcup22{suffix}")
    print_summary(df)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
