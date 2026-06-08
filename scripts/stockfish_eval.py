#!/usr/bin/env python3
"""Evaluate a chess position with Stockfish via UCI."""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

DEFAULT_STOCKFISH = (
    Path(__file__).resolve().parent.parent / "engines/stockfish/src/stockfish"
)
STARTPOS_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"


@dataclass(frozen=True)
class Evaluation:
    fen: str
    depth: int
    score_cp: Optional[int]
    mate_in: Optional[int]
    pv: tuple[str, ...]
    nodes: Optional[int]
    time_ms: Optional[int]
    wdl: Optional[tuple[int, int, int]]

    def summary(self) -> str:
        if self.mate_in is not None:
            score = f"mate {self.mate_in}"
        elif self.score_cp is not None:
            score = f"{self.score_cp:+d} cp"
        else:
            score = "unknown"

        pv = " ".join(self.pv) if self.pv else "(none)"
        parts = [
            f"depth {self.depth}",
            f"score {score}",
            f"pv {pv}",
        ]
        if self.nodes is not None:
            parts.append(f"nodes {self.nodes}")
        if self.time_ms is not None:
            parts.append(f"time {self.time_ms}ms")
        if self.wdl is not None:
            w, d, l = self.wdl
            parts.append(f"wdl {w}/{d}/{l}")
        return ", ".join(parts)


class StockfishEvaluator:
    def __init__(self, stockfish_path: Path = DEFAULT_STOCKFISH, threads: int = 1):
        self.stockfish_path = Path(stockfish_path)
        if not self.stockfish_path.is_file():
            raise FileNotFoundError(f"Stockfish binary not found: {self.stockfish_path}")

        self.process = subprocess.Popen(
            [str(self.stockfish_path)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        self._send("uci")
        self._wait_for("uciok")
        if threads != 1:
            self._send(f"setoption name Threads value {threads}")
        self._send("isready")
        self._wait_for("readyok")

    def _send(self, command: str) -> None:
        assert self.process.stdin is not None
        self.process.stdin.write(command + "\n")
        self.process.stdin.flush()

    def _readline(self) -> str:
        assert self.process.stdout is not None
        if self.process.poll() is not None:
            raise RuntimeError("Stockfish process exited unexpectedly")
        line = self.process.stdout.readline()
        if not line:
            raise RuntimeError("Stockfish produced no output")
        return line.rstrip("\n")

    def _wait_for(self, token: str) -> None:
        while True:
            line = self._readline()
            if token in line:
                return

    def evaluate(
        self,
        fen: str = STARTPOS_FEN,
        depth: int = 12,
        show_wdl: bool = False,
    ) -> Evaluation:
        if show_wdl:
            self._send("setoption name UCI_ShowWDL value true")

        self._send(f"position fen {fen}")
        self._send(f"go depth {depth}")

        last_info: Optional[dict[str, object]] = None
        last_at_depth: Optional[dict[str, object]] = None
        while True:
            line = self._readline()
            if "CRITICAL ERROR" in line:
                raise RuntimeError(line.removeprefix("info string ").strip())
            if line.startswith("info "):
                parsed = _parse_info_line(line)
                if parsed and _has_score(parsed):
                    last_info = parsed
                    if parsed.get("depth") == depth:
                        last_at_depth = parsed
            elif line.startswith("bestmove "):
                break

        chosen = last_at_depth or last_info
        if chosen is None:
            raise RuntimeError(f"No evaluation received at depth {depth}")

        reported_depth = int(chosen.get("depth", depth))
        return Evaluation(
            fen=fen,
            depth=reported_depth,
            score_cp=chosen.get("score_cp"),  # type: ignore[arg-type]
            mate_in=chosen.get("mate_in"),  # type: ignore[arg-type]
            pv=chosen.get("pv", ()),  # type: ignore[arg-type]
            nodes=chosen.get("nodes"),  # type: ignore[arg-type]
            time_ms=chosen.get("time_ms"),  # type: ignore[arg-type]
            wdl=chosen.get("wdl"),  # type: ignore[arg-type]
        )

    def close(self) -> None:
        if self.process.poll() is None:
            self._send("quit")
            self.process.wait(timeout=5)

    def __enter__(self) -> "StockfishEvaluator":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


def _has_score(info: dict[str, object]) -> bool:
    return "score_cp" in info or "mate_in" in info


def _parse_info_line(line: str) -> Optional[dict[str, object]]:
    depth_match = re.search(r"\bdepth (\d+)\b", line)
    if not depth_match:
        return None

    result: dict[str, object] = {"depth": int(depth_match.group(1))}

    cp_match = re.search(r"\bscore cp (-?\d+)\b", line)
    mate_match = re.search(r"\bscore mate (-?\d+)\b", line)
    if cp_match:
        result["score_cp"] = int(cp_match.group(1))
    elif mate_match:
        result["mate_in"] = int(mate_match.group(1))

    nodes_match = re.search(r"\bnodes (\d+)\b", line)
    if nodes_match:
        result["nodes"] = int(nodes_match.group(1))

    time_match = re.search(r"\btime (\d+)\b", line)
    if time_match:
        result["time_ms"] = int(time_match.group(1))

    wdl_match = re.search(r"\bwdl (\d+) (\d+) (\d+)\b", line)
    if wdl_match:
        result["wdl"] = (
            int(wdl_match.group(1)),
            int(wdl_match.group(2)),
            int(wdl_match.group(3)),
        )

    pv_match = re.search(r"\bpv (.+)$", line)
    if pv_match:
        result["pv"] = tuple(pv_match.group(1).split())

    return result


def evaluate_position(
    fen: str,
    depth: int = 12,
    stockfish_path: Path = DEFAULT_STOCKFISH,
    show_wdl: bool = False,
) -> Evaluation:
    with StockfishEvaluator(stockfish_path=stockfish_path) as engine:
        return engine.evaluate(fen=fen, depth=depth, show_wdl=show_wdl)


def run_tests(stockfish_path: Path) -> None:
    """Smoke-test the evaluator against a few known positions."""
    tests = [
        {
            "name": "starting position",
            "fen": STARTPOS_FEN,
            "depth": 8,
            "expect_cp_near": 0,
            "tolerance": 80,
        },
        {
            "name": "black is slightly worse after 1.e4",
            "fen": "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1",
            "depth": 8,
            "max_cp": -10,
        },
        {
            "name": "terminal checkmate on board",
            "fen": "rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3",
            "depth": 10,
            "expect_mate": 0,
        },
    ]

    with StockfishEvaluator(stockfish_path=stockfish_path) as engine:
        for test in tests:
            result = engine.evaluate(fen=test["fen"], depth=test["depth"])
            print(f"[ok] {test['name']}: {result.summary()}")

            if "expect_cp_near" in test:
                if result.score_cp is None:
                    raise AssertionError(f"{test['name']}: expected cp score")
                delta = abs(result.score_cp - test["expect_cp_near"])
                if delta > test["tolerance"]:
                    raise AssertionError(
                        f"{test['name']}: score {result.score_cp} cp "
                        f"not near {test['expect_cp_near']} +/- {test['tolerance']}"
                    )

            if "max_cp" in test:
                if result.score_cp is None or result.score_cp > test["max_cp"]:
                    raise AssertionError(
                        f"{test['name']}: expected score <= {test['max_cp']} cp, "
                        f"got {result.score_cp}"
                    )

            if "expect_mate" in test:
                if result.mate_in != test["expect_mate"]:
                    raise AssertionError(
                        f"{test['name']}: expected mate {test['expect_mate']}, "
                        f"got {result.mate_in}"
                    )

    print("All tests passed.")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fen",
        default=STARTPOS_FEN,
        help="FEN of the position to evaluate",
    )
    parser.add_argument("--depth", type=int, default=12, help="Search depth")
    parser.add_argument(
        "--stockfish",
        type=Path,
        default=DEFAULT_STOCKFISH,
        help="Path to the Stockfish binary",
    )
    parser.add_argument(
        "--wdl",
        action="store_true",
        help="Include WDL probabilities in the search output",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Run built-in smoke tests",
    )
    args = parser.parse_args(argv)

    try:
        if args.test:
            run_tests(args.stockfish)
            return 0

        result = evaluate_position(
            fen=args.fen,
            depth=args.depth,
            stockfish_path=args.stockfish,
            show_wdl=args.wdl,
        )
        print(result.summary())
        return 0
    except (FileNotFoundError, RuntimeError, AssertionError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
