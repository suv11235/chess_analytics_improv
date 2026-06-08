# Chess engine interpretability setup

Local copies of **Stockfish** (classical + NNUE chess engine) and **Leela Chess Zero / lc0** (AlphaZero-style CNN chess engine) for running inference and interpretability experiments.

Model weights are **not** stored in git. Download them once using the setup steps below.

## Layout

```
engines/
  stockfish/            # official-stockfish/Stockfish source
    src/stockfish       # binary (after build)
    src/nn-*.nnue       # NNUE weights (after make net)
  lc0/                  # LeelaChessZero/lc0 source
    build/release/lc0   # binary (after build)
    weights/            # network weights (after download)
      t1-256x10-distilled-swa-2432500.pb.gz
```

## Setup

### 1. Clone

```bash
git clone https://github.com/suv11235/chess_analytics_improv.git
cd chess_analytics_improv
```

### 2. Build Stockfish (macOS Apple Silicon)

```bash
cd engines/stockfish/src
make -j$(sysctl -n hw.ncpu) build ARCH=apple-silicon
```

For Linux, replace `ARCH=apple-silicon` with the output of `make help | grep ARCH`.

### 3. Download Stockfish NNUE weights

```bash
cd engines/stockfish/src
make net          # downloads nn-83a0d6daf7e5.nnue from tests.stockfishchess.org
```

The NNUE is also embedded in the binary — run `export_net out.nnue` inside the UCI shell to extract it.

### 4. Build lc0 (macOS)

```bash
brew install meson ninja
cd engines/lc0
./build.sh        # produces build/release/lc0
```

### 5. Download lc0 network weights

```bash
mkdir -p engines/lc0/weights
# Small/fast net — 256 filters × 10 blocks, ~35 MB, good for CPU
curl -fL -o engines/lc0/weights/t1-256x10-distilled-swa-2432500.pb.gz \
  "https://storage.lczero.org/files/networks-contrib/t1-256x10-distilled-swa-2432500.pb.gz"
```

Browse [lczero.org best nets](https://lczero.org/dev/wiki/best-nets-for-lc0/) for larger networks if you have a GPU.

---

## Running the engines

### Stockfish (UCI)

```bash
cd engines/stockfish/src
./stockfish
# then:
# uci
# position startpos moves e2e4 e7e5
# go depth 15
```

### lc0 (UCI)

```bash
cd engines/lc0/build/release
./lc0 --weights=../../weights/t1-256x10-distilled-swa-2432500.pb.gz
# then identical UCI commands as Stockfish
```

### Stockfish vs lc0 game (command-line)

Both engines speak UCI. Use any UCI arbiter (e.g. `cutechess-cli`) or the script at `scripts/play_game.sh` (see below).

---

## Accessing model outputs for a given position

### Stockfish — NNUE eval trace

```
position fen <FEN>
eval
```

Returns: NNUE bucket breakdown (PSQT / Layers), final centipawn evaluation.
Per-move during search: `setoption name MultiPV value 5` + `go depth 20` → `info` lines with `score cp`.

### lc0 — policy + value

lc0 returns **both** on every node evaluation:

- **Value head** (win probability): `score cp` in search output, or `score wdl` with `setoption name VerboseMoveStats value true`
- **Policy head** (per-move prior): enable `setoption name VerboseMoveStats value true` — each `info` line then includes `P:` (prior probability from the net)
- **Raw net outputs** without search: set very low nodes `go nodes 1` — the single net evaluation is surfaced directly

---

## Notes for interpretability

- **Stockfish NNUE**: sparse HalfKAv2 features → 2 small MLPs → scalar eval. Code in `engines/stockfish/src/nnue/`. PyTorch training tooling: [glinscott/nnue-pytorch](https://github.com/glinscott/nnue-pytorch).
- **lc0**: residual CNN with policy + value heads; weights in protobuf format (`.pb.gz`). Architecture and layer access: `engines/lc0/src/neural/`. Python loading: [lczero-training](https://github.com/LeelaChessZero/lczero-training).
- Both engines speak **UCI** — easy to drive programmatically from Python via `subprocess` or the [`chess`](https://python-chess.readthedocs.io/) + [`chess.engine`](https://python-chess.readthedocs.io/en/latest/engine.html) library.
