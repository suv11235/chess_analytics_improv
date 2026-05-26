# Chess engine interpretability setup

Local copies of **Stockfish** (chess, NNUE) and **Leela Zero** (Go, AlphaZero-style CNN) for running inference and interpretability experiments.

## Layout

```
engines/
  stockfish/          # official-stockfish/Stockfish source + built binary
  leela-zero/         # leela-zero/leela-zero source + leelaz binary
    src/leelaz        # inference binary (built for macOS ARM)
    weights/
      best-network.gz # strongest LZ net (40×256, hash 0e9ea880)
      weights.txt     # human-supervised baseline (weaker)
```

## Stockfish

| Item | Path |
|------|------|
| Source | `engines/stockfish/` |
| Binary | `engines/stockfish/src/stockfish` |
| Default NNUE | `engines/stockfish/src/nn-83a0d6daf7e5.nnue` (~86 MB) |

**Run (UCI):**

```bash
cd engines/stockfish/src
./stockfish
```

**Export embedded net for custom analysis:**

```
export_net my_export.nnue
```

**Re-download default net:** `make net` from `engines/stockfish/src`.

Network architecture lives under `engines/stockfish/src/nnue/`. For training / PyTorch tooling see [nodchip/nnue-pytorch](https://github.com/glinscott/nnue-pytorch).

## Leela Zero

| Item | Path |
|------|------|
| Source | `engines/leela-zero/` |
| Binary | `engines/leela-zero/src/leelaz` |
| Best weights | `engines/leela-zero/weights/best-network.gz` |
| Human weights | `engines/leela-zero/weights/weights.txt` |

The public weight server (`zero.sjeng.org`) is offline; `best-network.gz` was fetched from the [Internet Archive](https://web.archive.org/web/20210215100000/http://zero.sjeng.org/best-network) (final promoted net, Feb 2021).

**Run with weights:**

```bash
cd engines/leela-zero/src
./leelaz -w ../weights/best-network.gz --gtp
```

**Network definitions (for reimplementation / probes):**

- Caffe: `engines/leela-zero/training/caffe/zero.prototxt` (40 blocks)
- TensorFlow: `engines/leela-zero/training/tf/tfprocess.py`

**Rebuild leelaz** (macOS, Homebrew boost/cmake):

```bash
brew install boost cmake zlib openblas
cd engines/leela-zero/src
make clean
make CC=clang CXX=clang++ \
  CXXFLAGS="-I$(brew --prefix boost)/include -I./Eigen -I. -std=c++17 -O3 -DNDEBUG" \
  LDFLAGS="-L$(brew --prefix boost)/lib" \
  leelaz
# Then link without libboost_system (removed in Boost 1.89+):
clang++ -L$(brew --prefix boost)/lib -o leelaz *.o -framework Accelerate -framework OpenCL \
  -lboost_filesystem -lboost_program_options -lpthread -lz
```

A one-line patch to `src/UCTNode.cpp` (remove deprecated `std::binary_function`) is included for C++17 builds.

## Notes for interpretability

- **Stockfish**: NNUE is a sparse feature → MLP eval; good for feature attribution and layer-wise probes on the eval net.
- **Leela Zero**: full-board residual CNN + policy/value heads; weights are plain text (one coefficient per line) — easy to parse without the engine.
- Leela Zero is **Go**, not chess. For chess + similar architecture use [Leela Chess Zero (lc0)](https://github.com/LeelaChessZero/lc0) separately if needed.
