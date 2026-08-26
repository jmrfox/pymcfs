# MCFS benchmarks (WSL)

Recorded with `uv run python scripts/bench_mcfs_iter.py --profile` on Aug 2026.

## sindorelax (`fixtures/parity/sindorelax/input.off`, n≈12k)

| Phase | ms | Share |
|-------|-----|-------|
| geometry | 110 | 0.6% |
| collapse | 7620 | 40% |
| split | 9017 | 48% |
| degen | 2168 | 11% |
| **total / iter** | **~18917** | 100% |

Full iteration (after Tier B split preallocation): **~17 s/iter** (1-iter mean).

CHOLMOD improves the geometry slice (~3× on solve) but not wall-clock (~1% of iteration).

## TS1 (`data/mesh/TS1.obj`, n≈2.4k)

| Phase | ms | Share |
|-------|-----|-------|
| geometry | 11 | 0.5% |
| collapse | 921 | 41% |
| split | 1170 | 53% |
| degen | 127 | 6% |
| **total / iter** | **~2228** | 100% |

Full iteration: **~2.0 s/iter**.

## Contains / pole gating

`contains_calls` stays at +2 per contract (collapse + split remesh). Not a bottleneck (<1% of runtime).

## Commands

```bash
uv run python scripts/bench_mcfs_iter.py --profile --iters 1
uv run python scripts/bench_mcfs_iter.py --mesh ts1 --profile --iters 3
uv run python scripts/bench_mcfs_iter.py --iters 5 --no-cholmod
```
