# dev

Development and parity tooling for pymcfs. Lives on **`main` only** — not on
the public `release` branch.

## Layout

- `fixtures/parity/` — Starlab vs pymcfs stage dumps for regression/parity
- `scripts/` — dump / compare / bench / mesh-generation helpers

```bash
uv run python dev/scripts/compare_starlab_parity.py --case sindorelax
uv run python dev/scripts/bench_mcfs_iter.py --profile --iters 5
uv run python dev/scripts/make_test_meshes.py --outdir toric_spines/data/mesh
```
