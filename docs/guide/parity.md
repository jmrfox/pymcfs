# Starlab parity

Stage-wise comparison against Starlab `mcfskel` dumps lives under
[`dev/fixtures/parity/`](https://github.com/jmrfox/pymcfs/tree/main/dev/fixtures/parity)
(see that folder’s README for case layout). `dev/` lives on **`main` only** —
not on the public `release` branch.

Parity uses **`profile="starlab"`**: `attraction_weight=0.1`,
`medial_weight=0.2`, ungated poles.
Do not conflate with robust / auto defaults.

```bash
uv run pytest tests/test_parity.py -q
uv run python dev/scripts/compare_starlab_parity.py --case sindorelax
uv run python dev/scripts/dump_pymcfs_parity.py --case sindorelax --iters 1,final
```

Stage 1 (poles) is the always-on gate. Later stages need matching Starlab
reference dumps on disk.
