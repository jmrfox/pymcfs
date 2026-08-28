# Starlab parity

Stage-wise comparison against Starlab `mcfskel` dumps lives under
[`fixtures/parity/`](https://github.com/jmrfox/pymcfs/tree/main/fixtures/parity)
(see that folder’s README for case layout).

Parity uses **`profile="starlab"`**: `w_H=0.1`, `w_M=0.2`, ungated poles.
Do not conflate with robust / auto TS defaults.

```bash
uv run pytest tests/test_parity.py -q
uv run python scripts/compare_starlab_parity.py --case sindorelax
uv run python scripts/dump_pymcfs_parity.py --case sindorelax --iters 1,final
```

Stage 1 (poles) is the always-on gate. Later stages need matching Starlab
reference dumps on disk.
