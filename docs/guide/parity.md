# Starlab parity

Parity fixtures and dump/compare scripts live on the **`main`** branch under
[`dev/fixtures/parity/`](https://github.com/jmrfox/pymcfs/tree/main/dev/fixtures/parity)
(not on this public `release` branch).

Parity uses **`profile="starlab"`**: `attraction_weight=0.1`,
`medial_weight=0.2`, ungated poles.
Do not conflate with robust / auto defaults.

```bash
git checkout main
uv run pytest tests/test_parity.py -q
uv run python dev/scripts/compare_starlab_parity.py --case sindorelax
```

Stage 1 (poles) is the always-on gate when fixtures are present. Later stages
need matching Starlab reference dumps on disk.
