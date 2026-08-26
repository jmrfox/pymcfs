# pymcfs development TODO

Progress checkpoint: robust MCFS (pole gating, CGAL-app defaults), Starlab parity
harness, Tier A speedups, and `ts_skeleton` notebook are in place. Tests:
`52 passed`, `1 skipped` (CHOLMOD), `1 deselected` (slow TS1).

Use this list after moving to WSL. Items are ordered roughly by priority for
**production use on TS biological meshes**, not a PyPI release.

---

## 1. WSL environment (do first)

- [ ] Clone/pull repo in WSL; use Linux paths (avoid `/mnt/c/...` for heavy MCFS runs if I/O is slow).
- [ ] Install system deps:
  ```bash
  sudo apt update
  sudo apt install -y libsuitesparse-dev build-essential
  ```
- [ ] Recreate env with uv:
  ```bash
  uv sync
  uv sync --extra cholmod
  ```
- [ ] Verify CHOLMOD:
  ```bash
  uv run python -c "from pymcfs.spd_solve import cholmod_available; print(cholmod_available())"
  uv run python -m pytest tests/test_spd_solve.py -q
  ```
- [ ] Optional Starlab reference (parity dumps / reading C++): clone mcfskel into
  `_ref_starlab-mcfskel/` (gitignored). See `fixtures/parity/README.md`.

---

## 2. Blockers before trusting TS production runs

### CHOLMOD / scikit-sparse 0.5 API

- [ ] **Fix `pymcfs/spd_solve.py` for scikit-sparse ≥ 0.5.**  
  Linux conda/pip builds often return `(R, perm)` from `cholesky()` instead of a
  callable factor object. Current code tries `factor(b)` and silently falls back
  to SuperLU on failure — you may get no speedup on WSL until this is fixed.
  - Target API: `cho_factor` / `cho_solve`, or `CholeskyFactor(...).factorize()` + solve.
  - Re-run `tests/test_spd_solve.py` with CHOLMOD installed (currently `skipped` if missing).
  - Benchmark: `uv run python scripts/bench_mcfs_iter.py --iters 5` (with vs without `--no-cholmod`).

### TS mesh validation

- [ ] Re-run `notebooks/ts_skeleton.py` on WSL for **TS1** (and at least one other `data/mesh/TS*.obj`).
  - Confirm no exterior spike branches with robust defaults (`W_H=0.5`, `W_M=5.0`, gating on).
  - Export SWC/CG under `data/swc/` (gitignored for `TS*/`).
- [ ] Run slow regression explicitly:
  ```bash
  uv run python -m pytest tests/test_pole_gating.py::test_ts1_robust_no_screenshot_exterior_spikes -q
  ```
- [ ] Spot-check `analyze_skeleton(mesh, skel)` containment on each TS mesh you care about.

### Correctness confidence

- [ ] Review skeleton quality on **non-trivial topology** (handles, branches), not only TS1.
- [ ] If residual exterior nodes appear after gating, consider a **post-prune pass**
  (deferred from robust MCFS plan — only if still needed after WSL runs).

---

## 3. Performance (remaining Tier A + Tier B)

Tier A (done): cached `mesh.contains`, faster Laplacian assembly, optional CHOLMOD
path, face-walk collapse order.

### Still worth doing

- [ ] **Confirm CHOLMOD actually used** in driver logs (`spd=cholmod` at init when verbose).
- [ ] **Numba collapse hot path** — `collapse_short_edges` still uses Python incremental
  adjacency for updates; Numba `link_condition_ok` is used elsewhere but not fully
  on the main collapse loop. Revisit only if remesh time dominates after CHOLMOD.
- [ ] **Batch point-in-mesh** for gating — if `contains` is still slow on huge TS meshes,
  cache a BVH / reuse trimesh ray engine across the run (same boolean mask required).
- [ ] **Incremental sparse factor** — refactor `AᵀA` only when sparsity pattern changes
  (largest win after CHOLMOD on long runs).

### Benchmark hygiene

- [ ] Record baseline on WSL: mean ms/iter for TS1 and sindorelax via `scripts/bench_mcfs_iter.py`.
- [ ] Do not use iterative CG unless parity proves identical geometry — new numerical error.

---

## 4. Starlab parity harness

- [ ] Re-run full parity suite on WSL:
  ```bash
  uv run pytest tests/test_parity.py -q
  uv run python scripts/compare_starlab_parity.py --case sindorelax
  ```
- [ ] **Stage 2–3 gates** skip when Starlab `meso_N0001` / `skeleton.cg` dumps are missing
  (Windows demo MCF often broken without CHOLMOD DLLs). Options:
  - Regenerate Starlab dumps from a working Linux/WSL Starlab build, or
  - Accept pymcfs-only gates until reference binaries exist.
- [ ] Keep parity scripts on **`profile="starlab"`** / ungated poles — do not conflate with robust TS defaults.

---

## 5. API and library polish

- [ ] Export `McfsProfile` and `resolve_mcfs_profile` from `pymcfs.__init__` if external scripts need them.
- [ ] Thread `use_cholmod` through `skeletonize` / `thin_mesh` kwargs (currently driver-only).
- [ ] Move `pytest` from main `[project.dependencies]` to optional `test` extra (cleanup, not urgent).
- [ ] Jupytext: open notebooks via `notebooks/*.py` only; `*.ipynb` stays gitignored.
  - Regenerate local `.ipynb` after edits: `uv run jupytext --sync notebooks/ts_skeleton.py`

---

## 6. Testing and CI

- [ ] Add GitHub Actions (or similar): `uv sync`, `pytest -m "not slow"`, optional weekly slow TS1 job.
- [ ] Document slow marker in README:
  ```bash
  uv run python -m pytest -m slow
  ```
- [ ] Consider TS2/TS3 smoke tests (marked slow) once TS1 path is stable on WSL.

---

## 7. Documentation

- [ ] Add short **WSL quickstart** section to README (copy from §1 above once verified).
- [ ] Note in ABOUT.md when CHOLMOD API fix lands and expected speedup range.
- [ ] Document notebook workflow: `ts_skeleton.py` inputs (`data/mesh/TS1.obj`), outputs (`data/swc/TS1/`, gitignored).

---

## 8. Deferred / explicit non-goals (for now)

- **JAX / GPU / iterative solvers** — no accuracy benefit for this linear MCFS step; risks parity drift.
- **CGAL `ω_L` area-ratio schedule** — not in current CGAL/Starlab source; do not invent unless reference proves it.
- **Windows native CHOLMOD** — use WSL + `libsuitesparse-dev` instead of micromamba side envs.
- **PyPI release** — no packaging/version bump until TS workflow is validated end-to-end.
- **Exterior-branch post-prune** — only if gated robust defaults still leave outliers on real TS data.

---

## 9. Definition of done (implementation-ready)

Treat pymcfs as ready for your TS implementation workflow when:

1. WSL env runs with **CHOLMOD enabled** and measurably faster than SuperLU on TS1-scale meshes.
2. **TS1 + ≥1 other TS mesh** produce inside-mesh skeletons with robust defaults (visual + `analyze_skeleton`).
3. Slow TS1 regression passes on WSL.
4. Parity Stage 1 (sindorelax poles) still passes; Stage 2 understood/documented if Starlab dumps absent.
5. Notebook path documented: mesh → MCFS → SWC/CG export reproducible on WSL.

---

## Quick reference commands (WSL)

```bash
uv sync --extra cholmod
uv run python -m pytest -m "not slow" -q
uv run python -m pytest -m slow -q
uv run python scripts/bench_mcfs_iter.py --mesh data/mesh/TS1.obj --iters 3
uv run python scripts/compare_starlab_parity.py --case sindorelax
```
