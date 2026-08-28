# Benchmarks

Recorded with `uv run python scripts/bench_mcfs_iter.py --profile` on Aug 2026.
Machine: WSL2, numba 0.66, CHOLMOD present, `embreex` absent.

## Attribution fix (read this before trusting older numbers)

Every table in this file before Aug 2026 misattributed the dominant cost.
`_profile_contract` timed `driver.collapse_edges()` and `driver.split_faces()`
as wholes, and **both call pole gating internally**. Gating therefore showed up
as "collapse" and "split" time, which is why the notes here previously claimed
`contains` was "<1% of runtime" and why several rounds of optimization went
into remeshing instead.

Gating was in fact **74-80% of wall clock**. `scripts/bench_mcfs_iter.py` now
patches `pymcfs.mcfs.points_inside_mesh`, bills gating to its own phase, and
reports collapse/split net of it. Do not remove that separation.

## Results

Mean ms per `contract()` iteration, 5-iteration run:

| Mesh | Before | After | Speedup |
|------|--------|-------|---------|
| TS1 (`data/mesh/TS1.obj`, n≈2.4k) | 2033 | **287** | 7.1x |
| sindorelax (`fixtures/parity/sindorelax/input.off`, n≈12k) | 16200 | **601** | 27x |

End-to-end TS1 `contract_until_convergence()`: **141 s -> 30 s** over the same
49 iterations, producing a bit-identical meso surface (n=4794, f=9620) and
curve graph (875 nodes, 881 edges). `tests/test_parity.py` and
`scripts/compare_starlab_parity.py --case sindorelax` are unchanged.

### Phase breakdown, after

| Phase | TS1 ms | share | sindorelax ms | share |
|-------|--------|-------|---------------|-------|
| geometry (Laplacian + AtA + solve) | 11.5 | 2.3% | 104.4 | 4.5% |
| collapse | 130.7 | 25.8% | 178.2 | 7.7% |
| split | 53.0 | 10.5% | 169.0 | 7.3% |
| degeneracy detection | 7.3 | 1.4% | 17.1 | 0.7% |
| gating (`contains`) | 304.2 | 60.0% | 1846.1 | 79.7% |

### Phase breakdown, before (gating hidden inside collapse/split)

| Phase | TS1 ms | sindorelax ms |
|-------|--------|---------------|
| geometry | 11 | 128 |
| collapse (incl. one gating call) | 921 | 8104 |
| split (incl. one gating call) | 1170 | 9336 |
| degeneracy detection | 127 | 2169 |

## What changed

**Gating is carried by index instead of re-tested.** Pole containment is a
property of a fixed point against the fixed input mesh. A collapse keeps
whichever of two *existing* poles is closer to the midpoint, so its validity
carries over by index; only edge splits create genuinely new interpolated
poles. `collapse_short_edges` now threads `pole_valid` (zero `contains` calls)
and `split_obtuse_faces` grows it, so the driver tests just the new tail. Calls
per contract went from 2 whole-mesh tests to one batch of a few hundred points:
50 calls for the whole TS1 run instead of 99, and 3,344 points tested over 5
iterations instead of ~14,000.

**The edge hash was degenerate.** `_hash_lookup` / `_hash_insert` used
`key % cap` with `cap` a power of two and `key = (lo << 32) | hi`, so the slot
came entirely from the low bits of the larger endpoint and every key landed in
`[0, nv)` of a much larger table. Average probe length on sindorelax's 72,780
edge visits was **10,668**. With a murmur3 finalizer and a `cap - 1` mask it is
**~1.1**, and `build_topology` went from **1141 ms to 8.1 ms** (141x). Edge
indexing is unaffected because `_build_topology_kernel` assigns edge slots in
face-walk first-seen order, independent of the hash
(`test_build_topology_edge_slots_follow_face_walk_order` locks this in).

**Collapse adjacency is built once, not per pass.** The pass loop rebuilt
`_vertex_neighbors` + `_edge_to_faces` + the edge list every sweep (~287 ms per
sweep on sindorelax, 5 sweeps). Those structures are maintained incrementally,
so only the visit order needs re-enumerating; compaction moved to a single pass
at the end, which is equivalent because `np.unique` remap is monotone.

**The collapse sweep is now a Numba kernel.** With `build_topology` at 8 ms
instead of 1141 ms, the reason the previous notes gave for deferring this
("`build_topology` is ~10x slower than Python `_vertex_neighbors` +
`_edge_to_faces`") no longer holds. `collapse_short_edge_sweep` does the length
screen, link condition, midpoint, closest-pole choice and incremental topology
update in one kernel.
`tests/test_topology.py::test_numba_collapse_sweep_matches_python_reference`
asserts exact equality against the Python pass across 4 meshes x 3 thresholds;
the Python helpers in `remesh.py` are kept as that reference.

**Smaller, exact wins.** `AtA` is assembled directly as
`L_w.T @ L_w + diag(w_H²) + diag(w_M²)` instead of building a `(3n, n)` stack
and transposing it, with terms combined in the stacked product's order so the
result is bit-identical; the Laplacian is row-scaled through CSR `data` instead
of three sparse ops; the `detect_degeneracies` length prefilter is vectorized;
and the duplicate `select_obtuse_split_batch` / `split_face_on_edge_numba`
definitions in `topology.py` (each defined twice, the second shadowing the
first) are gone, with the kernel now actually wired into `split_obtuse_faces`.

## The remaining bottleneck: pole containment

Gating is now 60-80% of an iteration and is the natural next target, but the
obvious fix is a trap.

`trimesh` silently switches `mesh.contains` to Embree whenever `embreex` is
importable, and **Embree traces in single precision**. TS1 spans x∈[5661, 7957],
so float32 has ~1e-3 absolute resolution there. Measured against the exact
float64 traverser on TS1's 2,378 poles:

| Backend | Disagreements | Time |
|---------|---------------|------|
| `ray_triangle` (float64), raw coordinates | reference | 866 ms |
| `ray_triangle` (float64), unit-normalized | 0 | 698 ms |
| Embree, unit-normalized | 38 (1.6%) | 13 ms |
| Embree, raw coordinates | **1531 (64%)** | 57 ms |

On sindorelax (unit-ish scale, near origin) Embree disagrees on **0** of 12,128
poles and is **165x** faster.

So `pymcfs` no longer routes gating through `mesh.contains`. `points_inside_mesh`
in `pymcfs/medial.py` uses the exact float64 traverser by default, and the fast
backend is opt-in via `fast_gating=True` on the driver, `skeletonize` and
`thin_mesh`, with `pymcfs[embree]` installed. Only enable it for meshes at
unit-ish scale near the origin.
`test_gating_defaults_to_exact_float64_backend` guards the default.

Making exact gating fast would need a float64 BVH ray traverser (the current one
does one rtree query per ray in Python). Interpolated split poles offer no
shortcut: measured across 4 iterations on both meshes, they are all distinct
points and essentially never coincide with an existing pole, so the batch
cannot be deduplicated.

## Out of scope

CHOLMOD reports `rcond ~ 2.5e-15` "nearly singular" on TS1. That comes from
pinned vertices getting `w_H = 1/zero_TH = 1e7`, which becomes `1e14` in `AtA`.
It is a conditioning issue, not a performance one, and predates this work.

## Commands

```bash
uv run python scripts/bench_mcfs_iter.py --profile --iters 5
uv run python scripts/bench_mcfs_iter.py --mesh ts1 --profile --iters 5
uv run python scripts/bench_mcfs_iter.py --iters 5 --no-cholmod
```
