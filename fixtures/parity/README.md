# Starlab ↔ pymcfs parity fixtures

Layout per case:

```
fixtures/parity/<case>/
  input.off|obj
  starlab/
    poles.off          # Starlab medial nOFF (x y z angle radius) or plain OFF
    meso_N0001.off     # contracted surface after N MCF iterations
    meso_final.off     # optional
    skeleton.cg        # raw curve graph
  pymcfs/
    poles.npy
    meso_N0001.off|.npz
    meso_final.off|.npz
    skeleton.cg
```

## Generating dumps

Parity dumps intentionally use the **Starlab profile**: `w_H=0.1`, `w_M=0.2`,
and `gate_exterior_poles=False` (ungated medial pull). That matches Starlab
`mcfskel` / published generic CGAL defaults, not the application-robust pymcfs
defaults (`w_H=0.5`, `w_M=5.0`, gated poles).

```bash
# pymcfs side (Starlab profile by default)
uv run python scripts/dump_pymcfs_parity.py --case sindorelax --iters 1,final

# Starlab side (Windows demo starterm)
uv run python scripts/dump_starlab_parity.py --case sindorelax --mcf-iters 1
```

Compare:

```bash
uv run python scripts/compare_starlab_parity.py --case sindorelax
```

## Notes

- Optional local Starlab/mcfskel source for reading reference C++ and running
  `dump_starlab_parity.py` (Windows demo). Clone into `_ref_starlab-mcfskel/`
  (gitignored); see [Starlab mcfskel](https://github.com/totoro87/mcfskel).
- `sindorelax/starlab/poles.off` is the shipped Starlab `sindorelax_poles.off` medial dump.
- The prebuilt `mcfskel-v1.1-win32` `starterm.exe` runs **Voronoi based MAT** but
  **MCF Skeletonization** access-violates here (likely missing CHOLMOD DLLs). Until a
  working Starlab build provides `meso_N****` / `skeleton.cg`, Stage 2–3 pytest gates
  skip when those files are absent.
- Do not remesh the input in Starlab unless the same remesh is applied for pymcfs.
