# Export and I/O

Skeletons stay in the **input mesh coordinate frame**. Export what your
downstream tools expect.

## Polylines

One maximal chain per line (junction-to-leaf or cycle), as
`N x y z x y z ...`:

```python
skel.write_polylines("skeleton.polylines.txt")

# Or get arrays in memory
polylines = skel.to_polylines()  # list of (k_i, 3) arrays
```

## Starlab curve graph (`.cg`)

```python
from pymcfs import read_cg, write_cg

skel.write_cg("skeleton.cg")

nodes, edges = read_cg("skeleton.cg")  # edges are 0-based
write_cg("copy.cg", nodes, edges)
```

Format: header `# D:3 NV:<n> NE:<m>`, then `v x y z` lines and `e i j` with
**1-based** indices on disk (`read_cg` / `write_cg` convert to/from 0-based).

## NetworkX graph

```python
G = skel.graph  # node attr "pos", edge "weight"
print(skel.nodes.shape, skel.edges.shape)
```

`Skeleton.from_graph(G)` rebuilds a densely indexed skeleton from a curve
graph with `pos` attributes.

## Visualization

Needs `pymcfs[viz]` (plotly):

```bash
pip install ".[viz]"
```

```python
fig = skel.plot_3d(mesh=mesh, autoshow=False)
fig.show()
```
