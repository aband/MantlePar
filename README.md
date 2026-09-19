# MantlePar
Restruct of the MantleSolver_cpp repo

To build and run dry preheat followed by full phase-coupled evolution, using the
preheat H/C as initial conditions, and generate individual and combined videos:

```sh
./run.sh                 # 20×20 for both stages
./run.sh --grid 40       # 40×40 for both stages
./run.sh --mesh 40 60    # 40×60 rectangular mesh
./run.sh --grid 30 --warm-end 0.05  # Warm surface ends at x=0.05 in both stages
./run.sh --grid 20 --preheat-final-time 15000 --final-time 5
```

All evolution meshes use the shared `src/couple/example/formalevolve/input.yaml`;
the runner saves each run's resolved settings in its output directory.

`--warm-end X` sets the warm top boundary to `0 <= x <= X` in both stages.
It also sets the solid/Stokes velocity ramp cutoff: the surface speed reaches
3.2 cm/year at X and stays at that speed for x >= X.
During full evolution, H and C use free outflow over this warm surface interval.
X is nondimensional (default 0.025, allowed range `0 < X <= 0.5`) and must
coincide with a mesh vertex: `X*nx/0.5` must be an integer. The fixed melt outlet
at x=0.05 also requires nx to be a multiple of 10; `ny >= 5`. The default warm
endpoint therefore still requires nx to be a multiple of 20.
Default times are preheat t=15000, then evolution t=1 with its clock reset to zero.
Use `--preheat-final-time` and `--final-time` to change them. Both values are
nondimensional; evolution time is measured from zero after preheat.
The earlier `--preheat-end-time` and `--end-time` options remain valid aliases.
`evolve.sh` runs evolution
from an existing completed preheat. Use `./run.sh --dry-run` to inspect commands
without running anything, or `./run.sh --help` for options.

For 20×20, the combined video is
`src/couple/example/formalevolve/output/20x20/visualization/combined_evolution.mp4`.
The adjacent `index.html` includes all five videos and a link to the PDF.
See `visualization/conditions.pdf` for all initial/boundary conditions, physical units,
and the H/C handoff for both stages. `./run.sh --grid 20 --report-only` generates
this PDF from an existing completed run without rerunning simulations or movies.
PDF output uses ReportLab in the current Python environment, `PDF_PYTHON`, or
the available Codex bundled runtime.
See
[the coupled case instructions](src/couple/example/formalevolve/README.md)
for prerequisites, output locations and plotting conventions.
