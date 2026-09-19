# Coupled initialization and visualization

`mantle_couple_init` initializes H/C and equilibrium phase quantities, solves
Darcy–Stokes using `src/mfem`, and exports phase data at **cell centers** and
finite-element velocities at **cell and edge Gauss points**. It performs no
time stepping. All plots use physical units.

`mantle_couple_preheat` adds **legacy dry preheat time stepping**: it solves flow
once, evolves H, holds C fixed, and exports the final state. Full phase-coupled
H/C evolution with flow updates is not part of this driver.

## Run

For the complete two-stage corner simulation, run `./run.sh --grid 20` from the
repository root. It runs dry preheat, imports its final H/C into full evolution,
and renders velocity, temperature, porosity, phase, and a synchronized combined
video. `--grid N` selects a square mesh; `--mesh NX NY` selects a rectangular mesh
(boundary endpoints must align with mesh vertices, ny at least 5).
`--warm-end X` changes the warm surface endpoint and solid-velocity ramp cutoff
in both stages (nondimensional x, default 0.025). The surface speed reaches
3.2 cm/year at X. For example, `./run.sh --grid 30 --warm-end 0.05` is valid. See
[the full evolution instructions](example/formalevolve/README.md) for output
locations and time controls. `./evolve.sh` reuses a completed preheat instead.

The two-stage runner also writes `visualization/conditions.pdf`: both stages'
initial conditions, Stokes/Darcy boundaries, H/C advection and thermal diffusion,
region extents, units, phase pressure, and imported H/C provenance. It reads each
run's saved input rather than the current templates. Use `./run.sh --report-only`
to regenerate this PDF from an existing completed evolution output. ReportLab is
required in the active Python, `PDF_PYTHON`, or the available Codex bundled runtime.

From the repository root with the usual PETSc/MPI configuration:

```bash
cmake --build build --target mantle_couple_init --parallel
./build/src/couple/mantle_couple_init -input src/couple/example/formalpreheat/input.yaml
python3 src/couple/plot_initialization.py \
  --input-dir src/couple/example/formalpreheat/output/rectangular
```

Examples:

- `example/column/input.yaml`: pseudo-1D melting column.
- `example/perturbed_quad/input.yaml`: nonuniform quadrilateral mesh.
- `example/formalpreheat/input.yaml`: dry 2D corner preheat; see its README for
  the mapping from `MantleSolver_cpp/src/corner/formalpreheat`.

The plotter requires Python, NumPy, Matplotlib and SciPy. Open the resulting
`visualization/index.html`, a self-contained report containing four figures
and solver tables. `visualization/report.md` contains the readable summary.
`--output-dir PATH` changes the report destination.

## Plots and physical units

| Figure | Location and quantities |
| --- | --- |
| `centers_phase.png` | Equilibrium porosity, temperature, phase pressure, solid fractions and phase region at cell centers |
| `centers_state.png` | Specific enthalpy, bulk/phase compositions and temperature derivatives at cell centers |
| `flow_cells_velocity.png` | Solid/liquid velocities and Darcy segregation flux: quivers and streamlines from cell Gauss points |
| `flow_edges_velocity.png` | The same quantities at edge Gauss points, retaining both adjacent-cell traces |

A cell center is the mapped reference origin: the mean of its four vertices,
matching the legacy corner module. Phase is evaluated on the prescribed H/C
profiles **at that point**, not on cell averages. Cell polygons are colored by
these center values without mesh outlines or point markers. Gray
cells indicate absent-phase compositions or undefined derivatives.

Velocities are reconstructed from the solved BR/H(div) coefficients, using
MPI ghosts and the existing physical basis functions. Quivers use a subset of the
actual Gauss points, with labeled physical arrow scales. Constant-color streamlines
interpolate the sampled vectors; magnitude heatmaps are not drawn. Stokes velocity is continuous across
an interior edge. Darcy normal velocity is continuous but its tangent can jump,
so edge output and quivers retain both traces. Coincident edge traces are averaged
only when constructing the streamline interpolation.

| Displayed quantity | Conversion and units |
| --- | --- |
| Coordinates | `x*l0/1000`, km |
| Temperature | `TD*dT`, K (no temperature offset) |
| Specific enthalpy | `H*h0`, J/kg, with `h0=cp*dT` |
| Phase pressure | Supplied pressure / `1e6`, MPa |
| Solid velocity | `u0*us`, displayed in cm/year |
| Liquid velocity | `vs + q/phi`, where `phi>0`; undefined in dry material |
| Darcy segregation flux | `u0*phi^(1+theta)*ud`, displayed in cm/year |
| Temperature derivatives | `dTD/dH * dT/h0`, K/(J/kg); `dTD/dC * dT`, K per unit composition |
| Time | `time*t0/(365*24*3600)`, years |
| Composition and phase fractions | Unscaled fractions |

Darcy segregation flux is a volume flux per unit area, not liquid velocity.
Its conversion uses the **flow porosity** sampled at each point, harmonic on
shared edges. The assembled Darcy unknown is not itself a physical flux. No
additional sign reversal is applied. The year is 365 days, matching the legacy
3.2 cm/year boundary speed. Numerical solver residuals refer to the assembled
nondimensional system; solver tolerances retain their configured meaning.

## Initial flow and porosity

By default, MFEM coefficients come from equilibrium phase evaluated at cell
Gauss points, with a Jacobian-weighted cell average. Interior edges use
`2*phi_minus*phi_plus/(phi_minus+phi_plus)`; a dry side gives zero. Supported
mesh-aligned profile jumps use explicit one-sided limits. Boundary edges use
the interior trace. This is startup sampling; no ML-WENO reconstruction occurs.

The explicit legacy preheat option is:

```yaml
porosity:
  source: prescribed_function
  prescribed_function: {name: constant, parameters: {value: 0.0}}
  cell_average: physical_quadrature
```

It sets every **flow** coefficient to zero porosity while retaining independent
equilibrium phase diagnostics. Reports label this distinction. Other prescribed
porosity profiles are not accepted by this initializer. Fully molten points
are outside the current two-phase flow operator unless the explicit dry-flow
mode is selected; coefficients are never clipped.

The MFEM routines assemble both momentum systems, pressure sources and pressure
coupling, apply boundary lifting with legacy signs, validate the configured
joint pressure gauge, and solve at `time.start`. The phase-law pressure remains
an independent dimensional input; solved flow pressures do not feed back into
initial thermodynamics.

`InitialState` retains `flowSystem`, `flowReport`, three MFEM DOF maps and
`flowPorosity` (row-major owned cells, CCW local edge samples). Access borrowed
vectors with `GetCoupledLinearSystemSolution(state.flowSystem, us, ps, ud, pd)`.
`SampleInitialFlow` reconstructs cell/edge Gauss samples without modifying the
state. Use `DestroyInitialState` before `PetscFinalize`. Rejected initialization
leaves the caller's state empty, even with generic solver nonconvergence errors
disabled.

## Files and MPI

The YAML-relative output directory contains:

- `cellH1.dat`, `cellC1.dat`: global H/C cell averages in the legacy
  `formalEvolve` input format, written once by rank zero.
- `transport_state.json`: H/C export completion status, units, ordering,
  mesh parameters, reference scales and actual state time.
- `initial_rank_*.csv`: distributed H/C/T/porosity cell averages, including
  physical `h_J_kg`, `temperature_K`, centroid coordinates in metres and area
  in square metres.
- `cell_center_rank_*.csv`: equilibrium phase at mapped cell centers.
- `cell_gauss_rank_*.csv`, `edge_gauss_rank_*.csv`: initial phase quadrature
  diagnostics retained for integration checks (not used for phase plots).
- `flow_cell_gauss_rank_*.csv`, `flow_edge_gauss_rank_*.csv`: reconstructed flow
  samples. Edge `entity_id` is the owning cell; `edge_id,q` identifies the shared
  physical point, and `side` is the cell's CCW edge number.
- `flow_dofs_rank_*.csv`: four solved coefficient fields, with natural DOF IDs.
- `mesh_rank_*.csv`, `visualization.json` (schema 2), `setup.txt`, `input_used.yaml`.

CSV files retain explicit nondimensional diagnostic columns alongside physical
columns: `x_m`, `y_m`, `h_J_kg`, `temperature_K`, `phase_pressure_Pa`,
`dT_dh_K_kg_J`, `dT_dC_K`, `vs_x_m_s`, `vs_y_m_s`, `q_x_m_s`, `q_y_m_s`.
Flow `ps,pd` remain assembled nondimensional P0 coefficients, distinct from
phase pressure. Pressure is constant within each cell. Its Darcy potential is
`pd/sqrt(phi_hat)`, with `phi_hat=1` for an exactly dry cell and otherwise the
physical quadrature average. An absent liquid has no physical liquid pressure.

The completion manifest lists current rank files and row counts. The plotter
rejects incomplete/older-schema output and ignores stale files from other runs.
Rerun initialization when upgrading old output. For MPI, set `parallel.ranks`
and `process_grid` in YAML, then use PETSc's MPI launcher. Run the plotter once
after all ranks finish; do not run simultaneous writers in one directory.

`setup.txt` and the report include measured convergence, true residuals,
acceptance thresholds, effective outer KSP/PC and inner-solver statistics.
Configured defaults are separate from runtime PETSc overrides under
`couple_flow_`. The melting examples use FGMRES/full Schur with CG blocks;
the exactly dry preheat example uses a GMRES pressure block.

## H/C handoff to formalEvolve

Every initialization writes `cellH1.dat` and `cellC1.dat` from the actual stored
enthalpy/composition Vecs, using physical quadrature cell averages rather than
the center samples used for plotting. Each file has `ny` rows of `nx` numbers,
with no header: x varies fastest, and rows proceed from the bottom to the top
(`j*nx+i`). Values retain full floating-point precision. Natural-order conversion
before gathering makes this layout independent of the MPI process grid.

The legacy `ReadVectorTransport` reads these exact filenames from its working
directory. Copy the pair from the output directory into the `formalEvolve` run
directory and use matching cell dimensions, domain, geometry and reference
scales. The legacy reader does not validate this metadata or interpolate fields;
for a perturbed mesh, its actual vertices must also match. Keep
`transport_state.json` with the pair and check that its status is `complete`.

These solver input files intentionally store **nondimensional H=h/(cp*dT)** and
unscaled composition C, as required by the legacy reader. For physical values,
use the cell-average CSV (`h_J_kg` and `C`); the plots also remain in physical
units. The metadata records the enthalpy scale and the actual state time.
For `mantle_couple_init` that time is `time.start`; for a successful
`mantle_couple_preheat` run it is `time.end`. `mantle_couple_evolve -preheat PATH`
imports these completed preheat files; the root `run.sh` automates this handoff.

## Legacy preheat time stepping

```bash
cmake --build build --target mantle_couple_preheat --parallel
./build/src/couple/mantle_couple_preheat -input src/couple/example/formalpreheat/input.yaml
python3 src/couple/plot_initialization.py \
  --input-dir src/couple/example/formalpreheat/output/rectangular
```

Choose separate `output.directory` values to retain both initialized and evolved
results. The preheat executable requires prescribed zero flow porosity. It uses
the already accepted solid velocity and advances
`dH/dt + div(us*H - kappa*grad(H)) = 0` with forward Euler. C remains exactly
unchanged. This is the legacy dry thermal approximation: diffusion differentiates
H, with thermal boundary data interpreted as H=T/dT. The equilibrium phase-law
temperature/porosity remain independent diagnostics, not thermal feedback.

Advection uses the existing ML-WENO centered 3×3 / four 2×2 reconstruction and
Lax–Friedrichs face kernels, with the configured stabilization. Unavailable
physical-boundary stencils are discarded. Diffusion uses constant cell-average
samples and the existing normal-line diffusion kernels with the configured
sample count/extent. This follows the legacy preheat stages and physics; MFEM
boundary handling and normal-line sampling retain MantlePar's implementation,
so the discrete result is not a bitwise reproduction of the old solver.

Each interior face is assembled once with opposite flux contributions to its
neighboring balances, including across MPI partitions. Reconstructions exchange
only a local halo. The current three-vertex halo requires at least three vertex
rows/columns per MPI partition and at least 2×2 global cells; unsupported process
grids fail with a specific error.

`time.step.control: fixed` uses `initial`; `cfl` selects a step from the supplied
maximum and the advective/diffusive rate bound (the first step is also capped by
`initial`). Use `0 < cfl <= 1`. Fixed steps above this bound and CFL steps below
`minimum` are rejected. The final step may be shorter than `minimum` to reach
`time.end` exactly. Exhausting `maximum_steps` is an error, not a completed run.
This rate check is a stability estimate; it is not a positivity/TVD guarantee
for nonlinear high-order reconstruction. No values are silently clipped.
The full preheat examples have positive cell averages but can have negative
reconstructed Gauss-point H near the cold boundary. Reports flag such samples;
Gauss-point phase diagnostics inherit these reconstruction artifacts. A
positivity limiter is not included in this legacy stepping path.

Only finite candidates satisfying the global heat/boundary-flux balance are
accepted. `preheat_history.csv` records each accepted step, time, step duration,
Courant rate, mean/min/max physical enthalpy, and the discrete balance residual.
H/C handoff files contain the evolved cell averages. Final phase samples and
diagnostic Vecs are rebuilt from the evolved H/C reconstructions. The report
adds a physical-unit enthalpy history plot; velocity/flow convergence still
describe the fixed initial Darcy–Stokes solution.

## Validation

Build with `BUILD_TESTING=ON`, including `test_couple_initial_flow`, then run:

```bash
ctest --test-dir build -R '^couple_' --output-on-failure
```

Checks cover manufactured nonzero Stokes/Darcy velocities and pressures on both
mesh families, Gauss-point reconstruction, shared-edge normal continuity,
physical unit conversions, phase center locations, harmonic wet/dry traces,
pressure gauges, failed-solve cleanup, legacy preheat boundaries, plotting, and
serial/two-rank agreement, including legacy H/C matrix ordering with an x-split
MPI decomposition. Python tests require NumPy and Matplotlib.
Preheat tests additionally cover a hand-calculated surface cooling step,
constant-state preservation, global heat balance, stationary C, exact end-time
clipping, oversized-step rejection, CFL control, first-order Euler convergence
for a known discrete thermal eigenmode, and both x/y MPI splits.

## Full phase-coupled evolution

`mantle_couple_evolve -input <yaml> -preheat <completed output directory>` imports
cell-average H/C and advances both with SSPRK2, refreshing phase and Darcy–Stokes
flow at every stage. See [formalevolve](example/formalevolve/README.md) for the
20×20/40×40 examples, equations, reference decisions, boundary conditions, limiter,
CFL control, conservation history, and serial/MPI solver choices. The existing
initializer and legacy dry-preheat driver retain their behavior.
