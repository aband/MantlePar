# Coupled initialization and visualization

`mantle_couple_init` initializes H/C and equilibrium phase quantities, solves
Darcy–Stokes using `src/mfem`, and exports phase data at **cell centers** and
finite-element velocities at **cell and edge Gauss points**. It performs no
time stepping. All plots use physical units.

## Run

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

The plotter requires Python, NumPy and Matplotlib. Open the resulting
`visualization/index.html`, a self-contained report containing four figures
and solver tables. `visualization/report.md` contains the readable summary.
`--output-dir PATH` changes the report destination.

## Plots and physical units

| Figure | Location and quantities |
| --- | --- |
| `centers_phase.png` | Equilibrium porosity, temperature, phase pressure, solid fractions and phase region at cell centers |
| `centers_state.png` | Specific enthalpy, bulk/phase compositions and temperature derivatives at cell centers |
| `flow_cells_velocity.png` | Solid velocity and Darcy segregation flux at cell Gauss points |
| `flow_edges_velocity.png` | The same quantities at edge Gauss points, retaining both adjacent-cell traces |

A cell center is the mapped reference origin: the mean of its four vertices,
matching the legacy corner module. Phase is evaluated on the prescribed H/C
profiles **at that point**, not on cell averages. Cell polygons are colored by
these center values, with small markers showing the sample locations. Gray
cells indicate absent-phase compositions or undefined derivatives.

Velocities are reconstructed from the solved BR/H(div) coefficients, using
MPI ghosts and the existing physical basis functions. All Gauss points are
colored; direction arrows are subsampled. Stokes velocity is continuous across
an interior edge. Darcy normal velocity is continuous but its tangent can jump,
so edge output retains both traces without averaging them.

| Displayed quantity | Conversion and units |
| --- | --- |
| Coordinates | `x*l0/1000`, km |
| Temperature | `TD*dT`, K (no temperature offset) |
| Specific enthalpy | `H*h0`, J/kg, with `h0=cp*dT` |
| Phase pressure | Supplied pressure / `1e6`, MPa |
| Solid velocity | `u0*us`, displayed in cm/year |
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

- `initial_rank_*.csv`: distributed H/C/T/porosity cell averages.
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

## Validation

Build with `BUILD_TESTING=ON`, including `test_couple_initial_flow`, then run:

```bash
ctest --test-dir build -R '^couple_' --output-on-failure
```

Checks cover manufactured nonzero Stokes/Darcy velocities and pressures on both
mesh families, Gauss-point reconstruction, shared-edge normal continuity,
physical unit conversions, phase center locations, harmonic wet/dry traces,
pressure gauges, failed-solve cleanup, legacy preheat boundaries, plotting, and
serial/two-rank agreement. Python tests require NumPy and Matplotlib.
