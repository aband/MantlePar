# Darcy manufactured-solution example

Place `input.yaml` and this `README.md` in `MantlePar/example/darcy_manufactured/`.
This case reuses the shared root-level `driver.cpp`, the schema-v2 core input
reader, and the Python helpers already supplied under `example/common/`.
Use the updated `plot_results.py` for the powers-of-ten log-log axes.

The latest shared-driver update changes only the signed vertical forcing
in the column reference. This example still uses the trigonometric
`manufactured` reference and the same `mms_darcy_*` callbacks. Its
physical problem, boundary conditions and numerical settings are unchanged.

## Problem

On the unit square, this example solves the standalone Darcy block of
`legacy_rescaled`, including its pressure reaction term. Let
`phi = 0.04`, `theta = 0`, and `a = phi^(theta+1/2) = 0.2`. The equations for
assembled Darcy velocity and pressure are

\[
\boldsymbol u_d+a\nabla p_d=\boldsymbol f_d,\qquad
-a\nabla\cdot\boldsymbol u_d-\frac{p_d}{1-\phi}=g_d.
\]

The exact solution, already implemented in the shared driver, is

\[
\boldsymbol u_d^\star=
\begin{pmatrix}
\cos(\pi x)\sin(\pi y)\\
\sin(\pi x)\cos(\pi y)
\end{pmatrix},\qquad
p_d^\star=\cos(\pi x)\sin(\pi y).
\]

Writing `sx = sin(pi*x)`, `cx = cos(pi*x)`, `sy = sin(pi*y)`, and
`cy = cos(pi*y)`, the analytic data are

\[
\boldsymbol f_d=
\begin{pmatrix}(c_x-a\pi s_x)s_y\\(s_x+a\pi c_x)c_y\end{pmatrix},
\qquad
g_d=2a\pi s_xs_y-\frac{c_xs_y}{1-\phi}.
\]

The velocity divergence is `-2*pi*sx*sy`, so the pressure-row source is
nonzero. The driver selects these formulas through `mms_darcy_force` and
`mms_darcy_source`; the right-hand side is independent of the discrete matrix.
Stokes flow and the cross-pressure coupling are disabled.

## Boundary conditions and variables

| Boundary | Prescribed data |
| --- | --- |
| Left, right, bottom | Exact outward normal assembled velocity `u_d* . n` |
| Top | Exact assembled pressure `p_d* = 0` |

The normal velocity is `-sin(pi*y)` on each vertical side and `-sin(pi*x)`
on the bottom. The driver computes these signs from the outward normal.
Tangential velocity is not prescribed.

The YAML names follow the mixed implementation: normal velocity is
`type: dirichlet`, while prescribed pressure is `type: neumann`. The natural
pressure load and its porosity scaling are handled by the boundary-condition
module; supply the assembled pressure itself in `mms_darcy_pressure`.

The output distinguishes the rescaled unknowns from the derived fields:

| Output field | Meaning in this case |
| --- | --- |
| `darcy_velocity` | Assembled rescaled unknown `u_d` |
| `darcy_pressure_assembled` | Assembled pressure unknown `p_d` |
| `darcy_segregation_flux` | `phi^(1+theta)*u_d = 0.04*u_d` |
| `darcy_pressure_potential` | `p_d/sqrt(phi) = 5*p_d` |

This system has no constant pressure nullspace; the pressure reaction term
is active. The input uses `pressure_nullspace: none`.

## Run

After replacing the shared driver, rebuild using your existing configuration:

```bash
cmake --build build --target mantle_driver -j
```

Then run from the MantlePar root:

```bash
python3 example/common/run.py example/darcy_manufactured --exe build/mantle_driver
```

The launcher uses the MPICH executable specified in YAML. The default is one
rank. A direct invocation is:

```bash
/home/renpo/system/mpich-install/bin/mpiexec -n 1 \
    ./build/mantle_driver -input example/darcy_manufactured/input.yaml
```

Regenerate plots from an existing run using:

```bash
python3 example/common/run.py example/darcy_manufactured --plot-only
```

The Python helpers require `PyYAML`, `numpy`, and `matplotlib`. No additional
C++ source or CMake registration is needed for this case.

## Meshes and parallel settings

The input runs six solves: `4x4`, `8x8`, and `16x16` cells on both rectangular
and perturbed quadrilateral meshes. The perturbation amplitude is `0.15`
with a fixed seed and fixed outer boundary. Assembly uses five Gauss points
per coordinate direction and five per edge; L2 error integration uses seven
per coordinate direction.

For one initial solve, set both `studies.convergence.enabled: false` and
`output.plots.convergence: false`; the driver then uses `mesh.family` and
`mesh.cells`. The input reader rejects a convergence plot when the study
is disabled. For a two-rank run, edit the existing
parallel section to `ranks: 2` and `process_grid: [2, 1]`. The Python launcher
reads both automatically; a direct launch must also use `-n 2`. Retain the
MPICH launcher matching PETSc.

For separate serial and parallel results, use separate output roots, such as
`output/{mesh_family}/level_{level}` and
`output_mpi2/{mesh_family}/level_{level}`. The updated `clean.sh` removes
directories named exactly `output`; it does not remove `output_mpi2`.

## Results

Each mesh writes to `output/{mesh_family}/level_{level}/` within this case:

| File | Contents |
| --- | --- |
| `fields.xdmf`, `fields_0.h5` | Parallel HDF5/XDMF mesh and fields; open the XDMF in ParaView |
| `solver_report.csv`, `subsolvers.csv` | Outer/inner convergence and true residuals |
| `metrics.csv` | Velocity/pressure L2 errors and discrete pressure-row residual |
| `profiles_0.csv` | Computed fields along the vertical line `x=0.25` |
| `input_used.yaml`, `resolved_run.txt` | Input and resolved simulation settings |

The Python helpers write `output/convergence.csv`, `convergence.png/.pdf`,
and `profiles.png/.pdf`. Rates use the maximum physical cell diameter.
Errors evaluate the finite-element solution with quadrature; ParaView fields
are sampled at cell centers. The mass diagnostic measures the complete
discrete pressure-row balance, including its pressure term and source.

Keep the input unchanged when using `--plot-only`; the plotter checks it
against the saved input. Rerun after changing the configuration. An
unconverged solve is rejected before an error table is accepted.

## Previous numerical verification

All six serial solves passed the outer, inner-solver and true-residual checks
with the previously generated shared driver and PETSc 3.23. Relative true
residuals ranged from `2.58e-14` to `2.27e-13`. Measured rates from `8x8` to
`16x16` were:

| Mesh | Velocity L2 rate | Pressure L2 rate |
| --- | ---: | ---: |
| Rectangular | 2.009 | 0.998 |
| Perturbed quadrilateral | 2.044 | 1.023 |

The updated plotting helper produced the CSV and PNG/PDF plots successfully.
These runs used the existing single-process HDF5 adapter required by the
verification environment. Multiple MPI ranks and collective HDF5 writes
remain unverified here; your normal driver build uses parallel HDF5.

## Check of this update

The YAML callback names, active fields, output filenames and profile columns
were checked against the updated shared driver and the schema-v2 input reader.
The shared Python helpers retain powers-of-ten log-log convergence axes.
The numerical results above are from the earlier driver; this update was not
rerun with PETSc/MPI or parallel HDF5 in the current environment.
