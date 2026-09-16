# Half-corner coupled flow

This example ports the active porosity, velocity and load functions from
[MantleSolver_cpp, parallel_debug, src/corner/half](https://github.com/aband/MantleSolver_cpp/tree/parallel_debug/src/corner/half)
to the shared MantlePar steady driver. It uses the piecewise upwelling/surface
velocity in `bndryVs`; the arctangent corner-flow formula commented out in the
old file is not used. Porosity is prescribed and fixed. Transport and phase
updates are disabled.

The supplied `driver.cpp` extends
[MantlePar commit 818ba395](https://github.com/aband/MantlePar/commit/818ba395bdec7013dea1c720adaabab5151c0eba).
Extract the archive at the repository root, replacing `driver.cpp` and adding
`example/half_corner/`. The included common runner and requirements are unchanged
copies from that commit. The existing CMake configuration and input reader work
with this update.

Rebuild using your existing PETSc/MPICH configuration, then run from the
repository root:

```bash
cmake --build build --target mantle_driver -j
python3 -m pip install -r example/common/requirements.txt
python3 example/common/run.py example/half_corner --exe build/mantle_driver --no-plots
python3 example/half_corner/plot.py
```

For the corresponding perturbed mesh:

```bash
python3 example/common/run.py example/half_corner/input_perturbed.yaml --exe build/mantle_driver --no-plots
python3 example/half_corner/plot.py example/half_corner/input_perturbed.yaml
```

Each launch writes `example/half_corner/run.log`, replacing the previous log.
The two mesh families have separate directories under `example/half_corner/output/`.
Use `--no-plots` on the common runner because this case has its own two-dimensional
field plotter. The plotter checks solver acceptance and the saved input before
reading the fields, and draws the actual quadrilateral cells, including mesh
perturbations.

The model settings are:

| Quantity | Value |
| --- | --- |
| Domain | `[0, 0.5] × [-0.5, 0]` |
| Initial grid | `20 × 20`, matching the old `half/fast.sh` |
| Formulation | Coupled Stokes–Darcy, `legacy_rescaled`, `theta = 0` |
| Stokes force callback | `(0, -1)`; assembly supplies the solid-fraction factor |
| Darcy force and pressure-row sources | Zero |
| Velocity scale `u0` | `5e-5 m/s` |
| Prescribed speed `V` | `(3.2 cm/year)/u0 = 2.0294266869609333e-5` |
| Porosity length scale | `316228 m`, retaining the rounded number in the old porosity file |

With `d = 120000/316228`, `a = 20/316228`, and `z = |y|`, the prescribed porosity is

```text
phi(x,y) = 0.05 * ((d-z)/d)^2 * (1-|x|/(z+a)), if z < d and |x| < z+a;
phi(x,y) = 0, otherwise.
```

The pointwise maximum is `0.05` at the origin. HDF5 stores physical cell-average
porosity, whose maximum is smaller on a finite grid. The dimensional cell width
on the initial grid is about `7.9 km`; this quick test cannot resolve the `20 m`
apex or establish spatial convergence. The wedge is sampled with physical cell
and edge quadrature, including cells cut by its wet/dry boundary.

The edge-based boundary conditions are:

| Boundary | Solid flow | Darcy condition |
| --- | --- | --- |
| Bottom | `u_s = (0, V)` | Zero assembled normal velocity |
| Top | `u_s,y = 0`; `u_s,x = V*erf(x)/erf(0.04)` for `x < 0.04`, otherwise `V` | Zero assembled normal velocity |
| Left, `-0.5 <= y <= -0.125` | `u_s,x = 0`, zero tangential traction | Zero assembled normal velocity |
| Left, `-0.125 <= y <= 0` | Prescribed legacy velocity: `(0,V)` below the surface cutoff, `(0,0)` at the top endpoint | Zero assembled normal velocity |
| Right | Total traction `t = -|y| n`, hence `(-|y|,0)` | Zero natural pressure potential |

The velocity callbacks retain the old comparisons at `y = -0.00005`, and the
`x > 0.45, y > -0.3` override. The old scalar natural load `|y|` was subtracted
from the momentum RHS. MantlePar adds a traction vector, which explains the
negative sign above. The right-wall Stokes traction fixes the pressure level:
`pressure_nullspace: none` and `remove_pressure_nullspace: false` are intentional.

There are three discrete boundary differences from the old `bndryTypeMarker`:

- At the left-wall transition vertex, `y = -0.125`, MantlePar applies the upper
  segment's prescribed vertical velocity; the legacy global-to-local mapping
  leaves that vertex's vertical DOF free.
- The bottommost right-edge normal bubble remains natural. The old bottom-row
  override constrained it.
- The right-wall vertex one cell below the top remains natural. The old special
  top-right-cell rule constrained both velocity components there while leaving
  the right-edge bubble natural.

Top and bottom corner vertex velocities are still prescribed. These explicit
edge-based rules provide a reproducible new example, but do not reproduce the
old DOF stencil exactly. The upper-left segment covers five cells on the initial
grid. Its endpoint stays at `-0.125` when refining, so the physical problem stays
fixed; the old `Ny-5` test changed its physical length with resolution. Grid sizes
such as `40 × 40` and `80 × 80` keep that endpoint on a boundary vertex.

After a successful run, inspect:

- `solver_report.csv`: `converged = 1`, `subsolvers_converged = 1`, and the true
  residual within the reported threshold.
- `metrics.csv`: discrete Stokes and Darcy pressure-equation mass residuals.
- `half_corner.png` / `half_corner.pdf`: cell-average porosity, solid velocity,
  physical segregation flux and assembled Stokes pressure.
- `fields.xdmf`: all requested fields for ParaView. The segregation flux is
  `q = phi^(1+theta) * darcy_velocity`. Pressure potential is undefined in dry
  cells; its NaNs are accompanied by `darcy_pressure_potential_valid`.

This case has no exact whole-domain solution attached, so convergence errors
are disabled. The screen's true relative residual measures the linear-system
solve, not spatial discretization error. Start with these two runs before adding
a refinement study or moving on to `mlweno_array`.

Offline validation covers the new C++ callbacks against the compiled legacy
functions, local C++ schema/driver validation, boundary-region resolution, and
plot rendering with synthetic data. The complete PETSc/MPI executable has not
been built or run in the preparation environment; that run remains the next
validation step on your installation.
