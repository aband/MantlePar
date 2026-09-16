# Stokes manufactured-solution example

This case uses the shared root-level `driver.cpp` and the schema-v2 input
reader. Put this folder at `MantlePar/example/stokes_manufactured/`, and put
the supplied Python helpers in `MantlePar/example/common/`. The helpers are
reusable for subsequent cases. This example needs no additional C++ source,
CMake registration, or CTest entry.

## Problem and exact solution

The domain is the unit square. The constant porosity is `phi = 0.04`.
This is the standalone Stokes block of MantlePar's `legacy_rescaled`
formulation: it retains the compaction pressure term. Its equations are

\[
-\nabla\cdot\sigma=(1-\phi)\boldsymbol f_s,\qquad
-\nabla\cdot\boldsymbol u_s-\frac{\phi}{1-\phi}p_s=g_s,
\]

\[
\sigma=2(1-\phi)\left(\varepsilon(\boldsymbol u_s)
-\tfrac13(\nabla\cdot\boldsymbol u_s)I\right)-p_s I,
\qquad
\varepsilon(\boldsymbol u_s)=\tfrac12(\nabla\boldsymbol u_s+
\nabla\boldsymbol u_s^T).
\]

The driver supplies the exact fields

\[
\boldsymbol u_s=
\begin{pmatrix}\sin(\pi x)\sin(\pi y)\\
\cos(\pi x)\sin(\pi y)\end{pmatrix},\qquad
p_s=\sin(\pi x)\cos(\pi y).
\]

It evaluates the matching analytic body force and pressure-row source from
these continuous equations. In particular, the velocity is not divergence
free, and `mms_stokes_source` is required. The body force in YAML follows the
driver convention above: the assembler multiplies it by `1-phi`.

| Boundary | Condition |
| --- | --- |
| Left, right, bottom | Both velocity components equal the exact velocity |
| Top | Both traction components equal `sigma*n` from the exact solution |

The pressure mass term and these boundary conditions require no constant
pressure nullspace; the input uses `pressure_nullspace: none`.

## Build and run

Use the existing PETSc/MPICH compiler configuration. The driver requires
yaml-cpp and parallel HDF5 built with the same MPI as PETSc. Register
`src/core/input.cpp` with `mantle_core` and the root-level `mantle_driver`
target as previously discussed, then run from the MantlePar root:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DMANTLE_BUILD_DRIVER=ON -DBUILD_TESTING=OFF
cmake --build build --target mantle_driver -j
```

For the Python launcher and plots, use an environment with `PyYAML`, `numpy`,
and `matplotlib` installed. Then run:

```bash
python3 example/common/run.py example/stokes_manufactured --exe build/mantle_driver
```

The launcher reads the MPICH path and rank count from `input.yaml`, records
the command and solver output in the case's `run.log`, and produces plots
only after a successful run. The default is one MPI rank. A direct launch is:

```bash
/home/renpo/system/mpich-install/bin/mpiexec -n 1 \
    ./build/mantle_driver -input example/stokes_manufactured/input.yaml
```

After a direct launch, generate plots with:

```bash
python3 example/common/run.py example/stokes_manufactured --plot-only
```

## Mesh study and changes to the input

The supplied study runs `4x4`, `8x8`, and `16x16` meshes for both `rectangular`
and `perturbed_quadrilateral`, for six solves in total. Interior perturbations
use amplitude `0.15` and a fixed seed; the outer boundary remains fixed.
Assembly uses five Gauss points per coordinate direction and five per edge.
Error integration uses seven points per coordinate direction.

For one initial rectangular solve, set `studies.convergence.enabled: false`.
For one quadrilateral solve, also set `mesh.family: perturbed_quadrilateral`.
The convergence plot requires an enabled study; profiles remain available
for a single solve. With the study enabled, its `mesh_families` list selects
the families to run.

For a two-rank run, edit the existing `parallel` section:

```yaml
parallel:
  launcher: /home/renpo/system/mpich-install/bin/mpiexec
  ranks: 2
  process_grid: [2, 1]
```

The Python launcher picks up these changes automatically. With a direct MPI
launch, also change `-n 1` to `-n 2`. A four-rank variant can use `ranks: 4`
and `process_grid: [2, 2]`. Preserve serial output when comparing partitions
by changing `output.directory`, for example to
`output_mpi2/{mesh_family}/level_{level}`. The YAML file remains the single
source for case settings; no recompilation is needed for these changes.

The `mms_*` function names select formulas already implemented in the driver;
they are not arbitrary expressions evaluated by the YAML reader. A different
analytic solution requires registering and implementing its matching functions.

## Output and interpretation

With the supplied input, per-mesh files are written below
`example/stokes_manufactured/output/{mesh_family}/level_{level}/`:

| File | Contents |
| --- | --- |
| `fields.xdmf`, `fields_0.h5` | Mesh and cell-centered fields; open the XDMF in ParaView |
| `solver_report.csv` | Outer convergence reason, iterations and true residual |
| `subsolvers.csv` | Inner-solver convergence and accumulated iteration counts |
| `metrics.csv` | Velocity/pressure L2 errors, maximum cell diameter and pressure-row residual |
| `profiles_0.csv` | Computed fields along the vertical line `x=0.5` |
| `input_used.yaml`, `resolved_run.txt` | Input and resolved run settings |

The Python helpers combine the six runs into `output/convergence.csv` and
write `convergence.png`, `convergence.pdf`, `profiles.png`, and `profiles.pdf`
in the same `output/` folder. Profile curves show the finest mesh from each
family. Changing the output root in YAML changes this destination too.

Errors are integrated from the finite-element solution, independently of the
cell-center samples used for ParaView. Observed rates use
`log(error_old/error_new)/log(h_old/h_new)`. Compare both mesh families and
check that solver residuals are small enough to resolve the spatial errors.
The pressure-row diagnostic is an absolute residual of the assembled discrete
balance, including its pressure term and source; it is not simply `div(u)`.

The driver writes solver diagnostics and stops if its convergence acceptance
criteria fail. The plotter refuses unconverged results or results whose saved
input differs from the current YAML. Keep the input unchanged when using
`--plot-only`, or rerun after editing it. Output from a previous configuration
is not automatically removed.

This case solves steady Stokes only. Transport, phase evolution and external
coupling remain disabled.

## Verification of this example

The supplied YAML was run with the previously generated shared driver and
PETSc 3.23 on one MPICH rank. All six solves passed the outer, inner-solver,
and true-residual acceptance checks. Relative true residuals ranged from
`1.24e-12` to `2.16e-12`. The Python helpers generated the combined CSV and
both PNG/PDF plots.

Observed rates between the `8x8` and `16x16` levels were:

| Mesh | Velocity L2 rate | Pressure L2 rate |
| --- | ---: | ---: |
| Rectangular | 1.997 | 1.031 |
| Perturbed quadrilateral | 2.075 | 1.012 |

These are measurements for this case and resolution range. The execution
environment required the existing single-process HDF5 adapter; collective
HDF5 output and multiple MPI ranks were not verified here. The adapter is not
included in this package. Your normal driver build continues to require
parallel HDF5.
