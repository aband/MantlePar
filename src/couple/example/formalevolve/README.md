# Phase-coupled corner evolution

## Run the 20×20 case and make all videos

From the repository root (the directory containing `src`):

```sh
./run.sh
```

`run.sh` builds both drivers in `build/evolution`, runs dry preheat to t=15000,
checks its completed H/C matrices, then imports those exact cell averages into
full phase-coupled evolution. The evolution clock resets to zero and advances to
t=1, recording every accepted state. It renders the final preheat report, four
evolution videos, and a fifth video containing all four synchronized views.
Both stages use the same selected mesh. Selected output directories are reused
on subsequent runs.

`input.yaml` is the single shared evolution template. It contains the 20×20
defaults; `--grid` or `--mesh` selects the run's mesh. The runner also applies the
selected final time, warm endpoint, matching velocity cutoff, and output path,
then saves the resolved settings as `evolve_input.yaml` in the run directory.

```sh
./run.sh --dry-run                  # Inspect commands without changing files.
./run.sh --grid 40                  # Run both stages on 40×40.
./run.sh --mesh 40 60               # Run both stages on 40×60.
./run.sh --grid 30 --warm-end 0.05  # Warm strip ends at x=0.05 in both stages.
./run.sh --preheat-final-time 15000 --final-time 5  # Nondimensional final times.
./evolve.sh --grid 40               # Evolve the existing completed 40×40 preheat.
./run.sh --render-only              # Re-render movies; skip both simulations.
./run.sh --grid 20 --report-only    # PDF from saved conditions; no solver/movies.
./run.sh --preheat /path/to/preheat --output /path/to/evolution
./run.sh --help
```

`--preheat-final-time` controls when preheat stops (default 15000).
`--final-time` controls when full evolution stops (default 1), measured from zero
after importing preheat H/C. Both times are nondimensional and must be finite and
positive. The earlier names `--preheat-end-time` and `--end-time` remain aliases.

`--grid N` selects a square mesh; `--mesh NX NY` selects a rectangular mesh, with
ny at least 5. `--warm-end X` sets the warm surface interval to `[0,X]` in both
generated stage inputs. X uses nondimensional coordinates on the domain `[0,0.5]`,
defaults to 0.025, and must satisfy `0 < X <= 0.5`. The warm temperature is 1742 K.
The same X sets the top solid/Stokes velocity cutoff in both stages:
`vs_x = V0*erf(x)/erf(X)` for x<X, and `vs_x = V0` for x>=X, with
V0=3.2 cm/year. Vertical solid velocity is zero along the top.
Full evolution uses `type: outflow` for both H and C on `warm_top`, so their
advective boundary states come from the reconstructed interior values. This
region follows `--warm-end`. The existing outflow-only policy rejects backflow
instead of prescribing incoming H/C. Thermal diffusion retains the warm
prescribed temperature; preheat retains its original advection conditions.

The selected endpoint must align with a mesh vertex: `X*nx/0.5` must be an integer.
The independent melt outlet remains `[0,0.05]` and must also align, so nx must be a
multiple of 10. The default warm endpoint requires nx to be a multiple of 20;
`--grid 30 --warm-end 0.05` is an example using another valid combination. Invalid
combinations fail before building or running; endpoints are not automatically
rounded. `--render-only` uses the recorded data and does not need the warm endpoint.
When using `evolve.sh` with existing preheat, pass the same `--warm-end` value to
retain its thermal boundary and velocity cutoff during evolution. The selected value is printed at
startup and saved in each generated input YAML. The upper five left edges are
updated to start at `ny-5`.
Preheat keeps the legacy maximum time step of 10, with CFL control to reduce it on
finer grids. Evolution retains its adaptive CFL control.

Default stage directories are:

- Preheat: `src/couple/example/formalpreheat/output/evolved_NXxNY`.
- Evolution: `src/couple/example/formalevolve/output/NXxNY`.

For the default 20×20 mesh, evolution uses `src/couple/example/formalevolve/output/20x20`. Open its
`visualization/index.html`; `visualization/report.md` defines the fields and
plotting conventions. All five MP4 files are in `visualization/`:

| Video | Display |
| --- | --- |
| `combined_evolution.mp4` | All four videos synchronized: velocity on the left, temperature/porosity/phase stacked on the right, preserving every original panel and label at 2560×2520 |
| `velocity_evolution.mp4` | Solid velocity, liquid velocity and Darcy segregation flux; labeled quivers at Gauss points and constant-color streamlines, with no magnitude heatmaps |
| `temperature_evolution.mp4` | Cell-center temperature in K and its change; local pressure-dependent eutectic temperature is marked |
| `porosity_evolution.mp4` | Cell-center porosity and its change in percentage points |
| `phase_evolution.mp4` | Cell-center solid 1, solid 2 and liquid fractions, plus categorical phase regions |

The movies cover full evolution after the preheat handoff; preheat exports its
final fields and history plot. Each stage retains its generated input
(`preheat_input.yaml` or `evolve_input.yaml`). Logs for build, preheat, evolution,
and plotting are in the evolution directory as `run_*.log`. A failed preheat or
invalid H/C handoff stops the pipeline before evolution.

`visualization/conditions.pdf` lists the initial and boundary conditions for both
stages: H/C profiles and imported matrix ranges, named region extents, solid and
Darcy flow conditions, advection policies, thermal diffusion, phase pressure,
porosity initialization, physical reference scales, and configuration hashes.
It explains the warm H/C outflow policy and the distinction between preheat H
diffusion and full-evolution temperature diffusion. The video gallery links it.
`conditions.json` beside it preserves the exact input mappings and provenance.

Reports read `input_used.yaml` from the completed output. Each new pipeline run
archives its preheat configuration as `preheat_input_used.yaml` in the evolution
directory, alongside `source_state.json` and `starting_cellH/C.dat`, so reusing
the preheat directory does not change an earlier report. Older outputs without
that archive use the source directory's saved input and are labeled accordingly.
`--report-only` and `--render-only` use these saved settings, not current CLI
boundary or final-time values. A PDF can also be generated directly:

```sh
python3 src/couple/conditions_report.py --input-dir /path/to/completed/evolution
```

PDF generation needs ReportLab. The renderer first uses the active Python when
ReportLab is available, or `PDF_PYTHON` when set, then the Codex bundled runtime
if installed. This machine already has the bundled dependency; no installation
or simulation is required for report-only use.

No videos draw mesh lines or point markers. Scalar values are interpolated in
space from cell centers for a smooth display. Categorical regions use nearest-center
labels. Streamlines interpolate the Gauss-point vectors only inside their sample
convex hull; liquid velocity is undefined at zero porosity. Actual Gauss locations
anchor the quiver arrows. Time snapshots are not interpolated. Every recorded state
appears, with fixed color limits and arrow references throughout each video.

The `-couple_movie_snapshots` solver option records compact little-endian float64
arrays in `evolution_centers_rank_*.bin` and `evolution_gauss_rank_*.bin`, with sample
coordinates, an accepted-step clock and `evolution_series.json` describing their
layout. Temperature includes both `T` and the phase model's `Te(P)` in K. Phase
data are evaluated at mapped cell centers; velocities are sampled at cell Gauss
points. These snapshots reuse each accepted state's flow solve.

Dependencies: the normal PETSc/MPI/HDF5 build environment, CMake, Python with
NumPy/Matplotlib/SciPy/PyYAML, and FFmpeg with libx264. `--hdf5-root`, `--build-dir`,
`--jobs`, `--preheat`, `--seconds`, and `--fps` override the defaults. `PYTHON`
selects the Python interpreter. Older porosity-only snapshot sets cannot supply
velocity/temperature/phase videos: run once with the new full snapshot option.

## Physical setup

These cases import the **completed** 20×20 and 40×40 dry-preheat solutions, not
the initialization-only outputs. The verification runs end at t=1 (about 200.55 years). The evolution clock starts at zero; the source
preheat time is 15000 (about 3.008 million years). `source_state.json` records it.
`starting_cellH.dat` and `starting_cellC.dat` preserve the exact imported averages.

From the MantlePar root, evolve existing preheat results on either mesh with:

```sh
./evolve.sh --grid 20
./evolve.sh --grid 40
```

Both commands use the shared `input.yaml`. To run the built driver directly with
the default 20×20 settings:

```sh
build/evolution/src/couple/mantle_couple_evolve \
  -input src/couple/example/formalevolve/input.yaml \
  -preheat src/couple/example/formalpreheat/output/evolved_20x20
```

The input loader checks the manifest's completion status, mesh, domain, reference
scales, units, field lengths, finite values, and admissible cell averages. It does
not interpolate between grids. Natural DMDA ordering is restored independently of
MPI partitioning. General checkpoints of an interrupted evolution are not implemented.

## Equations and reference decisions

The authority is `MantleSolver_cpp/doc/writing/dissertation/formal/reviewed.pdf`,
especially (2.26), (2.81), (3.123)–(3.131), (3.210), and (4.1), (4.11)–(4.14).
The corner geometry and boundary profiles follow `corner/formalEvolve`; the PDF's
section 4.3 uses a different 2×2 domain and is not substituted for this preheated
corner domain.

With dimensionless `T = temperature / dT`, `L = latent_heat / (cp*dT)`, and
`q = phi^(1+theta) * ud`, the implemented equations are

```
v     = us + q
ce    = phi * mean(cl) + (1-phi) * mean(cs)
veff  = us + mean(cl) * q / ce
C_t   = -div(veff * C)
H_t   = -div(v*T - L*(1-phi)*us - kappa*grad(T))
        - alpha0*l0*g/cp * T*v_y
```

The `ce=0` pure-component limit uses `veff=us`; there is no division by zero.
`ud` is the assembled scaled relative velocity, **not** the segregation flux or
liquid velocity. For nonzero theta the factor in `q` is essential (equation 2.26).
Gravity points down while the mesh y coordinate points up. Lithostatic pressure,
in Pa, controls phase equilibrium; solved flow pressures are not substituted.

Interior fluxes are computed once and applied with opposite signs to both cells.
Composition uses local Lax–Friedrichs flux; enthalpy uses the temperature LF flux
minus the latent solid flux, as in (4.11). The LF dissipative term uses an absolute
normal speed as in (3.9); the apparent parentheses/sign and repeated-trace typos in
(4.11) are not copied. Both phase compositions use arithmetic edge averages, while
porosity uses the harmonic mean. An exactly quadrature-dry cell additionally closes
its incident melt fluxes on both sides, implementing the zero-divergence branch of
(3.123); otherwise a phase boundary between cell Gauss points and an edge can create
spurious mixture divergence. The example sets both compaction cutoffs to zero.

Chemical diffusion is neglected following section 4.2.2. Thermal conduction is
**retained** from (2.81): this corner has the cooled, impermeable surface for which
the document explicitly cautions against the conduction-free simplification.
Conduction samples the actual phase temperature on the existing normal-line
stencil. Adiabatic cooling uses physical cell quadrature.

Each SSPRK2 stage reconstructs H/C, evaluates phase, assembles and solves the
Darcy–Stokes system, and evaluates both transport residuals. A further solve at the
accepted state supplies consistent output and the next step's first residual.
The CFL controller rechecks both stages and halves rejected steps. Rejection never
clips the conserved cell averages. The legacy loop's unchanged H, overwritten
updated temperature, and uninitialized diffusion flux are not carried over.

## Bounds and boundary conditions

The imported preheat averages are untouched. ML-WENO polynomials are scaled about
their averages to the neighboring-cell bounds at every used cell, edge and diffusion
sample; H remains nonnegative and C stays in `[0,Xe]`. This extra limiter addresses
negative Gauss-point H overshoots present in the unlimited preheat reconstruction.
It can reduce spatial order near extrema and boundaries. The flow formulation still
requires `phi < 1`; fully liquid states and singular pure-component phase derivatives
are rejected explicitly.

Stokes boundary conditions retain the preheat corner setup, including the upper
five left-side edges. Darcy pressure potential is zero on the upper-left outlet
`0 <= x <= 0.05`; other Darcy boundaries have zero normal velocity. The fixed outlet
width matches the legacy two-edge outlet at 20×20 and spans four edges at 40×40.
The warm top region has free H/C outflow using the interior state, with no
prescribed incoming H or C; backflow there is rejected. On other boundaries,
H inflow is 3.35 on the bottom and zero elsewhere; C inflow is 0.01 as in legacy
`formalEvolve/trans.cpp`. Outflow extrapolates the interior state. The default thermal top
boundary is T=3.35 for x<0.025 and zero elsewhere, with insulated other sides.
The run scripts' `--warm-end` option changes the warm interval endpoint and the
top solid-velocity ramp cutoff together in both stages.
These inherited boundary choices are example conditions, not a validated ridge model.

## Solver and output

The examples select serial `fgmres / sparse_lu` to make repeated wet-flow solves
practical. The solver converts copies of the nested MFEM matrices to AIJ, maps fields
through their original global index sets, and checks the residual in the original
nested system. Nested-dissection ordering controls fill. A 1e-12 shift applies only to LU factors to
handle zero pressure pivots; the outer FGMRES iteration corrects the factor error
against the unchanged operator.
No matrix signs, physics or pressure gauges are altered. For MPI runs use
`fgmres / schur`, CG/Jacobi velocity and CG pressure blocks, and set ranks/process_grid.
Sparse LU currently requires one rank and a nonsingular pressure system.

`evolution_history.csv` includes both conserved integrals, physical enthalpy values,
composition and porosity ranges, time in years, CFL estimate, rejected-step count,
mixture-divergence error and flow residual. Boundary fluxes and adiabatic source on
accepted rows are averages of the two RK stages, so the recorded budgets can be
checked independently. Row zero records the initial instantaneous rates.

Final H/C matrices use the same nondimensional restart units as preheat. Plots use
J/kg, K, km, MPa, cm/year and years; composition and phase fractions are unscaled.
Phase plots are at cell centers; flow plots are at Gauss points. The report shows the
**final** flow solve, rather than the original dry preheat solve. Incomplete runs
leave incomplete manifests and cannot be mistaken for a completed final state.

## Verification

The coupling CTest suite covers scaled relative velocity (including nonzero theta),
exact H/C import, serial/MPI agreement for both partition directions, rectangular
and perturbed quadrilateral meshes, wet evolution of both conserved fields,
mixture continuity, finite-volume energy/species budgets, mean preservation of the
limited reconstruction, and SSPRK2 temporal order against an exact semidiscrete
thermal-decay solution. It also checks invalid inputs and report generation.

Both preheat-driven cases completed t=1 (200.5503 years), with exact H/C import
and no rejected steps. The largest absolute per-step integral balance errors are:

| Grid | Accepted steps | H balance error | C balance error | Final relative flow residual |
| --- | ---: | ---: | ---: | ---: |
| 20×20 | 80 | 1.80e-15 | 2.93e-17 | 1.21e-14 |
| 40×40 | 249 | 4.00e-15 | 6.63e-17 | 3.66e-14 |

Balance errors use the nondimensional conserved integrals and include boundary
fluxes and the adiabatic source. Each output directory contains `verification.json`
with these checks; `visualization/index.html` includes final fields, changes from
the preheat state, and the time history. These are short verification runs.

For the actual 20×20 case at t=1, reducing CFL from 0.4 to 0.2 increases the accepted
step count from 80 to 160. Maximum final differences are 0.0544 J/kg in specific
enthalpy and 2.42e-8 in composition (RMS: 0.00353 J/kg and 1.95e-9). This is a time-step
sensitivity check on one grid, not a spatial convergence result.

## Porosity videos

Pass `-couple_porosity_snapshots` to `mantle_couple_evolve` to record the initial
porosity and every accepted step. The optional `porosity_series_rank_*.csv` files
store phase porosity from the bounded H/C reconstruction at mapped cell centers,
matching the static phase plots. A completed `porosity_series.json` lists all MPI
parts and frames. Recording does not add flow solves or change the time steps.

Render a completed snapshot series with NumPy, Matplotlib and FFmpeg:

```sh
python3 src/couple/animate_porosity.py \
  --input-dir src/couple/example/formalevolve/output/porosity_movie_20x20 \
  --output src/couple/example/formalevolve/output/20x20/visualization/porosity_evolution.mp4 \
  --porosity-max 0.12

python3 src/couple/animate_porosity.py \
  --input-dir src/couple/example/formalevolve/output/porosity_movie_40x40 \
  --output src/couple/example/formalevolve/output/40x40/visualization/porosity_evolution.mp4 \
  --porosity-max 0.12
```

The first videos use separate reruns in `output/porosity_movie_20x20` and
`output/porosity_movie_40x40`, preserving the original verification outputs.
`input_used.yaml` in each rerun directory records its full configuration.
Future runs can write the snapshots directly into the normal output directories.

The MP4s use H.264 at 1280×720, 24 fps, and 12 seconds of evolution plus one-second
pauses at the endpoints. `--seconds` and `--fps` control playback. Both panels use
fixed color limits: porosity as a fraction and its change in percentage points.
The 0.12 porosity limit is shared by these two movies. Geometry is in km and time
in years after preheat. Snapshot times are quantized to video frames, reserving at
least one frame per accepted state without interpolating the fields. If needed,
playback duration increases to fit every state. Each video has an HTML
player, a PNG poster and a JSON file mapping every video frame to its recorded state.
The renderer checks complete MPI coverage, timestamps, bounds, and agreement of
the last porosity frame with the final cell-center export before encoding.
