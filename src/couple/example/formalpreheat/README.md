# Legacy dry corner preheat initialization

This case follows `MantleSolver_cpp/src/corner/formalpreheat/{flow,trans,porosity}.cpp`
and its `corner/advdiff.cpp` driver. The legacy driver exports its `myCAdv` field
as `cellH` and `myH` as `cellC`; therefore the initial physical interpretation is
**H=3.35, C=0.01−0.12y**, despite the reversed `InitCD`/`InitHD` function names.

| Setting | This example |
| --- | --- |
| Domain | x=[0,0.5], y=[−0.5,0], approximately 158.1 km square |
| Mesh | 20×20 rectangular cells, a smaller startup example than the legacy 32×32/50×50 scripts |
| Flow porosity | Explicitly zero everywhere, as in legacy `porosity.cpp` |
| Initial specific enthalpy | H=3.35, or 2,090,400 J/kg |
| Initial composition | C=0.01−0.12y, from 0.01 at the top to 0.07 at the bottom |
| Bottom Stokes velocity | (0, 3.2 cm/year) |
| Top Stokes velocity | (V0 erf(x)/erf(Xwarm), 0) for x<Xwarm; (V0,0) beyond; Xwarm is the warm endpoint (default 0.025) |
| Left Stokes boundary | Zero normal speed and zero tangential traction; upper five edges prescribe the legacy vertical velocity profile |
| Right Stokes boundary | Outflow traction (y,0), matching subtraction of `abs(y)*n` in the legacy natural-load assembly |
| Darcy boundary | Zero assembled normal velocity on every side |
| Body forces | Stokes (0,−1), Darcy (0,0) in the legacy nondimensional system |
| Thermal surface | Prescribed T/dT=3.35 on x<0.025, zero elsewhere; all other sides zero flux |
| Thermal diffusion | 8e−8 nondimensional, approximately 1.265e−6 m²/s |

The reference scales match legacy values: l0=316227.766 m, u0=5e−5 m/s,
p0=1.58114e9 Pa, dT=520 K, h0=624000 J/kg. The phase reference density is
3000 kg/m³, as in the legacy phase package; the flow density contrast is
3300−2800=500 kg/m³. The top cold boundary retains the benchmark's literal
zero temperature value; in the current absolute T/dT convention this is 0 K.
The warm strip is 1742 K.
The top solid-velocity ramp now reaches 3.2 cm/year at the warm-strip endpoint;
this replaces the legacy velocity cutoff of 0.1.

Equilibrium phase quantities are diagnostic and can show melt near the surface;
the **flow operator still uses zero porosity**, so the physical Darcy segregation
flux is zero. This choice intentionally reproduces the preheating flow assumption.

Boundary conditions use the current module's whole-edge rules and consistent
Dirichlet corner precedence. The legacy extra top-right vertex-row overrides
are represented by the top Dirichlet/right traction junction, rather than
introducing special constraints inside the neighboring cell. If changing mesh
resolution, set `left_upper.start=ny−5` to retain the five-edge prescription;
the warm-strip endpoint (default x=0.025) must coincide with a boundary vertex.

The stored time settings follow 1500 legacy steps of size 10.
`mantle_couple_init` initializes and solves the initial flow only.
`mantle_couple_preheat` additionally runs the forward Euler preheat loop to
t=15000: dry flow stays fixed, H advects and diffuses, and C stays fixed.
Build and run that target to obtain evolved H/C for the `formalEvolve` handoff.
Use a separate output directory to preserve the initial-state result.

The root `./run.sh --grid 20` automates both stages: it runs this dry preheat and
passes the completed H/C matrices directly into full evolution on the same mesh.
Use `--grid 40`, or `--mesh 40 60`, to change both meshes together. The script sets
`left_upper.start=ny-5`. `--warm-end X` changes the endpoint of the warm top strip
and the solid/Stokes velocity ramp cutoff in both stages, using nondimensional x
(default 0.025, allowed `0 < X <= 0.5`). The horizontal solid velocity reaches
3.2 cm/year at X; the vertical solid velocity is zero along the top.
The endpoint must coincide with a boundary vertex: `X*nx/0.5` must be an integer.
The evolution melt outlet stays at x=0.05, which also requires nx to be a multiple
of 10. Thus `./run.sh --grid 30 --warm-end 0.05` is valid, while the default
endpoint still requires nx to be a multiple of 20. Generated preheat inputs use CFL control with the legacy
maximum step of 10 and a smaller minimum, so finer meshes can reduce the step.
The stored example YAML retains its original fixed-step settings.

Default preheat output is `output/evolved_NXxNY`, with `preheat_input.yaml`,
`cellH1.dat`, `cellC1.dat`, `transport_state.json`, and a final visualization report.
`--preheat PATH` selects that output directory; `--preheat-final-time T` changes the
default 15000. Evolution starts its own clock at zero. The combined movie covers
the full evolution after the preheat handoff.

The legacy thermal approximation diffuses H directly (H=T/dT for this thermal
step), while equilibrium temperature and porosity are diagnostic. Advection uses
ML-WENO (3,2); diffusion uses constant cell samples and MantlePar's normal-line
flux kernels. Time steps are checked against an advective/diffusive rate bound,
and `preheat_history.csv` records the heat balance and physical enthalpy history.

The output directory also contains `cellH1.dat` and `cellC1.dat`, compatible
with legacy `formalEvolve`'s `ReadVectorTransport` format. These are the stored
cell-average fields at the actual output time, with x varying
fastest and rows going from bottom to top. H remains nondimensional (initially 3.35)
and C is an unscaled fraction. `transport_state.json` records the mesh, units,
scales and time. Use matching geometry, resolution and scales when transferring
the pair. The metadata's `time` and `accepted_steps` distinguish the initial
state from the evolved preheat state.
