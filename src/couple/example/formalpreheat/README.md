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
| Top Stokes velocity | (V0 erf(x)/erf(0.1), 0) for x<0.1; (V0,0) beyond |
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

Equilibrium phase quantities are diagnostic and can show melt near the surface;
the **flow operator still uses zero porosity**, so the physical Darcy segregation
flux is zero. This choice intentionally reproduces the preheating flow assumption.

Boundary conditions use the current module's whole-edge rules and consistent
Dirichlet corner precedence. The legacy extra top-right vertex-row overrides
are represented by the top Dirichlet/right traction junction, rather than
introducing special constraints inside the neighboring cell. If changing mesh
resolution, set `left_upper.start=ny−5` to retain the five-edge prescription;
the warm-strip endpoint x=0.025 must coincide with a boundary vertex.

The stored time settings follow 1500 legacy steps of size 10, but
`mantle_couple_init` only initializes and solves the initial flow. It does not
run the legacy advection/diffusion preheating loop or claim its evolved result.
