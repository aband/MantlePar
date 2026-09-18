# MantlePar transport flux tests

Place these eight files in `src/mlweno/test/`:

- `test_diffusiveflux.cpp`
- `test_diffusiveflux_convergence.cpp`
- `test_advection_diffusion.cpp`
- `transport_test_support.h`
- `diffusiveflux_tests.cmake`
- `advection_diffusion_tests.cmake`
- `plot_transport_tests.py`
- `transport_flux_tests.md`

The tests use the previously supplied `advectiveflux.h/.cpp` and
`diffusiveflux.h/.cpp` in `src/mlweno/`, plus the current `Reconstruction` API.
They target MantlePar main commit `295ec46767df906cefd384492854fd22bf1bff4b`.

## Build and run

Add the two flux sources to the existing source list in `src/mlweno/CMakeLists.txt`:

```cmake
add_library(mantle_mlweno
    reconstruction.cpp
    tensorstencilpoly.cpp
    advectiveflux.cpp
    diffusiveflux.cpp
)
```

Keep its existing include directories and `PUBLIC mantle_core` link. Add these
two lines to `src/mlweno/test/CMakeLists.txt`, alongside the existing tests:

```cmake
include("${CMAKE_CURRENT_LIST_DIR}/diffusiveflux_tests.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/advection_diffusion_tests.cmake")
```

From the repository root, retain your usual PETSc/LAPACK configuration options:

```bash
cmake -S . -B build -DBUILD_TESTING=ON
cmake --build build --target test_diffusiveflux test_diffusiveflux_convergence test_advection_diffusion -j2
ctest --test-dir build -R '^mlweno_(diffusion|advection_diffusion)_' --output-on-failure -j2
```

These are **serial numerical tests using real PETSc**. They reject execution on
more than one rank. They do not validate distributed ownership or halo exchange.
Run the executables directly or through CTest, without `mpiexec -n 2`.
Double precision or better is required. All assertions remain active in Release.

The generated test files contain no PETSc substitutes. Local development
validation used an isolated serial compatibility harness because this workspace
does not have a native PETSc installation; native integration must be checked in
your build.

## Coverage

`test_diffusiveflux.cpp` checks Lagrange polynomial exactness, multiple sample
counts, normal geometry, thin cells, constant/linear/quadratic accuracy, reversed
orientation, all four boundary directions, nonlinear diffused quantities,
prescribed state/quantity/flux conditions, zero diffusivity, invalid arguments
and output preservation. Sample, coefficient and boundary-data partials are
checked against finite differences. Exact analytical samples separately verify
the order of centered and endpoint sampling rules.

`test_diffusiveflux_convergence.cpp` has two modes:

- Mode 0: exact cell averages -> production ML-WENO -> sampled diffusive flux.
  Reports reconstructed value errors and **edge-average flux** errors
  `(numerical_integral-exact_integral)/edge_length` on interior and boundary
  faces. Cell-average conservation is checked independently.
- Mode 1: true discontinuities. Includes vertical, horizontal and oblique jumps,
  aligned interfaces, cut cells, and a linear background plus a jump. Polygon
  clipping gives exact cell averages. Error integrals are split geometrically
  at the jump. Outputs include one-sided traces where the interface is an edge,
  overshoot/undershoot, transition width, near-interface profiles and nonlinear
  candidate weights. Diffusive-flux errors are measured only where sampling and
  reconstruction remain on a smooth side of the jump.

`test_advection_diffusion.cpp` checks the spatial operator for
`u_t + div(a*u) = kappa*Laplacian(u)`:

- Mode 0: translating/decaying sine waves and positive-time translated erf
  fronts. Covers local/global LF, several diffusivities, two-dimensional meshes
  and pseudo-1D strips. A single oriented face flux contributes with opposite
  signs to its neighboring cells. The resulting RHS is compared with an
  independently integrated exact cell-average `u_t`. This is a spatial accuracy
  test, not a time-integration test.
- Mode 1: assemble the full cell-average Jacobian by chaining production flux
  partials and `Reconstruction::EvaluateWithJacobian`. Check coordinate and
  multi-cell directions with finite differences, recomputing nonlinear weights
  for every perturbed residual. Richardson extrapolation controls FD truncation
  error. Global LF uses the fixed bound `norm(a)`, so no state-dependent maximum
  derivative is omitted.
- Mode 2: constant-state preservation, a quadratic manufactured spatial
  operator, boundary/global mass balance and the zero-advection/zero-diffusion
  limits. Both large and small candidates reproduce the quadratic exactly.

No velocity reconstruction is introduced: the prescribed test velocity is
supplied at physical edge quadrature points, using the same interface that can
later accept Darcy-Stokes velocities.

## Reconstruction settings and interpretation

Order 3 means the `(3,2)` size pair; order 5 means `(5,3)`. Pseudo-1D uses the
corresponding `p x 1` or `1 x p` candidates. The constant candidate is tested on
and off. Production `epsilon=1e-2` and constant linear-weight defaults are retained.
The small candidates use the target-cell smoothness indicator.

The large candidate is centered when it fits and explicitly selected from a
one-sided stencil at physical boundaries. This avoids silently dropping to a
lower reconstruction degree in a boundary accuracy test; it does not change the
production reconstruction class. Boundary rates are nevertheless reported
separately. In pseudo-1D the two inactive-direction faces use zero diffusive flux.

With `N` interior-edge samples, exact smooth samples yield order `N` for the
centered derivative. A Dirichlet face uses `N` interior samples plus its face
value and has the same formal endpoint derivative order. Sufficient independent
edge quadrature is used. Reconstructed fluxes generally lose one spatial order;
the combined discrete divergence can lose another on perturbed meshes. The
tests use these conservative order floors, not the value-reconstruction order.
Rates below the roundoff floor are not asserted.

A true jump has no classical derivative at its interface, and a polynomial in a
cut cell cannot reproduce its two one-sided values everywhere. Cut-cell maximum
errors and overshoots are therefore reported rather than subjected to a false
smooth-order or TVD requirement. Global jump `L1` error must decrease (or already
be at roundoff). Positive-time erf fronts provide a well-defined steep-gradient
flux and spatial-RHS accuracy test. Their convergence is checked without
imposing a smooth asymptotic rate while the layer is underresolved.

## Results and plots

CSV files are written under the test build directory in `diffusion_results/`
and `advection_diffusion_results/`. Output paths are unique per CTest case, so
parallel CTest execution is safe.

- Main CSV: `n,h,quantity,region,l1,l2,linf,rate_l2,rate_linf`.
  Norms use area weights for values/RHS and length weights for edge-average
  fluxes, normalized by the total measure of the reported region.
- Jump regions: `all`, `cut`, `near_jump`, `smooth_stencil`. The last region
  requires the complete reconstruction support to stay on one side. Extra
  columns give measured ringing, transition width, conservation error and region
  area; rows for flux use region length in that final measure column.
- `_profile.csv`: exact and reconstructed normal profiles, scaled distance
  `distance/h`, cell indices and large/small/constant weight sums.
- `_weights.csv`: individual candidate smoothness indicators and nonlinear
  weights in cut/near-interface cells, including the optional constant weight.
- `_traces.csv`: left/right reconstruction values and exact one-sided states
  on mesh-aligned jump edges. It can contain just a header for a cut interface.
- `_front_metrics.csv`: measured and exact 10%-90% transition widths and profile
  overshoot/undershoot. Profile spacing is recorded; the width is a sampled
  diagnostic, not an exact polynomial-extremum calculation.
- Jacobian CSV: direction index and maximum scaled derivative discrepancy.

With Python and matplotlib installed, CTest produces log-log error plots and
profile/weight plots as PNG and PDF. Numerical tests remain enabled if plotting
dependencies are absent. Disable figures with
`-DMANTLE_TRANSPORT_TEST_PLOTS=OFF`, or run the script manually:

```bash
python3 src/mlweno/test/plot_transport_tests.py --input-dir PATH/TO/diffusion_results
```

All executables accept standard PETSc options. The convergence programs also
accept the following controls (defaults in parentheses):

| Option | Meaning |
|---|---|
| `-flux_mode` | Mode listed above (0) |
| `-flux_mesh` | 0 rectangular, 1 perturbed quadrilateral (0) |
| `-flux_axis` | 0 full 2D, 1 x strip, 2 y strip (0) |
| `-flux_order` | Large stencil size: 3 or 5 (3) |
| `-flux_constant` | Constant candidate: 0 or 1 (0) |
| `-flux_samples` | Even interior-edge sample count (4) |
| `-flux_n0`, `-flux_levels` | Initial active resolution and doubling levels (12, 3) |
| `-flux_field` | 0 sine/pure jump; 1 erf front/jump with linear background (0) |
| `-flux_orientation` | Front/jump: 0 vertical, 1 horizontal, 2 oblique (0); axis overrides |
| `-flux_cut` | Jump: 0 centered aligned location, 1 shifted location (1) |
| `-flux_lf` | Combined test: 0 local, 1 global LF (0) |
| `-flux_kappa`, `-flux_time` | Diffusivity and analytical snapshot time (0.02, 0.2) |
| `-flux_output` | CSV filename (`transport.csv`) |

Use distinct output paths when running custom cases concurrently. Extremely
thin fronts may require more refinement before asymptotic orders emerge.
