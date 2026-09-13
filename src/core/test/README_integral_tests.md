# Integral verification for MantlePar

These tests target the generated `integral.h` and `integral.cpp` interface and
the current `mesh.h` in MantlePar. The integral files were not yet on GitHub at
the time of this review. Install those two files in `src/core/` before building
the tests.

## Install and build

1. Copy `test_integral.cpp`, `integral_tests.cmake`, and this document into
   `src/core/test/`.
2. Add `integral.cpp` to the existing `mantle_core` source list in
   `src/core/CMakeLists.txt`:

   ```cmake
   add_library(mantle_core
       mesh.cpp
       integral.cpp
   )
   ```

3. Append this line to the existing `src/core/test/CMakeLists.txt`:

   ```cmake
   include("${CMAKE_CURRENT_LIST_DIR}/integral_tests.cmake")
   ```

The include adds the new test target and registrations. It uses your existing
PETSc compiler settings and `MPIEXEC_EXECUTABLE` from the root CMake file.

From the MantlePar root, configure and build in Release mode:

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=ON \
  -DHDF5_ROOT=/home/renpo/system/hdf5-install
cmake --build build --target test_integral -j
ctest --test-dir build -R '^core_integral_' --output-on-failure
```

Use your actual HDF5 prefix if it differs. HDF5 is required by the mesh exporter
already included in the root project; these integral tests use PETSc only.

All assertions in this test file use PETSc error checks, so they remain active
with `-DNDEBUG` and `CMAKE_BUILD_TYPE=Release`.

## Registered tests

With an MPI launcher available, CTest registers **13 tests**:

| Tests | Process grids | Coverage |
| --- | --- | --- |
| `core_integral_unit` | One process | Rules, individual cells, edges, scaling, error handling |
| `core_integral_rectangular_*` | `1x1`, `2x1`, `1x2`, `2x2` | `CreateFullMesh` |
| `core_integral_quadrilateral_*` | `1x1`, `2x1`, `1x2`, `2x2` | `LogicRectMesh` |
| `core_integral_stretched_*` | `1x1`, `2x1`, `1x2`, `2x2` | `RefineMesh` |

Four serial tests are registered if no MPI launcher is available.

To inspect the planned runs:

```sh
ctest --test-dir build -N -R '^core_integral_'
```

To run only the quadrilateral cases:

```sh
ctest --test-dir build -R '^core_integral_quadrilateral_' --output-on-failure
```

For a direct four-process quadrilateral run, use the MPICH launcher matching
your PETSc installation:

```sh
/home/renpo/system/mpich-install/bin/mpiexec -n 4 \
  ./build/src/core/test/test_integral \
  -integral_mesh_type 1 -mesh_px 2 -mesh_py 2
```

Running `test_integral` without options runs the unit tests. The mesh modes
are `-integral_mesh_type 0` (rectangular), `1` (quadrilateral), and `2`
(stretched). Use CTest to run the complete suite. The mesh tests deliberately
support the four process grids listed above, including their smallest grid.

## Numerical coverage

| Area | Checks |
| --- | --- |
| Adjustable rules | 1-8 Gauss points; reference monomials through degree `2*n-1`; actual `n` and `n*n` callback counts |
| Tensor-product exactness | Square monomials through degree `min(2*n-1,7)` in each coordinate, including `x^5*y^5` for the three-point rule |
| Individual cell geometry | Translated rectangle, rotated rectangle, skew parallelogram, trapezoid, general convex quadrilateral, extremely thin rectangle |
| Physical cell integration | All 15 monomials of total degree at most four, using three- and five-point rules |
| Non-polynomial integration | `exp(x+y)` on a rectangle and a general quadrilateral; comparison with a closed-form boundary integral |
| Edges | Horizontal, vertical and slanted segments; mapped endpoints, length, normal, scalar integrals and quadratic vector flux |
| Orientation | Reversal preserves scalar line integrals and reverses normal flux |
| Normalized integration | Area scaling, callback coordinates, and distinction from physical integration of a scaled callback |
| Invalid inputs | Clockwise, folded, concave and collapsed cells; repeated/non-finite vertices; zero edges; invalid scale, rule, and callback values |

Each generated mesh test includes a nonzero origin, unequal coordinate
spacings, and a domain with a high aspect ratio. The quadrilateral tests use
seeds 7, 991, and 42, perturbations 0.15 and 0.249, and a zero-perturbation
case. Positive-perturbation tests check that slanted edges actually occur.

Grid sizes count **vertices**: the cases include `13x9`, `8x6`, and `2x2`,
with `10x7` added for the larger perturbation. The `2x2` vertex case has one
physical cell. On two or four processes, the other ranks own no cells but
must still participate in the reductions.

## Independent expected values

The expected cell moments come from Green's theorem and exact polynomial
integration along polygon edges:

```text
integral_K x^a y^b dA = sum_edges integral x^(a+1) y^b dy / (a+1)
```

The test expands each edge polynomial and integrates its powers analytically.
It does not use the production mapping, Jacobian, or Gauss points to construct
these expected values.

For generated meshes the fields are polynomials in
`X=(x-xstart)/L` and `Y=(y-ystart)/H`. This controls roundoff on translated and
thin domains. Their exact domain integrals are
`L*H/((a+1)*(b+1))`. Physical area is the `a=b=0` case.

Every generated cell is checked against its own polygon moments, so errors
cannot pass merely by cancelling in a global total. The tests also check a
quadratic vector field with `div(F)=X+Y`: each cell's flux must match its
volume integral, and internal-edge fluxes must cancel in the global sum.

## MPI coverage

- A NaN-filled local vector is populated by a global-to-local scatter before
  reading any cell corners. Partition-boundary cells therefore exercise ghost
  coordinates.
- Each cell belongs to the rank owning its lower-left vertex. An independently
  reduced coverage array must contain exactly one contribution for every cell.
- Every distributed cell's 15 numerical moments are compared with the same
  cell on a complete serial mesh built on `PETSC_COMM_SELF`.
- Global moments are compared both with the serial calculation and with the
  analytic domain integrals.
- All reductions occur outside the local cell loops. Ranks with no cells
  still participate, and unexpected failures abort the test communicator.

The complete serial reference and global coverage arrays are intentionally
small test diagnostics, not a proposed production integration strategy.

## Checks performed during generation

The delivered test source, generated integral sources, and the current GitHub
`mesh.cpp` passed GCC C++17 compilation with `-O2 -DNDEBUG`,
`-Wall -Wextra -Wpedantic -Werror`, and address/undefined-behavior sanitizers.
The unit mode and all three mesh modes passed in a one-rank test facade.

The checking environment has no native PETSc, MPI, or CMake installation.
The facade supplies PETSc declarations, one-rank vector operations and
reductions, and SciPy-generated Gauss data in place of `PetscDTGaussQuadrature`.
Consequently, these checks verify the numerical test paths and Release-mode
checks, but **do not establish native PETSc linking, real MPI communication,
or successful CMake/CTest execution**. Run the complete CTest suite on your
PETSc/MPICH installation to verify those parts.
