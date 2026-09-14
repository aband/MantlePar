# HDivMixed visualizer

Samples the existing C++ HDivMixed implementation and exports parallel
HDF5/XDMF for ParaView. A Python script reads the exported samples and renders
PNG/PDF figures. It does not reimplement the basis functions.

## Files and CMake integration

Copy this directory to tools/hdivmixed_export/ in MantlePar:

- hdivmixed_output.h — options and collective exporter interface.
- hdivmixed_output.cpp — geometry, sampling, trace diagnostics, HDF5/XDMF output.
- export_hdivmixed.cpp — PETSc command-line executable.
- plot_hdivmixed.py — plots from the HDF5 file.
- CMakeLists.txt — exporter library and executable.

In the root CMakeLists.txt, after add_subdirectory(src/mfem), add:

    add_subdirectory(tools/hdivmixed_export)

Ensure hdivmixed.cpp is present in the existing mantle_mfem source list in
src/mfem/CMakeLists.txt. The exporter reuses that target, MPI::MPI_CXX, and
HDF5::HDF5. HDF5 must have parallel support and use the same MPI as PETSc.
The root's existing MPI compiler selection remains the source of configuration.

## Build and run

From the repository root, using the existing MPI/HDF5 configuration:

    cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
    cmake --build build --target export_hdivmixed --parallel

If HDF5 discovery needs an explicit installation, supply
-DHDF5_ROOT=/path/to/parallel-hdf5 on the configure command.

Serial:

    ./build/tools/hdivmixed_export/export_hdivmixed \
        -output output/hdivmixed/hdivmixed

Parallel, with the MPI launcher matching PETSc:

    mpiexec -n 2 ./build/tools/hdivmixed_export/export_hdivmixed \
        -expected_ranks 2 \
        -output output/hdivmixed/hdivmixed

Cell A is sampled on rank 0 and cell B on rank 1. Both are sampled on rank 0
in serial. With more than two ranks, additional ranks participate in collective
I/O using empty selections. All ranks need the same output path on a shared
filesystem. Avoid simultaneous runs with the same output stem.

| Option | Default | Meaning |
|---|---:|---|
| -hdiv_n | 36 | Display subdivisions per cell axis; 2–512 |
| -hdiv_trace_n | 200 | Shared-edge sampling intervals; 2–100000 |
| -hdiv_perturb | 0.18 | Fraction of logical spacing; 0 ≤ value < 0.25 |
| -hdiv_tolerance | about 9.09e-13 with double precision | Normalized diagnostic tolerance |
| -output | output/hdivmixed/hdivmixed | Output stem without extension |
| -expected_ranks | unset | Optional check for a mismatched MPI launcher |

The geometry is an interior two-cell patch of a larger logically rectangular
mesh. Perturbation moves its six vertices while keeping the common edge
identical in both cells. Cases: rectangular_i, quadrilateral_i, rectangular_j,
and quadrilateral_j. A zero perturbation also makes the quadrilateral cases
rectangular.

## Render figures

Install plotting dependencies if needed:

    python3 -m pip install numpy matplotlib h5py

Render:

    python3 tools/hdivmixed_export/plot_hdivmixed.py \
        output/hdivmixed/hdivmixed.h5

This writes six figures, each as PNG and PDF, under output/hdivmixed/figures/.
Use --out-dir path/to/figures to select another directory.

| Figure | Content |
|---|---|
| basis_rectangular | All eight local vector basis functions on a rectangle |
| basis_quadrilateral | All eight functions on a perturbed quadrilateral |
| shared_basis_i | Matched modes and a combined field on left/right neighbors |
| shared_basis_j | Matched modes and a combined field on lower/upper neighbors |
| conformity | Normal traces, normal jumps, and tangential traces in all four cases |
| divergence | Physical divergence of the two shared modes and combined field |

Arrows show the physical vector field; color shows its magnitude. The gold
dashed segment marks the shared edge; the red arrow shows the common normal.
Cell samples remain separate.

## What confirms H(div) continuity

At the same physical point on the shared edge, the diagnostic compares

    u_A · n = u_B · n

using a single normal n directed from A to B. With separate outward normals,
the equivalent condition is u_A · n_A + u_B · n_B = 0.
Tangential components and cellwise divergence may differ across the interface.
Those differences appear in the plots and reports but do not cause failure.

The two shared DOFs are matched as follows (indices start at zero):

| Neighbors | Shared sides | Linear mode A / B | Constant mode A / B |
|---|---|---|---|
| i | A Right / B Left | 1 / 3 | 5 / 7 |
| j | A Top / B Bottom | 2 / 0 | 6 / 4 |

Along the canonical edge parameter s from 0 to 1, the linear normal trace is
1-2s for i neighbors and 2s-1 for j neighbors. The constant mode has trace 1.
These orientations are already included in HDivMixed; no extra sign is applied.

The combined field uses all eight local basis functions. Coefficients depend
on physical edge midpoints, so the shared coefficients agree on A and B.
Values are evaluated independently on each side, without averaging.

Diagnostics cover the sampled points of the two shared modes and combined
field. The pass criterion checks normal directions, normal-trace agreement,
and agreement with the prescribed linear/constant traces. Normal discrepancies
are divided by max(1, |a|, |b|) before comparison with the tolerance.
Both programs retain output and return a nonzero status if this diagnostic
fails. Python recomputes normal projections from the saved vectors.

## ParaView and output data

Open any of the four .xdmf files beside hdivmixed.h5, then click Apply.
Each file contains two separate cell blocks.

- Use Glyph with shared_linear, shared_constant, or conforming_velocity
  to inspect a matched vector field across the shared edge.
- basis_00 through basis_07 are local indices. The matching index in the
  neighboring cell can differ, as shown in the table.
- Fields ending in _divergence contain analytic physical divergence.
- pressure_basis is the constant P0 basis function, equal to 1.

Coloring by vector magnitude can show a jump even when the normal component
is continuous. Keep A and B separate when inspecting interface traces.
The trace plots and hdivmixed_traces.csv provide the explicit comparison.

hdivmixed_summary.csv records the maximum normal, tangential, and divergence
jumps, normal-direction error, prescribed-trace error, and diagnostic status.
HDF5 stores Float64 coordinates/fields and Int32 display connectivity. The
/metadata array records schema version, display subdivisions, trace intervals,
perturbation, tolerance, rank count, local DOF count, and displayed mode count.

## Included example output

examples/ contains ready-to-open HDF5/XDMF data, CSV reports, and the six
PNG/PDF figures generated with default options and the supplied C++ sampler.
The largest sampled normal jump is about 1.22e-15.

Local verification used the actual C++ basis/export code with a PETSc/MPI
compatibility adapter and serial HDF5. Checks of collective ordering, ownership,
and empty selections produced identical data with 1, 2, and 4 simulated ranks.
VTK's XDMF reader loaded all four cases with two cell blocks and 23 fields per
block, and all six rendered figures were visually inspected.
Native PETSc and parallel HDF5 execution must be checked on the target
installation.
