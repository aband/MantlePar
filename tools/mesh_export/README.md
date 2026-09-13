# Mesh visualization: HDF5 + XDMF

This tool uses the `MeshParam`, `CreateFullMesh`, `LogicRectMesh`, `RefineMesh`,
and `ValidateMesh` interfaces already reviewed for MantlePar. It adds a mesh
export utility before the integral module, without changing the mesh generator.

## Add to MantlePar

Copy this directory to `tools/mesh_export/`. Add the following line to the root
`CMakeLists.txt`, after `add_subdirectory(src/core)`:

```cmake
add_subdirectory(tools/mesh_export)
```

Retain your existing PETSc `pkg-config` compiler selection and the matching
MPICH launcher. The exporter does not replace those settings.

You need **parallel HDF5 built with the same MPI installation as PETSc**.
Your reviewed PETSc installation uses `/home/renpo/system/mpich-install`.
An HDF5 library built for a different MPI implementation is not suitable.
The code uses HDF5's C API from C++; HDF5's C++ bindings are unnecessary.
PETSc itself does not need HDF5 support enabled.

If you already have compatible HDF5, set `HDF5_ROOT` to its install prefix.
For example, if you install it in `/home/renpo/system/hdf5-install`:

```sh
cmake -S . -B build -DBUILD_TESTING=ON \
  -DHDF5_ROOT=/home/renpo/system/hdf5-install
cmake --build build --target export_mesh
```

For an HDF5 source distribution that uses `configure`, the corresponding
installation commands, run in a separate HDF5 build directory, are:

```sh
/path/to/hdf5-source/configure \
  CC=/home/renpo/system/mpich-install/bin/mpicc \
  --enable-parallel --enable-shared \
  --prefix=/home/renpo/system/hdf5-install
make -j4
make install
```

Use the build instructions supplied with your HDF5 release if it uses CMake
instead. You can check an installed wrapper with `h5pcc -showconfig`.

## Generate meshes

Run these commands from the MantlePar root. `mesh_nx` and `mesh_ny` are cell
counts, so the output has `(nx+1)*(ny+1)` vertices and `nx*ny` cells.

Rectangular, one MPI process:

```sh
./build/tools/mesh_export/export_mesh \
  -mesh_type 0 -mesh_nx 32 -mesh_ny 16 \
  -mesh_length 2 -mesh_height 1 -output output/rectangular
```

Perturbed quadrilateral, four MPI processes:

```sh
/home/renpo/system/mpich-install/bin/mpiexec -n 4 \
  ./build/tools/mesh_export/export_mesh \
  -mesh_type 1 -mesh_nx 32 -mesh_ny 16 -mesh_px 2 -mesh_py 2 \
  -mesh_length 2 -mesh_height 1 -mesh_perturbation 0.15 -mesh_seed 42 \
  -output output/quadrilateral
```

Use `-mesh_type 2` for the sine-stretched rectangular mesh. Optional origins
are `-mesh_xstart` and `-mesh_ystart`. The exporter creates output directories;
reusing an output stem replaces the corresponding two files.

## Open in ParaView

1. Open `output/quadrilateral.xdmf` and click **Apply**. Select an XDMF reader
   if ParaView asks which reader to use.
2. Select **Surface With Edges** or **Wireframe** as the representation.
3. Select **VertexRank** under Coloring to inspect vertex ownership.

Keep the `.xdmf` and `.h5` files together. The XML uses a relative reference
to the HDF5 filename, so both files can be moved to another machine together.
Open the `.xdmf` file: it supplies the topology and geometry description.

`VertexRank` is a nodal diagnostic. Its surface coloring can interpolate
between ranks; use the **Points** representation to see individual owners.
It is not a cell-partition label. The coordinates are independent of MPI
partitioning, but the rank field changes when the process layout changes.

## Call from your solver

Link the calling target to `mantle_mesh_output` and include `mesh_output.h`:

```cpp
// All ranks in dmMesh's communicator participate in this call.
PetscCall(WriteMeshXdmf(dmMesh, vertices, "output/mesh"));
```

The function exports the coordinates already in the global Vec. It does not
regenerate coordinates from MeshParam. It also works on a subcommunicator,
provided each independently exported mesh uses a different output stem.

## File layout and MPI behaviour

| Dataset | Dimensions | Stored type | Meaning |
| --- | --- | --- | --- |
| `/Coordinates` | `ny+1, nx+1, 3` | Float64 | Physical x, y, z=0 |
| `/VertexRank` | `ny+1, nx+1` | Int32 | Owning MPI rank in the DM communicator |

XDMF uses `2DSMesh`, a curvilinear structured grid. Connectivity is implicit
in the logical vertex ordering, so the same format represents rectangles
and perturbed quadrilaterals. It does not assume uniform spacing or orthogonal
physical edges. Array ordering is `[j][i][component]` with i varying fastest.

Each rank packs only its owned DMDA vertices and writes a disjoint hyperslab
in one shared HDF5 file. No ghost vertices are written and no mesh is gathered
onto rank 0. Every rank participates in metadata operations and collective
HDF5 writes. Rank 0 writes the small XDMF file after the HDF5 file is closed.
All ranks must access the same output filesystem and use the same output stem.

## Verification

The CMake file adds two export smoke tests (one serial, one four-process):

```sh
ctest --test-dir build -R '^mesh_export_' --output-on-failure
```

These tests run the writer. Inspect their files in
`build/tools/mesh_export/output/` using ParaView for a visual check.

Development-environment checks and their limitations are recorded in
`VERIFICATION.md`. Actual PETSc/parallel-HDF5 execution still needs to be
verified with your MPI installation.

References:
- [Parallel HDF5](https://support.hdfgroup.org/documentation/hdf5/latest/_intro_par_h_d_f5.html)
- [XDMF topology and geometry](https://www.xdmf.org/index.php/XDMF_Model_and_Format)
- [CMake FindHDF5](https://cmake.org/cmake/help/latest/module/FindHDF5.html)
