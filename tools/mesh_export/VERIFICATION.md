# Verification record

The exporter was checked against the previously reviewed MantlePar mesh API.
These checks establish the file layout and reader compatibility; they do not
replace a build and MPI run against your PETSc and parallel HDF5 installations.

## Checks completed

- Both C++ sources passed a C++17 syntax check with GCC 13.3 using
  `-Wall -Wextra -Wpedantic -Werror` and minimal test declarations for the
  unavailable development libraries.
- The actual `mesh_output.cpp` implementation wrote real HDF5/XDMF files using
  VTK's bundled serial HDF5 library. A serial PETSc/MPI facade supplied the
  coordinates, ownership extents, communicator and successful mesh validation.
- VTK 9.3.1's `vtkXdmfReader` loaded both a rectangular grid and an interior-
  perturbed quadrilateral grid, each with 63 points and 48 quadrilateral cells.
  Every coordinate agreed with the supplied input to an absolute tolerance of
  `1e-14`. Every cell had the expected four vertex IDs and positive signed area;
  the sum of cell areas was 3.12, matching the rectangular outer boundary.
- The nodal `VertexRank` array was recovered as integer data with the expected
  serial owner, rank zero.
- A filename containing spaces and an ampersand loaded correctly, exercising
  XML escaping and the relative HDF5 reference.
- A separate partial-write fixture verified a nonzero hyperslab origin:
  coordinates occupied exactly `j=[2,5), i=[3,7)` in the global array, with
  HDF5's default zero fill elsewhere. This fixture is a storage-layout check,
  not a complete mesh intended for visualization.

The XDMF reader reports its internal structured dimensions as `(1,9,7)` for
these `2DSMesh` fixtures. The physical points and all 48 quadrilateral cell
connections were checked directly; there is no transposition of their data.

## Checks still needed on the target installation

Real PETSc, MPI, CMake and parallel HDF5 development packages were unavailable
in the checking environment. The serial facade made the two HDF5 MPI-property
setters no-ops; it did not emulate concurrent MPI ranks. Consequently, real
collective I/O, MPI linking, the production `ValidateMesh` call and the supplied
CMake configuration have not been executed here. The ParaView application
itself was not launched; compatibility was checked through VTK's XDMF reader.

Build the utility and run the two supplied CTest export smoke tests using the
same MPICH installation as PETSc. Then open their `.xdmf` files in ParaView.
The commands are in `README.md`. For the four-process test, the output should
have 63 vertices, 48 cells, and `VertexRank` values spanning ranks 0 through 3.
