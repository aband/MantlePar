#ifndef MANTLE_MESH_OUTPUT_H
#define MANTLE_MESH_OUTPUT_H

#include <petscdm.h>
#include <string>

// Collective on vertexDM's communicator. Every rank must pass the same stem.
// Accepts the vertex DMDA/global Vec used by core/mesh.h, including all physical
// boundary vertices. The coordinates must describe a valid, logically
// rectangular quadrilateral mesh.
//
// Writes/replaces stem.h5 and stem.xdmf; creates parent directories on rank 0.
// All ranks must see the same output filesystem. Open stem.xdmf in ParaView.
// A parallel HDF5 build using the same MPI as PETSc is required, even for a
// one-process run. PETSc itself does not need to be configured with HDF5.
PetscErrorCode WriteMeshXdmf(DM vertexDM, Vec vertices, const std::string& stem);

#endif
