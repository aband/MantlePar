#ifndef MANTLE_CORE_MESH_H
#define MANTLE_CORE_MESH_H

#include <cstdint>
#include <petscdmda.h>

struct MeshParam {
    PetscReal xstart = 0.0;
    PetscReal ystart = 0.0;
    PetscReal L = 1.0;
    PetscReal H = 1.0;

    // Maximum displacement as a fraction of the uniform spacing in each axis.
    // Used only by LogicRectMesh. Require 0 <= perturbation < 0.25.
    PetscReal perturbation = 0.15;
    std::uint64_t seed = 1;
};

struct Point {
    PetscReal p[2]{};
};

// Common contract for all functions below:
// - vertexDM is a set-up 2-D DMDA with dof=2, DMDA_STENCIL_BOX,
//   stencil width >= 1, and DM_BOUNDARY_NONE in both directions.
// - DMDA global sizes M,N count VERTICES, including all boundary vertices.
//   Thus the physical mesh contains (M-1)*(N-1) quadrilateral cells.
// - vertices is an existing global Vec created from vertexDM. These functions
//   fill it; they neither create nor destroy the DM or the Vec.
// - Calls are collective on the DM communicator; parameters must agree on
//   every rank. Coordinates are stored as [j][i][0=x,1=y] using the DOF API.
// - Cell (i,j) has counterclockwise vertices
//   (i,j), (i+1,j), (i+1,j+1), (i,j+1).
// - Interior MPI ghosts are updated by DMGlobalToLocalBegin/End before use.
//   Exterior physical ghost coordinates are outside this interface.

PetscErrorCode CreateFullMesh(DM vertexDM, Vec vertices,
                              const MeshParam& parameters);

// Perturb only interior vertices; all four boundaries stay fixed.
// A given seed and global vertex index produce the same coordinates regardless
// of MPI partitioning. A zero perturbation gives the uniform rectangular mesh.
PetscErrorCode LogicRectMesh(DM vertexDM, Vec vertices,
                             const MeshParam& parameters);

// Keep the original sine-stretch option, normalized to the physical domain:
// x=xstart+L*sin(pi*i/(2*(M-1))), and similarly for y.
// These cells remain rectangular and become finer toward the top and right.
PetscErrorCode RefineMesh(DM vertexDM, Vec vertices,
                          const MeshParam& parameters);

// Check the ACTUAL vector, including cells crossing MPI partitions.
// Reject non-real/non-finite coordinates and non-positive or nearly singular
// corner Jacobians. For a bilinear quadrilateral, det(J) is affine on the
// reference square, so its minimum occurs at a corner.
PetscErrorCode ValidateMesh(DM vertexDM, Vec vertices);

#endif

