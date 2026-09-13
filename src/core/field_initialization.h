#ifndef MANTLE_CORE_FIELD_INITIALIZATION_H
#define MANTLE_CORE_FIELD_INITIALIZATION_H

#include "mesh_info.h"

#include <functional>
#include <vector>

// A real scalar profile evaluated at PHYSICAL coordinates. Capturing lambdas
// can carry material parameters, time, or problem-specific initial conditions.
// Coordinates retain the units/scaling of the supplied mesh.
using CellInitialFunction = std::function<PetscReal(const Point&)>;

// Both functions are collective on cellDM's communicator. cellDM must be a
// valid, set-up 2-D DMDA with ONE degree of freedom per physical cell. field
// must be an assembled global Vec compatible with that DM, with no outstanding
// array access. A Vec from DMCreateGlobalVector(cellDM, ...) is suitable.
//
// Only the global field is filled. Neither function initializes physical ghost
// values or applies boundary conditions. Refresh any separate local/ghost Vec
// with DMGlobalToLocalBegin/End afterwards, then apply physical boundary data.
// Input validation, callback, and quadrature failures leave field unchanged.
// This guarantee does not cover failures in PETSc/MPI during the final VecCopy.

// Set each physical cell to its volume average:
//     field(K) = integral_K initialValue(x,y) dA / area(K).
//
// If mesh has M-by-N VERTICES, cellDM must have (M-1)-by-(N-1) CELLS.
// mesh must be a current BuildMeshInfo snapshot on each participating rank.
// Those snapshots must describe the same mesh and collectively own every cell
// exactly once; ownership coverage is checked before field is changed.
//
// The mesh and cell DM may use DIFFERENT MPI partitions. Quadrature is evaluated
// on mesh.OwnedCells(), where geometry is available; PETSc's DMDA application
// ordering and vector assembly route results to the field owners. Mesh ranks
// owning zero cells still participate. No mesh data is gathered onto rank zero.
//
// gaussPoints is the number of Gauss-Legendre points PER reference direction
// (gaussPoints squared evaluations per cell), not the polynomial degree.
// All ranks must pass the same gaussPoints and the same mathematical profile.
// The callback must be nonempty, return finite values, and make NO collective
// MPI/PETSc calls; its invocation count differs between ranks. Exceptions are
// converted to collective PETSc errors. Captured parameters are not modified by
// this helper; the callback should not modify mesh or field.
PetscErrorCode InitializeCellAverages(
    DM cellDM, Vec field, const MeshInfo& mesh,
    const CellInitialFunction& initialValue,
    PetscInt gaussPoints = DefaultGaussPoints);

// Assign already computed real values owned by the CELL FIELD's rank.
// Obtain xs,ys,xm,ym with DMDAGetCorners(cellDM, ...). Require exactly xm*ym
// entries, ordered with i varying fastest:
//     ownedValues[(j-ys)*xm + (i-xs)] -> cell (i,j).
//
// This is the field DM's ownership, not mesh.OwnedCells(), a global natural
// array, or a ghosted array. An empty ownership range requires an empty vector.
// Values must be finite. No file reading, interpolation, or ghost exchange is
// performed. Real values are stored with zero imaginary part in complex PETSc.
PetscErrorCode AssignOwnedCellValues(
    DM cellDM, Vec field, const std::vector<PetscReal>& ownedValues);

#endif

