#ifndef MANTLE_MFEM_ASSEMBLY_H
#define MANTLE_MFEM_ASSEMBLY_H

#include "dof_map.h"
#include "local_matrix.h"

#include <petscmat.h>
#include <petscvec.h>

#include <functional>

// Independent, FULL-size blocks for either Stokes or Darcy. Construct separate
// instances for the two systems. Every field uses its own DofMap numbering.
//
// A: velocity rows x velocity columns
// B: pressure rows x velocity columns (TRANSPOSE of the legacy stored B)
// C: pressure rows x pressure columns
// f: velocity load; g: pressure load, initially zero
//
// A, B, C and f retain exactly the signs/scaling of ComputeLocalStokes/Darcy.
// They do not yet specify the signs in a complete saddle-point system. B^T,
// pressure-block signs, and inter-system coupling are composed separately.
// All boundary DOFs remain present. These raw domain blocks can be singular;
// apply boundary conditions and any required pressure gauge before solving.
//
// Owns PETSc handles with EXPLICIT collective cleanup: DestroyMixedBlocks().
// There is intentionally no collective work in the C++ destructor. Do not copy
// handles into another owner; destroy before PetscFinalize. PETSc Mat/Vec
// operations may use these public handles directly.
struct MixedBlocks {
    Mat A = nullptr;
    Mat B = nullptr;
    Mat C = nullptr;
    Vec f = nullptr;
    Vec g = nullptr;

    MixedBlocks() = default;
    MixedBlocks(const MixedBlocks&) = delete;
    MixedBlocks& operator=(const MixedBlocks&) = delete;
    MixedBlocks(MixedBlocks&&) = delete;
    MixedBlocks& operator=(MixedBlocks&&) = delete;

    bool IsEmpty() const noexcept { return !A && !B && !C && !f && !g; }
};

// Collective on the objects' communicator; call on every participating rank.
// Safe on an empty object. Clears the handles; attempts to destroy all five
// objects even if one destruction reports an error.
PetscErrorCode DestroyMixedBlocks(MixedBlocks& blocks);

// Supply reconstructed porosity for ONE owned cell in global logical indices.
// Called exactly once per owned cell, in j-outer/i-inner order, with a fresh
// LocalPorositySamples. Capture the mesh, quadrature rules, fields, time and
// material parameters as needed. Sample locations and CCW edge directions are
// precisely those documented in local_matrix.h; no averaging is done here.
//
// Stokes needs cell samples only; Darcy needs cell/edge samples and average;
// pressure coupling needs cell samples and average. A callback may supply a
// superset. A nonempty callback is required even on ranks with no owned cells.
// Refresh required field ghosts BEFORE assembly. Make no MPI collectives in
// this callback or in force callbacks, and do not modify assembly inputs.
// Return PETSC_SUCCESS or a PETSc error; C++ exceptions are caught at this
// boundary. Callbacks using PetscCall/PetscCheck must follow PETSc's usual
// PetscFunctionBeginUser/PetscFunctionReturn stack discipline.
using CellPorosityFunction =
    std::function<PetscErrorCode(MeshIndex, LocalPorositySamples&)>;

// All assembly routines are COLLECTIVE, including on empty cell/DOF owners.
// Use the SAME communicator and rank ordering used to initialize every DofMap.
// MeshInfo and maps must describe the same topology and partition. Rebuild
// MeshInfo after mesh motion, and maps after topology/partition changes.
// Rules and model parameters should describe the same model on every rank.
//
// Only mesh.OwnedCells() contribute. Shared DOF contributions are summed by
// PETSc COO assembly, including off-process rows; no extra orientation signs
// are applied. Matrices are sparse AIJ, vectors are ordinary distributed Vecs
// (not ghost vectors). COO sparsity includes every element entry, even zeros.
// Requires PETSc 3.23 or later; no hard-coded MPI launcher or quadrature order.
//
// All local evaluation/validation completes, and errors are agreed across
// ranks, BEFORE creating PETSc objects. This prevents a local callback or
// quadrature failure from leaving peers waiting in matrix assembly. An MPI or
// PETSc internal collective failure is subject to PETSc's normal error policy.
//
// Output must be empty on entry. On success it owns newly assembled objects.
// To reassemble with changed parameters, destroy the old output or assemble
// into another empty object. On a reported failure the output stays empty.
// No boundary elimination, natural-boundary loads, or pressure gauge is applied.

PetscErrorCode AssembleStokesBlocks(
    MPI_Comm comm, const MeshInfo& mesh,
    const DofMap& velocityMap, const DofMap& pressureMap,
    const GaussRule1D& cellRule, const CellPorosityFunction& porosity,
    const LocalForceFunction& force, MixedBlocks& result);

PetscErrorCode AssembleDarcyBlocks(
    MPI_Comm comm, const MeshInfo& mesh,
    const DofMap& velocityMap, const DofMap& pressureMap,
    const GaussRule1D& cellRule, const GaussRule1D& edgeRule,
    const CellPorosityFunction& porosity,
    const LocalMatrixParameters& parameters,
    const LocalForceFunction& force, MixedBlocks& result);

// Optional K_sd: Stokes-pressure rows x Darcy-pressure columns, using the raw
// NEGATIVE scalar from ComputeLocalCoupling. The two pressure maps may be the
// same object. Each cell pairs its own two pressure DOFs; no velocity maps or
// Stokes/Darcy blocks are needed. Use its transpose for the reverse coupling
// when composing the symmetric coupled formulation, with the chosen signs.
// K_sd must be nullptr on entry; the caller later calls MatDestroy(&K_sd).
PetscErrorCode AssemblePressureCoupling(
    MPI_Comm comm, const MeshInfo& mesh,
    const DofMap& stokesPressureMap, const DofMap& darcyPressureMap,
    const GaussRule1D& cellRule, const CellPorosityFunction& porosity,
    const LocalMatrixParameters& parameters, Mat& K_sd);

#endif

