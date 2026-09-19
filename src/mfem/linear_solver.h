#ifndef MANTLE_MFEM_LINEAR_SOLVER_H
#define MANTLE_MFEM_LINEAR_SOLVER_H

#include "linear_system.h"
#include <petscksp.h>
#include <string>
#include <vector>

// SparseLU converts a COPY to AIJ and maps the original nested fields through
// their global ISs. It requires one MPI rank and a nonsingular pressure system;
// use Schur for distributed or pressure-nullspace problems. A small factor-only
// shift supports zero pressure pivots; use an outer Krylov method (e.g. FGMRES)
// to correct its error against the original operator.
enum class LinearPreconditioner { None, Schur, SparseLU };

struct LinearBlockSolverOptions {
    std::string kspType = KSPGMRES;
    std::string pcType = PCNONE;
    PetscReal relativeTolerance = PetscReal(1e-10);
    PetscReal absoluteTolerance = PetscReal(1e-14);
    PetscReal divergenceTolerance = PetscReal(1e8);
    PetscInt maximumIterations = 1000;
};

struct LinearSolverOptions {
    // PETSc option names become -<optionsPrefix>ksp_type, etc. No leading '-'.
    // Use distinct prefixes, e.g. "stokes_", "darcy_", "coupled_". Prefixes
    // contain only ASCII letters/digits/underscores and have at most 64 chars.
    std::string optionsPrefix = "mantle_";
    std::string kspType = KSPFGMRES;
    LinearPreconditioner preconditioner = LinearPreconditioner::Schur;
    PetscReal relativeTolerance = PetscReal(1e-8);
    PetscReal absoluteTolerance = PetscReal(1e-12);
    PetscReal divergenceTolerance = PetscReal(1e8);
    PetscInt maximumIterations = 1000;
    bool initialGuessNonzero = false;

    // Schur preset: outer [velocity,pressure] split, full factorization,
    // implicit S=E-D*A^{-1}*G (PETSc Schur preconditioner SELF).
    // Defaults work with both ordinary and nested blocks. Runtime PETSc
    // options override these defaults; no global options are inserted.
    LinearBlockSolverOptions velocity{KSPGMRES, PCJACOBI,
        PetscReal(1e-12), PetscReal(1e-14), PetscReal(1e8), 1000};
    LinearBlockSolverOptions pressure{};

    // Acceptance policies applied AFTER collecting the report. PETSc's
    // -ksp_error_if_not_converged is disabled on the solvers managed here so
    // ordinary divergence can be reported and all resources cleaned up.
    // These policies are controlled by this struct, not PETSc CLI options.
    bool errorIfNotConverged = true;
    bool requireSubsolverConvergence = true;
    bool requireTrueResidual = true;

    // Optional algebraic gauge, using the mode already validated/attached by
    // linear_system. Does not invent a mode or project/change the supplied RHS.
    // Applied only after successful outer convergence and accepted inner solves.
    bool removePressureNullspace = false;
};

struct LinearSubsolverReport {
    std::string optionsPrefix;
    PetscInt64 solves = 0;
    PetscInt64 totalIterations = 0;
    PetscInt64 failures = 0;
    KSPConvergedReason firstFailure = KSP_CONVERGED_ITERATING;
    KSPConvergedReason lastReason = KSP_CONVERGED_ITERATING;
    PetscInt lastIterations = 0;
    PetscReal lastResidualNorm = 0;
};

struct LinearSolveReport {
    bool solveCompleted = false; // KSPSolve returned without a PETSc API error.
    bool converged = false;      // ALL enabled acceptance policies passed.
    bool trueResidualSatisfied = false;
    bool subsolversConverged = true;
    bool pressureGaugeRemoved = false;
    PetscErrorCode petscError = PETSC_SUCCESS; // API/setup error, not a KSP reason.
    KSPConvergedReason reason = KSP_CONVERGED_ITERATING;
    PetscInt iterations = 0;
    PetscReal kspResidualNorm = 0; // PETSc's selected norm; may be preconditioned.
    PetscReal rhsNorm = 0;
    PetscReal trueResidualNorm = 0;
    PetscReal relativeTrueResidualNorm = 0; // ||b-Ax||/||b||; absolute if b=0.
    PetscReal trueResidualThreshold = 0;   // max(actual KSP atol, rtol*||b||).
    std::string kspType;
    std::string pcType;

    // Every COMPLETED call to the managed velocity, pressure, optional inner
    // velocity, and optional upper velocity KSP is observed, including earlier
    // KSP_DIVERGED_ITS later overwritten by a successful call. Deeper solvers
    // inside user-selected PCs are not individually instrumented.
    std::vector<LinearSubsolverReport> subsolvers;
};

// COLLECTIVE on the communicator of system.matrix, including empty owners.
// One entry point for Stokes, Darcy, and the coupled saddle system. Requires
// the corrected linear_system module with outer 2x2 [velocity,pressure] nests.
// All ranks must supply a built system on the same comm/rank ordering and
// identical C++ options and applicable PETSc options.
//
// Uses system.matrix/rhs and writes system.solution. It never reassembles,
// changes block signs, reapplies boundary lifting, rescales physical variables,
// or reverses Darcy velocity. A zero initial guess is the default; PETSc's
// prefixed -ksp_initial_guess_nonzero can override it.
//
// Each call creates/destroys its KSP and work vectors. report is reset first
// and populated before ordinary nonconvergence is returned. The last iterate
// is retained on nonconvergence. Default return: PETSC_ERR_NOT_CONVERGED if
// any acceptance condition fails. With errorIfNotConverged=false, ordinary
// divergence returns PETSC_SUCCESS; inspect report.converged/reason. PETSc
// API/setup errors always propagate regardless of this flag.
//
// Runtime configuration: standard PETSc options under optionsPrefix, with
// fieldsplit_velocity_ / fieldsplit_pressure_ suffixes for the managed splits.
// The managed fieldsplit must be SCHUR, with Schur preconditioner SELF or A11.
// Other outer PC types may be selected if they support the supplied nests.
// Factorization of a MATNEST is not automatically converted to AIJ. KSP diagonal
// scaling on managed solvers is rejected to preserve the assembled operator.
//
// A validated pressure-only system nullspace is transferred to the implicit
// Schur operator. No additional modes, pressure pinning, or rigid-body modes
// are inferred. Matrices, RHS, and nullspace metadata must stay consistent.
PetscErrorCode SolveLinearSystem(
    LinearSystem& system, const LinearSolverOptions& options,
    LinearSolveReport& report);

#endif

