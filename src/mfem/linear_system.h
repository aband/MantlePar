#ifndef MANTLE_MFEM_LINEAR_SYSTEM_H
#define MANTLE_MFEM_LINEAR_SYSTEM_H

#include "assembly.h"

enum class LinearSystemKind { Stokes, Darcy, Coupled };
enum class PressureNullspaceMode { None, Constant, Provided };

struct PressureNullspaceOptions {
    PressureNullspaceMode mode = PressureNullspaceMode::None;

    // Used ONLY with Provided. An ordinary distributed pressure Vec with the
    // SAME layout as MixedBlocks::g. Contains ONE nonzero, finite, REAL mode
    // in the ASSEMBLED pressure variable, including any pressure rescaling.
    // It is copied and normalized; the caller retains ownership. Constant
    // constructs a vector of ones in pressure only, with zero velocity.
    Vec pressureMode = nullptr;

    // Each relevant signed block T must satisfy
    // ||T*q||_2 <= absoluteTolerance + relativeTolerance*||T||_F
    // for the normalized q, for both right and transpose nullspace actions.
    // These are validation tolerances, not regularization parameters.
    PetscReal absoluteTolerance = PetscReal(1e-12);
    PetscReal relativeTolerance = PetscReal(1e-10);

    // Compatibility requires |q^T*g| <= absoluteTolerance +
    // relativeTolerance*||g||_2, using the pressure RHS and normalized q.
    // Default: reject an incompatible RHS. If explicitly enabled, project the
    // SYSTEM'S RHS onto the compatible subspace. This changes the forcing;
    // the input MixedBlocks are never changed. The removed component's norm
    // is recorded in LinearSystem::removedRhsComponent.
    bool projectRhs = false;
};

struct LinearSystemOptions {
    // Complete system, using ALREADY boundary-adjusted raw-sign blocks:
    //   [ A                 pressureGradientSign * B^T ] [u] = [f]
    //   [ pressureRowSign*B pressureBlockScale    * C   ] [p]   [g]
    // B has pressure rows and velocity columns. Signs are applied once to
    // owned copies; f and g are NOT negated during composition.
    PetscReal pressureGradientSign = -1; // +1 or -1

    // REQUIRED: explicitly set +1 or -1, and use this SAME value for
    // BoundaryApplicationOptions::pressureRowSign. MixedBlocks does not
    // retain boundary-option metadata, so this consistency is the caller's
    // responsibility. For Darcy, match the natural boundary pressureSign to
    // pressureGradientSign as well.
    PetscReal pressureRowSign = 0;

    // -1 preserves the legacy pressure-block sign; +1 is also supported.
    // Zero EXPLICITLY omits the compaction term, useful for a chosen classical
    // mixed benchmark. This changes the equations; it is never automatic.
    PetscReal pressureBlockScale = -1; // -1, 0, or +1

    PressureNullspaceOptions pressureNullspace{};
};

struct CoupledPressureNullspaceOptions {
    PressureNullspaceMode mode = PressureNullspaceMode::None;

    // Provided: two ordinary distributed Vecs, matching stokes.g and darcy.g.
    // BOTH handles are required; either component may be the zero vector,
    // but their combined norm must be nonzero. Values must be finite and real,
    // expressed in the ASSEMBLED pressure variables. They are copied.
    // Constant: ones in BOTH pressure fields, zero in both velocity fields.
    // The complete [[0, 0], [q_s, q_d]] mode is normalized ONCE, preserving the
    // relative scaling of its components. Pressure rescaling will often make
    // Provided more appropriate than Constant.
    Vec stokesPressureMode = nullptr;
    Vec darcyPressureMode = nullptr;

    // For each field of M*z and M^T*z, require residual norm <= atol + rtol
    // times the Frobenius norm of that field's pressure-column block row.
    // Pressure-block actions are SUMMED before checking: C and K terms can
    // cancel. The candidate must be a mode on BOTH sides of the full operator.
    PetscReal absoluteTolerance = PetscReal(1e-12);
    PetscReal relativeTolerance = PetscReal(1e-10);

    // Default: reject |q_s^T*g_s + q_d^T*g_d| > atol + rtol*||(g_s,g_d)||_2.
    // Opt-in projection changes only the system's copied RHS and records the
    // removed norm in LinearSystem::removedRhsComponent.
    bool projectRhs = false;
};

struct CoupledLinearSystemOptions {
    // Use the corresponding boundary-adjusted blocks and pressure-row signs.
    // These component options select block signs/scales only: their individual
    // pressureNullspace settings MUST remain None, with no vector/projection.
    // Configure a JOINT mode below; individual gauges must not be imposed
    // before coupling because K can create/remove their pressure freedom.
    LinearSystemOptions stokes{};
    LinearSystemOptions darcy{};

    // Raw K_sd has Stokes-pressure rows, Darcy-pressure columns and the
    // NEGATIVE local sign from AssemblePressureCoupling. The legacy complete
    // matrix uses -K_sd and -K_sd^T, hence both defaults are -1.
    // Values -1, 0, +1 are supported, independently for each pressure row.
    // Setting both to zero explicitly produces a decoupled four-field matrix.
    // Match any pressure-row sign changes in ALL of that row's terms.
    PetscReal stokesDarcyCouplingScale = -1;
    PetscReal darcyStokesCouplingScale = -1;

    CoupledPressureNullspaceOptions pressureNullspace{};
};

// Every system has the outer saddle-point layout [velocity, pressure].
// For a coupled system these groups are U=[u_s,u_d] and P=[p_s,p_d]. Its
// outer 2x2 MATNEST contains four inner 2x2 MATNESTs; rhs and solution are
// two-level VECNESTs [[f_s,f_d],[g_s,g_d]] and [[u_s,u_d],[p_s,p_d]].
// For independent systems the outer blocks/vectors are ordinary field objects.
// Each field retains ALL of
// its original global DOFs, including essential boundary DOFs, and its MPI
// ownership. Combined indices come from the matching field-IS accessor; do
// not assume a global pressure offset equal to the velocity DOF count in MPI.
//
// Owns all handles. The matrix blocks and RHS are independent copies of the
// input, so that input may be destroyed after a successful build. solution
// starts at zero. There is no implicit Darcy velocity sign conversion or
// pressure rescaling when retrieving its fields.
//
// Explicit collective cleanup before PetscFinalize; no collective destructor.
// Treat matrix/rhs as fixed after construction. Rebuild after changing the
// equations, coefficients, partition, or boundary values. Direct modification
// invalidates any previously validated nullspace/compatibility information.
struct LinearSystem {
    Mat matrix = nullptr;
    Vec rhs = nullptr;
    Vec solution = nullptr;
    MatNullSpace pressureNullspace = nullptr;
    // Owned full-system leaf index sets for Coupled only, in the accessor's
    // original argument order [u_s,p_s,u_d,p_d]. Do not replace/destroy them.
    IS coupledFieldIS[4]{};
    LinearSystemKind kind = LinearSystemKind::Stokes;
    PetscReal removedRhsComponent = 0;

    LinearSystem() = default;
    LinearSystem(const LinearSystem&) = delete;
    LinearSystem& operator=(const LinearSystem&) = delete;
    LinearSystem(LinearSystem&&) = delete;
    LinearSystem& operator=(LinearSystem&&) = delete;

    bool IsEmpty() const noexcept
    {
        return !matrix && !rhs && !solution && !pressureNullspace &&
               !coupledFieldIS[0] && !coupledFieldIS[1] &&
               !coupledFieldIS[2] && !coupledFieldIS[3];
    }
};

PetscErrorCode DestroyLinearSystem(LinearSystem& system);

// COLLECTIVE on the same communicator/rank ordering as EVERY input object.
// Requires the assembled AIJ blocks produced by assembly/boundary_conditions,
// with consistent row/column/vector ownership. Empty owners participate.
// Apply boundary conditions first; no boundary loads or lifting happen here.
// No coupling matrix or data from the other system is required.
//
// Pass identical scalar options on every rank. Output must be empty. It is
// published only after success. Input handles and values are never modified.
// The two entry points share their algebra and record the selected kind;
// MixedBlocks carries no element-space tag, so supply the appropriate blocks.
//
// Optional pressure nullspace: validates ONE pressure-only mode against BOTH
// sides of the completed operator before attaching it. Does not infer a
// constant mode, pin pressure, or handle velocity rigid-body modes. Default
// None is appropriate when C/boundary conditions make pressure unique. Other
// singularities and additional modes need their own
// model-specific handling. Current operators and provided modes are real;
// PETSc builds with complex scalar storage are supported.
PetscErrorCode BuildStokesLinearSystem(
    MPI_Comm comm, const MixedBlocks& blocks, const LinearSystemOptions& options,
    LinearSystem& result);

PetscErrorCode BuildDarcyLinearSystem(
    MPI_Comm comm, const MixedBlocks& blocks, const LinearSystemOptions& options,
    LinearSystem& result);

// COLLECTIVE. Constructs the legacy velocity-pressure saddle-point grouping:
//   [ A  G ] [ U ] = [ F ], U=[u_s,u_d], P=[p_s,p_d],
//   [ D  E ] [ P ]   [ H ]  F=[f_s,f_d], H=[g_s,g_d].
// A=diag(A_s,A_d), G=diag(a_s*B_s^T,a_d*B_d^T),
// D=diag(b_s*B_s,b_d*B_d),
// E=[[c_s*C_s,k_sd*K_sd],[k_ds*K_sd^T,c_d*C_d]].
// a,b,c are pressureGradientSign, pressureRowSign, pressureBlockScale.
// The outer matrix and EACH of A,G,D,E are 2x2 MATNESTs.
// With legacy signs (a=b=c=k=-1), G=D^T and E is the negative coupled
// compaction block. Independent sign choices may make the operator nonsymmetric.
//
// stokes and darcy must be boundary-adjusted MixedBlocks with RAW assembly
// block signs, not already signed diagonal blocks extracted from a system.
// K_sd must be a nonnull, assembled AIJ matrix, even when coupling scales are
// zero. Its row ownership must match stokes.g and columns must match darcy.g;
// the two pressure spaces may have different sizes/partitions on the same comm.
// No boundary contribution is reapplied. All inputs, including K_sd, are
// independently copied and may be destroyed after successful construction.
//
// Optional nullspace handling supports ONE joint real pressure-only mode,
// validated on both sides of the complete coupled matrix. It does not combine
// independently projected RHSs or independently pinned pressure systems.
// Additional modes or different left/right modes require separate handling;
// leave mode=None when attaching such model-specific nullspaces yourself.
// Output must be empty. Other collective/lifetime rules above also apply.
PetscErrorCode BuildCoupledLinearSystem(
    MPI_Comm comm, const MixedBlocks& stokes, const MixedBlocks& darcy,
    Mat K_sd, const CoupledLinearSystemOptions& options, LinearSystem& result);

// Noncollective borrowed views. Do NOT destroy or replace these handles.
// They remain valid until system is destroyed. Copy a field with VecDuplicate
// and VecCopy when an independent, longer-lived result is required.
// These return the OUTER velocity/pressure groups for every system kind.
// For Coupled, velocity=[u_s,u_d] and pressure=[p_s,p_d] are VECNESTs.
PetscErrorCode GetLinearSystemSolution(
    const LinearSystem& system, Vec& velocity, Vec& pressure);

PetscErrorCode GetLinearSystemRhs(
    const LinearSystem& system, Vec& velocity, Vec& pressure);

// Borrowed OUTER [velocity,pressure] index sets, e.g. for a saddle-point
// PCFIELDSPLIT. For Coupled these select the whole U and P groups.
PetscErrorCode GetLinearSystemFieldIS(
    const LinearSystem& system, IS& velocity, IS& pressure);

// Borrowed SIGNED outer blocks of [A G; D E], for every system kind.
// For Coupled each is a 2x2 MATNEST in [Stokes,Darcy] order. In particular
// pressure=E contains both compaction and inter-system pressure coupling.
PetscErrorCode GetLinearSystemBlocks(
    const LinearSystem& system, Mat& velocity, Mat& gradient,
    Mat& divergence, Mat& pressure);

// Noncollective borrowed LEAF views for kind=Coupled. Keep the original
// argument order [u_s,p_s,u_d,p_d] for source compatibility; this is NOT the
// grouped matrix order [[u_s,u_d],[p_s,p_d]]. These reject independent systems.
// Do not destroy the returned Vec/IS handles.
PetscErrorCode GetCoupledLinearSystemSolution(
    const LinearSystem& system, Vec& stokesVelocity, Vec& stokesPressure,
    Vec& darcyVelocity, Vec& darcyPressure);

PetscErrorCode GetCoupledLinearSystemRhs(
    const LinearSystem& system, Vec& stokesVelocity, Vec& stokesPressure,
    Vec& darcyVelocity, Vec& darcyPressure);

// Leaf IS use the FULL coupled numbering, composed through the outer U/P IS.
// IS obtained directly from an inner block instead use that group's numbering.
// These full indices are for numbering/scatters; PETSc nested extraction uses
// immediate-level ISs. Use the outer U/P IS for a saddle split, then the inner
// group's IS for its Stokes/Darcy split; use the leaf Vec accessors above.
PetscErrorCode GetCoupledLinearSystemFieldIS(
    const LinearSystem& system, IS& stokesVelocity, IS& stokesPressure,
    IS& darcyVelocity, IS& darcyPressure);

// COLLECTIVE. Optionally call after solving: removes the VALIDATED pressure
// mode from system.solution, giving an algebraic orthogonality gauge. This
// leaves the velocity unchanged. For Constant it is an unweighted coefficient
// mean, NOT an area-weighted physical-pressure mean on a nonuniform mesh.
// Requires a built system with an attached pressureNullspace. Does not change
// the matrix or RHS. Pressure pinning/physical-mean constraints are separate.
// Works for both independent and coupled systems. For a coupled mode it removes
// ONE joint component; it does not impose a separate mean on each pressure.
PetscErrorCode RemoveLinearSystemPressureNullspace(LinearSystem& system);

#endif

