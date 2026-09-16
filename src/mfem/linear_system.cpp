#include "linear_system.h"

#include <initializer_list>
#include <limits>
#include <utility>
#include <vector>

namespace {

bool Finite(PetscReal value) { return !PetscIsInfOrNanReal(value); }

PetscErrorCode AgreeError(MPI_Comm comm, PetscErrorCode local, const char* stage)
{
    PetscFunctionBeginUser;
    const int own = static_cast<int>(local);
    int all = 0;
    PetscCallMPI(MPI_Allreduce(&own, &all, 1, MPI_INT, MPI_MAX, comm));
    PetscCheck(!all, comm, static_cast<PetscErrorCode>(all),
               "%s failed on at least one rank; see the originating error", stage);
    PetscFunctionReturn(PETSC_SUCCESS);
}

void KeepFirst(PetscErrorCode& first, PetscErrorCode next)
{ if (!first) first = next; }

PetscErrorCode CheckComm(MPI_Comm comm, PetscObject object)
{
    PetscFunctionBeginUser;
    int comparison = MPI_UNEQUAL;
    PetscCallMPI(MPI_Comm_compare(comm, PetscObjectComm(object), &comparison));
    PetscCheck(comparison == MPI_IDENT || comparison == MPI_CONGRUENT,
               PETSC_COMM_SELF, PETSC_ERR_ARG_INCOMP,
               "All system objects must use the same communicator and rank ordering");
    PetscFunctionReturn(PETSC_SUCCESS);
}

struct Layout {
    PetscInt velocity = 0, pressure = 0;
    PetscInt velocityBegin = 0, velocityEnd = 0;
    PetscInt pressureBegin = 0, pressureEnd = 0;
};

PetscErrorCode CheckMatrix(MPI_Comm comm, Mat matrix, PetscInt rows, PetscInt columns,
                           PetscInt rb, PetscInt re, PetscInt cb, PetscInt ce)
{
    PetscFunctionBeginUser;
    PetscCheck(matrix, PETSC_COMM_SELF, PETSC_ERR_ARG_NULL, "Missing system matrix block");
    PetscCall(CheckComm(comm, reinterpret_cast<PetscObject>(matrix)));
    PetscInt M=0, N=0, m=0, n=0, begin=0, end=0, first=0, last=0;
    PetscBool assembled = PETSC_FALSE;
    PetscCall(MatGetSize(matrix, &M, &N));
    PetscCall(MatGetLocalSize(matrix, &m, &n));
    PetscCall(MatGetOwnershipRange(matrix, &begin, &end));
    PetscCall(MatGetOwnershipRangeColumn(matrix, &first, &last));
    PetscCall(MatAssembled(matrix, &assembled));
    PetscCheck(assembled && M==rows && N==columns && m==re-rb && n==ce-cb &&
               begin==rb && end==re && first==cb && last==ce,
               PETSC_COMM_SELF, PETSC_ERR_ARG_SIZ,
               "System block is unassembled or its dimensions/ownership differ");
    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode CheckVector(MPI_Comm comm, Vec vector, PetscInt size,
                           PetscInt begin, PetscInt end, bool requireReal)
{
    PetscFunctionBeginUser;
    PetscCheck(vector, PETSC_COMM_SELF, PETSC_ERR_ARG_NULL, "Missing system vector");
    PetscCall(CheckComm(comm, reinterpret_cast<PetscObject>(vector)));
    PetscInt N=0, n=0, first=0, last=0;
    PetscCall(VecGetSize(vector, &N));
    PetscCall(VecGetLocalSize(vector, &n));
    PetscCall(VecGetOwnershipRange(vector, &first, &last));
    PetscCheck(N==size && n==end-begin && first==begin && last==end,
               PETSC_COMM_SELF, PETSC_ERR_ARG_SIZ, "System vector layout differs from its field");
    const PetscScalar* values = nullptr;
    PetscCall(VecGetArrayRead(vector, &values));
    bool valid = true;
    for (PetscInt i=0; i<n; ++i)
        if (PetscIsInfOrNanScalar(values[i]) || (requireReal && PetscImaginaryPart(values[i]) != 0)) valid = false;
    PetscCall(VecRestoreArrayRead(vector, &values));
    PetscCheck(valid, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE,
               "Vector values must be finite; a provided pressure nullspace mode must also be real");
    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode CheckInput(MPI_Comm comm, const MixedBlocks& blocks,
                          const LinearSystemOptions& options, const LinearSystem& result,
                          Layout& layout)
{
    PetscFunctionBeginUser;
    PetscCheck(result.IsEmpty(), PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG,
               "LinearSystem output must be empty; call DestroyLinearSystem first");
    PetscCheck(options.pressureGradientSign==1 || options.pressureGradientSign==-1,
               PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "pressureGradientSign must be +1 or -1");
    PetscCheck(options.pressureRowSign==1 || options.pressureRowSign==-1,
               PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE,
               "Set pressureRowSign explicitly, matching BoundaryApplicationOptions");
    PetscCheck(options.pressureBlockScale==-1 || options.pressureBlockScale==0 || options.pressureBlockScale==1,
               PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "pressureBlockScale must be -1, 0, or +1");
    const auto& ns = options.pressureNullspace;
    PetscCheck(ns.mode==PressureNullspaceMode::None || ns.mode==PressureNullspaceMode::Constant ||
               ns.mode==PressureNullspaceMode::Provided, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE,
               "Invalid pressure nullspace mode");
    PetscCheck(Finite(ns.absoluteTolerance) && ns.absoluteTolerance>=0 &&
               Finite(ns.relativeTolerance) && ns.relativeTolerance>=0,
               PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Nullspace tolerances must be finite and nonnegative");
    PetscCheck((ns.mode==PressureNullspaceMode::Provided) == (ns.pressureMode!=nullptr),
               PETSC_COMM_SELF, PETSC_ERR_ARG_INCOMP, "Provide pressureMode exactly when mode is Provided");
    PetscCheck(ns.mode!=PressureNullspaceMode::None || !ns.projectRhs,
               PETSC_COMM_SELF, PETSC_ERR_ARG_INCOMP, "RHS projection requires a pressure nullspace mode");
    PetscCheck(blocks.B, PETSC_COMM_SELF, PETSC_ERR_ARG_NULL, "Missing pressure-by-velocity B block");
    PetscCall(CheckComm(comm, reinterpret_cast<PetscObject>(blocks.B)));
    PetscCall(MatGetSize(blocks.B, &layout.pressure, &layout.velocity));
    PetscCall(MatGetOwnershipRange(blocks.B, &layout.pressureBegin, &layout.pressureEnd));
    PetscCall(MatGetOwnershipRangeColumn(blocks.B, &layout.velocityBegin, &layout.velocityEnd));
    PetscCheck(layout.velocity>0 && layout.pressure>0 &&
               layout.velocity<=std::numeric_limits<PetscInt>::max()-layout.pressure,
               PETSC_COMM_SELF, PETSC_ERR_ARG_SIZ, "Invalid or overflowing mixed-system size");
    PetscCall(CheckMatrix(comm, blocks.A, layout.velocity, layout.velocity,
                          layout.velocityBegin, layout.velocityEnd, layout.velocityBegin, layout.velocityEnd));
    PetscCall(CheckMatrix(comm, blocks.B, layout.pressure, layout.velocity,
                          layout.pressureBegin, layout.pressureEnd, layout.velocityBegin, layout.velocityEnd));
    PetscCall(CheckMatrix(comm, blocks.C, layout.pressure, layout.pressure,
                          layout.pressureBegin, layout.pressureEnd, layout.pressureBegin, layout.pressureEnd));
    PetscCall(CheckVector(comm, blocks.f, layout.velocity, layout.velocityBegin, layout.velocityEnd, false));
    PetscCall(CheckVector(comm, blocks.g, layout.pressure, layout.pressureBegin, layout.pressureEnd, false));
    if (ns.mode==PressureNullspaceMode::Provided)
        PetscCall(CheckVector(comm, ns.pressureMode, layout.pressure, layout.pressureBegin, layout.pressureEnd, true));
    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode CheckOptions(MPI_Comm comm, LinearSystemKind kind, const LinearSystemOptions& options)
{
    PetscFunctionBeginUser;
    const auto& ns = options.pressureNullspace;
    const PetscReal own[5] = {options.pressureGradientSign, options.pressureRowSign,
        options.pressureBlockScale, ns.absoluteTolerance, ns.relativeTolerance};
    PetscReal low[5]{}, high[5]{};
    PetscCallMPI(MPI_Allreduce(own, low, 5, MPIU_REAL, MPI_MIN, comm));
    PetscCallMPI(MPI_Allreduce(own, high, 5, MPIU_REAL, MPI_MAX, comm));
    for (int i=0; i<5; ++i)
        PetscCheck(low[i]==high[i], comm, PETSC_ERR_ARG_INCOMP, "Linear-system scalar options differ across ranks");
    const int flags[3] = {static_cast<int>(kind), static_cast<int>(ns.mode), ns.projectRhs ? 1 : 0};
    int lowest[3]{}, highest[3]{};
    PetscCallMPI(MPI_Allreduce(flags, lowest, 3, MPI_INT, MPI_MIN, comm));
    PetscCallMPI(MPI_Allreduce(flags, highest, 3, MPI_INT, MPI_MAX, comm));
    for (int i=0; i<3; ++i)
        PetscCheck(lowest[i]==highest[i], comm, PETSC_ERR_ARG_INCOMP, "Linear-system kind/nullspace options differ across ranks");
    PetscFunctionReturn(PETSC_SUCCESS);
}

struct Work {
    LinearSystem system;
    Mat blocks[4]{}; // A, signed B^T, signed B, signed C; nest retains references.
    Vec mode = nullptr, velocityWork = nullptr, pressureWork = nullptr;
};

PetscErrorCode DestroyScratch(Work& work)
{
    PetscFunctionBeginUser;
    PetscErrorCode error = PETSC_SUCCESS;
    KeepFirst(error, VecDestroy(&work.mode));
    KeepFirst(error, VecDestroy(&work.velocityWork));
    KeepFirst(error, VecDestroy(&work.pressureWork));
    for (auto& block : work.blocks) KeepFirst(error, MatDestroy(&block));
    PetscCall(error);
    PetscFunctionReturn(PETSC_SUCCESS);
}

// The mode has zero velocity, so these block actions are exactly the nonzero
// parts of M*z and M^T*z. Checking each separately avoids letting a very large
// velocity block hide an invalid pressure mode in a whole-matrix tolerance.
PetscErrorCode CheckModeAction(MPI_Comm comm, Mat block, Vec mode, Vec action,
                               bool transpose, const PressureNullspaceOptions& options,
                               const char* description)
{
    PetscFunctionBeginUser;
    PetscReal blockNorm=0, residual=0;
    PetscCall(AgreeError(comm, MatNorm(block, NORM_FROBENIUS, &blockNorm), "Nullspace block norm"));
    if (transpose)
        PetscCall(AgreeError(comm, MatMultTranspose(block, mode, action), "Transpose nullspace action"));
    else PetscCall(AgreeError(comm, MatMult(block, mode, action), "Right nullspace action"));
    PetscCall(AgreeError(comm, VecNorm(action, NORM_2, &residual), "Nullspace action norm"));
    const PetscReal limit = options.absoluteTolerance + options.relativeTolerance*blockNorm;
    PetscCheck(Finite(blockNorm) && Finite(residual) && Finite(limit) && residual<=limit,
               comm, PETSC_ERR_ARG_INCOMP,
               "Pressure mode is invalid for %s: residual %g, allowed %g; check C, pressure scaling and boundary conditions",
               description, static_cast<double>(residual), static_cast<double>(limit));
    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode AttachPressureNullspace(MPI_Comm comm, const PressureNullspaceOptions& options, Work& work)
{
    PetscFunctionBeginUser;
    if (options.mode==PressureNullspaceMode::None) PetscFunctionReturn(PETSC_SUCCESS);
    auto& system = work.system;
    PetscCall(AgreeError(comm, VecDuplicate(system.solution, &work.mode), "Create pressure-only nullspace vector"));
    PetscCall(AgreeError(comm, VecSet(work.mode, 0), "Zero nullspace velocity"));
    Vec velocity=nullptr, pressure=nullptr;
    PetscCall(AgreeError(comm, VecNestGetSubVec(work.mode, 0, &velocity), "Get nullspace velocity"));
    PetscCall(AgreeError(comm, VecNestGetSubVec(work.mode, 1, &pressure), "Get nullspace pressure"));
    if (options.mode==PressureNullspaceMode::Constant)
        PetscCall(AgreeError(comm, VecSet(pressure, 1), "Set constant pressure mode"));
    else PetscCall(AgreeError(comm, VecCopy(options.pressureMode, pressure), "Copy pressure mode"));
    PetscReal norm = 0;
    PetscCall(AgreeError(comm, VecNorm(pressure, NORM_2, &norm), "Pressure mode norm"));
    PetscCheck(Finite(norm) && norm>0 && Finite(PetscReal(1)/norm), comm, PETSC_ERR_ARG_OUTOFRANGE,
               "Pressure nullspace mode must have a finite, nonzero, normalizable norm");
    PetscCall(AgreeError(comm, VecScale(pressure, PetscReal(1)/norm), "Normalize pressure mode"));
    PetscCall(AgreeError(comm, VecDuplicate(velocity, &work.velocityWork), "Create velocity mode workspace"));
    PetscCall(AgreeError(comm, VecDuplicate(pressure, &work.pressureWork), "Create pressure mode workspace"));
    PetscCall(CheckModeAction(comm, work.blocks[1], pressure, work.velocityWork, false, options, "momentum pressure coupling"));
    PetscCall(CheckModeAction(comm, work.blocks[3], pressure, work.pressureWork, false, options, "pressure block"));
    PetscCall(CheckModeAction(comm, work.blocks[2], pressure, work.velocityWork, true, options, "transposed continuity coupling"));
    PetscCall(CheckModeAction(comm, work.blocks[3], pressure, work.pressureWork, true, options, "transposed pressure block"));

    // This is a pressure-only explicit basis, NOT has_cnst=PETSC_TRUE, which
    // would incorrectly put ones in every velocity component as well.
    PetscCall(AgreeError(comm, MatNullSpaceCreate(comm, PETSC_FALSE, 1, &work.mode,
                                                &system.pressureNullspace), "Create validated pressure nullspace"));
    Vec pressureRhs = nullptr;
    PetscCall(AgreeError(comm, VecNestGetSubVec(system.rhs, 1, &pressureRhs), "Get pressure RHS for compatibility"));
    PetscScalar component = 0;
    PetscReal rhsNorm = 0;
    PetscCall(AgreeError(comm, VecDot(pressure, pressureRhs, &component), "RHS nullspace component"));
    PetscCall(AgreeError(comm, VecNorm(pressureRhs, NORM_2, &rhsNorm), "Pressure RHS norm"));
    const PetscReal limit = options.absoluteTolerance + options.relativeTolerance*rhsNorm;
    PetscCheck(!PetscIsInfOrNanScalar(component) && Finite(rhsNorm) && Finite(limit),
               comm, PETSC_ERR_FP, "Nonfinite RHS compatibility calculation");
    PetscCheck(options.projectRhs || PetscAbsScalar(component)<=limit,
               comm, PETSC_ERR_ARG_INCOMP,
               "RHS is incompatible with pressure nullspace: component %g, allowed %g; correct the data or explicitly enable projectRhs",
               static_cast<double>(PetscAbsScalar(component)), static_cast<double>(limit));
    if (options.projectRhs) {
        PetscCall(AgreeError(comm, MatNullSpaceRemove(system.pressureNullspace, system.rhs), "Project system RHS"));
        system.removedRhsComponent = PetscAbsScalar(component);
    }
    PetscCall(AgreeError(comm, MatSetNullSpace(system.matrix, system.pressureNullspace), "Attach right pressure nullspace"));
    PetscCall(AgreeError(comm, MatSetTransposeNullSpace(system.matrix, system.pressureNullspace), "Attach transpose pressure nullspace"));
    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode CopySignedBlocks(MPI_Comm comm, const MixedBlocks& input,
                                const LinearSystemOptions& options, Mat (&blocks)[4])
{
    PetscFunctionBeginUser;
    PetscCall(AgreeError(comm, MatDuplicate(input.A, MAT_COPY_VALUES, &blocks[0]), "Copy velocity block"));
    // Explicit transpose of the CONSTRAINED B. No dependence on an obsolete
    // pre-elimination transpose, and no mutation of the caller's matrix.
    PetscCall(AgreeError(comm, MatTranspose(input.B, MAT_INITIAL_MATRIX, &blocks[1]), "Transpose constrained B"));
    PetscCall(AgreeError(comm, MatScale(blocks[1], options.pressureGradientSign), "Set gradient sign"));
    PetscCall(AgreeError(comm, MatDuplicate(input.B, MAT_COPY_VALUES, &blocks[2]), "Copy continuity block"));
    PetscCall(AgreeError(comm, MatScale(blocks[2], options.pressureRowSign), "Set pressure-row sign"));
    PetscCall(AgreeError(comm, MatDuplicate(input.C, MAT_COPY_VALUES, &blocks[3]), "Copy pressure block"));
    if (options.pressureBlockScale==0)
        PetscCall(AgreeError(comm, MatZeroEntries(blocks[3]), "Omit compaction block explicitly"));
    else PetscCall(AgreeError(comm, MatScale(blocks[3], options.pressureBlockScale), "Set pressure-block sign"));
    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode Compose(MPI_Comm comm, LinearSystemKind kind, const MixedBlocks& input,
                       const LinearSystemOptions& options, Work& work)
{
    PetscFunctionBeginUser;
    auto& system = work.system;
    system.kind = kind;
    PetscCall(CopySignedBlocks(comm, input, options, work.blocks));
    PetscCall(AgreeError(comm, MatCreateNest(comm, 2, nullptr, 2, nullptr, work.blocks, &system.matrix), "Create mixed matrix nest"));
    PetscCall(AgreeError(comm, MatNestSetVecType(system.matrix, VECNEST), "Choose matching nested vectors"));
    PetscCall(AgreeError(comm, MatCreateVecs(system.matrix, &system.solution, &system.rhs), "Create compatible mixed vectors"));
    PetscCall(AgreeError(comm, VecSet(system.solution, 0), "Initialize mixed solution"));
    Vec f=nullptr, g=nullptr;
    PetscCall(AgreeError(comm, VecNestGetSubVec(system.rhs, 0, &f), "Get velocity RHS"));
    PetscCall(AgreeError(comm, VecNestGetSubVec(system.rhs, 1, &g), "Get pressure RHS"));
    PetscCall(AgreeError(comm, VecCopy(input.f, f), "Copy boundary-adjusted velocity RHS"));
    PetscCall(AgreeError(comm, VecCopy(input.g, g), "Copy boundary-adjusted pressure RHS"));
    PetscCall(AttachPressureNullspace(comm, options.pressureNullspace, work));
    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode Build(MPI_Comm comm, LinearSystemKind kind, const MixedBlocks& input,
                     const LinearSystemOptions& options, LinearSystem& result)
{
    PetscFunctionBeginUser;
    Layout layout;
    PetscCall(AgreeError(comm, CheckInput(comm, input, options, result, layout), "Linear-system inputs"));
    PetscCall(CheckOptions(comm, kind, options));
    Work work;
    const PetscErrorCode error = Compose(comm, kind, input, options, work);
    const PetscErrorCode cleanup = DestroyScratch(work);
    if (error || cleanup) {
        const PetscErrorCode destroy = DestroyLinearSystem(work.system);
        (void)destroy;
        PetscCall(error ? error : cleanup);
    }
    std::swap(result.matrix, work.system.matrix);
    std::swap(result.rhs, work.system.rhs);
    std::swap(result.solution, work.system.solution);
    std::swap(result.pressureNullspace, work.system.pressureNullspace);
    result.kind = kind;
    result.removedRhsComponent = work.system.removedRhsComponent;
    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode CheckCoupledInput(MPI_Comm comm, const MixedBlocks& stokes,
                                 const MixedBlocks& darcy, Mat coupling,
                                 const CoupledLinearSystemOptions& options,
                                 const LinearSystem& result)
{
    PetscFunctionBeginUser;
    PetscCheck(options.stokes.pressureNullspace.mode==PressureNullspaceMode::None &&
               options.darcy.pressureNullspace.mode==PressureNullspaceMode::None,
               PETSC_COMM_SELF, PETSC_ERR_ARG_INCOMP,
               "Configure the coupled pressureNullspace, not independent subsystem modes");
    Layout s, d;
    PetscCall(CheckInput(comm, stokes, options.stokes, result, s));
    PetscCall(CheckInput(comm, darcy, options.darcy, result, d));
    PetscCheck(s.velocity+s.pressure <= std::numeric_limits<PetscInt>::max()-(d.velocity+d.pressure),
               PETSC_COMM_SELF, PETSC_ERR_ARG_SIZ, "Coupled system size overflows PetscInt");
    PetscCall(CheckMatrix(comm, coupling, s.pressure, d.pressure,
                          s.pressureBegin, s.pressureEnd, d.pressureBegin, d.pressureEnd));
    for (const PetscReal scale : {options.stokesDarcyCouplingScale, options.darcyStokesCouplingScale})
        PetscCheck(scale==-1 || scale==0 || scale==1, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE,
                   "Pressure coupling scales must be -1, 0, or +1");
    const auto& ns = options.pressureNullspace;
    PetscCheck(ns.mode==PressureNullspaceMode::None || ns.mode==PressureNullspaceMode::Constant ||
               ns.mode==PressureNullspaceMode::Provided, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE,
               "Invalid coupled pressure nullspace mode");
    PetscCheck(Finite(ns.absoluteTolerance) && ns.absoluteTolerance>=0 &&
               Finite(ns.relativeTolerance) && ns.relativeTolerance>=0,
               PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Coupled nullspace tolerances must be finite and nonnegative");
    const bool provided = ns.mode==PressureNullspaceMode::Provided;
    PetscCheck(provided==(ns.stokesPressureMode!=nullptr) && provided==(ns.darcyPressureMode!=nullptr),
               PETSC_COMM_SELF, PETSC_ERR_ARG_INCOMP,
               "Provide both pressure-mode vectors exactly when coupled mode is Provided");
    PetscCheck(ns.mode!=PressureNullspaceMode::None || !ns.projectRhs,
               PETSC_COMM_SELF, PETSC_ERR_ARG_INCOMP, "Coupled RHS projection requires a joint pressure mode");
    if (provided) {
        PetscCall(CheckVector(comm, ns.stokesPressureMode, s.pressure, s.pressureBegin, s.pressureEnd, true));
        PetscCall(CheckVector(comm, ns.darcyPressureMode, d.pressure, d.pressureBegin, d.pressureEnd, true));
    }
    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode CheckCoupledOptions(MPI_Comm comm, const CoupledLinearSystemOptions& options)
{
    PetscFunctionBeginUser;
    PetscCall(CheckOptions(comm, LinearSystemKind::Coupled, options.stokes));
    PetscCall(CheckOptions(comm, LinearSystemKind::Coupled, options.darcy));
    const auto& ns = options.pressureNullspace;
    const PetscReal own[4] = {options.stokesDarcyCouplingScale, options.darcyStokesCouplingScale,
                              ns.absoluteTolerance, ns.relativeTolerance};
    PetscReal low[4]{}, high[4]{};
    PetscCallMPI(MPI_Allreduce(own, low, 4, MPIU_REAL, MPI_MIN, comm));
    PetscCallMPI(MPI_Allreduce(own, high, 4, MPIU_REAL, MPI_MAX, comm));
    for (int i=0; i<4; ++i)
        PetscCheck(low[i]==high[i], comm, PETSC_ERR_ARG_INCOMP, "Coupled scalar options differ across ranks");
    const int flags[2] = {static_cast<int>(ns.mode), ns.projectRhs ? 1 : 0};
    int lowest[2]{}, highest[2]{};
    PetscCallMPI(MPI_Allreduce(flags, lowest, 2, MPI_INT, MPI_MIN, comm));
    PetscCallMPI(MPI_Allreduce(flags, highest, 2, MPI_INT, MPI_MAX, comm));
    PetscCheck(lowest[0]==highest[0] && lowest[1]==highest[1], comm, PETSC_ERR_ARG_INCOMP,
               "Coupled nullspace options differ across ranks");
    PetscFunctionReturn(PETSC_SUCCESS);
}

struct CoupledWork {
    LinearSystem system;
    Mat stokes[4]{}, darcy[4]{}, coupling[2]{};
    Mat groups[4]{}; // Outer A,G,D,E; each is itself a 2x2 MATNEST.
    Vec mode = nullptr, action = nullptr;
};

PetscErrorCode DestroyScratch(CoupledWork& work)
{
    PetscFunctionBeginUser;
    PetscErrorCode error = PETSC_SUCCESS;
    KeepFirst(error, VecDestroy(&work.mode));
    KeepFirst(error, VecDestroy(&work.action));
    for (auto& block : work.stokes) KeepFirst(error, MatDestroy(&block));
    for (auto& block : work.darcy) KeepFirst(error, MatDestroy(&block));
    for (auto& block : work.coupling) KeepFirst(error, MatDestroy(&block));
    for (auto& group : work.groups) KeepFirst(error, MatDestroy(&group));
    PetscCall(error);
    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode GetFourFields(Vec vector, Vec (&fields)[4])
{
    PetscFunctionBeginUser;
    Vec velocity=nullptr, pressure=nullptr;
    PetscCall(VecNestGetSubVec(vector, 0, &velocity));
    PetscCall(VecNestGetSubVec(vector, 1, &pressure));
    // Preserve the public leaf accessor order [u_s,p_s,u_d,p_d].
    PetscCall(VecNestGetSubVec(velocity, 0, &fields[0]));
    PetscCall(VecNestGetSubVec(pressure, 0, &fields[1]));
    PetscCall(VecNestGetSubVec(velocity, 1, &fields[2]));
    PetscCall(VecNestGetSubVec(pressure, 1, &fields[3]));
    PetscFunctionReturn(PETSC_SUCCESS);
}

// Compose one leaf IS through its outer group's IS. MatCreateNest uses
// rank-local concatenation, so neither a global field-size offset nor a
// concatenation of the unmodified inner IS gives full-system indices in MPI.
PetscErrorCode MapLeafIndices(IS parent, IS child, PetscInt begin, PetscInt end,
                              std::vector<PetscInt>& mapped)
{
    PetscFunctionBeginUser;
    PetscInt parentSize=0, childSize=0;
    PetscCall(ISGetLocalSize(parent, &parentSize));
    PetscCall(ISGetLocalSize(child, &childSize));
    PetscCheck(parentSize==end-begin, PETSC_COMM_SELF, PETSC_ERR_PLIB,
               "Outer field IS and grouped ownership disagree");
    mapped.resize(static_cast<std::size_t>(childSize));
    const PetscInt *outer=nullptr, *inner=nullptr;
    PetscCall(ISGetIndices(parent, &outer));
    PetscCall(ISGetIndices(child, &inner));
    bool valid = true;
    for (PetscInt i=0; i<childSize; ++i) {
        if (inner[i]<begin || inner[i]>=end) valid = false;
        else mapped[static_cast<std::size_t>(i)] = outer[inner[i]-begin];
    }
    PetscCall(ISRestoreIndices(child, &inner));
    PetscCall(ISRestoreIndices(parent, &outer));
    PetscCheck(valid, PETSC_COMM_SELF, PETSC_ERR_PLIB,
               "Inner field IS contains an index outside the locally owned group");
    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode CreateCoupledFieldIS(MPI_Comm comm, CoupledWork& work)
{
    PetscFunctionBeginUser;
    IS outer[2]{};
    PetscCall(AgreeError(comm, MatNestGetISs(work.system.matrix, nullptr, outer), "Get outer saddle field IS"));
    for (PetscInt group=0; group<2; ++group) {
        Mat diagonal = work.groups[group==0 ? 0 : 3];
        IS inner[2]{};
        PetscInt begin=0, end=0;
        PetscCall(AgreeError(comm, MatNestGetISs(diagonal, nullptr, inner), "Get inner field IS"));
        PetscCall(AgreeError(comm, MatGetOwnershipRangeColumn(diagonal, &begin, &end), "Get grouped field ownership"));
        for (PetscInt part=0; part<2; ++part) {
            std::vector<PetscInt> mapped;
            PetscCall(AgreeError(comm, MapLeafIndices(outer[group], inner[part], begin, end, mapped), "Map leaf field IS"));
            PetscCall(AgreeError(comm, ISCreateGeneral(comm, static_cast<PetscInt>(mapped.size()), mapped.data(),
                                                       PETSC_COPY_VALUES, &work.system.coupledFieldIS[2*part+group]),
                                 "Create full-system leaf IS"));
        }
    }
    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode CheckCoupledModeAction(MPI_Comm comm, CoupledWork& work, bool transpose,
                                      const CoupledPressureNullspaceOptions& options)
{
    PetscFunctionBeginUser;
    if (transpose)
        PetscCall(AgreeError(comm, MatMultTranspose(work.system.matrix, work.mode, work.action), "Coupled transpose nullspace action"));
    else PetscCall(AgreeError(comm, MatMult(work.system.matrix, work.mode, work.action), "Coupled right nullspace action"));
    Vec actions[4]{};
    PetscCall(AgreeError(comm, GetFourFields(work.action, actions), "Get coupled nullspace action fields"));
    // Only pressure columns act on z=[[0,0],[q_s,q_d]]. For M^T the matching
    // rows of M are used. Leaf arrays retain [u_s,p_s,u_d,p_d] accessor order.
    // Frobenius norms are unchanged by transposition.
    Mat first[4] = {work.stokes[transpose ? 2 : 1], work.stokes[3],
                    work.darcy[transpose ? 2 : 1], work.darcy[3]};
    Mat second[4] = {nullptr, work.coupling[transpose ? 1 : 0],
                     nullptr, work.coupling[transpose ? 0 : 1]};
    for (PetscInt field=0; field<4; ++field) {
        PetscReal n1=0, n2=0, residual=0;
        PetscCall(AgreeError(comm, MatNorm(first[field], NORM_FROBENIUS, &n1), "Coupled nullspace block norm"));
        if (second[field])
            PetscCall(AgreeError(comm, MatNorm(second[field], NORM_FROBENIUS, &n2), "Coupled cross-block norm"));
        PetscCall(AgreeError(comm, VecNorm(actions[field], NORM_2, &residual), "Coupled mode residual norm"));
        const PetscReal limit = options.absoluteTolerance + options.relativeTolerance*PetscHypotReal(n1, n2);
        PetscCheck(Finite(n1) && Finite(n2) && Finite(residual) && Finite(limit) && residual<=limit,
                   comm, PETSC_ERR_ARG_INCOMP,
                   "Joint pressure mode fails %s field %" PetscInt_FMT ": residual %g, allowed %g; check coupling, scaling and boundary conditions",
                   transpose ? "transpose" : "right", field, static_cast<double>(residual), static_cast<double>(limit));
    }
    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode AttachCoupledPressureNullspace(MPI_Comm comm,
                                              const CoupledPressureNullspaceOptions& options,
                                              CoupledWork& work)
{
    PetscFunctionBeginUser;
    if (options.mode==PressureNullspaceMode::None) PetscFunctionReturn(PETSC_SUCCESS);
    auto& system = work.system;
    PetscCall(AgreeError(comm, VecDuplicate(system.solution, &work.mode), "Create joint pressure mode"));
    PetscCall(AgreeError(comm, VecSet(work.mode, 0), "Zero both nullspace velocity fields"));
    Vec mode[4]{};
    PetscCall(AgreeError(comm, GetFourFields(work.mode, mode), "Get joint pressure-mode fields"));
    if (options.mode==PressureNullspaceMode::Constant) {
        PetscCall(AgreeError(comm, VecSet(mode[1], 1), "Set constant Stokes pressure mode"));
        PetscCall(AgreeError(comm, VecSet(mode[3], 1), "Set constant Darcy pressure mode"));
    } else {
        PetscCall(AgreeError(comm, VecCopy(options.stokesPressureMode, mode[1]), "Copy Stokes pressure mode"));
        PetscCall(AgreeError(comm, VecCopy(options.darcyPressureMode, mode[3]), "Copy Darcy pressure mode"));
    }
    PetscReal norm = 0;
    PetscCall(AgreeError(comm, VecNorm(work.mode, NORM_2, &norm), "Joint pressure-mode norm"));
    PetscCheck(Finite(norm) && norm>0 && Finite(PetscReal(1)/norm), comm, PETSC_ERR_ARG_OUTOFRANGE,
               "Joint pressure mode must have a finite, nonzero, normalizable norm");
    PetscCall(AgreeError(comm, VecScale(work.mode, PetscReal(1)/norm), "Normalize complete joint mode"));
    PetscCall(AgreeError(comm, VecDuplicate(system.rhs, &work.action), "Create joint mode workspace"));
    PetscCall(CheckCoupledModeAction(comm, work, false, options));
    PetscCall(CheckCoupledModeAction(comm, work, true, options));
    PetscCall(AgreeError(comm, MatNullSpaceCreate(comm, PETSC_FALSE, 1, &work.mode,
                                                &system.pressureNullspace), "Create validated joint nullspace"));
    Vec rhs[4]{};
    PetscCall(AgreeError(comm, GetFourFields(system.rhs, rhs), "Get coupled RHS fields"));
    PetscScalar cs=0, cd=0;
    PetscReal ns=0, nd=0;
    PetscCall(AgreeError(comm, VecDot(mode[1], rhs[1], &cs), "Stokes RHS mode component"));
    PetscCall(AgreeError(comm, VecDot(mode[3], rhs[3], &cd), "Darcy RHS mode component"));
    PetscCall(AgreeError(comm, VecNorm(rhs[1], NORM_2, &ns), "Stokes pressure RHS norm"));
    PetscCall(AgreeError(comm, VecNorm(rhs[3], NORM_2, &nd), "Darcy pressure RHS norm"));
    const PetscScalar component = cs+cd; // Individual nonzero components can cancel.
    const PetscReal limit = options.absoluteTolerance + options.relativeTolerance*PetscHypotReal(ns, nd);
    PetscCheck(!PetscIsInfOrNanScalar(component) && Finite(ns) && Finite(nd) && Finite(limit),
               comm, PETSC_ERR_FP, "Nonfinite coupled RHS compatibility calculation");
    PetscCheck(options.projectRhs || PetscAbsScalar(component)<=limit, comm, PETSC_ERR_ARG_INCOMP,
               "Coupled RHS is incompatible with joint mode: component %g, allowed %g; correct data or explicitly enable projectRhs",
               static_cast<double>(PetscAbsScalar(component)), static_cast<double>(limit));
    if (options.projectRhs) {
        PetscCall(AgreeError(comm, MatNullSpaceRemove(system.pressureNullspace, system.rhs), "Project coupled RHS jointly"));
        system.removedRhsComponent = PetscAbsScalar(component);
    }
    PetscCall(AgreeError(comm, MatSetNullSpace(system.matrix, system.pressureNullspace), "Attach right joint pressure nullspace"));
    PetscCall(AgreeError(comm, MatSetTransposeNullSpace(system.matrix, system.pressureNullspace), "Attach transpose joint pressure nullspace"));
    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode ComposeCoupled(MPI_Comm comm, const MixedBlocks& stokes,
                              const MixedBlocks& darcy, Mat coupling,
                              const CoupledLinearSystemOptions& options, CoupledWork& work)
{
    PetscFunctionBeginUser;
    auto& system = work.system;
    system.kind = LinearSystemKind::Coupled;
    PetscCall(CopySignedBlocks(comm, stokes, options.stokes, work.stokes));
    PetscCall(CopySignedBlocks(comm, darcy, options.darcy, work.darcy));
    PetscCall(AgreeError(comm, MatDuplicate(coupling, MAT_COPY_VALUES, &work.coupling[0]), "Copy K_sd"));
    // Transpose the ORIGINAL raw K_sd before applying either independent scale.
    PetscCall(AgreeError(comm, MatTranspose(coupling, MAT_INITIAL_MATRIX, &work.coupling[1]), "Transpose K_sd"));
    const PetscReal scales[2] = {options.stokesDarcyCouplingScale, options.darcyStokesCouplingScale};
    for (int i=0; i<2; ++i) {
        if (scales[i]==0) PetscCall(AgreeError(comm, MatZeroEntries(work.coupling[i]), "Omit pressure coupling explicitly"));
        else PetscCall(AgreeError(comm, MatScale(work.coupling[i], scales[i]), "Set pressure coupling sign"));
    }
    Mat velocity[4] = {work.stokes[0], nullptr, nullptr, work.darcy[0]};
    Mat pressure[4] = {work.stokes[3], work.coupling[0], work.coupling[1], work.darcy[3]};
    PetscCall(AgreeError(comm, MatCreateNest(comm, 2, nullptr, 2, nullptr, velocity, &work.groups[0]), "Create coupled velocity block"));
    PetscCall(AgreeError(comm, MatCreateNest(comm, 2, nullptr, 2, nullptr, pressure, &work.groups[3]), "Create coupled pressure block"));
    // Reuse the diagonal groups' ISs so ALL inner matrices share U/P layouts.
    IS velocityRows[2]{}, velocityColumns[2]{}, pressureRows[2]{}, pressureColumns[2]{};
    PetscCall(AgreeError(comm, MatNestGetISs(work.groups[0], velocityRows, velocityColumns), "Get velocity group layout"));
    PetscCall(AgreeError(comm, MatNestGetISs(work.groups[3], pressureRows, pressureColumns), "Get pressure group layout"));
    Mat gradient[4] = {work.stokes[1], nullptr, nullptr, work.darcy[1]};
    Mat divergence[4] = {work.stokes[2], nullptr, nullptr, work.darcy[2]};
    PetscCall(AgreeError(comm, MatCreateNest(comm, 2, velocityRows, 2, pressureColumns, gradient, &work.groups[1]), "Create coupled gradient block"));
    PetscCall(AgreeError(comm, MatCreateNest(comm, 2, pressureRows, 2, velocityColumns, divergence, &work.groups[2]), "Create coupled divergence block"));
    for (auto group : work.groups)
        PetscCall(AgreeError(comm, MatNestSetVecType(group, VECNEST), "Choose inner nested vectors"));
    PetscCall(AgreeError(comm, MatCreateNest(comm, 2, nullptr, 2, nullptr, work.groups, &system.matrix), "Create outer saddle-point matrix"));
    PetscCall(AgreeError(comm, MatNestSetVecType(system.matrix, VECNEST), "Choose coupled nested vectors"));
    PetscCall(AgreeError(comm, MatCreateVecs(system.matrix, &system.solution, &system.rhs), "Create compatible coupled vectors"));
    PetscCall(AgreeError(comm, VecSet(system.solution, 0), "Initialize coupled solution"));
    Vec rhs[4]{};
    PetscCall(AgreeError(comm, GetFourFields(system.rhs, rhs), "Get coupled RHS fields"));
    const Vec source[4] = {stokes.f, stokes.g, darcy.f, darcy.g};
    for (int i=0; i<4; ++i)
        PetscCall(AgreeError(comm, VecCopy(source[i], rhs[i]), "Copy boundary-adjusted coupled RHS"));
    PetscCall(AttachCoupledPressureNullspace(comm, options.pressureNullspace, work));
    PetscCall(CreateCoupledFieldIS(comm, work));
    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode CheckBuilt(const LinearSystem& system)
{
    PetscFunctionBeginUser;
    PetscCheck(system.matrix && system.rhs && system.solution, PETSC_COMM_SELF,
               PETSC_ERR_ARG_WRONG, "Build a LinearSystem first");
    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode CheckKind(const LinearSystem& system, bool coupled)
{
    PetscFunctionBeginUser;
    PetscCall(CheckBuilt(system));
    const bool matches = coupled ? system.kind==LinearSystemKind::Coupled :
        (system.kind==LinearSystemKind::Stokes || system.kind==LinearSystemKind::Darcy);
    PetscCheck(matches, PETSC_COMM_SELF, PETSC_ERR_ARG_INCOMP,
               "This accessor requires a %s system", coupled ? "four-field coupled" : "two-field independent");
    PetscFunctionReturn(PETSC_SUCCESS);
}

} // namespace

PetscErrorCode DestroyLinearSystem(LinearSystem& system)
{
    PetscFunctionBeginUser;
    PetscErrorCode error = PETSC_SUCCESS;
    KeepFirst(error, VecDestroy(&system.rhs));
    KeepFirst(error, VecDestroy(&system.solution));
    KeepFirst(error, MatDestroy(&system.matrix));
    KeepFirst(error, MatNullSpaceDestroy(&system.pressureNullspace));
    for (auto& field : system.coupledFieldIS) KeepFirst(error, ISDestroy(&field));
    system.kind = LinearSystemKind::Stokes;
    system.removedRhsComponent = 0;
    PetscCall(error);
    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode BuildStokesLinearSystem(MPI_Comm comm, const MixedBlocks& blocks,
                                      const LinearSystemOptions& options, LinearSystem& result)
{
    PetscFunctionBeginUser;
    PetscCall(Build(comm, LinearSystemKind::Stokes, blocks, options, result));
    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode BuildDarcyLinearSystem(MPI_Comm comm, const MixedBlocks& blocks,
                                     const LinearSystemOptions& options, LinearSystem& result)
{
    PetscFunctionBeginUser;
    PetscCall(Build(comm, LinearSystemKind::Darcy, blocks, options, result));
    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode BuildCoupledLinearSystem(MPI_Comm comm, const MixedBlocks& stokes,
                                       const MixedBlocks& darcy, Mat K_sd,
                                       const CoupledLinearSystemOptions& options,
                                       LinearSystem& result)
{
    PetscFunctionBeginUser;
    PetscCall(AgreeError(comm, CheckCoupledInput(comm, stokes, darcy, K_sd, options, result), "Coupled-system inputs"));
    PetscCall(CheckCoupledOptions(comm, options));
    CoupledWork work;
    const PetscErrorCode error = ComposeCoupled(comm, stokes, darcy, K_sd, options, work);
    const PetscErrorCode cleanup = DestroyScratch(work);
    if (error || cleanup) {
        const PetscErrorCode destroy = DestroyLinearSystem(work.system);
        (void)destroy;
        PetscCall(error ? error : cleanup);
    }
    std::swap(result.matrix, work.system.matrix);
    std::swap(result.rhs, work.system.rhs);
    std::swap(result.solution, work.system.solution);
    std::swap(result.pressureNullspace, work.system.pressureNullspace);
    for (int i=0; i<4; ++i) std::swap(result.coupledFieldIS[i], work.system.coupledFieldIS[i]);
    result.kind = LinearSystemKind::Coupled;
    result.removedRhsComponent = work.system.removedRhsComponent;
    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode GetLinearSystemSolution(const LinearSystem& system, Vec& velocity, Vec& pressure)
{
    PetscFunctionBeginUser;
    PetscCall(CheckBuilt(system));
    Vec u=nullptr, p=nullptr;
    PetscCall(VecNestGetSubVec(system.solution, 0, &u));
    PetscCall(VecNestGetSubVec(system.solution, 1, &p));
    velocity = u; pressure = p;
    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode GetLinearSystemRhs(const LinearSystem& system, Vec& velocity, Vec& pressure)
{
    PetscFunctionBeginUser;
    PetscCall(CheckBuilt(system));
    Vec f=nullptr, g=nullptr;
    PetscCall(VecNestGetSubVec(system.rhs, 0, &f));
    PetscCall(VecNestGetSubVec(system.rhs, 1, &g));
    velocity = f; pressure = g;
    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode GetLinearSystemFieldIS(const LinearSystem& system, IS& velocity, IS& pressure)
{
    PetscFunctionBeginUser;
    PetscCall(CheckBuilt(system));
    IS fields[2]{};
    PetscCall(MatNestGetISs(system.matrix, nullptr, fields));
    velocity = fields[0]; pressure = fields[1];
    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode GetLinearSystemBlocks(const LinearSystem& system, Mat& velocity,
                                     Mat& gradient, Mat& divergence, Mat& pressure)
{
    PetscFunctionBeginUser;
    PetscCall(CheckBuilt(system));
    Mat A=nullptr, G=nullptr, D=nullptr, E=nullptr;
    PetscCall(MatNestGetSubMat(system.matrix, 0, 0, &A));
    PetscCall(MatNestGetSubMat(system.matrix, 0, 1, &G));
    PetscCall(MatNestGetSubMat(system.matrix, 1, 0, &D));
    PetscCall(MatNestGetSubMat(system.matrix, 1, 1, &E));
    velocity=A; gradient=G; divergence=D; pressure=E;
    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode GetCoupledLinearSystemSolution(const LinearSystem& system,
                                              Vec& stokesVelocity, Vec& stokesPressure,
                                              Vec& darcyVelocity, Vec& darcyPressure)
{
    PetscFunctionBeginUser;
    PetscCall(CheckKind(system, true));
    Vec fields[4]{};
    PetscCall(GetFourFields(system.solution, fields));
    stokesVelocity=fields[0]; stokesPressure=fields[1];
    darcyVelocity=fields[2]; darcyPressure=fields[3];
    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode GetCoupledLinearSystemRhs(const LinearSystem& system,
                                         Vec& stokesVelocity, Vec& stokesPressure,
                                         Vec& darcyVelocity, Vec& darcyPressure)
{
    PetscFunctionBeginUser;
    PetscCall(CheckKind(system, true));
    Vec fields[4]{};
    PetscCall(GetFourFields(system.rhs, fields));
    stokesVelocity=fields[0]; stokesPressure=fields[1];
    darcyVelocity=fields[2]; darcyPressure=fields[3];
    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode GetCoupledLinearSystemFieldIS(const LinearSystem& system,
                                             IS& stokesVelocity, IS& stokesPressure,
                                             IS& darcyVelocity, IS& darcyPressure)
{
    PetscFunctionBeginUser;
    PetscCall(CheckKind(system, true));
    stokesVelocity=system.coupledFieldIS[0]; stokesPressure=system.coupledFieldIS[1];
    darcyVelocity=system.coupledFieldIS[2]; darcyPressure=system.coupledFieldIS[3];
    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode RemoveLinearSystemPressureNullspace(LinearSystem& system)
{
    PetscFunctionBeginUser;
    // As with any PETSc collective, all ranks must pass a built object with
    // the same communicator. Recover it from matrix before agreeing options.
    PetscCall(CheckBuilt(system));
    const MPI_Comm comm = PetscObjectComm(reinterpret_cast<PetscObject>(system.matrix));
    const int present = system.pressureNullspace ? 1 : 0;
    int all = 0;
    PetscCallMPI(MPI_Allreduce(&present, &all, 1, MPI_INT, MPI_MIN, comm));
    PetscCheck(all, comm, PETSC_ERR_ARG_WRONG, "No validated pressure nullspace is attached");
    PetscCall(MatNullSpaceRemove(system.pressureNullspace, system.solution));
    PetscFunctionReturn(PETSC_SUCCESS);
}
