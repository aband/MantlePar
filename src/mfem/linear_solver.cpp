#include "linear_solver.h"

#include <algorithm>
#include <deque>
#include <limits>

namespace {

bool Finite(PetscReal x) { return !PetscIsInfOrNanReal(x); }

PetscErrorCode AgreeError(MPI_Comm comm, PetscErrorCode local, const char* stage)
{
    PetscFunctionBeginUser;
    const int own=static_cast<int>(local); int all=0;
    PetscCallMPI(MPI_Allreduce(&own, &all, 1, MPI_INT, MPI_MAX, comm));
    PetscCheck(!all, comm, static_cast<PetscErrorCode>(all), "%s failed on at least one rank", stage);
    PetscFunctionReturn(PETSC_SUCCESS);
}

void KeepFirst(PetscErrorCode& first, PetscErrorCode next) { if (!first) first=next; }

bool Name(const std::string& s, bool allowEmpty)
{
    if ((!allowEmpty && s.empty()) || s.size()>64) return false;
    for (const unsigned char c : s)
        if (!((c>='a' && c<='z') || (c>='A' && c<='Z') || (c>='0' && c<='9') || c=='_')) return false;
    return true;
}

PetscErrorCode CheckTolerances(PetscReal r, PetscReal a, PetscReal d, PetscInt n)
{
    PetscFunctionBeginUser;
    PetscCheck(Finite(r) && r>0 && r<1 && Finite(a) && a>=0 && Finite(d) && d>1 && n>0,
               PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE,
               "Require 0<rtol<1, atol>=0, dtol>1, max_it>0, with finite tolerances");
    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode CheckInput(const LinearSystem& s, const LinearSolverOptions& o)
{
    PetscFunctionBeginUser;
    PetscCheck(s.rhs && s.solution, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Build the complete LinearSystem first");
    PetscBool nest=PETSC_FALSE;
    PetscCall(PetscObjectTypeCompare(reinterpret_cast<PetscObject>(s.matrix), MATNEST, &nest));
    PetscCheck(nest, PETSC_COMM_SELF, PETSC_ERR_ARG_INCOMP, "Expected the linear_system MATNEST operator");
    PetscInt nr=0,nc=0;
    PetscCall(MatNestGetSize(s.matrix, &nr, &nc));
    PetscCheck(nr==2 && nc==2, PETSC_COMM_SELF, PETSC_ERR_ARG_SIZ,
               "Use the corrected outer 2x2 velocity-pressure linear_system module");
    PetscCheck(Name(o.optionsPrefix,true) && Name(o.kspType,false) &&
               Name(o.velocity.kspType,false) && Name(o.velocity.pcType,false) &&
               Name(o.pressure.kspType,false) && Name(o.pressure.pcType,false),
               PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Invalid or overlong solver prefix/type name");
    PetscCheck(o.preconditioner==LinearPreconditioner::None || o.preconditioner==LinearPreconditioner::Schur,
               PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Unknown linear preconditioner preset");
    PetscCall(CheckTolerances(o.relativeTolerance,o.absoluteTolerance,o.divergenceTolerance,o.maximumIterations));
    for (const auto* b : {&o.velocity,&o.pressure})
        PetscCall(CheckTolerances(b->relativeTolerance,b->absoluteTolerance,b->divergenceTolerance,b->maximumIterations));
    PetscCheck(!o.removePressureNullspace || s.pressureNullspace, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG,
               "Pressure gauge removal requires an already validated system pressure nullspace");
    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode AgreeString(MPI_Comm comm, const std::string& value)
{
    PetscFunctionBeginUser;
    // All accepted strings are at most 64 bytes; compare exactly, without hashes.
    PetscMPIInt rank=0; PetscCallMPI(MPI_Comm_rank(comm,&rank));
    char reference[65]{};
    if (rank==0) std::copy(value.begin(),value.end(),reference);
    PetscCallMPI(MPI_Bcast(reference,65,MPI_CHAR,0,comm));
    const int same=value==reference ? 1 : 0; int all=0;
    PetscCallMPI(MPI_Allreduce(&same,&all,1,MPI_INT,MPI_MIN,comm));
    PetscCheck(all,comm,PETSC_ERR_ARG_INCOMP,"Solver string options differ across ranks");
    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode AgreeOptions(MPI_Comm comm,const LinearSolverOptions& o)
{
    PetscFunctionBeginUser;
    for (const auto* value : {&o.optionsPrefix,&o.kspType,&o.velocity.kspType,
                              &o.velocity.pcType,&o.pressure.kspType,&o.pressure.pcType})
        PetscCall(AgreeString(comm,*value));
    const PetscReal own[9]={o.relativeTolerance,o.absoluteTolerance,o.divergenceTolerance,
        o.velocity.relativeTolerance,o.velocity.absoluteTolerance,o.velocity.divergenceTolerance,
        o.pressure.relativeTolerance,o.pressure.absoluteTolerance,o.pressure.divergenceTolerance};
    PetscReal lo[9]{},hi[9]{};
    PetscCallMPI(MPI_Allreduce(own,lo,9,MPIU_REAL,MPI_MIN,comm));
    PetscCallMPI(MPI_Allreduce(own,hi,9,MPIU_REAL,MPI_MAX,comm));
    const PetscInt flags[9]={static_cast<PetscInt>(o.preconditioner),o.maximumIterations,
        o.velocity.maximumIterations,o.pressure.maximumIterations,o.initialGuessNonzero?1:0,
        o.errorIfNotConverged?1:0,o.requireSubsolverConvergence?1:0,o.requireTrueResidual?1:0,
        o.removePressureNullspace?1:0};
    PetscInt flo[9]{},fhi[9]{};
    PetscCallMPI(MPI_Allreduce(flags,flo,9,MPIU_INT,MPI_MIN,comm));
    PetscCallMPI(MPI_Allreduce(flags,fhi,9,MPIU_INT,MPI_MAX,comm));
    for (int i=0;i<9;++i)
        PetscCheck(lo[i]==hi[i] && flo[i]==fhi[i],comm,PETSC_ERR_ARG_INCOMP,"Solver numeric options differ across ranks");
    PetscFunctionReturn(PETSC_SUCCESS);
}

struct Tracked {
    KSP ksp=nullptr; // Borrowed from the owning outer KSP.
    LinearSubsolverReport report;
};

struct Work {
    KSP ksp=nullptr;
    Vec residual=nullptr;
    MatNullSpace schurNullspace=nullptr;
    KSP *sub=nullptr, *inner=nullptr; // PETSc-allocated arrays, not KSP ownership.
    std::deque<Tracked> tracked;     // Stable callback addresses until KSPDestroy.
};

PetscErrorCode DestroyWork(Work& w)
{
    PetscFunctionBeginUser;
    PetscErrorCode error=PETSC_SUCCESS;
    KeepFirst(error,KSPDestroy(&w.ksp));
    KeepFirst(error,VecDestroy(&w.residual));
    KeepFirst(error,MatNullSpaceDestroy(&w.schurNullspace));
    KeepFirst(error,PetscFree(w.sub));
    KeepFirst(error,PetscFree(w.inner));
    PetscCall(error);
    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode ObserveSolve(KSP ksp, Vec, Vec, void* context)
{
    PetscFunctionBeginUser;
    auto& r=static_cast<Tracked*>(context)->report;
    PetscCall(KSPGetConvergedReason(ksp,&r.lastReason));
    PetscCall(KSPGetIterationNumber(ksp,&r.lastIterations));
    PetscCall(KSPGetResidualNorm(ksp,&r.lastResidualNorm));
    ++r.solves;
    r.totalIterations+=static_cast<PetscInt64>(r.lastIterations);
    if (r.lastReason<0) {
        ++r.failures;
        if (r.firstFailure==KSP_CONVERGED_ITERATING) r.firstFailure=r.lastReason;
    }
    // Do not return an error for divergence from this callback. Let PETSc finish
    // unwinding the nested solve, then apply the module's acceptance policy.
    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode PreserveOperator(KSP ksp)
{
    PetscFunctionBeginUser;
    PetscBool scaling=PETSC_FALSE;
    PetscCall(KSPGetDiagonalScale(ksp,&scaling));
    PetscCheck(!scaling,PetscObjectComm(reinterpret_cast<PetscObject>(ksp)),PETSC_ERR_SUP,
               "KSP diagonal scaling is unsupported here because it changes the assembled operator");
    PetscCall(KSPSetErrorIfNotConverged(ksp,PETSC_FALSE));
    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode ConfigureSubsolver(KSP ksp,const LinearBlockSolverOptions& options,Work& w)
{
    PetscFunctionBeginUser;
    for (const auto& t : w.tracked) if (t.ksp==ksp) PetscFunctionReturn(PETSC_SUCCESS);
    PetscCall(KSPSetType(ksp,options.kspType.c_str()));
    PetscCall(KSPSetTolerances(ksp,options.relativeTolerance,options.absoluteTolerance,
                               options.divergenceTolerance,options.maximumIterations));
    PetscCall(KSPSetInitialGuessNonzero(ksp,PETSC_FALSE));
    PC pc=nullptr; PetscCall(KSPGetPC(ksp,&pc));
    PetscCall(PCSetType(pc,options.pcType.c_str()));
    // PETSc already read child options while constructing the Schur PC. Reapply
    // them after our defaults; first remove those option-created monitors to
    // avoid registering the same monitor twice. No global options are changed.
    PetscCall(KSPMonitorCancel(ksp));
    PetscCall(KSPSetFromOptions(ksp));
    PetscCall(PreserveOperator(ksp));
    w.tracked.emplace_back();
    auto& t=w.tracked.back(); t.ksp=ksp;
    const char* prefix=nullptr; PetscCall(KSPGetOptionsPrefix(ksp,&prefix));
    t.report.optionsPrefix=prefix ? prefix : "";
    PetscCall(KSPSetPostSolve(ksp,ObserveSolve,&t));
    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode AttachSchurNullspace(const LinearSystem& system,KSP pressure,Work& w)
{
    PetscFunctionBeginUser;
    if (!system.pressureNullspace) PetscFunctionReturn(PETSC_SUCCESS);
    PetscBool constant=PETSC_FALSE; PetscInt count=0; const Vec* modes=nullptr;
    PetscCall(MatNullSpaceGetVecs(system.pressureNullspace,&constant,&count,&modes));
    PetscCheck(!constant && count==1,PetscObjectComm(reinterpret_cast<PetscObject>(system.matrix)),
               PETSC_ERR_ARG_INCOMP,"Expected the single explicit pressure-only mode from linear_system");
    Vec q=nullptr; PetscCall(VecNestGetSubVec(modes[0],1,&q));
    Mat schur=nullptr; PetscCall(KSPGetOperators(pressure,&schur,nullptr));
    const MPI_Comm comm=PetscObjectComm(reinterpret_cast<PetscObject>(system.matrix));
    // The validated full mode has zero velocity. Its normalized pressure part
    // is therefore also a two-sided mode of E-D*A^{-1}*G. For Coupled q is nest.
    PetscCall(MatNullSpaceCreate(comm,PETSC_FALSE,1,&q,&w.schurNullspace));
    PetscCall(MatSetNullSpace(schur,w.schurNullspace));
    PetscCall(MatSetTransposeNullSpace(schur,w.schurNullspace));
    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode Configure(const LinearSystem& system,const LinearSolverOptions& options,Work& w)
{
    PetscFunctionBeginUser;
    const MPI_Comm comm=PetscObjectComm(reinterpret_cast<PetscObject>(system.matrix));
    PetscCall(KSPCreate(comm,&w.ksp));
    PetscCall(KSPSetOptionsPrefix(w.ksp,options.optionsPrefix.c_str()));
    PetscCall(KSPSetOperators(w.ksp,system.matrix,system.matrix));
    PetscCall(KSPSetType(w.ksp,options.kspType.c_str()));
    PetscCall(KSPSetTolerances(w.ksp,options.relativeTolerance,options.absoluteTolerance,
                               options.divergenceTolerance,options.maximumIterations));
    PetscCall(KSPSetInitialGuessNonzero(w.ksp,options.initialGuessNonzero?PETSC_TRUE:PETSC_FALSE));
    PC pc=nullptr; PetscCall(KSPGetPC(w.ksp,&pc));
    IS velocity=nullptr,pressure=nullptr;
    PetscCall(GetLinearSystemFieldIS(system,velocity,pressure));
    if (options.preconditioner==LinearPreconditioner::Schur) {
        PetscCall(PCSetType(pc,PCFIELDSPLIT));
        PetscCall(PCFieldSplitSetType(pc,PC_COMPOSITE_SCHUR));
        PetscCall(PCFieldSplitSetSchurFactType(pc,PC_FIELDSPLIT_SCHUR_FACT_FULL));
        PetscCall(PCFieldSplitSetSchurPre(pc,PC_FIELDSPLIT_SCHUR_PRE_SELF,nullptr));
        PetscCall(PCFieldSplitSetIS(pc,"velocity",velocity));
        PetscCall(PCFieldSplitSetIS(pc,"pressure",pressure));
    } else PetscCall(PCSetType(pc,PCNONE));
    PetscCall(KSPSetFromOptions(w.ksp));
    PetscCall(PreserveOperator(w.ksp));
    PetscBool split=PETSC_FALSE;
    PetscCall(PetscObjectTypeCompare(reinterpret_cast<PetscObject>(pc),PCFIELDSPLIT,&split));
    if (split) {
        if (options.preconditioner==LinearPreconditioner::None) {
            PetscCall(PCFieldSplitSetIS(pc,"velocity",velocity));
            PetscCall(PCFieldSplitSetIS(pc,"pressure",pressure));
        }
        PCCompositeType kind; PCFieldSplitSchurPreType pre;
        PetscCall(PCFieldSplitGetType(pc,&kind));
        PetscCheck(kind==PC_COMPOSITE_SCHUR,comm,PETSC_ERR_SUP,"The managed fieldsplit must use -pc_fieldsplit_type schur");
        PetscCall(PCFieldSplitGetSchurPre(pc,&pre,nullptr));
        PetscCheck(pre==PC_FIELDSPLIT_SCHUR_PRE_SELF || pre==PC_FIELDSPLIT_SCHUR_PRE_A11,
                   comm,PETSC_ERR_SUP,"Use Schur preconditioner self or a11 with this nested solver");
        IS actualU=nullptr,actualP=nullptr; PetscBool sameU=PETSC_FALSE,sameP=PETSC_FALSE;
        PetscCall(PCFieldSplitGetIS(pc,"velocity",&actualU));
        PetscCall(PCFieldSplitGetIS(pc,"pressure",&actualP));
        PetscCheck(actualU && actualP,comm,PETSC_ERR_ARG_INCOMP,"The velocity/pressure splits are reserved by linear_solver");
        PetscCall(ISEqual(actualU,velocity,&sameU)); PetscCall(ISEqual(actualP,pressure,&sameP));
        PetscCheck(sameU && sameP,comm,PETSC_ERR_ARG_INCOMP,"Runtime fields must retain the system's outer velocity/pressure split");
    }
    // SELF/A11 construct the managed children before setting up their PCs.
    // This lets us replace default ILU before it is applied to a MATNEST.
    PetscCall(KSPSetUp(w.ksp));
    if (split) {
        PetscInt count=0,innerCount=0;
        PetscCall(PCFieldSplitSchurGetSubKSP(pc,&count,&w.sub));
        PetscCheck(count==2 || count==3,comm,PETSC_ERR_PLIB,"Expected velocity, pressure and optional upper velocity KSPs");
        PetscCall(ConfigureSubsolver(w.sub[0],options.velocity,w));
        PetscCall(ConfigureSubsolver(w.sub[1],options.pressure,w));
        if (count==3) PetscCall(ConfigureSubsolver(w.sub[2],options.velocity,w));
        // A separately configured Schur-inner velocity solve is not necessarily
        // the outer velocity KSP returned above. Configure and observe both.
        PetscCall(PCFieldSplitGetSubKSP(pc,&innerCount,&w.inner));
        PetscCheck(innerCount==2,comm,PETSC_ERR_PLIB,"Expected two Schur subsolvers");
        PetscCall(ConfigureSubsolver(w.inner[0],options.velocity,w));
        PetscCall(AttachSchurNullspace(system,w.sub[1],w));
        PetscCall(PetscFree(w.sub)); PetscCall(PetscFree(w.inner));
    }
    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode Snapshot(Work& w,LinearSolveReport& report)
{
    PetscFunctionBeginUser;
    report.subsolvers.clear(); report.subsolversConverged=true;
    for (const auto& t : w.tracked) {
        report.subsolvers.push_back(t.report);
        if (t.report.failures) report.subsolversConverged=false;
    }
    if (w.ksp) {
        PetscCall(KSPGetConvergedReason(w.ksp,&report.reason));
        PetscCall(KSPGetIterationNumber(w.ksp,&report.iterations));
        PetscCall(KSPGetResidualNorm(w.ksp,&report.kspResidualNorm));
        KSPType kt=nullptr; PCType pt=nullptr; PC pc=nullptr;
        PetscCall(KSPGetType(w.ksp,&kt)); PetscCall(KSPGetPC(w.ksp,&pc)); PetscCall(PCGetType(pc,&pt));
        report.kspType=kt ? kt : ""; report.pcType=pt ? pt : "";
    }
    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode Run(LinearSystem& system,const LinearSolverOptions& options,Work& w,LinearSolveReport& report)
{
    PetscFunctionBeginUser;
    const MPI_Comm comm=PetscObjectComm(reinterpret_cast<PetscObject>(system.matrix));
    PetscCall(VecNorm(system.rhs,NORM_2,&report.rhsNorm));
    PetscCheck(Finite(report.rhsNorm),comm,PETSC_ERR_ARG_OUTOFRANGE,"The RHS must have a finite norm");
    PetscCall(Configure(system,options,w));
    PetscBool nonzero=PETSC_FALSE; PetscCall(KSPGetInitialGuessNonzero(w.ksp,&nonzero));
    if (nonzero) {
        PetscReal norm=0; PetscCall(VecNorm(system.solution,NORM_2,&norm));
        PetscCheck(Finite(norm),comm,PETSC_ERR_ARG_OUTOFRANGE,"The initial guess must have a finite norm");
    } else PetscCall(VecSet(system.solution,0));
    PetscCall(VecDuplicate(system.rhs,&w.residual));
    PetscCall(KSPSolve(w.ksp,system.rhs,system.solution));
    report.solveCompleted=true;
    PetscCall(Snapshot(w,report));
    if (options.removePressureNullspace && report.reason>0 &&
        (!options.requireSubsolverConvergence || report.subsolversConverged)) {
        PetscCall(RemoveLinearSystemPressureNullspace(system));
        report.pressureGaugeRemoved=true;
    }
    PetscCall(MatMult(system.matrix,system.solution,w.residual));
    PetscCall(VecAYPX(w.residual,-1,system.rhs)); // b-A*x in the original grouped layout.
    PetscCall(VecNorm(w.residual,NORM_2,&report.trueResidualNorm));
    PetscReal rtol=0,atol=0;
    PetscCall(KSPGetTolerances(w.ksp,&rtol,&atol,nullptr,nullptr));
    report.trueResidualThreshold=std::max(atol,rtol*report.rhsNorm);
    report.relativeTrueResidualNorm=report.rhsNorm>0 ? report.trueResidualNorm/report.rhsNorm : report.trueResidualNorm;
    report.trueResidualSatisfied=Finite(report.trueResidualNorm) && Finite(report.trueResidualThreshold) &&
        report.trueResidualNorm<=report.trueResidualThreshold;
    report.converged=report.reason>0 && Finite(report.trueResidualNorm) &&
        (!options.requireTrueResidual || report.trueResidualSatisfied) &&
        (!options.requireSubsolverConvergence || report.subsolversConverged);
    PetscFunctionReturn(PETSC_SUCCESS);
}

} // namespace

PetscErrorCode SolveLinearSystem(LinearSystem& system,const LinearSolverOptions& options,LinearSolveReport& report)
{
    PetscFunctionBeginUser;
    report=LinearSolveReport{};
    // As with a PETSc collective, a built object on every participating rank
    // is a precondition for recovering its communicator.
    PetscCheck(system.matrix,PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Build a LinearSystem first");
    const MPI_Comm comm=PetscObjectComm(reinterpret_cast<PetscObject>(system.matrix));
    PetscErrorCode error=AgreeError(comm,CheckInput(system,options),"Linear-solver inputs");
    if (!error) error=AgreeOptions(comm,options);
    Work work;
    if (!error) error=AgreeError(comm,Run(system,options,work,report),"Linear solve");
    KeepFirst(error,Snapshot(work,report));
    KeepFirst(error,DestroyWork(work));
    report.petscError=error;
    if (error) { report.converged=false; PetscCall(error); }
    PetscCheck(report.converged || !options.errorIfNotConverged,comm,PETSC_ERR_NOT_CONVERGED,
               "Linear solve not accepted: outer reason %s, true residual %g (limit %g), managed subsolvers %s; inspect LinearSolveReport",
               KSPConvergedReasons[report.reason],static_cast<double>(report.trueResidualNorm),
               static_cast<double>(report.trueResidualThreshold),report.subsolversConverged?"converged":"had failures");
    PetscFunctionReturn(PETSC_SUCCESS);
}
