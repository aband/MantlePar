#include "initialization.h"
#include "visualization.h"

#include <cmath>
#include <stdexcept>
#include <string>

using namespace mantle::couple;
namespace {
const MPI_Comm& comm=PETSC_COMM_WORLD;
PetscErrorCode Check(bool ok,const char* message)
{
    PetscFunctionBeginUser;
    int bad=ok?0:1,any=0;
    PetscCallMPI(MPI_Allreduce(&bad,&any,1,MPI_INT,MPI_MAX,comm));
    PetscCheck(!any,comm,PETSC_ERR_PLIB,"%s",message);
    PetscFunctionReturn(PETSC_SUCCESS);
}
mantle::input::Value Scalar(const std::string& value)
{
    mantle::input::Value out; out.kind=mantle::input::Value::Kind::Scalar; out.scalar=value; return out;
}
ScalarProfile Constant(PetscReal value)
{ return [value](const Point&,PetscReal) { return value; }; }

PetscErrorCode Residual(const InitialState& s)
{
    PetscFunctionBeginUser;
    Vec r=nullptr; PetscReal norm=0;
    PetscCall(VecDuplicate(s.flowSystem.rhs,&r));
    const auto operation=[&]() -> PetscErrorCode {
        PetscFunctionBeginUser;
        PetscCall(MatMult(s.flowSystem.matrix,s.flowSystem.solution,r));
        PetscCall(VecAXPY(r,-1,s.flowSystem.rhs)); PetscCall(VecNorm(r,NORM_2,&norm));
        PetscFunctionReturn(PETSC_SUCCESS);
    };
    const auto error=operation(); PetscCall(VecDestroy(&r)); PetscCall(error);
    PetscCall(Check(s.flowReport.converged && s.flowReport.subsolversConverged &&
        norm<=s.flowReport.trueResidualThreshold && std::abs(norm-s.flowReport.trueResidualNorm)<1e-12,
        "Initial flow failed independently recomputed residual/acceptance checks"));
    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode Manufactured(Configuration c,InitialState& s)
{
    PetscFunctionBeginUser;
    c.initialEnthalpy=Constant(3.3); c.initialComposition=Constant(.04); c.pressure=Constant(0);
    const auto phi=c.phase.evaluate(3.3,.04,0).phil;
    const PetscReal ps=.7,pd=.3;
    c.flow.stokesForce=c.flow.darcyForce=[](const Point&) { return Point{{0,0}}; };
    c.flow.darcyForce=[](const Point&) { return Point{{.005,-.002}}; };
    c.flow.stokesPressureSource=Constant((-phi*ps+std::sqrt(phi)*pd)/(1-phi));
    c.flow.darcyPressureSource=Constant((std::sqrt(phi)*ps-pd)/(1-phi));
    for (auto& rule:c.flow.stokes.rules) for (int k=0;k<2;++k)
        if (rule.component[k].type==BoundaryType::Neumann)
            rule.component[k].value.function=[ps,k](const BoundaryPoint& p) { return -ps*p.outwardNormal.p[k]; };
    for (auto& rule:c.flow.darcy.rules) {
        if (rule.condition.type==BoundaryType::Neumann) rule.condition.value=BoundaryValue{pd/std::sqrt(phi),{}};
        else rule.condition.value.function=[](const BoundaryPoint& p) { return .005*p.outwardNormal.p[0]-.002*p.outwardNormal.p[1]; };
    }
    PetscCall(Initialize(comm,c,s)); PetscCall(Residual(s));
    Vec us=nullptr,psv=nullptr,ud=nullptr,pdv=nullptr;
    PetscCall(GetCoupledLinearSystemSolution(s.flowSystem,us,psv,ud,pdv));
    bool good=true;
    const Vec fields[4]={us,psv,ud,pdv};
    const DofMap* maps[4]={&s.stokesVelocityMap,&s.pressureMap,&s.darcyVelocityMap,&s.pressureMap};
    for (int k=0;k<4;++k) {
        const PetscScalar* data=nullptr; PetscCall(VecGetArrayRead(fields[k],&data));
        for (PetscInt n=0;n<maps[k]->OwnedDofs();++n) {
            DofInfo dof;
            const auto error=maps[k]->GetDofInfo(maps[k]->OwnershipBegin()+n,dof);
            if (error) { good=false; continue; }
            PetscReal exact=k==1?ps:k==3?pd:0;
            if (k==0 && dof.entity==DofEntity::Vertex && dof.component==1) exact=.01;
            if (k!=2) good=good && PetscAbsScalar(data[n]-exact)<2e-8;
        }
        PetscCall(VecRestoreArrayRead(fields[k],&data));
    }
    PetscCall(Check(good,"Manufactured constant flow/pressure solution is incorrect (sources, boundary lifting, signs or scaling)"));
    InitialFlowSamples samples; PetscCall(SampleInitialFlow(comm,c,s,samples));
    for (const auto* points:{&samples.cells,&samples.edges}) for (const auto& p:*points)
        good=good && std::abs(p.stokesVelocity.p[0])<2e-8 && std::abs(p.stokesVelocity.p[1]-.01)<2e-8
             && std::abs(p.darcyVelocity.p[0]-.005)<2e-8 && std::abs(p.darcyVelocity.p[1]+.002)<2e-8
             && std::abs(p.stokesPressure-ps)<2e-8 && std::abs(p.darcyPressure-pd)<2e-8;
    PetscCall(Check(good,"Gauss-point BR/H(div) reconstruction is incorrect"));
    PetscCall(DestroyInitialState(s));
    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode Harmonic(Configuration c,InitialState& s,bool dry)
{
    PetscFunctionBeginUser;
    auto& initial=c.input.transport.settings.mapping["initial_conditions"].mapping["enthalpy"];
    initial.mapping["name"]=Scalar("piecewise_constant");
    auto& p=initial.mapping["parameters"]; p.mapping.clear();
    p.mapping["interface_y"]=Scalar("-0.1"); p.mapping["value_below"]=Scalar(dry?"2.0":"3.1");
    p.mapping["value_above"]=Scalar("3.6");
    c=MakeInitializationConfiguration(c.input); c.pressure=Constant(0);
    c.flow.stokesForce=c.flow.darcyForce=[](const Point&) { return Point{{0,0}}; };
    // Closed, motionless domain: nonconstant phi still has one joint pressure gauge.
    c.flow.stokes.rules.clear(); c.flow.darcy.rules.clear();
    for (int side=0;side<4;++side) {
        StokesBoundaryRule a; a.region.side=static_cast<CellSide>(side);
        for (auto& component:a.component) component.type=BoundaryType::Dirichlet;
        c.flow.stokes.rules.push_back(a);
        DarcyBoundaryRule b; b.region.side=static_cast<CellSide>(side); b.condition.type=BoundaryType::Dirichlet;
        c.flow.darcy.rules.push_back(b);
    }
    c.input.flow.pressureNullspace="provided";
    c.input.flow.pressureModes.kind=mantle::input::Value::Kind::Mapping;
    c.input.flow.pressureModes.mapping["stokes"]=Scalar("1");
    c.input.flow.pressureModes.mapping["darcy"]=Scalar("sqrt_cell_average_porosity");
    c.flow.solver.removePressureNullspace=true;
    PetscCall(Initialize(comm,c,s)); PetscCall(Residual(s));
    const auto lo=c.phase.evaluate(dry?2.0:3.1,.04,0).phil, hi=c.phase.evaluate(3.6,.04,0).phil;
    const auto harmonic=(lo+hi)>0?2*lo*hi/(lo+hi):0;
    const auto range=s.mesh.OwnedCells(); bool good=true; PetscInt count=0,total=0;
    for (PetscInt j=range.begin.j;j<range.end.j;++j) for (PetscInt i=range.begin.i;i<range.end.i;++i) {
        const auto& sample=s.flowPorosity[(j-range.begin.j)*range.Size().i+i-range.begin.i];
        good=good && std::abs(sample.average-(j<3?lo:hi))<1e-12;
        if (j==2 || j==3) {
            for (auto value:sample.edge[j==2?2:0]) good=good && std::abs(value-harmonic)<1e-12;
            ++count;
        }
        if (j==0) for (auto value:sample.edge[0]) good=good && std::abs(value-lo)<1e-12;
        if (j==3) for (auto value:sample.edge[2]) good=good && std::abs(value-hi)<1e-12;
    }
    PetscCallMPI(MPI_Allreduce(&count,&total,1,MPIU_INT,MPI_SUM,comm));
    PetscCall(Check(good && total==6,"Harmonic/interior-boundary porosity traces or physical averages are incorrect"));
    PetscCall(Check(s.flowSystem.pressureNullspace && s.flowReport.pressureGaugeRemoved,"Joint pressure gauge was not applied"));
    PetscCall(DestroyInitialState(s)); PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode Failures(Configuration c,InitialState& s)
{
    PetscFunctionBeginUser;
    // Initializer must reject nonconvergence even if the generic solver merely reports it.
    c.flow.solver.maximumIterations=1; c.flow.solver.velocity.maximumIterations=1;
    c.flow.solver.pressure.maximumIterations=1; c.flow.solver.errorIfNotConverged=false;
    PetscCall(PetscPushErrorHandler(PetscReturnErrorHandler,nullptr));
    auto error=Initialize(comm,c,s);
    PetscCall(PetscPopErrorHandler());
    PetscCall(Check(error==PETSC_ERR_NOT_CONVERGED && s.IsEmpty(),"Failed flow solve published a partial initial state"));
    // Fully molten phase is valid thermodynamics but outside this two-phase operator.
    c.initialEnthalpy=Constant(6); c.pressure=Constant(0);
    PetscCall(PetscPushErrorHandler(PetscReturnErrorHandler,nullptr));
    error=Initialize(comm,c,s); PetscCall(PetscPopErrorHandler());
    PetscCall(Check(error!=PETSC_SUCCESS && s.IsEmpty(),"Fully molten flow coefficients were silently accepted"));
    // A local source exception must reach every rank before the next collective.
    c.initialEnthalpy=Constant(3.3);
    PetscMPIInt rank=0; PetscCallMPI(MPI_Comm_rank(comm,&rank));
    c.flow.stokesPressureSource=[rank](const Point&,PetscReal) -> PetscReal {
        if (!rank) throw std::runtime_error("deliberate pressure source failure");
        return 0;
    };
    PetscCall(PetscPushErrorHandler(PetscReturnErrorHandler,nullptr));
    error=Initialize(comm,c,s); PetscCall(PetscPopErrorHandler());
    PetscCall(Check(error==PETSC_ERR_USER && s.IsEmpty(),"A rank-local source error was not propagated/cleaned up"));
    PetscCall(DestroyInitialState(s)); PetscCall(DestroyInitialState(s));
    PetscFunctionReturn(PETSC_SUCCESS);
}
PetscErrorCode Run(InitialState& s)
{
    PetscFunctionBeginUser;
    char filename[PETSC_MAX_PATH_LEN];
    PetscCall(PetscOptionsGetString(nullptr,nullptr,"-input",filename,sizeof(filename),nullptr));
    mantle::input::InputConfig input;
    auto options=InitializationReaderOptions(); options.requireMatchingMpiSize=false;
    PetscCall(mantle::input::ReadInput(comm,filename,input,options));
    PetscMPIInt size=0; PetscCallMPI(MPI_Comm_size(comm,&size));
    input.parallel.ranks=size; input.parallel.processGrid={{1,size}};
    input.mesh.cells={{3,4}}; input.quadrature.cellPointsPerAxis=3; input.quadrature.edgePoints=3;
    auto c=MakeInitializationConfiguration(input);
    PetscCall(Manufactured(c,s));
    auto warped=c; warped.input.mesh.family="perturbed_quadrilateral";
    PetscCall(Manufactured(warped,s));
    PetscCall(Harmonic(c,s,false)); PetscCall(Harmonic(c,s,true));
    PetscCall(Failures(c,s));
    PetscCall(PetscPrintf(comm,"Passed initial flow: manufactured fields, pressure sources, true residual, harmonic wet/dry traces, gauge and failure cleanup.\n"));
    PetscFunctionReturn(PETSC_SUCCESS);
}
}
int main(int argc,char** argv)
{
    auto error=PetscInitialize(&argc,&argv,nullptr,nullptr); if (error) return error;
    InitialState state; error=Run(state);
    const auto cleanup=DestroyInitialState(state), finalize=PetscFinalize();
    return error?error:cleanup?cleanup:finalize;
}
