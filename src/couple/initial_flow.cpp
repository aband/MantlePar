#include "initialization.h"
#include "state_reconstruction.h"

#include <cmath>
#include <exception>
#include <map>

namespace mantle::couple {
namespace {
// Catch profile/phase exceptions before returning to collective assembly.
template<class F> PetscErrorCode Local(F&& f)
{
    PetscFunctionBeginUser;
    try { const auto error=f(); PetscFunctionReturn(error); }
    catch (const std::bad_alloc&) { SETERRQ(PETSC_COMM_SELF,PETSC_ERR_MEM,"Initial flow allocation failed"); }
    catch (const std::exception& e) { SETERRQ(PETSC_COMM_SELF,PETSC_ERR_USER,"%s",e.what()); }
    catch (...) { SETERRQ(PETSC_COMM_SELF,PETSC_ERR_USER,"Initial flow callback failed"); }
}
PetscErrorCode Agree(MPI_Comm comm, PetscErrorCode error)
{
    PetscFunctionBeginUser;
    const int local=error; int global=0;
    PetscCallMPI(MPI_Allreduce(&local,&global,1,MPI_INT,MPI_MAX,comm));
    PetscCheck(!global,comm,static_cast<PetscErrorCode>(global),"Initial flow setup failed on at least one rank");
    PetscFunctionReturn(PETSC_SUCCESS);
}

// At a mesh-aligned discontinuity use exact one-sided profile limits. Merely
// evaluating the profile at the edge selects its upper branch on BOTH cells.
PetscReal Trace(const input::Value& specification, const ScalarProfile& profile,
                const Point& p, const EdgeVertices& edge, PetscReal time,
                PetscReal tolerance, bool below)
{
    const auto name=specification.At("name").AsString();
    if (name=="piecewise_constant" || name=="quadratic_below_interface") {
        const auto& parameters=specification.At("parameters");
        const auto y=parameters.At("interface_y").AsReal();
        if (std::abs(edge[0].p[1]-y)<=tolerance && std::abs(edge[1].p[1]-y)<=tolerance) {
            if (name=="piecewise_constant") return parameters.At(below?"value_below":"value_above").AsReal();
            return below?0:parameters.At("value_above_interface").AsReal();
        }
    }
    return profile(p,time);
}

PetscErrorCode SamplePorosity(const Configuration& c, InitialState& s,
                             const GaussRule1D& cellRule, const GaussRule1D& edgeRule,
                             const StateReconstruction* reconstruction)
{
    PetscFunctionBeginUser;
    PetscCall(Local([&]() -> PetscErrorCode {
        const auto range=s.mesh.OwnedCells();
        const auto& initial=c.input.transport.settings.At("initial_conditions");
        const auto nc=cellRule.points.size(), ne=edgeRule.points.size();
        std::map<std::pair<PetscInt,PetscInt>,bool> wetCells;
        const auto isWet=[&](MeshIndex index,bool& wet) -> PetscErrorCode {
            const auto key=std::make_pair(index.i,index.j);
            const auto found=wetCells.find(key);
            if (found!=wetCells.end()) { wet=found->second; return PETSC_SUCCESS; }
            QuadVertices v; PetscCall(reconstruction->Mesh().GetCellCorners(index,v));
            wet=false;
            for (auto x:cellRule.points) for (auto y:cellRule.points) {
                const auto p=MapCellPoint(Point{{x,y}},v); PetscReal h,composition;
                PetscCall(reconstruction->Evaluate(index,p,h,composition));
                wet=wet || c.phase.evaluate(h,composition,c.pressure(p,s.time)).phil>0;
            }
            wetCells[key]=wet; return PETSC_SUCCESS;
        };
        for (PetscInt j=range.begin.j;j<range.end.j;++j) for (PetscInt i=range.begin.i;i<range.end.i;++i) {
            QuadVertices v; auto error=s.mesh.GetCellCorners({i,j},v); if (error) return error;
            LocalPorositySamples samples; samples.cell.resize(nc*nc);
            PetscReal sum=0,area=0;
            for (std::size_t b=0;b<nc;++b) for (std::size_t a=0;a<nc;++a) {
                const Point reference{{cellRule.points[a],cellRule.points[b]}};
                const auto p=MapCellPoint(reference,v);
                PetscReal h=c.initialEnthalpy(p,s.time), composition=c.initialComposition(p,s.time);
                if (reconstruction) PetscCall(reconstruction->Evaluate({i,j},p,h,composition));
                const auto phi=c.flow.dryPorosity?0:c.phase.evaluate(h,composition,c.pressure(p,s.time)).phil;
                samples.cell[b*nc+a]=phi;
                const auto w=cellRule.weights[a]*cellRule.weights[b]*CellJacobian(reference,v);
                sum+=w*phi; area+=w;
            }
            samples.average=sum/area;
            std::array<OrientedEdge,4> edges;
            error=s.mesh.GetCellEdges({i,j},edges); if (error) return error;
            const auto center=MapCellPoint(Point{{0,0}},v);
            for (int e=0;e<4;++e) {
                EdgeTopology topology;
                error=s.mesh.GetEdgeTopology(edges[e].id,topology); if (error) return error;
                const EdgeVertices edge{{v[e],v[(e+1)%4]}};
                samples.edge[e].resize(ne);
                for (std::size_t q=0;q<ne;++q) {
                    const auto p=MapEdgePoint(edgeRule.points[q],edge);
                    if (reconstruction) {
                        PetscReal h,composition;
                        PetscCall(reconstruction->Evaluate({i,j},p,h,composition));
                        const auto a=c.phase.evaluate(h,composition,c.pressure(p,s.time)).phil;
                        auto b=a;
                        if (!topology.IsBoundary()) {
                            auto neighbor=*topology.leftCell;
                            if (neighbor.i==i && neighbor.j==j) neighbor=*topology.rightCell;
                            PetscCall(reconstruction->Evaluate(neighbor,p,h,composition));
                            b=c.phase.evaluate(h,composition,c.pressure(p,s.time)).phil;
                        }
                        samples.edge[e][q]=(a+b)>0?2*a*b/(a+b):0;
                        // (3.123): a quadrature-dry cell has zero scaled
                        // divergence. Apply this on BOTH sides of a shared
                        // face, preserving normal-flux continuity and (3.131).
                        bool wet; PetscCall(isWet({i,j},wet));
                        if (!wet) samples.edge[e][q]=0;
                        if (!topology.IsBoundary()) {
                            auto neighbor=*topology.leftCell;
                            if (neighbor.i==i && neighbor.j==j) neighbor=*topology.rightCell;
                            PetscCall(isWet(neighbor,wet));
                            if (!wet) samples.edge[e][q]=0;
                        }
                        continue;
                    }
                    const auto phaseTrace=[&](bool below) {
                        if (c.flow.dryPorosity) return 0.0;
                        const auto h=Trace(initial.At("enthalpy"),c.initialEnthalpy,p,edge,s.time,c.input.boundaryRegions.coordinateTolerance,below);
                        const auto composition=Trace(initial.At("composition"),c.initialComposition,p,edge,s.time,c.input.boundaryRegions.coordinateTolerance,below);
                        return c.phase.evaluate(h,composition,c.pressure(p,s.time)).phil;
                    };
                    if (topology.IsBoundary()) samples.edge[e][q]=phaseTrace(center.p[1]<p.p[1]);
                    else {
                        const auto a=phaseTrace(true), b=phaseTrace(false);
                        samples.edge[e][q]=(a+b)>0?2*a*b/(a+b):0;
                    }
                }
            }
            s.flowPorosity.push_back(std::move(samples));
        }
        return PETSC_SUCCESS;
    }));
    PetscFunctionReturn(PETSC_SUCCESS);
}

struct Workspace {
    MixedBlocks rawS,rawD,stokes,darcy;
    BoundaryData bcS,bcD;
    Mat coupling=nullptr;
    Vec modeS=nullptr,modeD=nullptr;
    PetscErrorCode Clear() {
        PetscFunctionBeginUser;
        PetscErrorCode first=PETSC_SUCCESS;
        const auto record=[&](PetscErrorCode e) { if (!first) first=e; };
        record(DestroyMixedBlocks(rawS)); record(DestroyMixedBlocks(rawD));
        record(DestroyMixedBlocks(stokes)); record(DestroyMixedBlocks(darcy));
        record(DestroyBoundaryData(bcS)); record(DestroyBoundaryData(bcD));
        record(MatDestroy(&coupling)); record(VecDestroy(&modeS)); record(VecDestroy(&modeD));
        PetscFunctionReturn(first);
    }
};

PetscErrorCode PressureLoad(MPI_Comm comm, const InitialState& s, const GaussRule1D& rule,
                            const ScalarProfile& source, Vec rhs)
{
    PetscFunctionBeginUser;
    std::vector<PetscInt> ids; std::vector<PetscScalar> values;
    PetscCall(Agree(comm,Local([&]() -> PetscErrorCode {
        const auto range=s.mesh.OwnedCells();
        for (PetscInt j=range.begin.j;j<range.end.j;++j) for (PetscInt i=range.begin.i;i<range.end.i;++i) {
            QuadVertices corners; auto error=s.mesh.GetCellCorners({i,j},corners); if (error) return error;
            PetscReal value=0;
            // Do not throw profile exceptions through PETSc integration frames.
            for (std::size_t b=0;b<rule.points.size();++b) for (std::size_t a=0;a<rule.points.size();++a) {
                const Point ref{{rule.points[a],rule.points[b]}};
                value+=rule.weights[a]*rule.weights[b]*CellJacobian(ref,corners)*source(MapCellPoint(ref,corners),s.time);
            }
            if (PetscIsInfOrNanReal(value)) return PETSC_ERR_FP;
            std::vector<PetscInt> dofs; error=s.pressureMap.GetCellGlobalDofs({i,j},dofs); if (error) return error;
            ids.push_back(dofs[0]); values.push_back(value);
        }
        return PETSC_SUCCESS;
    })));
    PetscCall(VecSetValues(rhs,static_cast<PetscInt>(ids.size()),ids.data(),values.data(),INSERT_VALUES));
    PetscCall(VecAssemblyBegin(rhs)); PetscCall(VecAssemblyEnd(rhs));
    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode Build(MPI_Comm comm, const Configuration& c, InitialState& s, Workspace& w,
                    const StateReconstruction* reconstruction = nullptr)
{
    PetscFunctionBeginUser;
    PetscCall(s.stokesVelocityMap.Initialize(s.vertexDM,s.mesh,DofSpace::BRVelocity));
    PetscCall(s.darcyVelocityMap.Initialize(s.vertexDM,s.mesh,DofSpace::HDivVelocity));
    PetscCall(s.pressureMap.Initialize(s.vertexDM,s.mesh,DofSpace::CellPressure));
    GaussRule1D cell,edge;
    PetscCall(Agree(comm,CreateGaussRule(c.input.quadrature.cellPointsPerAxis,cell)));
    PetscCall(Agree(comm,CreateGaussRule(c.input.quadrature.edgePoints,edge)));
    PetscCall(Agree(comm,SamplePorosity(c,s,cell,edge,reconstruction)));
    const CellPorosityFunction porosity=[&](MeshIndex index,LocalPorositySamples& samples) -> PetscErrorCode {
        const auto range=s.mesh.OwnedCells();
        if (!range.Contains(index)) return PETSC_ERR_ARG_OUTOFRANGE;
        samples=s.flowPorosity[(index.j-range.begin.j)*range.Size().i+index.i-range.begin.i];
        return PETSC_SUCCESS;
    };
    PetscCall(AssembleStokesBlocks(comm,s.mesh,s.stokesVelocityMap,s.pressureMap,cell,porosity,c.flow.stokesForce,w.rawS));
    PetscCall(PressureLoad(comm,s,cell,c.flow.stokesPressureSource,w.rawS.g));
    PetscCall(AssembleDarcyBlocks(comm,s.mesh,s.darcyVelocityMap,s.pressureMap,cell,edge,porosity,c.flow.material,c.flow.darcyForce,w.rawD));
    PetscCall(PressureLoad(comm,s,cell,c.flow.darcyPressureSource,w.rawD.g));
    PetscCall(BuildStokesBoundaryData(comm,s.mesh,s.stokesVelocityMap,edge,c.flow.stokes,s.time,w.bcS));
    PetscCall(BuildDarcyBoundaryData(comm,s.mesh,s.darcyVelocityMap,edge,c.flow.darcy,s.time,porosity,c.flow.material,w.bcD));
    BoundaryApplicationOptions application; application.pressureRowSign=-1;
    PetscCall(ApplyBoundaryConditions(comm,w.rawS,w.bcS,application,w.stokes));
    PetscCall(ApplyBoundaryConditions(comm,w.rawD,w.bcD,application,w.darcy));
    PetscCall(AssemblePressureCoupling(comm,s.mesh,s.pressureMap,s.pressureMap,cell,porosity,c.flow.material,w.coupling));
    CoupledLinearSystemOptions options;
    options.stokes.pressureRowSign=-1; options.darcy.pressureRowSign=-1;
    auto& gauge=options.pressureNullspace;
    gauge.mode=c.input.flow.pressureNullspace=="none"?PressureNullspaceMode::None:
        c.input.flow.pressureNullspace=="constant"?PressureNullspaceMode::Constant:PressureNullspaceMode::Provided;
    gauge.projectRhs=c.input.flow.projectRhs;
    if (gauge.mode==PressureNullspaceMode::Provided) {
        PetscCall(VecDuplicate(w.stokes.g,&w.modeS));
        PetscCall(VecSet(w.modeS,c.input.flow.pressureModes.At("stokes").AsReal()));
        PetscCall(VecDuplicate(w.darcy.g,&w.modeD));
        const auto& value=c.input.flow.pressureModes.At("darcy");
        if (value.AsString()=="sqrt_cell_average_porosity") {
            // P0 ownership follows the same row-major owned cell order.
            PetscScalar* data=nullptr; PetscCall(VecGetArray(w.modeD,&data));
            for (std::size_t k=0;k<s.flowPorosity.size();++k) data[k]=std::sqrt(s.flowPorosity[k].average);
            PetscCall(VecRestoreArray(w.modeD,&data));
        } else PetscCall(VecSet(w.modeD,value.AsReal()));
        gauge.stokesPressureMode=w.modeS; gauge.darcyPressureMode=w.modeD;
    }
    PetscCall(BuildCoupledLinearSystem(comm,w.stokes,w.darcy,w.coupling,options,s.flowSystem));
    PetscCall(SolveLinearSystem(s.flowSystem,c.flow.solver,s.flowReport));
    // A nonconverged iterate cannot become a valid initialized transport state,
    // even with the lower-level solver's errorIfNotConverged policy disabled.
    PetscCheck(s.flowReport.converged,comm,PETSC_ERR_NOT_CONVERGED,"Initial Darcy-Stokes solve did not satisfy its acceptance policy");
    if (!reconstruction) PetscCall(PetscPrintf(comm,"  Initial Darcy-Stokes solve: %" PetscInt_FMT " iterations, true relative residual %.3e\n",
        s.flowReport.iterations,static_cast<double>(s.flowReport.relativeTrueResidualNorm)));
    PetscFunctionReturn(PETSC_SUCCESS);
}
} // namespace

// InitialState owns the system; assembly work objects are always cleaned up.
PetscErrorCode InitializeFlow(MPI_Comm comm, const Configuration& c, InitialState& s)
{
    PetscFunctionBeginUser;
    Workspace workspace;
    const auto error=Build(comm,c,s,workspace);
    const auto cleanup=workspace.Clear();
    PetscCall(error); PetscCall(cleanup);
    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode SolveReconstructedFlow(MPI_Comm comm,const Configuration& c,InitialState& s,
                                    const StateReconstruction& reconstruction)
{
    PetscFunctionBeginUser;
    // Build an independent candidate; retain the old accepted solve on failure.
    InitialState candidate;
    candidate.vertexDM=s.vertexDM; candidate.mesh=s.mesh; candidate.time=s.time;
    Workspace workspace;
    const auto error=Build(comm,c,candidate,workspace,&reconstruction);
    const auto cleanup=workspace.Clear();
    if (error || cleanup) {
        (void)DestroyLinearSystem(candidate.flowSystem);
        PetscCall(error); PetscCall(cleanup);
    }
    auto& a=s.flowSystem; auto& b=candidate.flowSystem;
    std::swap(a.matrix,b.matrix); std::swap(a.rhs,b.rhs); std::swap(a.solution,b.solution);
    std::swap(a.pressureNullspace,b.pressureNullspace); std::swap(a.kind,b.kind);
    std::swap(a.removedRhsComponent,b.removedRhsComponent);
    for (int k=0;k<4;++k) std::swap(a.coupledFieldIS[k],b.coupledFieldIS[k]);
    s.stokesVelocityMap=std::move(candidate.stokesVelocityMap);
    s.darcyVelocityMap=std::move(candidate.darcyVelocityMap);
    s.pressureMap=std::move(candidate.pressureMap);
    s.flowPorosity=std::move(candidate.flowPorosity);
    s.flowReport=std::move(candidate.flowReport);
    PetscCall(DestroyLinearSystem(candidate.flowSystem));
    PetscFunctionReturn(PETSC_SUCCESS);
}
} // namespace mantle::couple
