#include "visualization.h"

#include <algorithm>
#include <exception>

namespace mantle::couple {
namespace {
template<class F> PetscErrorCode Local(F&& f)
{
    PetscFunctionBeginUser;
    try { const auto error=f(); PetscFunctionReturn(error); }
    catch (const std::exception& e) { SETERRQ(PETSC_COMM_SELF,PETSC_ERR_USER,"%s",e.what()); }
}
PetscErrorCode Agree(MPI_Comm comm,PetscErrorCode error)
{
    PetscFunctionBeginUser;
    const int local=error; int global=0;
    PetscCallMPI(MPI_Allreduce(&local,&global,1,MPI_INT,MPI_MAX,comm));
    PetscCheck(!global,comm,static_cast<PetscErrorCode>(global),"Flow sampling failed on at least one rank");
    PetscFunctionReturn(PETSC_SUCCESS);
}
PetscErrorCode LocalValues(MPI_Comm comm,const DofMap& map,Vec field,std::vector<PetscScalar>& output)
{
    PetscFunctionBeginUser;
    Vec ghost=nullptr,local=nullptr;
    PetscCall(Agree(comm,Local([&]() { output.resize(map.LocalDofs()); return PETSC_SUCCESS; })));
    const auto operation=[&]() -> PetscErrorCode {
        PetscFunctionBeginUser;
        PetscCall(VecCreateGhost(comm,map.OwnedDofs(),map.GlobalDofs(),map.GhostDofs(),map.GhostGlobalIds().data(),&ghost));
        PetscCall(VecCopy(field,ghost));
        PetscCall(VecGhostUpdateBegin(ghost,INSERT_VALUES,SCATTER_FORWARD));
        PetscCall(VecGhostUpdateEnd(ghost,INSERT_VALUES,SCATTER_FORWARD));
        PetscCall(VecGhostGetLocalForm(ghost,&local));
        const PetscScalar* values=nullptr; PetscCall(VecGetArrayRead(local,&values));
        std::copy_n(values,output.size(),output.begin());
        PetscCall(VecRestoreArrayRead(local,&values));
        PetscFunctionReturn(PETSC_SUCCESS);
    };
    const auto error=operation();
    const auto restore=local?VecGhostRestoreLocalForm(ghost,&local):PETSC_SUCCESS;
    const auto cleanup=VecDestroy(&ghost);
    PetscCall(error); PetscCall(restore); PetscCall(cleanup);
    PetscFunctionReturn(PETSC_SUCCESS);
}
PetscErrorCode Evaluate(const BRMixed& br,const HDivMixed& hd,
                        const std::array<PetscScalar,12>& us,const std::array<PetscScalar,8>& ud,
                        FlowPointSample& sample)
{
    PetscFunctionBeginUser;
    BRMixed::Values sb; HDivMixed::Values db;
    PetscCall(br.EvaluateAll(sample.position,sb)); PetscCall(hd.EvaluateAll(sample.position,db));
    for (int d=0;d<2;++d) {
        for (int k=0;k<12;++k) sample.stokesVelocity.p[d]+=PetscRealPart(us[k])*sb[k].value.p[d];
        for (int k=0;k<8;++k) sample.darcyVelocity.p[d]+=PetscRealPart(ud[k])*db[k].value.p[d];
        PetscCheck(!PetscIsInfOrNanReal(sample.stokesVelocity.p[d]) && !PetscIsInfOrNanReal(sample.darcyVelocity.p[d]),
                   PETSC_COMM_SELF,PETSC_ERR_FP,"Nonfinite reconstructed velocity");
    }
    PetscFunctionReturn(PETSC_SUCCESS);
}
} // namespace

PetscErrorCode SampleInitialFlow(MPI_Comm comm,const Configuration& c,const InitialState& s,InitialFlowSamples& output)
{
    PetscFunctionBeginUser;
    PetscCall(Agree(comm,s.flowReport.converged && !s.flowSystem.IsEmpty()?PETSC_SUCCESS:PETSC_ERR_ARG_WRONGSTATE));
    Vec us=nullptr,ps=nullptr,ud=nullptr,pd=nullptr;
    PetscCall(GetCoupledLinearSystemSolution(s.flowSystem,us,ps,ud,pd));
    std::vector<PetscScalar> vs,vd,qs,qd;
    PetscCall(LocalValues(comm,s.stokesVelocityMap,us,vs)); PetscCall(LocalValues(comm,s.darcyVelocityMap,ud,vd));
    PetscCall(LocalValues(comm,s.pressureMap,ps,qs)); PetscCall(LocalValues(comm,s.pressureMap,pd,qd));
    GaussRule1D cell,edge; InitialFlowSamples result;
    PetscCall(Agree(comm,CreateGaussRule(c.input.quadrature.cellPointsPerAxis,cell)));
    PetscCall(Agree(comm,CreateGaussRule(c.input.quadrature.edgePoints,edge)));
    PetscCall(Agree(comm,Local([&]() -> PetscErrorCode {
        const auto range=s.mesh.OwnedCells(); std::size_t local=0;
        for (PetscInt j=range.begin.j;j<range.end.j;++j) for (PetscInt i=range.begin.i;i<range.end.i;++i,++local) {
            const MeshIndex index{i,j}; QuadVertices corners; PetscInt id=0;
            auto error=s.mesh.GetCellCorners(index,corners); if (error) return error;
            error=s.mesh.CellId(index,id); if (error) return error;
            BRMixed br; HDivMixed hd;
            error=br.Initialize(corners); if (error) return error;
            error=hd.Initialize(corners); if (error) return error;
            std::vector<PetscInt> ids; std::array<PetscScalar,12> su; std::array<PetscScalar,8> du;
            error=s.stokesVelocityMap.GetCellLocalDofs(index,ids); if (error) return error;
            for (int k=0;k<12;++k) su[k]=vs[ids[k]];
            error=s.darcyVelocityMap.GetCellLocalDofs(index,ids); if (error) return error;
            for (int k=0;k<8;++k) du[k]=vd[ids[k]];
            error=s.pressureMap.GetCellLocalDofs(index,ids); if (error) return error;
            FlowPointSample base; base.cellId=id;
            base.stokesPressure=PetscRealPart(qs[ids[0]]); base.darcyPressure=PetscRealPart(qd[ids[0]]);
            const auto& phi=s.flowPorosity.at(local); const auto n=cell.points.size();
            for (std::size_t b=0;b<n;++b) for (std::size_t a=0;a<n;++a) {
                auto sample=base; sample.q=b*n+a;
                const Point ref{{cell.points[a],cell.points[b]}};
                sample.position=MapCellPoint(ref,corners);
                sample.weight=cell.weights[a]*cell.weights[b]*CellJacobian(ref,corners);
                sample.flowPorosity=phi.cell.at(b*n+a);
                error=Evaluate(br,hd,su,du,sample); if (error) return error;
                result.cells.push_back(sample);
            }
            std::array<OrientedEdge,4> edges;
            error=s.mesh.GetCellEdges(index,edges); if (error) return error;
            for (int e=0;e<4;++e) {
                const EdgeVertices vertices{{corners[e],corners[(e+1)%4]}};
                PetscReal length=0; Point normal;
                error=GetEdgeGeometry(vertices,length,normal); if (error) return error;
                for (std::size_t q=0;q<edge.points.size();++q) {
                    auto sample=base; sample.edgeId=edges[e].id; sample.side=e;
                    sample.q=edges[e].direction==1?q:edge.points.size()-1-q;
                    sample.position=MapEdgePoint(edge.points[q],vertices);
                    sample.weight=.5*length*edge.weights[q]; sample.flowPorosity=phi.edge[e].at(q);
                    error=Evaluate(br,hd,su,du,sample); if (error) return error;
                    result.edges.push_back(sample);
                }
            }
        }
        return PETSC_SUCCESS;
    })));
    output=std::move(result);
    PetscFunctionReturn(PETSC_SUCCESS);
}
} // namespace mantle::couple
