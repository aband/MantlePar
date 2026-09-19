#include "state_reconstruction.h"
#include <petscao.h>
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace mantle::couple {
namespace {
template<class F> PetscErrorCode Local(F&& f)
{
    PetscFunctionBeginUser;
    try { const auto e=f(); PetscFunctionReturn(e); }
    catch (const std::exception& e) { SETERRQ(PETSC_COMM_SELF,PETSC_ERR_USER,"%s",e.what()); }
}
PetscErrorCode Agree(MPI_Comm comm,PetscErrorCode error)
{
    PetscFunctionBeginUser;
    int local=error,global=0;
    PetscCallMPI(MPI_Allreduce(&local,&global,1,MPI_INT,MPI_MAX,comm));
    PetscCheck(!global,comm,static_cast<PetscErrorCode>(global),"State reconstruction failed on at least one rank");
    PetscFunctionReturn(PETSC_SUCCESS);
}
MeshRange Expand(MeshRange r,MeshIndex n,PetscInt width)
{
    return {{std::max(PetscInt(0),r.begin.i-width),std::max(PetscInt(0),r.begin.j-width)},
            {std::min(n.i,r.end.i+width),std::min(n.j,r.end.j+width)}};
}
std::size_t Offset(MeshRange r,MeshIndex p)
{
    if (!r.Contains(p)) throw std::out_of_range("Cell outside reconstruction halo");
    return static_cast<std::size_t>(p.j-r.begin.j)*r.Size().i+p.i-r.begin.i;
}
}

PetscErrorCode StateReconstruction::Initialize(MPI_Comm comm,const InitialState& s)
{
    PetscFunctionBeginUser;
    comm_=comm;
    PetscCheck(!geometryDM_,comm,PETSC_ERR_ARG_WRONGSTATE,"Reconstruction already initialized");
    PetscInt nx,ny,px,py;
    PetscCall(DMDAGetInfo(s.vertexDM,nullptr,&nx,&ny,nullptr,&px,&py,nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,nullptr));
    const PetscInt *lx=nullptr,*ly=nullptr;
    PetscCall(DMDAGetOwnershipRanges(s.vertexDM,&lx,&ly,nullptr));
    // Width three covers a neighboring cell and its centered 3x3 stencil,
    // including the far vertex. No replication of the global mesh or fields.
    PetscCheck(nx>=3 && ny>=3,comm,PETSC_ERR_ARG_SIZ,"Transport requires at least 2x2 cells");
    for (PetscInt i=0;i<px;++i) PetscCheck(lx[i]>=3,comm,PETSC_ERR_ARG_SIZ,
        "Transport needs at least three vertex columns per MPI partition; reduce process_grid[0]");
    for (PetscInt j=0;j<py;++j) PetscCheck(ly[j]>=3,comm,PETSC_ERR_ARG_SIZ,
        "Transport needs at least three vertex rows per MPI partition; reduce process_grid[1]");
    PetscCall(DMDACreate2d(comm,DM_BOUNDARY_NONE,DM_BOUNDARY_NONE,DMDA_STENCIL_BOX,
        nx,ny,px,py,2,3,lx,ly,&geometryDM_));
    PetscCall(DMSetUp(geometryDM_));
    PetscCall(DMCreateGlobalVector(geometryDM_,&coordinates_));
    PetscCall(VecCopy(s.vertices,coordinates_));
    PetscCall(BuildMeshInfo(geometryDM_,coordinates_,mesh_));
    targets_=Expand(mesh_.OwnedCells(),mesh_.CellDimensions(),1);
    valuesRange_=mesh_.AvailableCells();
    std::vector<PetscInt> ids;
    PetscCall(Agree(comm,Local([&]() -> PetscErrorCode {
        const auto count=static_cast<std::size_t>(targets_.Size().i)*targets_.Size().j;
        h_.resize(count); c_.resize(count); areas_.resize(count);
        limitH_.assign(count,1); limitC_.assign(count,1);
        ReconstructionOptions options;
        options.large.size={3,3}; options.large.order=2; options.large.offsets={{-1,-1}};
        options.small.size={2,2}; options.small.order=1;
        options.small.offsets={{0,-1},{0,0},{-1,-1},{-1,0}};
        options.small.smoothness=ReconstructionSmoothness::TargetCell;
        // Match legacy (3,2) advection: unavailable boundary stencils are
        // discarded; the remaining 2x2 candidates still contain the target.
        for (PetscInt j=targets_.begin.j;j<targets_.end.j;++j) for (PetscInt i=targets_.begin.i;i<targets_.end.i;++i) {
            const auto k=Index({i,j});
            PetscCall(mesh_.GetCellArea({i,j},areas_[k]));
            PetscCall(h_[k].Initialize(mesh_,{i,j},options,std::sqrt(areas_[k])));
            c_[k]=h_[k];
        }
        for (PetscInt j=valuesRange_.begin.j;j<valuesRange_.end.j;++j)
            for (PetscInt i=valuesRange_.begin.i;i<valuesRange_.end.i;++i) ids.push_back(j*(nx-1)+i);
        averagesH_.resize(ids.size()); averagesC_.resize(ids.size());
        return PETSC_SUCCESS;
    })));
    AO ao=nullptr; PetscCall(DMDAGetAO(s.cellDM,&ao));
    PetscCall(AOApplicationToPetsc(ao,static_cast<PetscInt>(ids.size()),ids.data()));
    IS selected=nullptr;
    PetscCall(ISCreateGeneral(comm,static_cast<PetscInt>(ids.size()),ids.data(),PETSC_COPY_VALUES,&selected));
    const auto operation=[&]() -> PetscErrorCode {
        PetscFunctionBeginUser;
        PetscCall(VecCreateSeq(PETSC_COMM_SELF,static_cast<PetscInt>(ids.size()),&local_));
        PetscCall(VecScatterCreate(s.enthalpy,selected,local_,nullptr,&scatter_));
        PetscFunctionReturn(PETSC_SUCCESS);
    };
    const auto error=operation(), cleanup=ISDestroy(&selected);
    PetscCall(error); PetscCall(cleanup);
    PetscCall(Update(s));
    PetscFunctionReturn(PETSC_SUCCESS);
}

std::size_t StateReconstruction::Index(MeshIndex p) const { return Offset(targets_,p); }
PetscReal StateReconstruction::AverageH(MeshIndex p) const { return averagesH_.at(Offset(valuesRange_,p)); }
PetscReal StateReconstruction::AverageC(MeshIndex p) const { return averagesC_.at(Offset(valuesRange_,p)); }
PetscErrorCode StateReconstruction::Update(const InitialState& s,bool updateComposition)
{
    PetscFunctionBeginUser;
    std::fill(limitH_.begin(),limitH_.end(),1);
    std::fill(limitC_.begin(),limitC_.end(),1);
    const Vec fields[2]={s.enthalpy,s.composition};
    for (int k=0;k<(updateComposition?2:1);++k) {
        PetscCall(VecScatterBegin(scatter_,fields[k],local_,INSERT_VALUES,SCATTER_FORWARD));
        PetscCall(VecScatterEnd(scatter_,fields[k],local_,INSERT_VALUES,SCATTER_FORWARD));
        auto& values=k?averagesC_:averagesH_; auto& recon=k?c_:h_;
        const PetscScalar* a=nullptr; PetscCall(VecGetArrayRead(local_,&a));
        for (std::size_t n=0;n<values.size();++n) values[n]=PetscRealPart(a[n]);
        PetscCall(VecRestoreArrayRead(local_,&a));
        PetscCall(Agree(comm_,Local([&]() -> PetscErrorCode {
            for (std::size_t n=0;n<recon.size();++n) PetscCall(recon[n].Update(values,valuesRange_,areas_[n]));
            return PETSC_SUCCESS;
        })));
    }
    PetscFunctionReturn(PETSC_SUCCESS);
}
PetscErrorCode StateReconstruction::Enthalpy(MeshIndex cell,const Point& p,PetscReal& H) const
{ return Local([&]() -> PetscErrorCode {
    PetscCall(h_.at(Index(cell)).Evaluate(p,H));
    H=AverageH(cell)+limitH_.at(Index(cell))*(H-AverageH(cell));
    return PETSC_SUCCESS;
}); }
PetscErrorCode StateReconstruction::Evaluate(MeshIndex cell,const Point& p,PetscReal& H,PetscReal& C) const
{
    return Local([&]() -> PetscErrorCode {
        PetscCall(Enthalpy(cell,p,H)); PetscCall(c_.at(Index(cell)).Evaluate(p,C));
        C=AverageC(cell)+limitC_.at(Index(cell))*(C-AverageC(cell));
        return PETSC_SUCCESS;
    });
}

PetscErrorCode StateReconstruction::Limit(const Configuration& config)
{
    PetscFunctionBeginUser;
    PetscCall(Agree(comm_,Local([&]() -> PetscErrorCode {
        GaussRule1D cellRule,edgeRule;
        PetscCall(CreateGaussRule(config.input.quadrature.cellPointsPerAxis,cellRule));
        PetscCall(CreateGaussRule(config.input.quadrature.edgePoints,edgeRule));
        const auto dims=mesh_.CellDimensions();
        for (PetscInt j=targets_.begin.j;j<targets_.end.j;++j) for (PetscInt i=targets_.begin.i;i<targets_.end.i;++i) {
            const MeshIndex cell{i,j}; const auto k=Index(cell);
            const PetscReal mean[2]={AverageH(cell),AverageC(cell)};
            PetscCheck(std::isfinite(mean[0]) && mean[0]>=0 && std::isfinite(mean[1]) && mean[1]>=0 && mean[1]<=config.phase.parameters().Xe,
                PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Inadmissible cell average in coupled reconstruction");
            PetscReal lo[2]={mean[0],mean[1]},hi[2]={mean[0],mean[1]};
            for (PetscInt y=std::max(PetscInt(0),j-1);y<std::min(dims.j,j+2);++y)
                for (PetscInt x=std::max(PetscInt(0),i-1);x<std::min(dims.i,i+2);++x) {
                    const PetscReal v[2]={AverageH({x,y}),AverageC({x,y})};
                    for (int z=0;z<2;++z) { lo[z]=std::min(lo[z],v[z]); hi[z]=std::max(hi[z],v[z]); }
                }
            lo[0]=std::max(PetscReal(0),lo[0]); lo[1]=std::max(PetscReal(0),lo[1]);
            hi[1]=std::min(PetscReal(config.phase.parameters().Xe),hi[1]);
            QuadVertices corners; PetscCall(mesh_.GetCellCorners(cell,corners));
            std::vector<Point> points{MapCellPoint(Point{{0,0}},corners)};
            for (auto a:cellRule.points) for (auto b:cellRule.points) points.push_back(MapCellPoint(Point{{a,b}},corners));
            std::array<OrientedEdge,4> edges; PetscCall(mesh_.GetCellEdges(cell,edges));
            for (int e=0;e<4;++e) {
                const EdgeVertices edge{{corners[e],corners[(e+1)%4]}};
                for (auto q:edgeRule.points) points.push_back(MapEdgePoint(q,edge));
                EdgeTopology t; PetscCall(mesh_.GetEdgeTopology(edges[e].id,t));
                DiffusiveSampling sampling;
                if (t.IsBoundary()) PetscCall(CreateBoundaryDiffusiveSampling(edge,corners,edgeRule,config.sampling,sampling));
                else {
                    auto neighbor=*t.leftCell; if (neighbor.i==i && neighbor.j==j) neighbor=*t.rightCell;
                    QuadVertices other; PetscCall(mesh_.GetCellCorners(neighbor,other));
                    PetscCall(CreateInteriorDiffusiveSampling(edge,corners,other,edgeRule,config.sampling,sampling));
                }
                for (std::size_t q=0;q<sampling.SamplePoints().size();++q)
                    if (sampling.SampleSides()[q]==DiffusiveSampleSide::Left) points.push_back(sampling.SamplePoints()[q]);
            }
            PetscReal theta[2]={1,1};
            for (const auto& p:points) {
                PetscReal v[2]; PetscCall(h_[k].Evaluate(p,v[0])); PetscCall(c_[k].Evaluate(p,v[1]));
                for (int z=0;z<2;++z) {
                    PetscCheck(std::isfinite(v[z]),PETSC_COMM_SELF,PETSC_ERR_FP,"Nonfinite reconstructed state");
                    const auto delta=v[z]-mean[z];
                    if (v[z]>hi[z]) theta[z]=std::min(theta[z],(hi[z]-mean[z])/delta);
                    if (v[z]<lo[z]) theta[z]=std::min(theta[z],(lo[z]-mean[z])/delta);
                }
            }
            // Round inward so admissible zero/Xe bounds survive floating-point evaluation.
            limitH_[k]=theta[0]<1?std::max(PetscReal(0),theta[0]*(1-1e-13)):1;
            limitC_[k]=theta[1]<1?std::max(PetscReal(0),theta[1]*(1-1e-13)):1;
        }
        return PETSC_SUCCESS;
    })));
    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode StateReconstruction::Destroy()
{
    PetscFunctionBeginUser;
    PetscErrorCode first=0;
    const auto record=[&](PetscErrorCode e) { if (!first) first=e; };
    record(VecScatterDestroy(&scatter_)); record(VecDestroy(&local_));
    record(VecDestroy(&coordinates_)); record(DMDestroy(&geometryDM_));
    h_.clear(); c_.clear(); areas_.clear(); averagesH_.clear(); averagesC_.clear();
    mesh_=MeshInfo{}; comm_=MPI_COMM_NULL;
    PetscFunctionReturn(first);
}
}
