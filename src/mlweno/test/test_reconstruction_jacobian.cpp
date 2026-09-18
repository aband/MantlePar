// Serial Jacobian tests. Uses the production PETSc mesh and numerical kernels.
// -jacobian_case: -1=all, 0=rectangular, 1=quadrilateral, 2=pseudo-1D,
//                 3=cache/errors, 4=constants/extreme scales/Taylor test.
#include "reconstruction.h"
#include <petscsys.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <exception>
#include <limits>
#include <numeric>
#include <utility>
#include <vector>

namespace {
using Mode = ReconstructionJacobianMode;
using Matrix = std::vector<std::vector<PetscReal>>;
struct Stats {
    std::size_t pointEntries=0, weightEntries=0;
    PetscReal pointError=0, weightError=0, actionError=0;
};
Stats stats;
struct Grid {
    DM dm=nullptr;
    Vec coordinates=nullptr;
    MeshInfo mesh;
    MeshIndex size{};
    ~Grid() { if(coordinates) (void)VecDestroy(&coordinates); if(dm) (void)DMDestroy(&dm); }
};
std::size_t Index(MeshIndex size, MeshIndex cell)
{ return static_cast<std::size_t>(cell.j)*size.i+cell.i; }
std::size_t PatchIndex(const MeshRange& range, MeshIndex cell)
{ return static_cast<std::size_t>(cell.j-range.begin.j)*range.Size().i+cell.i-range.begin.i; }

PetscErrorCode Require(bool condition, const char* message)
{
    PetscFunctionBeginUser;
    PetscCheck(condition,PETSC_COMM_SELF,PETSC_ERR_PLIB,"%s",message);
    PetscFunctionReturn(PETSC_SUCCESS);
}
PetscErrorCode Near(PetscReal a, PetscReal b, const char* message, PetscReal tolerance=3e-9)
{
    PetscFunctionBeginUser;
    PetscCheck(std::isfinite(a) && std::isfinite(b) &&
               std::abs(a-b)<=tolerance*(1+std::abs(a)+std::abs(b)),
               PETSC_COMM_SELF,PETSC_ERR_PLIB,"%s: %.17g versus %.17g",message,
               static_cast<double>(a),static_cast<double>(b));
    PetscFunctionReturn(PETSC_SUCCESS);
}
template<class Function>
PetscErrorCode ExpectError(Function&& function)
{
    PetscFunctionBeginUser;
    PetscCall(PetscPushErrorHandler(PetscReturnErrorHandler,nullptr));
    const PetscErrorCode error=function();
    PetscCall(PetscPopErrorHandler());
    PetscCall(Require(error!=PETSC_SUCCESS,"Expected an error"));
    PetscFunctionReturn(PETSC_SUCCESS);
}
PetscErrorCode MakeGrid(MeshIndex size, bool quad, Grid& grid)
{
    PetscFunctionBeginUser;
    grid.size=size;
    PetscCall(DMDACreate2d(PETSC_COMM_WORLD,DM_BOUNDARY_NONE,DM_BOUNDARY_NONE,
                           DMDA_STENCIL_BOX,size.i+1,size.j+1,1,1,2,1,
                           nullptr,nullptr,&grid.dm));
    PetscCall(DMSetUp(grid.dm));
    PetscCall(DMCreateGlobalVector(grid.dm,&grid.coordinates));
    MeshParam parameters;
    parameters.L=.13*size.i; parameters.H=.19*size.j;
    parameters.perturbation=.14; parameters.seed=37;
    if(quad) PetscCall(LogicRectMesh(grid.dm,grid.coordinates,parameters));
    else PetscCall(CreateFullMesh(grid.dm,grid.coordinates,parameters));
    PetscCall(BuildMeshInfo(grid.dm,grid.coordinates,grid.mesh));
    PetscFunctionReturn(PETSC_SUCCESS);
}
std::vector<PetscReal> Data(MeshIndex size)
{
    std::vector<PetscReal> u(static_cast<std::size_t>(size.i)*size.j);
    for(PetscInt j=0;j<size.j;++j) for(PetscInt i=0;i<size.i;++i)
        u[Index(size,{i,j})]=.5+(static_cast<int>((13*i+7*j+3*i*j)%23)-11)/128.;
    return u;
}
ReconstructionOptions Options(PetscInt axis=0, bool constant=true)
{
    ReconstructionOptions o;
    o.useConstant=constant; o.constantLinearWeight=.02;
    o.large.smoothness=ReconstructionSmoothness::ReferenceSquare;
    o.small.smoothness=ReconstructionSmoothness::TargetCell;
    if(axis==0) {
        o.large.size={4,3}; o.large.order=3;
        o.large.offsets={{-1,-1},{-2,0},{0,-2},{0,0}};
        o.large.linearWeights={1.2,.8,.6,.9};
        o.small.size={2,2}; o.small.order=1;
        // Includes duplicate offsets and a disabled candidate deliberately.
        o.small.offsets={{-1,-1},{0,0},{0,-1},{-1,0},{0,0}};
        o.small.linearWeights={.7,.5,0,1.1,.3};
    } else if(axis==1) {
        o.large.size={5,1}; o.large.order=4;
        o.large.offsets={{-4,0},{-2,0},{0,0}};
        o.small.size={3,1}; o.small.order=2;
        o.small.offsets={{-2,0},{-1,0},{0,0}};
    } else {
        o.large.size={1,5}; o.large.order=4;
        o.large.offsets={{0,-4},{0,-2},{0,0}};
        o.small.size={1,3}; o.small.order=2;
        o.small.offsets={{0,-2},{0,-1},{0,0}};
    }
    return o;
}
PetscErrorCode Weights(const Reconstruction& r, std::vector<PetscReal>& weights)
{
    PetscFunctionBeginUser;
    weights.resize(r.CandidateCount()+1);
    for(std::size_t k=0;k<r.CandidateCount();++k) {
        ReconstructionCandidateInfo info;
        PetscCall(r.GetCandidate(k,info)); weights[k]=info.nonlinearWeight;
    }
    PetscCall(r.ConstantWeight(weights.back()));
    PetscFunctionReturn(PETSC_SUCCESS);
}
struct FrozenCandidate {
    TensorStencilPoly polynomial;
    ReconstructionCandidateInfo info;
};
PetscErrorCode FrozenCandidates(const Grid& grid, const Reconstruction& r,
                                std::vector<FrozenCandidate>& candidates)
{
    PetscFunctionBeginUser;
    candidates.resize(r.CandidateCount());
    for(std::size_t k=0;k<candidates.size();++k) {
        auto& c=candidates[k]; PetscCall(r.GetCandidate(k,c.info));
        PetscCall(c.polynomial.Initialize(grid.mesh,c.info.start,c.info.size,
                                          c.info.order,std::sqrt(.13*.19)));
    }
    PetscFunctionReturn(PETSC_SUCCESS);
}
PetscErrorCode FrozenValue(const Grid& grid, MeshIndex target,
                           const std::vector<FrozenCandidate>& candidates,
                           const std::vector<PetscReal>& u, const Point& p,
                           PetscInt dx, PetscInt dy, PetscReal& value)
{
    PetscFunctionBeginUser;
    const PetscReal mean=u[Index(grid.size,target)];
    long double result=(dx==0 && dy==0)?mean:0;
    for(const auto& c:candidates) {
        std::vector<PetscReal> local;
        for(PetscInt j=0;j<c.info.size.j;++j) for(PetscInt i=0;i<c.info.size.i;++i)
            local.push_back(u[Index(grid.size,{c.info.start.i+i,c.info.start.j+j})]-mean);
        PetscReal residual=0;
        PetscCall(c.polynomial.EvaluateDerivative(local,p,dx,dy,residual));
        result+=static_cast<long double>(c.info.nonlinearWeight)*residual;
    }
    value=static_cast<PetscReal>(result);
    PetscFunctionReturn(PETSC_SUCCESS);
}
PetscErrorCode Conservation(const Grid& grid, const Reconstruction& r)
{
    PetscFunctionBeginUser;
    QuadVertices cell; PetscCall(grid.mesh.GetCellCorners(r.Target(),cell));
    GaussRule1D rule; PetscCall(CreateGaussRule(6,rule));
    std::vector<long double> integral(r.JacobianCells().size(),0);
    long double area=0;
    for(std::size_t y=0;y<rule.points.size();++y) for(std::size_t x=0;x<rule.points.size();++x) {
        const Point q{{rule.points[x],rule.points[y]}};
        const Point p=MapCellPoint(q,cell);
        const PetscReal weight=rule.weights[x]*rule.weights[y]*CellJacobian(q,cell);
        std::vector<PetscReal> jac; PetscReal value;
        PetscCall(r.EvaluateWithJacobian(p,value,jac));
        area+=weight;
        for(std::size_t c=0;c<jac.size();++c) integral[c]+=weight*jac[c];
    }
    for(std::size_t c=0;c<integral.size();++c)
        PetscCall(Near(static_cast<PetscReal>(integral[c]/area),
                       r.JacobianCells()[c]==r.Target()?1:0,"Mean Jacobian conservation",2e-8));
    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode FiniteDifferences(const Grid& grid, MeshIndex target,
                                 const ReconstructionOptions& options, bool conserve)
{
    PetscFunctionBeginUser;
    const MeshRange range{{0,0},grid.size};
    const PetscReal area=.13*.19;
    // Central differences are second order. Step-halving resolves the
    // sharper quadrilateral weight transitions without relaxing tolerances.
    PetscReal step=5e-7;
    PetscCall(PetscOptionsGetReal(nullptr,nullptr,"-jacobian_fd_step",&step,nullptr));
    PetscCall(Require(std::isfinite(step) && step>0,"Invalid finite-difference step"));
    auto u=Data(grid.size);
    Reconstruction r;
    PetscCall(r.Initialize(grid.mesh,target,options,std::sqrt(area)));
    PetscCall(r.UpdateWithJacobian(u,range,area));
    const auto cells=r.JacobianCells();
    PetscCall(Require(!cells.empty(),"Missing Jacobian support"));
    for(std::size_t k=1;k<cells.size();++k)
        PetscCall(Require(cells[k-1].j<cells[k].j ||
                           (cells[k-1].j==cells[k].j && cells[k-1].i<cells[k].i),
                           "Jacobian cells are not sorted and unique"));
    QuadVertices cell; PetscCall(grid.mesh.GetCellCorners(target,cell));
    const std::vector<Point> points={MapCellPoint(Point{{.37,-.41}},cell),
                                     MapCellPoint(Point{{-1,.23}},cell)};
    std::vector<FrozenCandidate> frozen;
    PetscCall(FrozenCandidates(grid,r,frozen));
    Matrix weightJac;
    PetscCall(r.WeightJacobian(weightJac));
    PetscCall(Require(weightJac.size()==r.CandidateCount()+1,"Missing constant weight row"));
    for(std::size_t c=0;c<cells.size();++c) {
        long double sum=0;
        for(const auto& row:weightJac) sum+=row[c];
        PetscCall(Near(static_cast<PetscReal>(sum),0,"Weight derivatives do not sum to zero",2e-8));
    }
    std::vector<PetscReal> direction(cells.size());
    for(std::size_t c=0;c<cells.size();++c) direction[c]=.3*std::sin(.8*(c+1));
    for(const Point& point:points) {
        for(MeshIndex derivative:{MeshIndex{0,0},{1,0},{0,1},{1,1}}) {
            PetscReal value, direct;
            std::vector<PetscReal> fullJac, frozenJac;
            PetscCall(r.EvaluateDerivativeWithJacobian(point,derivative.i,derivative.j,value,fullJac));
            PetscCall(r.EvaluateDerivative(point,derivative.i,derivative.j,direct));
            PetscCall(Require(value==direct,"Jacobian path changed the reconstructed value"));
            PetscCall(r.EvaluateDerivativeWithJacobian(point,derivative.i,derivative.j,value,
                                                       frozenJac,Mode::FrozenWeights));
            const PetscReal sum=std::accumulate(fullJac.begin(),fullJac.end(),PetscReal(0));
            PetscCall(Near(sum,derivative==MeshIndex{0,0}?1:0,"Constant-shift Jacobian identity",2e-8));
            for(Mode mode:{Mode::Full,Mode::FrozenWeights}) {
                PetscReal action=0; long double product=0;
                const auto& jac=mode==Mode::Full?fullJac:frozenJac;
                PetscCall(r.ApplyDerivativeJacobian(point,derivative.i,derivative.j,direction,action,mode));
                for(std::size_t c=0;c<cells.size();++c) product+=static_cast<long double>(jac[c])*direction[c];
                const PetscReal expected=static_cast<PetscReal>(product);
                stats.actionError=std::max(stats.actionError,std::abs(action-expected)/(1+std::abs(expected)));
                PetscCall(Near(action,expected,"Direct J*v differs from the assembled row",2e-9));
            }
            for(std::size_t c=0;c<cells.size();++c) {
                auto plus=u, minus=u;
                const auto id=Index(grid.size,cells[c]); plus[id]+=step; minus[id]-=step;
                Reconstruction perturbed=r;
                PetscReal plusValue,minusValue;
                PetscCall(perturbed.Update(plus,range,area));
                PetscCall(perturbed.EvaluateDerivative(point,derivative.i,derivative.j,plusValue));
                PetscCall(perturbed.Update(minus,range,area));
                PetscCall(perturbed.EvaluateDerivative(point,derivative.i,derivative.j,minusValue));
                const PetscReal fd=(plusValue-minusValue)/(2*step);
                stats.pointError=std::max(stats.pointError,std::abs(fd-fullJac[c])/(1+std::abs(fullJac[c])));
                ++stats.pointEntries;
                if(std::abs(fd-fullJac[c])>2e-6*(1+std::abs(fd)+std::abs(fullJac[c])))
                    PetscCall(PetscPrintf(PETSC_COMM_WORLD,
                        "FD context: target=(%lld,%lld), derivative=(%lld,%lld), "
                        "column=(%lld,%lld), point=(%.6g,%.6g), step=%.6g\n",
                        static_cast<long long>(target.i),static_cast<long long>(target.j),
                        static_cast<long long>(derivative.i),static_cast<long long>(derivative.j),
                        static_cast<long long>(cells[c].i),static_cast<long long>(cells[c].j),
                        static_cast<double>(point.p[0]),static_cast<double>(point.p[1]),
                        static_cast<double>(step)));
                PetscCall(Near(fd,fullJac[c],"Full Jacobian finite difference",2e-6));
                PetscCall(FrozenValue(grid,target,frozen,plus,point,derivative.i,derivative.j,plusValue));
                PetscCall(FrozenValue(grid,target,frozen,minus,point,derivative.i,derivative.j,minusValue));
                PetscCall(Near((plusValue-minusValue)/(2*step),frozenJac[c],
                               "Frozen-weight finite difference",2e-7));
            }
        }
    }
    for(std::size_t c=0;c<cells.size();++c) {
        auto plus=u,minus=u; const auto id=Index(grid.size,cells[c]);
        plus[id]+=step; minus[id]-=step;
        Reconstruction perturbed=r;
        std::vector<PetscReal> wp,wm;
        PetscCall(perturbed.Update(plus,range,area)); PetscCall(Weights(perturbed,wp));
        PetscCall(perturbed.Update(minus,range,area)); PetscCall(Weights(perturbed,wm));
        for(std::size_t row=0;row<weightJac.size();++row) {
            const PetscReal fd=(wp[row]-wm[row])/(2*step);
            stats.weightError=std::max(stats.weightError,
                std::abs(fd-weightJac[row][c])/(1+std::abs(weightJac[row][c])));
            ++stats.weightEntries;
            PetscCall(Near(fd,weightJac[row][c],"Weight Jacobian finite difference",2e-6));
        }
    }
    Matrix batchJac; std::vector<PetscReal> batchValues;
    PetscCall(r.EvaluateWithJacobian(points,batchValues,batchJac));
    for(std::size_t p=0;p<points.size();++p) {
        PetscReal value; std::vector<PetscReal> jac;
        PetscCall(r.EvaluateWithJacobian(points[p],value,jac));
        PetscCall(Require(value==batchValues[p] && jac==batchJac[p],"Batch evaluation differs"));
    }
    // A patch with a nonzero global origin must give the same column ordering.
    const MeshRange patch=r.RequiredCells();
    std::vector<PetscReal> local(static_cast<std::size_t>(patch.Size().i)*patch.Size().j,
                                std::numeric_limits<PetscReal>::quiet_NaN());
    for(MeshIndex c:cells) local[PatchIndex(patch,c)]=u[Index(grid.size,c)];
    Reconstruction localR=r;
    PetscCall(localR.UpdateWithJacobian(local,patch,area));
    Matrix localJac; std::vector<PetscReal> localValues;
    PetscCall(localR.EvaluateWithJacobian(points,localValues,localJac));
    PetscCall(Require(localR.JacobianCells()==cells && localValues==batchValues && localJac==batchJac,
                       "Global column mapping depends on patch origin or unused entries"));
    // Exactly representable common shifts leave weights and derivatives alone.
    for(auto& v:u) v+=1024;
    PetscCall(localR.UpdateWithJacobian(u,range,area));
    PetscCall(localR.EvaluateWithJacobian(points,localValues,localJac));
    PetscCall(Require(localJac==batchJac,"Constant offset changed the solution Jacobian"));
    if(conserve) PetscCall(Conservation(grid,r));
    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode Geometries(bool quad)
{
    PetscFunctionBeginUser;
    Grid grid; PetscCall(MakeGrid({10,9},quad,grid));
    for(bool constant:{false,true}) {
        PetscCall(FiniteDifferences(grid,{4,4},Options(0,constant),true));
        PetscCall(FiniteDifferences(grid,{0,0},Options(0,constant),false));
    }
    // Reverse the two smoothness-region choices as a separate check.
    auto options=Options();
    options.large.smoothness=ReconstructionSmoothness::TargetCell;
    options.small.smoothness=ReconstructionSmoothness::ReferenceSquare;
    PetscCall(FiniteDifferences(grid,{4,4},options,false));
    PetscFunctionReturn(PETSC_SUCCESS);
}
PetscErrorCode Pseudo1D()
{
    PetscFunctionBeginUser;
    for(PetscInt axis:{1,2}) {
        Grid grid; PetscCall(MakeGrid(axis==1?MeshIndex{14,1}:MeshIndex{1,14},false,grid));
        for(bool constant:{false,true})
            PetscCall(FiniteDifferences(grid,axis==1?MeshIndex{6,0}:MeshIndex{0,6},
                                         Options(axis,constant),false));
    }
    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode StateAndErrors()
{
    PetscFunctionBeginUser;
    Grid grid; PetscCall(MakeGrid({10,9},true,grid));
    const MeshRange range{{0,0},grid.size}; const PetscReal area=.13*.19;
    auto u=Data(grid.size);
    Reconstruction r;
    const Point p{{.6,.8}};
    PetscReal value=73,action=74;
    std::vector<PetscReal> jac{8,9}; Matrix matrix{{5,6}};
    PetscCall(ExpectError([&]{return r.PrepareJacobian();}));
    PetscCall(ExpectError([&]{return r.EvaluateWithJacobian(p,value,jac);}));
    PetscCall(Require(value==73 && jac==std::vector<PetscReal>{8,9},"Error changed outputs"));
    PetscCall(r.Initialize(grid.mesh,{4,4},Options(),std::sqrt(area)));
    const auto cells=r.JacobianCells();
    PetscCall(r.Update(u,range,area));
    PetscCall(Require(!r.HasJacobian(),"Ordinary update unexpectedly prepared a Jacobian"));
    PetscCall(ExpectError([&]{return r.WeightJacobian(matrix);}));
    PetscCall(Require(matrix==Matrix{{5,6}},"Weight query error changed output"));
    PetscCall(r.EvaluateWithJacobian(p,value,jac,Mode::FrozenWeights));
    PetscCall(r.PrepareJacobian()); PetscCall(r.EvaluateWithJacobian(p,value,jac));
    const auto savedJac=jac; const PetscReal savedValue=value;
    PetscCall(r.PrepareJacobian());
    PetscCall(ExpectError([&]{return r.UpdateWithJacobian(u,range,0);}));
    auto bad=u; bad[Index(grid.size,{4,4})]=std::numeric_limits<PetscReal>::quiet_NaN();
    PetscCall(ExpectError([&]{return r.UpdateWithJacobian(bad,range,area);}));
    PetscCall(ExpectError([&]{return r.Initialize(grid.mesh,{-1,0},Options(),std::sqrt(area));}));
    PetscCall(r.EvaluateWithJacobian(p,value,jac));
    PetscCall(Require(r.HasJacobian() && value==savedValue && jac==savedJac &&
                       r.JacobianCells()==cells,"Failed update/setup changed valid state"));
    const Point invalid{{std::numeric_limits<PetscReal>::quiet_NaN(),0}};
    PetscCall(ExpectError([&]{return r.EvaluateWithJacobian(invalid,value,jac);}));
    PetscCall(ExpectError([&]{return r.EvaluateDerivativeWithJacobian(p,-1,0,value,jac);}));
    PetscCall(ExpectError([&]{return r.EvaluateWithJacobian(p,value,jac,static_cast<Mode>(99));}));
    PetscCall(Require(value==savedValue && jac==savedJac,"Point error changed outputs"));
    PetscCall(ExpectError([&]{return r.ApplyJacobian(p,{1,2},action);}));
    std::vector<PetscReal> direction(cells.size(),1);
    direction[0]=std::numeric_limits<PetscReal>::infinity();
    PetscCall(ExpectError([&]{return r.ApplyJacobian(p,direction,action);}));
    PetscCall(Require(action==74,"Direction error changed output"));
    std::vector<PetscReal> values{2,3}; matrix={{4,5}};
    PetscCall(ExpectError([&]{return r.EvaluateWithJacobian(std::vector<Point>{p,invalid},values,matrix);}));
    PetscCall(Require(values==std::vector<PetscReal>{2,3} && matrix==Matrix{{4,5}},
                       "Batch error partially published outputs"));
    Reconstruction copied=r;
    u[Index(grid.size,{4,4})]+=.03;
    PetscCall(copied.UpdateWithJacobian(u,range,area));
    PetscCall(r.EvaluateWithJacobian(p,value,jac));
    PetscCall(Require(value==savedValue && jac==savedJac,"Copy shares mutable Jacobian state"));
    Reconstruction moved(std::move(r));
    PetscCall(Require(!r.IsInitialized() && !r.HasJacobian() && r.JacobianCells().empty(),
                       "Moved-from object retains Jacobian state"));
    r=std::move(moved);
    PetscCall(r.EvaluateWithJacobian(p,value,jac));
    PetscCall(Require(value==savedValue && jac==savedJac,"Move lost Jacobian state"));
    PetscCall(r.Update(u,range,area));
    PetscCall(Require(!r.HasJacobian(),"Update retained a stale Jacobian"));
    PetscCall(ExpectError([&]{return r.EvaluateWithJacobian(p,value,jac);}));
    PetscCall(r.PrepareJacobian());
    PetscCall(r.EvaluateWithJacobian(p,value,jac));
    PetscReal value2; std::vector<PetscReal> jac2;
    PetscCall(copied.EvaluateWithJacobian(p,value2,jac2));
    PetscCall(Require(value==value2 && jac==jac2,"PrepareJacobian differs from UpdateWithJacobian"));
    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode ConstantsScalesAndTaylor()
{
    PetscFunctionBeginUser;
    Grid grid; PetscCall(MakeGrid({10,9},true,grid));
    const MeshRange range{{0,0},grid.size}; const PetscReal area=.13*.19;
    const MeshIndex target{4,4}; const Point point{{.62,.87}};
    for(PetscInt kind=0;kind<3;++kind) {
        auto o=Options();
        if(kind==1) {o.large.offsets.clear();o.large.linearWeights.clear();
                     o.small.offsets.clear();o.small.linearWeights.clear();}
        if(kind==2) {o.useConstant=false;o.large.offsets={{-1,-1}};
                     o.large.linearWeights.clear();o.small.offsets.clear();o.small.linearWeights.clear();}
        Reconstruction r; PetscCall(r.Initialize(grid.mesh,target,o,std::sqrt(area)));
        auto u=Data(grid.size); std::fill(u.begin(),u.end(),7);
        PetscCall(r.UpdateWithJacobian(u,range,area));
        Matrix weights; PetscCall(r.WeightJacobian(weights));
        for(const auto& row:weights) for(PetscReal entry:row)
            PetscCall(Require(entry==0,"Constant data has nonzero weight sensitivity"));
        PetscReal value,action; std::vector<PetscReal> full,frozen;
        PetscCall(r.EvaluateWithJacobian(point,value,full));
        PetscCall(Require(value==7,"Constant reconstruction changed"));
        PetscCall(r.EvaluateWithJacobian(point,value,frozen,Mode::FrozenWeights));
        PetscCall(Require(full==frozen,"Constant full/frozen Jacobians differ"));
        std::vector<PetscReal> ones(r.JacobianCells().size(),1);
        PetscCall(r.ApplyJacobian(point,ones,action));
        PetscCall(Require(action==1,"Constant direction must give J*v=1"));
        PetscCall(r.ApplyDerivativeJacobian(point,1,0,ones,action));
        PetscCall(Require(action==0,"Constant direction must give derivative J*v=0"));
        PetscCall(r.EvaluateDerivativeWithJacobian(point,10,10,value,full));
        PetscCall(Require(value==0,"Degree-exceeding derivative is nonzero"));
        for(PetscReal entry:full) PetscCall(Require(entry==0,"Degree-exceeding Jacobian is nonzero"));
        if(kind==1) PetscCall(Require(r.JacobianCells().size()==1 && frozen[0]==1,
                                     "Constant-only Jacobian is not a unit target entry"));
    }
    // Exercise log normalization where epsilon*area is outside double range.
    for(bool tiny:{true,false}) {
        auto o=Options(); o.epsilon=tiny?1e-300:1e300;
        Reconstruction r; PetscCall(r.Initialize(grid.mesh,target,o,std::sqrt(area)));
        auto u=Data(grid.size);
        if(tiny) for(auto& entry:u) entry=(entry-.5)*1e-150;
        PetscCall(r.UpdateWithJacobian(u,range,tiny?1e-300:1e300));
        PetscReal value,action; std::vector<PetscReal> jac;
        PetscCall(r.EvaluateWithJacobian(point,value,jac));
        std::vector<PetscReal> direction(jac.size());
        for(std::size_t k=0;k<direction.size();++k) direction[k]=.1*std::cos(PetscReal(k));
        PetscCall(r.ApplyJacobian(point,direction,action));
        long double product=0;
        for(std::size_t k=0;k<jac.size();++k) product+=jac[k]*direction[k];
        PetscCall(Near(action,static_cast<PetscReal>(product),"Extreme-scale J*v",2e-8));
    }
    Reconstruction r; PetscCall(r.Initialize(grid.mesh,target,Options(),std::sqrt(area)));
    auto u=Data(grid.size); PetscCall(r.UpdateWithJacobian(u,range,area));
    std::vector<PetscReal> direction(r.JacobianCells().size());
    for(std::size_t k=0;k<direction.size();++k) direction[k]=.2*std::sin(.53*(k+1));
    PetscReal value,action; PetscCall(r.Evaluate(point,value));
    PetscCall(r.ApplyJacobian(point,direction,action));
    std::array<PetscReal,3> error{};
    for(std::size_t level=0;level<error.size();++level) {
        const PetscReal step=.004/std::pow(2.,static_cast<PetscReal>(level));
        auto shifted=u;
        for(std::size_t k=0;k<direction.size();++k)
            shifted[Index(grid.size,r.JacobianCells()[k])]+=step*direction[k];
        Reconstruction perturbed=r; PetscCall(perturbed.Update(shifted,range,area));
        PetscReal result; PetscCall(perturbed.Evaluate(point,result));
        error[level]=std::abs(result-value-step*action);
    }
    PetscCall(Require(error[0]>1e-12 && error[1]/error[0]<.35 && error[2]/error[1]<.35,
                       "Taylor remainder did not decay quadratically"));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD,"Taylor remainder ratios: %.6g, %.6g\n",
                          static_cast<double>(error[1]/error[0]),static_cast<double>(error[2]/error[1])));
    PetscFunctionReturn(PETSC_SUCCESS);
}
PetscErrorCode Run()
{
    PetscFunctionBeginUser;
    PetscMPIInt ranks=0; PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD,&ranks));
    PetscCall(Require(ranks==1,"These Jacobian tests are serial"));
    PetscInt selection=-1;
    PetscCall(PetscOptionsGetInt(nullptr,nullptr,"-jacobian_case",&selection,nullptr));
    PetscCall(Require(selection>=-1 && selection<=4,"Invalid Jacobian test selector"));
    if(selection==-1 || selection==0) PetscCall(Geometries(false));
    if(selection==-1 || selection==1) PetscCall(Geometries(true));
    if(selection==-1 || selection==2) PetscCall(Pseudo1D());
    if(selection==-1 || selection==3) PetscCall(StateAndErrors());
    if(selection==-1 || selection==4) PetscCall(ConstantsScalesAndTaylor());
    PetscCall(PetscPrintf(PETSC_COMM_WORLD,
        "PASS: %zu point Jacobian entries, %zu weight Jacobian entries; "
        "max scaled errors %.6g, %.6g; direct J*v error %.6g\n",
        stats.pointEntries,stats.weightEntries,static_cast<double>(stats.pointError),
        static_cast<double>(stats.weightError),static_cast<double>(stats.actionError)));
    PetscFunctionReturn(PETSC_SUCCESS);
}
}

int main(int argc,char** argv)
{
    PetscErrorCode error=PetscInitialize(&argc,&argv,nullptr,nullptr);
    if(error) return static_cast<int>(error);
    try {error=Run();}
    catch(const std::exception& exception) {
        (void)PetscPrintf(PETSC_COMM_WORLD,"Jacobian test exception: %s\n",exception.what());
        error=PETSC_ERR_PLIB;
    }
    const PetscErrorCode finalized=PetscFinalize();
    return static_cast<int>(error?error:finalized);
}
