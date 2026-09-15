#include "boundary_conditions.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <utility>
#include <vector>

namespace {

constexpr PetscReal tolerance = 131072 * PETSC_MACHINE_EPSILON;
enum class Field { Stokes, Darcy };

PetscErrorCode Require(bool condition, const char* message)
{
    PetscFunctionBeginUser;
    PetscCheck(condition, PETSC_COMM_SELF, PETSC_ERR_PLIB, "%s", message);
    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode Near(PetscScalar actual, PetscScalar expected, const char* message)
{
    PetscFunctionBeginUser;
    PetscCheck(!PetscIsInfOrNanScalar(actual) && !PetscIsInfOrNanScalar(expected) &&
               PetscAbsScalar(actual-expected) <= tolerance*std::max(PetscReal(1),PetscAbsScalar(expected)),
               PETSC_COMM_SELF, PETSC_ERR_PLIB,
               "%s: got %.17g%+.17gi, expected %.17g%+.17gi", message,
               static_cast<double>(PetscRealPart(actual)), static_cast<double>(PetscImaginaryPart(actual)),
               static_cast<double>(PetscRealPart(expected)), static_cast<double>(PetscImaginaryPart(expected)));
    PetscFunctionReturn(PETSC_SUCCESS);
}

template<class Function>
PetscErrorCode ExpectError(Function&& function, PetscErrorCode expected, const char* message)
{
    PetscFunctionBeginUser;
    PetscCall(PetscPushErrorHandler(PetscReturnErrorHandler, nullptr));
    const PetscErrorCode error = function();
    PetscCall(PetscPopErrorHandler());
    const int wrong = error == expected ? 0 : 1;
    int anyWrong = 0;
    PetscCallMPI(MPI_Allreduce(&wrong, &anyWrong, 1, MPI_INT, MPI_MAX, PETSC_COMM_WORLD));
    PetscCheck(!anyWrong, PETSC_COMM_WORLD, PETSC_ERR_PLIB,
               "%s: expected error %d on every rank; this rank returned %d",
               message, static_cast<int>(expected), static_cast<int>(error));
    PetscFunctionReturn(PETSC_SUCCESS);
}

MeshParam Domain()
{
    MeshParam p;
    p.xstart = -0.3; p.ystart = 0.1; p.L = 2.0; p.H = 1.2;
    p.perturbation = 0.20; p.seed = 817;
    return p;
}

PetscErrorCode MakeMesh(MPI_Comm comm, PetscInt nx, PetscInt ny, PetscInt px, PetscInt py,
                        PetscInt kind, DM& dm, MeshInfo& mesh)
{
    PetscFunctionBeginUser;
    Vec coordinates = nullptr;
    PetscCall(DMDACreate2d(comm, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE, DMDA_STENCIL_BOX,
                          nx, ny, px, py, 2, 1, nullptr, nullptr, &dm));
    PetscCall(DMSetUp(dm));
    PetscCall(DMCreateGlobalVector(dm, &coordinates));
    if (kind == 0) PetscCall(CreateFullMesh(dm, coordinates, Domain()));
    else if (kind == 2) PetscCall(RefineMesh(dm, coordinates, Domain()));
    else PetscCall(LogicRectMesh(dm, coordinates, Domain()));
    if (kind == 3) {
        // A nonsingular affine transform also slants the OUTER edges. This
        // exposes missing normal cross terms hidden by interior perturbations.
        PetscInt xs=0,ys=0,xm=0,ym=0;
        PetscCall(DMDAGetCorners(dm,&xs,&ys,nullptr,&xm,&ym,nullptr));
        PetscScalar*** values = nullptr;
        PetscCall(DMDAVecGetArrayDOF(dm,coordinates,&values));
        for (PetscInt j=ys;j<ys+ym;++j) for (PetscInt i=xs;i<xs+xm;++i) {
            const PetscScalar x=values[j][i][0], y=values[j][i][1];
            values[j][i][0]=x+PetscReal(0.3)*y;
            values[j][i][1]=y+PetscReal(0.2)*x;
        }
        PetscCall(DMDAVecRestoreArrayDOF(dm,coordinates,&values));
    }
    PetscCall(BuildMeshInfo(dm,coordinates,mesh));
    PetscCall(VecDestroy(&coordinates));
    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscReal Dot(const Point& a, const Point& b)
{ return a.p[0]*b.p[0]+a.p[1]*b.p[1]; }

Point Velocity(const Point& p, PetscReal time)
{
    const auto x=p.p[0], y=p.p[1];
    // Quartic data distinguish flux-moment interpolation from midpoint data.
    return {{1+time+x+PetscReal(0.2)*x*x+PetscReal(0.03)*y*y*y*y,
             PetscReal(-0.4)+PetscReal(0.2)*time+2*y+PetscReal(0.15)*x*x*x*x+PetscReal(0.3)*y*y}};
}

Point Traction(const Point& p, PetscReal time)
{ return {{2+time+PetscReal(0.1)*p.p[0]*p.p[0], -3+PetscReal(0.2)*time+PetscReal(0.2)*p.p[1]*p.p[1]}}; }

PetscReal Pressure(const Point& p, PetscReal time)
{ return PetscReal(0.8)+PetscReal(0.2)*p.p[0]-PetscReal(0.1)*p.p[1]+PetscReal(0.15)*time; }

PetscReal Phi(const Point& p)
{ return PetscReal(0.22)+PetscReal(0.013)*p.p[0]+PetscReal(0.017)*p.p[1]; }

Point Force(const Point& p) { return {{1+p.p[0],-2+p.p[1]}}; }

Point Centroid(const QuadVertices& q)
{
    PetscReal twiceArea=0,cx=0,cy=0;
    for (std::size_t k=0;k<4;++k) {
        const auto& a=q[k]; const auto& b=q[(k+1)%4];
        const auto cross=a.p[0]*b.p[1]-b.p[0]*a.p[1];
        twiceArea+=cross; cx+=(a.p[0]+b.p[0])*cross; cy+=(a.p[1]+b.p[1])*cross;
    }
    return {{cx/(3*twiceArea),cy/(3*twiceArea)}};
}

struct PorositySource {
    const MeshInfo& mesh;
    const GaussRule1D& cellRule;
    const GaussRule1D& edgeRule;
    PetscReal uniform = -1;
    PetscReal averageOverride = -1;
    std::vector<int>* visits = nullptr;

    PetscErrorCode operator()(MeshIndex cell, LocalPorositySamples& samples) const
    {
        PetscFunctionBeginUser;
        PetscCall(Require(mesh.OwnsCell(cell), "Porosity requested from a nonowned cell"));
        PetscCall(Require(samples.cell.empty(), "Porosity callback did not receive fresh cell storage"));
        for (const auto& edge : samples.edge)
            PetscCall(Require(edge.empty(), "Porosity callback did not receive fresh edge storage"));
        PetscCall(Require(PetscIsInfOrNanReal(samples.average), "Porosity average was not initially unset"));
        PetscInt id=0; PetscCall(mesh.CellId(cell,id));
        if (visits) ++visits->at(static_cast<std::size_t>(id));
        QuadVertices corners; PetscCall(mesh.GetCellCorners(cell,corners));
        const auto value=[&](const Point& p) -> PetscReal { return uniform < 0 ? Phi(p) : uniform; };
        const auto nc=cellRule.points.size(), ne=edgeRule.points.size();
        samples.cell.resize(nc*nc);
        for (std::size_t j=0;j<nc;++j) for (std::size_t i=0;i<nc;++i)
            samples.cell[j*nc+i]=value(MapCellPoint({{cellRule.points[i],cellRule.points[j]}},corners));
        for (std::size_t e=0;e<4;++e) {
            samples.edge[e].resize(ne);
            for (std::size_t i=0;i<ne;++i)
                samples.edge[e][i]=value(MapEdgePoint(edgeRule.points[i],{corners[e],corners[(e+1)%4]}));
        }
        samples.average=averageOverride >= 0 ? averageOverride : value(Centroid(corners));
        PetscFunctionReturn(PETSC_SUCCESS);
    }
};

std::pair<PetscInt,PetscInt> Segment(const MeshInfo& mesh)
{
    const auto n=mesh.CellDimensions().i;
    return n > 2 ? std::make_pair(PetscInt(1),n-1) : std::make_pair(PetscInt(0),n);
}

// These test cases are classified directly, independently of the production
// rule-matching function. Modes: all D, all N, side/component mix, segment.
bool IsEssential(Field field, int mode, CellSide side, PetscInt index, int component,
                  const MeshInfo& mesh)
{
    if (mode==0 || mode==4) return true;
    if (mode==1) return false;
    if (mode==2) {
        if (field==Field::Darcy) return side==CellSide::Bottom || side==CellSide::Left;
        return component==0 ? (side==CellSide::Left || side==CellSide::Right) : side==CellSide::Bottom;
    }
    const auto segment=Segment(mesh);
    return side==CellSide::Top && index>=segment.first && index<segment.second &&
           (field==Field::Darcy || component==0);
}

StokesBoundarySpecification StokesSpec(const MeshInfo& mesh, int mode)
{
    StokesBoundarySpecification spec;
    for (int e=0;e<4;++e) {
        StokesBoundaryRule rule; rule.region.side=static_cast<CellSide>(e);
        for (int d=0;d<2;++d) {
            const bool essential=IsEssential(Field::Stokes,mode==3?1:mode,rule.region.side,0,d,mesh);
            auto& condition=rule.component[d];
            condition.type=essential?BoundaryType::Dirichlet:BoundaryType::Neumann;
            if (mode!=4) condition.value.function=[d,essential](const BoundaryPoint& p) -> PetscReal {
                return essential?Velocity(p.position,p.time).p[d]:Traction(p.position,p.time).p[d];
            };
        }
        spec.rules.push_back(rule);
    }
    if (mode==3) {
        StokesBoundaryRule override;
        override.region.side=CellSide::Top;
        const auto segment=Segment(mesh);
        override.region.firstEdge=segment.first; override.region.endEdge=segment.second;
        override.region.priority=10;
        override.component[0].type=BoundaryType::Dirichlet;
        override.component[0].value.function=[](const BoundaryPoint& p) -> PetscReal { return Velocity(p.position,p.time).p[0]; };
        spec.rules.push_back(override);
        // Reversing the rules ensures the result is not just last-write-wins.
        std::reverse(spec.rules.begin(),spec.rules.end());
    }
    return spec;
}

DarcyBoundarySpecification DarcySpec(const MeshInfo& mesh, int mode,
                                     DarcyNeumannVariable variable, PetscReal sign)
{
    DarcyBoundarySpecification spec; spec.neumannVariable=variable; spec.pressureSign=sign;
    for (int e=0;e<4;++e) {
        DarcyBoundaryRule rule; rule.region.side=static_cast<CellSide>(e);
        const bool essential=IsEssential(Field::Darcy,mode==3?1:mode,rule.region.side,0,0,mesh);
        rule.condition.type=essential?BoundaryType::Dirichlet:BoundaryType::Neumann;
        if (mode!=4) rule.condition.value.function=[essential](const BoundaryPoint& p) -> PetscReal {
            return essential?Dot(Velocity(p.position,p.time),p.outwardNormal):Pressure(p.position,p.time);
        };
        spec.rules.push_back(rule);
    }
    if (mode==3) {
        DarcyBoundaryRule override;
        override.region.side=CellSide::Top;
        const auto segment=Segment(mesh);
        override.region.firstEdge=segment.first; override.region.endEdge=segment.second;
        override.region.priority=10;
        override.condition.type=BoundaryType::Dirichlet;
        override.condition.value.function=[](const BoundaryPoint& p) -> PetscReal { return Dot(Velocity(p.position,p.time),p.outwardNormal); };
        spec.rules.push_back(override);
        std::reverse(spec.rules.begin(),spec.rules.end());
    }
    return spec;
}

struct Reference {
    std::vector<int> constrained;        // Natural numbering, full velocity space.
    std::vector<PetscScalar> values,load;
};

PetscErrorCode Prescribe(Reference& ref, PetscInt id, PetscReal value)
{
    PetscFunctionBeginUser;
    const auto k=static_cast<std::size_t>(id);
    if (ref.constrained[k]) PetscCall(Near(ref.values[k],value,"Inconsistent reference endpoint"));
    ref.constrained[k]=1; ref.values[k]=value;
    PetscFunctionReturn(PETSC_SUCCESS);
}

// Five equally spaced points, Boole's rule, exact through degree five. The
// oracle uses analytic polynomial traces, not BR/HDiv Evaluate or Gauss rules.
// All tested edge integrands have degree <=5 (theta=1 for variable porosity).
constexpr std::array<PetscReal,5> meanWeights{{PetscReal(7)/90,PetscReal(32)/90,
    PetscReal(12)/90,PetscReal(32)/90,PetscReal(7)/90}};

PetscErrorCode MakeReference(Field field, const MeshInfo& serial, const DofMap& map,
                             int mode, PetscReal time, DarcyNeumannVariable variable,
                             PetscReal pressureSign, PetscReal theta,
                             PetscReal uniform, PetscReal averageOverride, Reference& ref)
{
    PetscFunctionBeginUser;
    const auto size=static_cast<std::size_t>(map.GlobalDofs());
    ref.constrained.assign(size,0); ref.values.assign(size,0); ref.load.assign(size,0);
    const auto dims=serial.CellDimensions();
    for (int side=0;side<4;++side) {
        const bool horizontal=side==0||side==2;
        const PetscInt count=horizontal?dims.i:dims.j;
        for (PetscInt index=0;index<count;++index) {
            const MeshIndex cell=horizontal?MeshIndex{index,side==0?0:dims.j-1}:MeshIndex{side==3?0:dims.i-1,index};
            QuadVertices corners; PetscCall(serial.GetCellCorners(cell,corners));
            const auto e=static_cast<std::size_t>(side), next=(e+1)%4;
            const Point a=corners[e], b=corners[next];
            const PetscReal dx=b.p[0]-a.p[0],dy=b.p[1]-a.p[1];
            const PetscReal length=PetscSqrtReal(dx*dx+dy*dy);
            const Point normal{{dy/length,-dx/length}};
            const PetscReal sigma=side==0||side==3?-1:1;
            const Point shared{{sigma*normal.p[0],sigma*normal.p[1]}};
            std::vector<PetscInt> ids; PetscCall(map.GetCellNaturalDofs(cell,ids));
            const Point ua=mode==4?Point{}:Velocity(a,time), ub=mode==4?Point{}:Velocity(b,time);
            const std::array<bool,2> essential{{IsEssential(field,mode,static_cast<CellSide>(side),index,0,serial),
                IsEssential(field,mode,static_cast<CellSide>(side),index,1,serial)}};
            PetscReal mean[2]{}, meanFlux=0, linearMoment=0;
            for (std::size_t q=0;q<5;++q) {
                const PetscReal s=PetscReal(q)/4, w=meanWeights[q];
                const Point point{{(1-s)*a.p[0]+s*b.p[0],(1-s)*a.p[1]+s*b.p[1]}};
                const auto u=mode==4?Point{}:Velocity(point,time);
                if (field==Field::Stokes) {
                    const auto traction=Traction(point,time);
                    for (std::size_t d=0;d<2;++d) {
                        mean[d]+=w*u.p[d];
                        if (!essential[d]) {
                            ref.load[ids[e+4*d]]+=length*w*(1-s)*traction.p[d];
                            ref.load[ids[next+4*d]]+=length*w*s*traction.p[d];
                            ref.load[ids[8+e]]+=length*w*4*s*(1-s)*shared.p[d]*traction.p[d];
                        }
                    }
                } else if (essential[0]) {
                    const auto flux=Dot(u,normal);
                    meanFlux+=w*flux; linearMoment+=w*flux*(1-2*s);
                } else {
                    PetscReal coefficient=1;
                    if (variable!=DarcyNeumannVariable::WeightedNormalLoad) {
                        const auto phi=uniform<0?Phi(point):uniform;
                        coefficient=std::pow(phi,1+theta);
                        if (variable==DarcyNeumannVariable::AssembledPressure) {
                            const auto average=averageOverride>=0?averageOverride:uniform<0?Phi(Centroid(corners)):uniform;
                            if (average!=0) coefficient/=PetscSqrtReal(average);
                        }
                    }
                    const auto value=length*w*pressureSign*coefficient*Pressure(point,time);
                    ref.load[ids[e]]+=value*(1-2*s); ref.load[ids[e+4]]+=value*sigma;
                }
            }
            if (field==Field::Stokes) {
                for (std::size_t d=0;d<2;++d) if (essential[d]) {
                    PetscCall(Prescribe(ref,ids[e+4*d],ua.p[d]));
                    PetscCall(Prescribe(ref,ids[next+4*d],ub.p[d]));
                }
                if (essential[0]&&essential[1]) {
                    const Point residual{{mean[0]-(ua.p[0]+ub.p[0])/2,mean[1]-(ua.p[1]+ub.p[1])/2}};
                    PetscCall(Prescribe(ref,ids[8+e],PetscReal(1.5)*Dot(residual,shared)));
                } else for (std::size_t d=0;d<2;++d) if (essential[d]&&shared.p[d]!=0)
                    PetscCall(Prescribe(ref,ids[8+e],PetscReal(1.5)*(mean[d]-(ua.p[d]+ub.p[d])/2)/shared.p[d]));
            } else if (essential[0]) {
                PetscCall(Prescribe(ref,ids[e],3*linearMoment));
                PetscCall(Prescribe(ref,ids[e+4],sigma*meanFlux));
            }
        }
    }
    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode ReadVector(Vec vector, const DofMap& map, std::vector<PetscScalar>& values)
{
    PetscFunctionBeginUser;
    PetscInt N=0,n=0,begin=0,end=0;
    PetscCall(VecGetSize(vector,&N)); PetscCall(VecGetLocalSize(vector,&n));
    PetscCall(VecGetOwnershipRange(vector,&begin,&end));
    PetscCall(Require(N==map.GlobalDofs()&&n==map.OwnedDofs()&&begin==map.OwnershipBegin()&&end==map.OwnershipEnd(),
                      "Vec has incorrect global/owned layout"));
    values.resize(static_cast<std::size_t>(n));
    const PetscScalar* array=nullptr; PetscCall(VecGetArrayRead(vector,&array));
    if (n) std::copy(array,array+n,values.begin());
    PetscCall(VecRestoreArrayRead(vector,&array));
    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode ReadMatrix(Mat matrix, const DofMap& rows, const DofMap& columns,
                          std::vector<PetscScalar>& values)
{
    PetscFunctionBeginUser;
    PetscInt M=0,N=0,m=0,n=0,begin=0,end=0,cb=0,ce=0; PetscBool assembled=PETSC_FALSE;
    PetscCall(MatGetSize(matrix,&M,&N)); PetscCall(MatGetLocalSize(matrix,&m,&n));
    PetscCall(MatGetOwnershipRange(matrix,&begin,&end)); PetscCall(MatGetOwnershipRangeColumn(matrix,&cb,&ce));
    PetscCall(MatAssembled(matrix,&assembled));
    PetscCall(Require(assembled&&M==rows.GlobalDofs()&&N==columns.GlobalDofs()&&m==rows.OwnedDofs()&&n==columns.OwnedDofs()&&
                      begin==rows.OwnershipBegin()&&end==rows.OwnershipEnd()&&cb==columns.OwnershipBegin()&&ce==columns.OwnershipEnd(),
                      "Mat has incorrect dimensions, ownership, orientation, or assembled state"));
    std::vector<PetscInt> ids(static_cast<std::size_t>(N)); std::iota(ids.begin(),ids.end(),PetscInt(0));
    values.resize(static_cast<std::size_t>(m)*static_cast<std::size_t>(N));
    for (PetscInt row=begin;row<end;++row)
        PetscCall(MatGetValues(matrix,1,&row,N,ids.data(),values.data()+static_cast<std::size_t>(row-begin)*N));
    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode CheckData(const BoundaryData& data, const DofMap& map, const Reference& ref)
{
    PetscFunctionBeginUser;
    PetscCall(Require(data.naturalLoad&&data.space==map.Space()&&data.globalDofs==map.GlobalDofs()&&
                      data.ownedBegin==map.OwnershipBegin()&&data.ownedEnd==map.OwnershipEnd()&&
                      data.essentialDofs.size()==data.essentialValues.size(),"Boundary metadata is incorrect"));
    std::vector<PetscScalar> load; PetscCall(ReadVector(data.naturalLoad,map,load));
    std::size_t k=0;
    for (PetscInt id=map.OwnershipBegin();id<map.OwnershipEnd();++id) {
        PetscInt natural=0; PetscCall(map.GlobalToNatural(id,natural));
        const auto ni=static_cast<std::size_t>(natural);
        PetscCall(Near(load[static_cast<std::size_t>(id-map.OwnershipBegin())],ref.load[ni],"Natural boundary load vs analytic edge traces"));
        if (!ref.constrained[ni]) continue;
        PetscCall(Require(k<data.essentialDofs.size()&&data.essentialDofs[k]==id,"Missing, duplicate, unordered or nonowned constraint"));
        PetscCall(Near(data.essentialValues[k],ref.values[ni],"Dirichlet coefficient vs independent moment reference")); ++k;
    }
    PetscCall(Require(k==data.essentialDofs.size(),"Unexpected constrained DOF"));
    PetscFunctionReturn(PETSC_SUCCESS);
}

struct Snapshot { std::vector<PetscScalar> A,B,C,f,g; };

PetscErrorCode Capture(const MixedBlocks& blocks, const DofMap& velocity, const DofMap& pressure, Snapshot& s)
{
    PetscFunctionBeginUser;
    PetscCall(ReadMatrix(blocks.A,velocity,velocity,s.A)); PetscCall(ReadMatrix(blocks.B,pressure,velocity,s.B));
    PetscCall(ReadMatrix(blocks.C,pressure,pressure,s.C)); PetscCall(ReadVector(blocks.f,velocity,s.f));
    PetscCall(ReadVector(blocks.g,pressure,s.g));
    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode CheckApplication(const MixedBlocks& raw, const BoundaryData& data,
                               const DofMap& velocity, const DofMap& pressure, const Reference& ref)
{
    PetscFunctionBeginUser;
    Snapshot before; PetscCall(Capture(raw,velocity,pressure,before));
    const auto N=static_cast<std::size_t>(velocity.GlobalDofs());
    std::vector<PetscInt> natural(N);
    for (std::size_t col=0;col<N;++col) PetscCall(velocity.GlobalToNatural(static_cast<PetscInt>(col),natural[col]));
    for (PetscReal sign : {PetscReal(-1),PetscReal(1)}) {
        BoundaryApplicationOptions options; options.pressureRowSign=sign; options.diagonal=PetscReal(1.7);
        MixedBlocks result; PetscCall(ApplyBoundaryConditions(PETSC_COMM_WORLD,raw,data,options,result));
        PetscCall(Require(result.A!=raw.A&&result.B!=raw.B&&result.C!=raw.C&&result.f!=raw.f&&result.g!=raw.g,
                          "Constrained output aliases an original handle"));
        Snapshot after; PetscCall(Capture(result,velocity,pressure,after));
        for (std::size_t row=0;row<before.f.size();++row) {
            const auto global=static_cast<std::size_t>(velocity.OwnershipBegin())+row;
            const auto nr=static_cast<std::size_t>(natural[global]);
            PetscScalar f=before.f[row]+ref.load[nr];
            for (std::size_t col=0;col<N;++col) {
                const auto nc=static_cast<std::size_t>(natural[col]);
                if (ref.constrained[nc]) f-=before.A[row*N+col]*ref.values[nc];
                const PetscScalar expected=ref.constrained[nr]||ref.constrained[nc]
                    ? (global==col?PetscScalar(options.diagonal):PetscScalar(0)) : before.A[row*N+col];
                PetscCall(Near(after.A[row*N+col],expected,"Dirichlet rows/columns of A"));
            }
            if (ref.constrained[nr]) f=options.diagonal*ref.values[nr];
            PetscCall(Near(after.f[row],f,"Momentum RHS lifting, including natural loads"));
        }
        for (std::size_t row=0;row<before.g.size();++row) {
            PetscScalar g=before.g[row];
            for (std::size_t col=0;col<N;++col) {
                const auto nc=static_cast<std::size_t>(natural[col]);
                g-=sign*before.B[row*N+col]*ref.values[nc];
                PetscCall(Near(after.B[row*N+col],ref.constrained[nc]?PetscScalar(0):before.B[row*N+col],"Constrained B columns"));
            }
            PetscCall(Near(after.g[row],g,"Pressure RHS lifting and explicit B sign"));
        }
        PetscCall(Require(after.C==before.C,"Boundary application modified C"));
        Snapshot unchanged; PetscCall(Capture(raw,velocity,pressure,unchanged));
        PetscCall(Require(unchanged.A==before.A&&unchanged.B==before.B&&unchanged.C==before.C&&unchanged.f==before.f&&unchanged.g==before.g,
                          "Boundary application changed its original blocks"));
        PetscCall(CheckData(data,velocity,ref));
        PetscCall(DestroyMixedBlocks(result));
        PetscCall(Require(result.IsEmpty(),"Constrained block cleanup left live handles"));
    }
    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode CheckVisits(const std::vector<int>& visits, const MeshInfo& mesh, int mode, bool weighted)
{
    PetscFunctionBeginUser;
    std::vector<int> total(visits.size());
    PetscCallMPI(MPI_Allreduce(visits.data(),total.data(),static_cast<int>(visits.size()),MPI_INT,MPI_SUM,PETSC_COMM_WORLD));
    const auto dims=mesh.CellDimensions();
    for (PetscInt id=0;id<mesh.CellCount();++id) {
        MeshIndex cell; PetscCall(mesh.CellIndex(id,cell));
        bool natural=false;
        if (cell.j==0) natural|=!IsEssential(Field::Darcy,mode,CellSide::Bottom,cell.i,0,mesh);
        if (cell.i==dims.i-1) natural|=!IsEssential(Field::Darcy,mode,CellSide::Right,cell.j,0,mesh);
        if (cell.j==dims.j-1) natural|=!IsEssential(Field::Darcy,mode,CellSide::Top,cell.i,0,mesh);
        if (cell.i==0) natural|=!IsEssential(Field::Darcy,mode,CellSide::Left,cell.j,0,mesh);
        PetscCall(Require(total[static_cast<std::size_t>(id)]==(weighted&&natural?1:0),
                          "Porosity must be requested exactly once per owned cell with a weighted Neumann edge"));
    }
    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode Assemble(Field field, const MeshInfo& mesh, const DofMap& velocity,
                        const DofMap& pressure, const PorositySource& source,
                        const LocalMatrixParameters& parameters, MixedBlocks& blocks)
{
    PetscFunctionBeginUser;
    if (field==Field::Stokes)
        PetscCall(AssembleStokesBlocks(PETSC_COMM_WORLD,mesh,velocity,pressure,source.cellRule,source,Force,blocks));
    else PetscCall(AssembleDarcyBlocks(PETSC_COMM_WORLD,mesh,velocity,pressure,source.cellRule,source.edgeRule,source,parameters,Force,blocks));
    // A nonzero starting pressure RHS catches accidental overwrite/reset.
    PetscCall(VecSet(blocks.g,field==Field::Stokes?PetscScalar(0.8):PetscScalar(-0.6)));
    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode CheckNumerics(Field field, DM dm, const MeshInfo& mesh, const MeshInfo& serial)
{
    PetscFunctionBeginUser;
    DofMap velocity,pressure;
    PetscCall(velocity.Initialize(dm,mesh,field==Field::Stokes?DofSpace::BRVelocity:DofSpace::HDivVelocity));
    PetscCall(pressure.Initialize(dm,mesh,DofSpace::CellPressure));
    GaussRule1D cellRule,edgeRule; PetscCall(CreateGaussRule(4,cellRule)); PetscCall(CreateGaussRule(4,edgeRule));
    LocalMatrixParameters parameters; parameters.theta=1;
    PorositySource source{mesh,cellRule,edgeRule};
    MixedBlocks raw; PetscCall(Assemble(field,mesh,velocity,pressure,source,parameters,raw));
    BoundaryData data;
    for (int pass=0;pass<2;++pass) {
        const PetscReal time=pass==0?PetscReal(0.2):PetscReal(0.9);
        if (pass) PetscCall(CreateGaussRule(5,edgeRule));
        for (int mode=0;mode<5;++mode) {
            const int variables=field==Field::Darcy&&mode==1?3:1;
            for (int v=0;v<variables;++v) {
                const auto variable=static_cast<DarcyNeumannVariable>(v);
                const PetscReal sign=pass==0?-1:1;
                Reference reference;
                PetscCall(MakeReference(field,serial,velocity,mode,time,variable,sign,parameters.theta,-1,-1,reference));
                std::vector<int> visits(static_cast<std::size_t>(mesh.CellCount()),0);
                if (field==Field::Stokes) {
                    PetscCall(BuildStokesBoundaryData(PETSC_COMM_WORLD,mesh,velocity,edgeRule,StokesSpec(mesh,mode),time,data));
                } else {
                    auto counted=source; counted.visits=&visits;
                    CellPorosityFunction provider=counted;
                    if (variable==DarcyNeumannVariable::WeightedNormalLoad) provider={};
                    PetscCall(BuildDarcyBoundaryData(PETSC_COMM_WORLD,mesh,velocity,edgeRule,
                        DarcySpec(mesh,mode,variable,sign),time,provider,parameters,data));
                    PetscCall(CheckVisits(visits,mesh,mode,variable!=DarcyNeumannVariable::WeightedNormalLoad));
                }
                PetscCall(CheckData(data,velocity,reference));
                // Also applies twice to the SAME original with opposite signs.
                PetscCall(CheckApplication(raw,data,velocity,pressure,reference));
                PetscCall(DestroyBoundaryData(data));
                PetscCall(Require(data.IsEmpty()&&data.globalDofs==0,"Boundary cleanup did not reset output"));
            }
        }
    }
    if (field==Field::Darcy) {
        // Dry edges, exact-zero average fallback, and the legacy pow(0,0)
        // branch. Uniform porosity permits an independent exact reference.
        const PetscReal cases[][3]={{0,0,1},{0,0,-1},{PetscReal(0.25),0,PetscReal(0.5)}};
        for (const auto& c:cases) {
            parameters.theta=c[2]; auto uniform=source; uniform.uniform=c[0]; uniform.averageOverride=c[1];
            const auto spec=DarcySpec(mesh,1,DarcyNeumannVariable::AssembledPressure,-1);
            PetscCall(BuildDarcyBoundaryData(PETSC_COMM_WORLD,mesh,velocity,edgeRule,spec,0,uniform,parameters,data));
            Reference reference; PetscCall(MakeReference(field,serial,velocity,1,0,spec.neumannVariable,-1,c[2],c[0],c[1],reference));
            PetscCall(CheckData(data,velocity,reference)); PetscCall(DestroyBoundaryData(data));
        }
        // A pure essential specification does not need porosity, even when dry.
        PetscCall(BuildDarcyBoundaryData(PETSC_COMM_WORLD,mesh,velocity,edgeRule,
            DarcySpec(mesh,0,DarcyNeumannVariable::AssembledPressure,-1),0,{},parameters,data));
        PetscCall(DestroyBoundaryData(data));
    }
    PetscCall(DestroyBoundaryData(data)); PetscCall(DestroyMixedBlocks(raw));
    if (mesh.CellCount()==1) {
        const auto size=mesh.OwnedCells().Size(); const int empty=size.i*size.j==0?1:0;
        int total=0; PetscMPIInt ranks=0; PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD,&ranks));
        PetscCallMPI(MPI_Allreduce(&empty,&total,1,MPI_INT,MPI_SUM,PETSC_COMM_WORLD));
        PetscCall(Require(total==ranks-1,"Minimal case did not exercise empty cell owners"));
        if (field==Field::Darcy&&ranks==4) {
            const int zero=velocity.OwnedDofs()==0?1:0;
            PetscCallMPI(MPI_Allreduce(&zero,&total,1,MPI_INT,MPI_SUM,PETSC_COMM_WORLD));
            PetscCall(Require(total==1,"Minimal 2x2 layout did not exercise an empty HDiv owner"));
        }
    }
    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode CheckErrors(DM dm, const MeshInfo& mesh, const MeshInfo& serial)
{
    PetscFunctionBeginUser;
    PetscMPIInt rank=0,ranks=0;
    PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD,&rank)); PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD,&ranks));
    DofMap stokes,darcy,pressure,uninitialized;
    PetscCall(stokes.Initialize(dm,mesh,DofSpace::BRVelocity));
    PetscCall(darcy.Initialize(dm,mesh,DofSpace::HDivVelocity));
    PetscCall(pressure.Initialize(dm,mesh,DofSpace::CellPressure));
    GaussRule1D rule,one; PetscCall(CreateGaussRule(4,rule)); PetscCall(CreateGaussRule(1,one));
    const auto goodStokes=StokesSpec(mesh,0);
    const auto goodDarcy=DarcySpec(mesh,1,DarcyNeumannVariable::AssembledPressure,-1);
    LocalMatrixParameters parameters; parameters.theta=1;
    PorositySource source{mesh,rule,rule};
    BoundaryData empty,live;
    const auto buildStokes=[&](const StokesBoundarySpecification& spec,BoundaryData& out) -> PetscErrorCode {
        return BuildStokesBoundaryData(PETSC_COMM_WORLD,mesh,stokes,rule,spec,0,out);
    };
    const auto buildDarcy=[&](const DarcyBoundarySpecification& spec,const CellPorosityFunction& provider,
                              const LocalMatrixParameters& param,BoundaryData& out) -> PetscErrorCode {
        return BuildDarcyBoundaryData(PETSC_COMM_WORLD,mesh,darcy,rule,spec,0,provider,param,out);
    };
    PetscCall(ExpectError([&]() -> PetscErrorCode {
        return BuildStokesBoundaryData(PETSC_COMM_WORLD,mesh,uninitialized,rule,goodStokes,0,empty);
    },PETSC_ERR_ARG_WRONG,"Uninitialized map"));
    PetscCall(ExpectError([&]() -> PetscErrorCode {
        return BuildStokesBoundaryData(PETSC_COMM_WORLD,mesh,darcy,rule,goodStokes,0,empty);
    },PETSC_ERR_ARG_INCOMP,"Wrong velocity space"));
    PetscCall(ExpectError([&]() -> PetscErrorCode {
        return BuildStokesBoundaryData(PETSC_COMM_WORLD,mesh,stokes,rank==0?one:rule,goodStokes,0,empty);
    },PETSC_ERR_ARG_SIZ,"One-point edge quadrature on one rank"));
    PetscCall(ExpectError([&]() -> PetscErrorCode {
        return BuildStokesBoundaryData(PETSC_COMM_WORLD,mesh,stokes,rule,goodStokes,
            rank==0?std::numeric_limits<PetscReal>::quiet_NaN():PetscReal(0),empty);
    },PETSC_ERR_ARG_OUTOFRANGE,"Nonfinite time"));

    auto spec=goodStokes;
    if (rank==0) spec.rules[0].component[0].type=BoundaryType::Unspecified;
    PetscCall(ExpectError([&]() -> PetscErrorCode { return buildStokes(spec,empty); },PETSC_ERR_ARG_INCOMP,"Uncovered Stokes component"));
    spec=goodStokes;
    if (rank==0) spec.rules.push_back(spec.rules[0]);
    PetscCall(ExpectError([&]() -> PetscErrorCode { return buildStokes(spec,empty); },PETSC_ERR_ARG_INCOMP,"Ambiguous overlapping rules"));
    spec=goodStokes;
    if (rank==0) spec.rules[0].region.firstEdge=-1;
    PetscCall(ExpectError([&]() -> PetscErrorCode { return buildStokes(spec,empty); },PETSC_ERR_ARG_OUTOFRANGE,"Invalid segment"));
    spec=goodStokes;
    if (rank==0) spec.rules[0].region.side=static_cast<CellSide>(7);
    PetscCall(ExpectError([&]() -> PetscErrorCode { return buildStokes(spec,empty); },PETSC_ERR_ARG_OUTOFRANGE,"Invalid side"));
    spec=goodStokes;
    if (rank==0) spec.rules[0].component[0].type=static_cast<BoundaryType>(7);
    PetscCall(ExpectError([&]() -> PetscErrorCode { return buildStokes(spec,empty); },PETSC_ERR_ARG_OUTOFRANGE,"Unsupported boundary type"));
    spec=goodStokes;
    if (rank==0) {
        spec.rules[0].component[0].value.function={};
        spec.rules[0].component[0].value.constant=99;
    }
    PetscCall(ExpectError([&]() -> PetscErrorCode { return buildStokes(spec,empty); },PETSC_ERR_ARG_INCOMP,"Conflicting corner data"));
    spec=goodStokes;
    if (rank==0) spec.rules[0].component[0].value.function=[](const BoundaryPoint&) -> PetscReal {
        return std::numeric_limits<PetscReal>::quiet_NaN();
    };
    PetscCall(ExpectError([&]() -> PetscErrorCode { return buildStokes(spec,empty); },PETSC_ERR_FP,"Nonfinite callback"));
    spec=goodStokes;
    // Throwing callbacks have no PetscFunctionBegin frame to unwind.
    if (rank==0) spec.rules[0].component[0].value.function=[](const BoundaryPoint&) -> PetscReal {
        throw std::runtime_error("intentional boundary test exception");
    };
    PetscCall(ExpectError([&]() -> PetscErrorCode { return buildStokes(spec,empty); },PETSC_ERR_USER,"Thrown callback error"));
    PetscCall(Require(empty.IsEmpty(),"Failed construction published partial boundary data"));

    // A live output must survive rejected reuse unchanged, even when only one
    // rank passes that live output and its peers pass an empty object.
    PetscCall(buildStokes(goodStokes,live));
    Reference reference;
    PetscCall(MakeReference(Field::Stokes,serial,stokes,0,0,DarcyNeumannVariable::AssembledPressure,-1,1,-1,-1,reference));
    const Vec liveLoad=live.naturalLoad;
    const auto liveDofs=live.essentialDofs; const auto liveValues=live.essentialValues;
    PetscCall(ExpectError([&]() -> PetscErrorCode { return buildStokes(goodStokes,rank==0?live:empty); },
                          PETSC_ERR_ARG_WRONG,"Nonempty boundary output"));
    PetscCall(Require(live.naturalLoad==liveLoad&&live.essentialDofs==liveDofs&&live.essentialValues==liveValues,
                      "Rejected boundary output reuse changed the original data"));
    PetscCall(CheckData(live,stokes,reference));

    PetscCall(ExpectError([&]() -> PetscErrorCode { return buildDarcy(goodDarcy,{},parameters,empty); },
                          PETSC_ERR_ARG_NULL,"Missing weighted Darcy porosity provider"));
    CellPorosityFunction provider=[&](MeshIndex cell,LocalPorositySamples& samples) -> PetscErrorCode {
        if (rank==0) return static_cast<PetscErrorCode>(PETSC_ERR_USER);
        return source(cell,samples);
    };
    PetscCall(ExpectError([&]() -> PetscErrorCode { return buildDarcy(goodDarcy,provider,parameters,empty); },
                          PETSC_ERR_USER,"Returned provider error on one rank"));
    // Callbacks containing PetscCall use BeginUser/Return, including failures.
    provider=[&](MeshIndex cell,LocalPorositySamples& samples) -> PetscErrorCode {
        PetscFunctionBeginUser;
        PetscCall(source(cell,samples));
        if (rank==0) for (auto& edge:samples.edge) edge.clear();
        PetscFunctionReturn(PETSC_SUCCESS);
    };
    PetscCall(ExpectError([&]() -> PetscErrorCode { return buildDarcy(goodDarcy,provider,parameters,empty); },
                          PETSC_ERR_ARG_SIZ,"Incorrect edge sample count"));
    provider=[&](MeshIndex cell,LocalPorositySamples& samples) -> PetscErrorCode {
        PetscFunctionBeginUser;
        PetscCall(source(cell,samples));
        if (rank==0) samples.average=std::numeric_limits<PetscReal>::quiet_NaN();
        PetscFunctionReturn(PETSC_SUCCESS);
    };
    PetscCall(ExpectError([&]() -> PetscErrorCode { return buildDarcy(goodDarcy,provider,parameters,empty); },
                          PETSC_ERR_ARG_OUTOFRANGE,"Unset pressure-scaling average"));
    provider=[&](MeshIndex cell,LocalPorositySamples& samples) -> PetscErrorCode {
        PetscFunctionBeginUser;
        PetscCall(source(cell,samples));
        if (rank==0) for (auto& edge:samples.edge) std::fill(edge.begin(),edge.end(),PetscReal(1.1));
        PetscFunctionReturn(PETSC_SUCCESS);
    };
    PetscCall(ExpectError([&]() -> PetscErrorCode { return buildDarcy(goodDarcy,provider,parameters,empty); },
                          PETSC_ERR_ARG_OUTOFRANGE,"Out-of-range edge porosity"));
    auto badParameters=parameters; if (rank==0) badParameters.theta=-2;
    auto dry=source; dry.uniform=0;
    PetscCall(ExpectError([&]() -> PetscErrorCode { return buildDarcy(goodDarcy,dry,badParameters,empty); },
                          PETSC_ERR_ARG_OUTOFRANGE,"Negative porosity power on dry edges"));
    auto ds=goodDarcy; if (rank==0) ds.pressureSign=0;
    PetscCall(ExpectError([&]() -> PetscErrorCode { return buildDarcy(ds,source,parameters,empty); },
                          PETSC_ERR_ARG_OUTOFRANGE,"Invalid Darcy natural sign"));
    ds=goodDarcy; if (rank==0) ds.neumannVariable=static_cast<DarcyNeumannVariable>(7);
    PetscCall(ExpectError([&]() -> PetscErrorCode { return buildDarcy(ds,source,parameters,empty); },
                          PETSC_ERR_ARG_OUTOFRANGE,"Invalid Darcy pressure variable"));
    PetscCall(Require(empty.IsEmpty(),"Failed Darcy construction left partial output"));

    // PressurePotential intentionally does not need a cell average; its edge
    // weighting remains well defined. WeightedNormalLoad needs no provider.
    provider=[&](MeshIndex cell,LocalPorositySamples& samples) -> PetscErrorCode {
        PetscFunctionBeginUser;
        PetscCall(source(cell,samples)); samples.average=std::numeric_limits<PetscReal>::quiet_NaN();
        PetscFunctionReturn(PETSC_SUCCESS);
    };
    ds=goodDarcy; ds.neumannVariable=DarcyNeumannVariable::PressurePotential;
    PetscCall(buildDarcy(ds,provider,parameters,empty));
    Reference dr; PetscCall(MakeReference(Field::Darcy,serial,darcy,1,0,ds.neumannVariable,-1,1,-1,-1,dr));
    PetscCall(CheckData(empty,darcy,dr)); PetscCall(DestroyBoundaryData(empty));
    ds.neumannVariable=DarcyNeumannVariable::WeightedNormalLoad;
    PetscCall(buildDarcy(ds,{},parameters,empty)); PetscCall(DestroyBoundaryData(empty));

    MixedBlocks raw,constrained;
    PetscCall(Assemble(Field::Stokes,mesh,stokes,pressure,source,parameters,raw));
    Snapshot before; PetscCall(Capture(raw,stokes,pressure,before));
    BoundaryApplicationOptions options;
    PetscCall(ExpectError([&]() -> PetscErrorCode { return ApplyBoundaryConditions(PETSC_COMM_WORLD,raw,live,options,constrained); },
                          PETSC_ERR_ARG_OUTOFRANGE,"Pressure-row sign must be explicit"));
    options.pressureRowSign=-1; if (rank==0) options.diagonal=0;
    PetscCall(ExpectError([&]() -> PetscErrorCode { return ApplyBoundaryConditions(PETSC_COMM_WORLD,raw,live,options,constrained); },
                          PETSC_ERR_ARG_OUTOFRANGE,"Nonpositive elimination diagonal"));
    options.diagonal=1;
    if (ranks>1) {
        options.diagonal=rank==0?2:1;
        PetscCall(ExpectError([&]() -> PetscErrorCode { return ApplyBoundaryConditions(PETSC_COMM_WORLD,raw,live,options,constrained); },
                              PETSC_ERR_ARG_INCOMP,"Inconsistent application options across ranks"));
        options.diagonal=1;
    }
    if (rank==0) live.essentialValues[0]=std::numeric_limits<PetscReal>::quiet_NaN();
    PetscCall(ExpectError([&]() -> PetscErrorCode { return ApplyBoundaryConditions(PETSC_COMM_WORLD,raw,live,options,constrained); },
                          PETSC_ERR_ARG_INCOMP,"Corrupted essential value"));
    live.essentialValues=liveValues;
    PetscCall(Require(constrained.IsEmpty(),"Failed application published partial blocks"));
    PetscCall(ApplyBoundaryConditions(PETSC_COMM_WORLD,raw,live,options,constrained));
    Snapshot applied; PetscCall(Capture(constrained,stokes,pressure,applied));
    PetscCall(ExpectError([&]() -> PetscErrorCode { return ApplyBoundaryConditions(PETSC_COMM_WORLD,raw,live,options,constrained); },
                          PETSC_ERR_ARG_WRONG,"Nonempty constrained output"));
    Snapshot unchanged; PetscCall(Capture(constrained,stokes,pressure,unchanged));
    PetscCall(Require(unchanged.A==applied.A&&unchanged.B==applied.B&&unchanged.C==applied.C&&unchanged.f==applied.f&&unchanged.g==applied.g,
                      "Rejected reuse changed constrained blocks"));
    PetscCall(Capture(raw,stokes,pressure,unchanged));
    PetscCall(Require(unchanged.A==before.A&&unchanged.B==before.B&&unchanged.C==before.C&&unchanged.f==before.f&&unchanged.g==before.g,
                      "Error/recovery modified original assembly"));
    PetscCall(CheckData(live,stokes,reference));
    PetscCall(DestroyMixedBlocks(constrained));
    PetscCall(CheckApplication(raw,live,stokes,pressure,reference));
    PetscCall(DestroyMixedBlocks(raw)); PetscCall(DestroyBoundaryData(live)); PetscCall(DestroyBoundaryData(empty));
    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode Run()
{
    PetscFunctionBeginUser;
    PetscInt kind=0,suite=0,px=1,py=1,nx=7,ny=5,expectedRanks=1;
    PetscCall(PetscOptionsGetInt(nullptr,nullptr,"-boundary_mesh_type",&kind,nullptr));
    PetscCall(PetscOptionsGetInt(nullptr,nullptr,"-boundary_case",&suite,nullptr));
    PetscCall(PetscOptionsGetInt(nullptr,nullptr,"-mesh_px",&px,nullptr));
    PetscCall(PetscOptionsGetInt(nullptr,nullptr,"-mesh_py",&py,nullptr));
    PetscCall(PetscOptionsGetInt(nullptr,nullptr,"-mesh_nx",&nx,nullptr));
    PetscCall(PetscOptionsGetInt(nullptr,nullptr,"-mesh_ny",&ny,nullptr));
    PetscCall(PetscOptionsGetInt(nullptr,nullptr,"-expected_ranks",&expectedRanks,nullptr));
    PetscMPIInt ranks=0; PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD,&ranks));
    PetscCheck(expectedRanks==ranks,PETSC_COMM_WORLD,PETSC_ERR_ARG_WRONG,
               "Expected %d ranks but PETSc sees %d; use the MPICH launcher matching PETSc",
               static_cast<int>(expectedRanks),static_cast<int>(ranks));
    PetscCall(Require(kind>=0&&kind<=3&&(suite==0||suite==1),"Use mesh type 0..3 and case 0 (numerics) or 1 (errors)"));
    PetscCall(Require((px==1||px==2)&&(py==1||py==2)&&px*py==ranks,"Use process grid 1x1, 2x1, 1x2 or 2x2"));
    PetscCall(Require(nx>=2&&ny>=2&&nx>=px&&ny>=py&&nx<=12&&ny<=12,"Tests require 2..12 vertices in each direction"));
    DM dm=nullptr,serialDM=nullptr; MeshInfo mesh,serial;
    PetscCall(MakeMesh(PETSC_COMM_WORLD,nx,ny,px,py,kind,dm,mesh));
    PetscCall(MakeMesh(PETSC_COMM_SELF,nx,ny,1,1,kind,serialDM,serial));
    PetscCall(DMDestroy(&serialDM));
    if (suite==0) {
        PetscCall(CheckNumerics(Field::Stokes,dm,mesh,serial));
        PetscCall(CheckNumerics(Field::Darcy,dm,mesh,serial));
    } else PetscCall(CheckErrors(dm,mesh,serial));
    PetscCall(DMDestroy(&dm));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD,"Boundary case %d, mesh type %d passed on %d rank(s)\n",
                          static_cast<int>(suite),static_cast<int>(kind),static_cast<int>(ranks)));
    PetscFunctionReturn(PETSC_SUCCESS);
}

} // namespace

int main(int argc, char** argv)
{
    const PetscErrorCode error=PetscInitialize(&argc,&argv,nullptr,
        "Boundary tests: -boundary_mesh_type 0 rectangular, 1 perturbed, 2 stretched, 3 skewed; -boundary_case 0/1.\n");
    if (error) return static_cast<int>(error);
    // Unexpected local failures abort WORLD instead of leaving peers waiting.
    // Expected failures are captured by ExpectError with balanced PETSc frames.
    PetscCallAbort(PETSC_COMM_WORLD,Run());
    return static_cast<int>(PetscFinalize());
}
