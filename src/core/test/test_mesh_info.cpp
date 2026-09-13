#include "mesh_info.h"

#include <algorithm>
#include <array>
#include <limits>
#include <utility>
#include <vector>

namespace {
using Generator = PetscErrorCode (*)(DM, Vec, const MeshParam&);

PetscErrorCode Require(bool condition, const char* message)
{
    PetscFunctionBeginUser;
    PetscCheck(condition, PETSC_COMM_SELF, PETSC_ERR_PLIB, "%s", message);
    PetscFunctionReturn(PETSC_SUCCESS);
}
PetscErrorCode Near(PetscReal actual, PetscReal expected, PetscReal scale)
{
    PetscFunctionBeginUser;
    const PetscReal tolerance = 4096*PETSC_MACHINE_EPSILON
        * std::max(PetscAbsReal(expected),PetscAbsReal(scale));
    PetscCheck(!PetscIsInfOrNanReal(actual) && PetscAbsReal(actual-expected) <= tolerance,
               PETSC_COMM_SELF, PETSC_ERR_PLIB,
               "Got %.17g, expected %.17g (tolerance %.3g)",
               static_cast<double>(actual), static_cast<double>(expected),
               static_cast<double>(tolerance));
    PetscFunctionReturn(PETSC_SUCCESS);
}
template<class Call>
PetscErrorCode ExpectError(Call call)
{
    PetscFunctionBeginUser;
    PetscCall(PetscPushErrorHandler(PetscReturnErrorHandler, nullptr));
    const PetscErrorCode error = call();
    PetscCall(PetscPopErrorHandler());
    PetscCall(Require(error != PETSC_SUCCESS, "Expected a rejected input"));
    PetscFunctionReturn(PETSC_SUCCESS);
}
PetscReal PolygonArea(const QuadVertices& q)
{
    // Independent triangle areas; no production mapping, Jacobian, or quadrature.
    const auto cross = [&](const Point& a, const Point& b) {
        return (a.p[0]-q[0].p[0])*(b.p[1]-q[0].p[1])
             - (a.p[1]-q[0].p[1])*(b.p[0]-q[0].p[0]);
    };
    return (cross(q[1],q[2])+cross(q[2],q[3]))/2;
}

PetscErrorCode CheckIndexErrors()
{
    PetscFunctionBeginUser;
    PetscInt id = -99;
    MeshIndex p{-9,-9};
    for (PetscInt j=0; j<3; ++j) for (PetscInt i=0; i<5; ++i) {
        PetscCall(FlattenIndex({5,3},{i,j},id));
        PetscCall(Require(id==5*j+i, "Flatten convention"));
        PetscCall(UnflattenIndex({5,3},id,p));
        PetscCall(Require(p==MeshIndex{i,j}, "Index round trip"));
    }
    id = -99;
    PetscCall(ExpectError([&]{ return FlattenIndex({0,3},{0,0},id); }));
    PetscCall(ExpectError([&]{ return FlattenIndex({5,3},{-1,0},id); }));
    PetscCall(ExpectError([&]{ return FlattenIndex({5,3},{5,0},id); }));
    PetscCall(ExpectError([&]{ return FlattenIndex({std::numeric_limits<PetscInt>::max(),2},{0,0},id); }));
    PetscCall(Require(id==-99,"Index error changed output"));
    p = {-9,-9};
    PetscCall(ExpectError([&]{ return UnflattenIndex({5,3},15,p); }));
    PetscCall(ExpectError([&]{ return UnflattenIndex({5,3},-1,p); }));
    PetscCall(Require(p==MeshIndex{-9,-9},"Inverse error changed output"));
    MeshInfo empty;
    Point point;
    PetscCall(Require(!empty.IsInitialized() && !empty.OwnsCell({0,0}),"Default snapshot"));
    PetscCall(ExpectError([&]{ return empty.GetVertex({0,0},point); }));
    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode MakeMesh(MPI_Comm comm, PetscInt M, PetscInt N, PetscInt px, PetscInt py,
                        PetscInt width, const MeshParam& mp, Generator generator,
                        DM& dm, Vec& vertices)
{
    PetscFunctionBeginUser;
    PetscCall(DMDACreate2d(comm,DM_BOUNDARY_NONE,DM_BOUNDARY_NONE,DMDA_STENCIL_BOX,
                          M,N,px,py,2,width,nullptr,nullptr,&dm));
    PetscCall(DMSetUp(dm));
    PetscCall(DMCreateGlobalVector(dm,&vertices));
    PetscCall(generator(dm,vertices,mp));
    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode CheckTopology(const MeshInfo& info, PetscInt M, PetscInt N)
{
    PetscFunctionBeginUser;
    const PetscInt cx=M-1, cy=N-1, horizontal=cx*N;
    PetscInt boundary=0;
    for (PetscInt e=0; e<info.EdgeCount(); ++e) {
        EdgeTopology t;
        PetscInt roundtrip;
        PetscCall(info.GetEdgeTopology(e,t));
        PetscCall(info.EdgeId(t.axis,t.vertices[0],roundtrip));
        PetscCall(Require(roundtrip==e,"Edge numbering round trip"));
        const bool alongI=e<horizontal;
        const MeshIndex start=alongI ? MeshIndex{e%cx,e/cx}
                                     : MeshIndex{(e-horizontal)%M,(e-horizontal)/M};
        const MeshIndex end=alongI ? MeshIndex{start.i+1,start.j} : MeshIndex{start.i,start.j+1};
        PetscCall(Require(t.vertices[0]==start && t.vertices[1]==end,"Canonical edge endpoints"));
        const MeshIndex left=alongI ? start : MeshIndex{start.i-1,start.j};
        const MeshIndex right=alongI ? MeshIndex{start.i,start.j-1} : start;
        const auto inside=[&](MeshIndex p){return p.i>=0 && p.i<cx && p.j>=0 && p.j<cy;};
        PetscCall(Require(t.leftCell.has_value()==inside(left),"Left boundary neighbor"));
        PetscCall(Require(t.rightCell.has_value()==inside(right),"Right boundary neighbor"));
        if (t.leftCell) PetscCall(Require(*t.leftCell==left,"Left cell"));
        if (t.rightCell) PetscCall(Require(*t.rightCell==right,"Right cell"));
        if (t.IsBoundary()) ++boundary;
    }
    PetscCall(Require(boundary==2*(cx+cy),"Physical boundary edge count"));
    for (PetscInt j=0;j<cy;++j) for (PetscInt i=0;i<cx;++i) {
        PetscInt id;
        MeshIndex p;
        PetscCall(info.CellId({i,j},id));
        PetscCall(Require(id==j*cx+i,"Natural cell ID"));
        PetscCall(info.CellIndex(id,p));
        PetscCall(Require(p==MeshIndex{i,j},"Cell ID inverse"));
        std::array<OrientedEdge,4> edges;
        PetscCall(info.GetCellEdges({i,j},edges));
        const std::array<PetscInt,4> expected{{j*cx+i,horizontal+j*M+i+1,(j+1)*cx+i,horizontal+j*M+i}};
        for (int side=0;side<4;++side) {
            PetscCall(Require(edges[side].id==expected[side],"Cell edge ID"));
            PetscCall(Require(edges[side].direction==(side<2 ? 1 : -1),"Cell edge orientation"));
        }
    }
    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode CheckCase(PetscInt M, PetscInt N, PetscInt px, PetscInt py, PetscInt width,
                         MeshParam mp, Generator generator, MeshInfo& info)
{
    PetscFunctionBeginUser;
    DM dm=nullptr, serialDM=nullptr;
    Vec vertices=nullptr, serialVertices=nullptr;
    PetscCall(MakeMesh(PETSC_COMM_WORLD,M,N,px,py,width,mp,generator,dm,vertices));
    PetscCall(BuildMeshInfo(dm,vertices,info)); // Reuse the object across sizes and partitions.
    MeshInfo serial;
    PetscCall(MakeMesh(PETSC_COMM_SELF,M,N,1,1,width,mp,generator,serialDM,serialVertices));
    PetscCall(BuildMeshInfo(serialDM,serialVertices,serial));
    PetscCall(Require(info.VertexDimensions()==MeshIndex{M,N},"Vertex dimensions must include boundaries"));
    PetscCall(Require(info.CellDimensions()==MeshIndex{M-1,N-1},"Physical cell dimensions"));
    PetscCall(Require(info.VertexCount()==M*N && info.CellCount()==(M-1)*(N-1),"Global counts"));
    PetscCall(Require(info.AlongIEdgeCount()==(M-1)*N && info.AlongJEdgeCount()==M*(N-1),"Edge family counts"));
    PetscCall(Require(info.StencilWidth()==width,"Stencil width"));

    PetscInt xs,ys,xm,ym,gx,gy,gxm,gym;
    PetscCall(DMDAGetCorners(dm,&xs,&ys,nullptr,&xm,&ym,nullptr));
    PetscCall(DMDAGetGhostCorners(dm,&gx,&gy,nullptr,&gxm,&gym,nullptr));
    PetscCall(Require(info.OwnedVertices().begin==MeshIndex{xs,ys} &&
                       info.OwnedVertices().end==MeshIndex{xs+xm,ys+ym},"Owned vertex range"));
    PetscCall(Require(info.GhostVertices().begin==MeshIndex{gx,gy} &&
                       info.GhostVertices().end==MeshIndex{gx+gxm,gy+gym},"Actual ghost range"));
    PetscCall(CheckTopology(info,M,N));

    // Compare every copied ghost point with a complete serial mesh, detecting
    // missing exchanges, wrong ghost origins, and physical-boundary padding.
    for (PetscInt j=gy;j<gy+gym;++j) for (PetscInt i=gx;i<gx+gxm;++i) {
        Point actual,expected;
        PetscInt id;
        MeshIndex p;
        PetscCall(info.VertexId({i,j},id));
        PetscCall(Require(id==j*M+i,"Natural vertex ID"));
        PetscCall(info.VertexIndex(id,p));
        PetscCall(Require(p==MeshIndex{i,j},"Vertex ID inverse"));
        PetscCall(info.GetVertex({i,j},actual));
        PetscCall(serial.GetVertex({i,j},expected));
        PetscCall(Near(actual.p[0],expected.p[0],mp.L));
        PetscCall(Near(actual.p[1],expected.p[1],mp.H));
    }
    const auto available=info.AvailableCells();
    for (PetscInt j=available.begin.j;j<available.end.j;++j)
        for (PetscInt i=available.begin.i;i<available.end.i;++i) {
            QuadVertices q;
            PetscReal area,expected;
            PetscCall(info.GetCellCorners({i,j},q));
            PetscCall(info.GetCellArea({i,j},area));
            PetscCall(serial.GetCellArea({i,j},expected));
            PetscCall(Near(area,PolygonArea(q),area));
            PetscCall(Near(area,expected,area));
        }

    std::vector<int> cells(static_cast<std::size_t>(info.CellCount()));
    std::vector<int> edgeOwners(static_cast<std::size_t>(info.EdgeCount()));
    std::vector<int> incidence(edgeOwners.size()), directions(edgeOwners.size());
    PetscReal localArea=0, localFlux=0;
    int ownedCount=0;
    GaussRule1D rule;
    PetscCall(CreateGaussRule(3,rule));
    for (PetscInt j=0;j<N-1;++j) for (PetscInt i=0;i<M-1;++i) {
        const bool owns=i>=xs && i<xs+xm && j>=ys && j<ys+ym;
        PetscCall(Require(info.OwnsCell({i,j})==owns,"Cell owner must own its lower-left vertex"));
        if (!owns) continue;
        ++ownedCount;
        cells[static_cast<std::size_t>(j*(M-1)+i)]=1;
        MeshIndex local,global;
        PetscCall(info.CellGlobalToLocal({i,j},local));
        PetscCall(Require(local==MeshIndex{i-xs,j-ys},"Owned cell offset"));
        PetscCall(info.CellLocalToGlobal(local,global));
        PetscCall(Require(global==MeshIndex{i,j},"Owned cell offset inverse"));
        PetscReal area;
        QuadVertices corners;
        PetscCall(info.GetCellArea({i,j},area));
        PetscCall(info.GetCellCorners({i,j},corners));
        localArea+=area;
        Point center{};
        for (const auto& p:corners) for(int d=0;d<2;++d) center.p[d]+=p.p[d]/4;
        std::array<OrientedEdge,4> edges;
        PetscCall(info.GetCellEdges({i,j},edges));
        PetscReal cellFlux=0;
        for (int side=0;side<4;++side) {
            const auto e=edges[side];
            ++incidence[static_cast<std::size_t>(e.id)];
            directions[static_cast<std::size_t>(e.id)]+=e.direction;
            EdgeVertices directed,canonical;
            PetscCall(info.GetCellEdgeVertices({i,j},static_cast<CellSide>(side),directed));
            PetscCall(info.GetEdgeVertices(e.id,canonical));
            if (e.direction<0) std::swap(canonical[0],canonical[1]);
            for(int k=0;k<2;++k) for(int d=0;d<2;++d)
                PetscCall(Require(directed[k].p[d]==canonical[k].p[d],"Cell and canonical edge agreement"));
            PetscReal length,flux;
            Point normal;
            PetscCall(GetEdgeGeometry(directed,length,normal));
            const PetscReal inward=(center.p[0]-(directed[0].p[0]+directed[1].p[0])/2)*normal.p[0]
                                  +(center.p[1]-(directed[0].p[1]+directed[1].p[1])/2)*normal.p[1];
            PetscCall(Require(inward<0,"Normal must point away from the cell interior"));
            PetscCall(IntegrateNormalFlux(directed,rule,[](const Point& p){return p;},flux));
            cellFlux+=flux;
        }
        PetscCall(Near(cellFlux,2*area,area)); // div((x,y)) = 2.
        localFlux+=cellFlux;
    }
    for (PetscInt e:info.OwnedEdgeIds()) {
        edgeOwners[static_cast<std::size_t>(e)]=1;
        PetscInt local,global;
        EdgeTopology t;
        EdgeVertices endpoints;
        PetscCall(info.EdgeGlobalToLocal(e,local));
        PetscCall(info.EdgeLocalToGlobal(local,global));
        PetscCall(Require(global==e && info.OwnsEdge(e),"Local edge ID round trip"));
        PetscCall(info.GetEdgeTopology(e,t));
        PetscCall(Require(info.OwnedVertices().Contains(t.vertices[0]),"Edge anchor owner"));
        PetscCall(info.GetEdgeVertices(e,endpoints)); // Endpoint must be available even on empty cell owners.
    }
    for(PetscInt e=0;e<info.EdgeCount();++e) {
        EdgeTopology t;
        PetscCall(info.GetEdgeTopology(e,t));
        PetscCall(Require(info.OwnsEdge(e)==info.OwnedVertices().Contains(t.vertices[0]),"Missing owned edge"));
    }
    // Reductions are outside loops, so ranks with zero owned cells participate.
    for (auto* values : {&cells,&edgeOwners,&incidence,&directions}) {
        std::vector<int> reduced(values->size());
        PetscCallMPI(MPI_Allreduce(values->data(),reduced.data(),static_cast<int>(values->size()),
                                   MPI_INT,MPI_SUM,PETSC_COMM_WORLD));
        *values=std::move(reduced);
    }
    for (int count:cells) PetscCall(Require(count==1,"Cell ownership coverage"));
    for (PetscInt e=0;e<info.EdgeCount();++e) {
        EdgeTopology t;
        PetscCall(info.GetEdgeTopology(e,t));
        PetscCall(Require(edgeOwners[static_cast<std::size_t>(e)]==1,"Edge ownership coverage"));
        PetscCall(Require(incidence[static_cast<std::size_t>(e)]==(t.IsBoundary()?1:2),"Cell-edge incidence count"));
        const int sign=directions[static_cast<std::size_t>(e)];
        PetscCall(Require(t.IsBoundary() ? (sign==1 || sign==-1) : sign==0,"Shared-edge orientation cancellation"));
    }
    PetscReal globalArea,globalFlux;
    PetscCallMPI(MPI_Allreduce(&localArea,&globalArea,1,MPIU_REAL,MPI_SUM,PETSC_COMM_WORLD));
    PetscCallMPI(MPI_Allreduce(&localFlux,&globalFlux,1,MPIU_REAL,MPI_SUM,PETSC_COMM_WORLD));
    PetscCall(Near(globalArea,mp.L*mp.H,mp.L*mp.H));
    PetscCall(Near(globalFlux,2*mp.L*mp.H,mp.L*mp.H));
    if(M==2 && N==2) {
        int empty=ownedCount==0?1:0, allEmpty;
        PetscMPIInt size;
        PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD,&size));
        PetscCallMPI(MPI_Allreduce(&empty,&allEmpty,1,MPI_INT,MPI_SUM,PETSC_COMM_WORLD));
        PetscCall(Require(allEmpty==size-1,"Single-cell test must exercise empty cell owners"));
    }

    // Rejected accesses preserve output, including valid global but unavailable cells.
    PetscReal sentinel=-77;
    PetscCall(ExpectError([&]{return info.GetCellArea({-1,0},sentinel);}));
    PetscCall(ExpectError([&]{return info.GetCellArea({M-1,0},sentinel);}));
    PetscCall(Require(sentinel==-77,"Rejected area lookup modified output"));
    PetscInt id=-77;
    PetscCall(ExpectError([&]{return info.EdgeGlobalToLocal(-1,id);}));
    PetscCall(ExpectError([&]{return info.EdgeLocalToGlobal(static_cast<PetscInt>(info.OwnedEdgeIds().size()),id);}));
    PetscCall(ExpectError([&]{return info.EdgeId(EdgeAxis::AlongJ,{M,0},id);}));
    PetscCall(Require(id==-77,"Rejected edge lookup modified output"));
    for(PetscInt j=0;j<N-1;++j) for(PetscInt i=0;i<M-1;++i) if(!info.HasCell({i,j}))
        PetscCall(ExpectError([&]{return info.GetCellArea({i,j},sentinel);}));

    const MeshIndex probe=available.begin;
    PetscReal before,after;
    PetscCall(info.GetCellArea(probe,before));
    PetscScalar*** array=nullptr;
    PetscCall(DMDAVecGetArrayDOF(dm,vertices,&array));
    if(info.OwnedVertices().Contains({0,0})) array[0][0][0]=std::numeric_limits<PetscReal>::quiet_NaN();
    PetscCall(DMDAVecRestoreArrayDOF(dm,vertices,&array));
    // Only one owner corrupts input; all ranks must return an error, with old snapshots intact.
    PetscCall(ExpectError([&]{return BuildMeshInfo(dm,vertices,info);}));
    PetscCall(info.GetCellArea(probe,after));
    PetscCall(Require(after==before,"Failed build replaced cached geometry"));
    mp.L*=1.5;
    PetscCall(generator(dm,vertices,mp));
    PetscCall(info.GetCellArea(probe,after));
    PetscCall(Require(after==before,"Snapshot changed when the input Vec changed"));
    PetscCall(BuildMeshInfo(dm,vertices,info));
    PetscCall(info.GetCellArea(probe,after));
    PetscCall(Near(after,1.5*before,before));

    MeshInfo copied=info;
    MeshInfo moved=std::move(copied);
    PetscCall(Require(!copied.IsInitialized(),"Moved-from snapshot must be empty"));
    PetscCall(moved.GetCellArea(probe,after));
    PetscCall(Near(after,1.5*before,before));
    info=std::move(moved);
    PetscCall(VecDestroy(&vertices));
    PetscCall(DMDestroy(&dm));
    PetscCall(VecDestroy(&serialVertices));
    PetscCall(DMDestroy(&serialDM));
    PetscCall(info.GetCellArea(probe,after)); // Geometry owns its storage beyond PETSc object lifetimes.
    PetscCall(Near(after,1.5*before,before));
    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode Run()
{
    PetscFunctionBeginUser;
    PetscInt kind=0,px=PETSC_DECIDE,py=PETSC_DECIDE;
    PetscCall(PetscOptionsGetInt(nullptr,nullptr,"-mesh_info_type",&kind,nullptr));
    PetscCall(PetscOptionsGetInt(nullptr,nullptr,"-mesh_px",&px,nullptr));
    PetscCall(PetscOptionsGetInt(nullptr,nullptr,"-mesh_py",&py,nullptr));
    PetscMPIInt size;
    PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD,&size));
    PetscCheck(kind>=0 && kind<=2 && (size==1 || size==2 || size==4),
               PETSC_COMM_WORLD,PETSC_ERR_ARG_OUTOFRANGE,"Use mesh type 0,1,2 and 1,2,4 processes");
    PetscCheck((px==PETSC_DECIDE || px==1 || px==2) && (py==PETSC_DECIDE || py==1 || py==2)
               && (px==PETSC_DECIDE || py==PETSC_DECIDE || px*py==size),
               PETSC_COMM_WORLD,PETSC_ERR_ARG_WRONG,"Use 1x1,2x1,1x2,2x2; check the MPI launcher");
    PetscCall(CheckIndexErrors());
    const std::array<Generator,3> generators{{CreateFullMesh,LogicRectMesh,RefineMesh}};
    const auto generator=generators[static_cast<std::size_t>(kind)];
    MeshParam mp;
    mp.xstart=-0.75; mp.ystart=0.2; mp.L=2.5; mp.H=1.3; mp.seed=7;
    MeshInfo info;
    PetscCall(CheckCase(13,9,px,py,1,mp,generator,info));
    mp.perturbation=0.249; mp.seed=991;
    PetscCall(CheckCase(10,7,px,py,2,mp,generator,info));
    mp.xstart=3.5; mp.ystart=-2.25; mp.L=17; mp.H=0.08;
    PetscCall(CheckCase(8,6,px,py,1,mp,generator,info));
    PetscCall(CheckCase(2,2,px,py,1,mp,generator,info));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD,"mesh_info type %d passed on %d rank(s)\n",
                          static_cast<int>(kind),static_cast<int>(size)));
    PetscFunctionReturn(PETSC_SUCCESS);
}
} // namespace

int main(int argc,char** argv)
{
    const PetscErrorCode error=PetscInitialize(&argc,&argv,nullptr,"Mesh metadata verification\n");
    if(error) return static_cast<int>(error);
    PetscCallAbort(PETSC_COMM_WORLD,Run());
    return static_cast<int>(PetscFinalize());
}
