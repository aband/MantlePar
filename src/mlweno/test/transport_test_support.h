#ifndef MANTLE_TRANSPORT_TEST_SUPPORT_H
#define MANTLE_TRANSPORT_TEST_SUPPORT_H

// Shared SERIAL test utilities. Production mesh, reconstruction and flux
// implementations are used unchanged. No test PETSc substitutes are shipped.
#include "advectiveflux.h"
#include "diffusiveflux.h"
#include "reconstruction.h"
#include <petscsys.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <exception>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace transport_test {
inline std::size_t checks=0;
inline void Require(bool condition,const std::string& what)
{ ++checks; if(!condition) throw std::runtime_error(what); }
inline void Call(PetscErrorCode code)
{ Require(code==PETSC_SUCCESS,"PETSc call failed: "+std::to_string(code)); }
inline void Near(PetscReal a,PetscReal b,const std::string& what,PetscReal tolerance=3e-10)
{
    Require(std::isfinite(a)&&std::isfinite(b)&&
            std::abs(a-b)<=tolerance*(1+std::abs(a)+std::abs(b)),
            what+": "+std::to_string(a)+" versus "+std::to_string(b));
}
template<class F> inline void ExpectError(F&& f,const std::string& what)
{
    Call(PetscPushErrorHandler(PetscReturnErrorHandler,nullptr));
    const auto code=f();
    Call(PetscPopErrorHandler());
    Require(code!=PETSC_SUCCESS,what);
}
template<class F> int Main(int argc,char** argv,F&& run)
{
    const auto initialized=PetscInitialize(&argc,&argv,nullptr,nullptr);
    if(initialized) return static_cast<int>(initialized);
    int status=0;
    try {
        PetscMPIInt size=0;Call(MPI_Comm_size(PETSC_COMM_WORLD,&size));
        Require(size==1,"These numerical tests require one MPI rank");
        Require(std::numeric_limits<PetscReal>::digits>=53,
                "These accuracy tolerances require double precision or better");
        run();
        std::cout<<"PASS: "<<checks<<" checks\n";
    } catch(const std::exception& e) {
        std::cerr<<"FAIL: "<<e.what()<<'\n';status=1;
    }
    const auto finalized=PetscFinalize();
    return status ? status : static_cast<int>(finalized);
}
inline std::ofstream CSV(const std::string& filename)
{
    const std::filesystem::path path(filename);
    if(!path.parent_path().empty())std::filesystem::create_directories(path.parent_path());
    std::ofstream file(path);Require(file.is_open(),"Cannot open "+filename);
    file<<std::scientific<<std::setprecision(17);return file;
}
inline void Finish(std::ofstream& file)
{ file.close();Require(!file.fail(),"CSV write failed"); }
inline std::string Sibling(const std::string& filename,const std::string& suffix)
{
    const std::filesystem::path p(filename);
    return (p.parent_path()/(p.stem().string()+suffix+".csv")).string();
}
inline GaussRule1D Rule(PetscInt n)
{ GaussRule1D q;Call(CreateGaussRule(n,q));return q; }
inline PetscReal Dot(const Point& a,const Point& b)
{ return a.p[0]*b.p[0]+a.p[1]*b.p[1]; }
inline std::size_t Id(MeshIndex size,MeshIndex p)
{ return static_cast<std::size_t>(p.j)*size.i+p.i; }

struct Options {
    PetscInt mode=0,mesh=0,axis=0,order=3,constant=0,samples=4;
    PetscInt n0=12,levels=3,field=0,lf=0,orientation=0,cut=1;
    PetscReal kappa=.02,time=.2;
    std::string output="transport.csv";
};
inline Options ReadOptions()
{
    Options o;
    const std::pair<const char*,PetscInt*> integers[]={
        {"-flux_mode",&o.mode},{"-flux_mesh",&o.mesh},{"-flux_axis",&o.axis},
        {"-flux_order",&o.order},{"-flux_constant",&o.constant},{"-flux_samples",&o.samples},
        {"-flux_n0",&o.n0},{"-flux_levels",&o.levels},{"-flux_field",&o.field},
        {"-flux_lf",&o.lf},{"-flux_orientation",&o.orientation},{"-flux_cut",&o.cut}};
    for(const auto& p:integers)Call(PetscOptionsGetInt(nullptr,nullptr,p.first,p.second,nullptr));
    Call(PetscOptionsGetReal(nullptr,nullptr,"-flux_kappa",&o.kappa,nullptr));
    Call(PetscOptionsGetReal(nullptr,nullptr,"-flux_time",&o.time,nullptr));
    char name[4096]="";
    Call(PetscOptionsGetString(nullptr,nullptr,"-flux_output",name,sizeof(name),nullptr));
    if(name[0])o.output=name;
    Require(o.mode>=0&&o.mode<=2&&o.mesh>=0&&o.mesh<=1&&o.axis>=0&&o.axis<=2&&
            (o.order==3||o.order==5)&&o.constant>=0&&o.constant<=1&&o.samples>=2&&
            o.samples%2==0&&o.n0>=8&&o.levels>=2&&o.levels<=7&&o.field>=0&&o.field<=1&&
            o.lf>=0&&o.lf<=1&&o.orientation>=0&&o.orientation<=2&&o.cut>=0&&o.cut<=1,
            "Invalid transport test options");
    Require(std::isfinite(o.kappa)&&o.kappa>=0&&std::isfinite(o.time)&&o.time>0,
            "Require finite kappa>=0 and time>0");
    Require(o.axis==0||o.mesh==0,"Pseudo-1D tests use rectangular strips");
    Require(o.field!=1||o.kappa>0,"The positive-time diffused front requires kappa>0");
    return o;
}

struct Grid {
    DM dm=nullptr;Vec coordinates=nullptr;MeshInfo mesh;
    MeshIndex size{};std::vector<QuadVertices> cells;std::vector<PetscReal> areas;
    PetscReal h=0;
    Grid()=default;Grid(const Grid&)=delete;Grid& operator=(const Grid&)=delete;
    ~Grid(){if(coordinates)(void)VecDestroy(&coordinates);if(dm)(void)DMDestroy(&dm);}
};
inline void MakeGrid(PetscInt n,const Options& o,Grid& g)
{
    g.size={o.axis==2?1:n,o.axis==1?1:n};g.h=1.0/n;
    Call(DMDACreate2d(PETSC_COMM_SELF,DM_BOUNDARY_NONE,DM_BOUNDARY_NONE,DMDA_STENCIL_BOX,
                     g.size.i+1,g.size.j+1,1,1,2,1,nullptr,nullptr,&g.dm));
    Call(DMSetUp(g.dm));Call(DMCreateGlobalVector(g.dm,&g.coordinates));
    MeshParam p;p.L=p.H=1;p.perturbation=.15;p.seed=37;
    if(o.mesh)Call(LogicRectMesh(g.dm,g.coordinates,p));
    else Call(CreateFullMesh(g.dm,g.coordinates,p));
    Call(BuildMeshInfo(g.dm,g.coordinates,g.mesh));
    g.cells.resize(static_cast<std::size_t>(g.size.i)*g.size.j);g.areas.resize(g.cells.size());
    const auto q=Rule(2);long double total=0;bool slanted=false;
    for(PetscInt j=0;j<g.size.j;++j)for(PetscInt i=0;i<g.size.i;++i){
        const auto k=Id(g.size,{i,j});Call(g.mesh.GetCellCorners({i,j},g.cells[k]));
        Call(IntegrateCell(g.cells[k],q,[](const Point&){return PetscReal(1);},g.areas[k]));
        total+=g.areas[k];
        for(std::size_t f=0;f<4;++f){const auto&a=g.cells[k][f];const auto&b=g.cells[k][(f+1)%4];
            slanted=slanted||(std::abs(a.p[0]-b.p[0])>1e-12*g.h&&std::abs(a.p[1]-b.p[1])>1e-12*g.h);}
    }
    Near(static_cast<PetscReal>(total),1,"Mesh total area",2e-12);
    Require(!o.mesh||slanted,"Perturbed case must contain slanted faces");
}
inline ReconstructionOptions Configuration(const Grid& g,MeshIndex target,const Options& o)
{
    ReconstructionOptions c;c.useConstant=o.constant!=0;
    const PetscInt small=o.order==3?2:3;
    c.large.size={o.axis==2?1:o.order,o.axis==1?1:o.order};c.large.order=o.order-1;
    c.small.size={o.axis==2?1:small,o.axis==1?1:small};c.small.order=small-1;
    // Keep a full-degree one-sided large candidate at physical boundaries.
    // This is explicit per-target candidate selection, not a production fallback.
    const PetscInt sx=std::clamp(target.i-c.large.size.i/2,PetscInt(0),g.size.i-c.large.size.i);
    const PetscInt sy=std::clamp(target.j-c.large.size.j/2,PetscInt(0),g.size.j-c.large.size.j);
    c.large.offsets={{sx-target.i,sy-target.j}};
    for(PetscInt j=0;j<c.small.size.j;++j)for(PetscInt i=0;i<c.small.size.i;++i)
        c.small.offsets.push_back({-i,-j});
    c.small.smoothness=ReconstructionSmoothness::TargetCell;
    return c; // Preserve production epsilon=1e-2 and constant weight defaults.
}
inline void InitializeReconstructions(const Grid& g,const Options& o,std::vector<Reconstruction>& r)
{
    r.resize(g.cells.size());const PetscReal scale=std::sqrt(1.0/(g.size.i*g.size.j));
    for(PetscInt j=0;j<g.size.j;++j)for(PetscInt i=0;i<g.size.i;++i)
        Call(r[Id(g.size,{i,j})].Initialize(g.mesh,{i,j},Configuration(g,{i,j},o),scale));
}
inline void Update(const Grid& g,const std::vector<PetscReal>& u,
                   std::vector<Reconstruction>& r,bool jacobian=false)
{
    const MeshRange range{{0,0},g.size};
    for(std::size_t c=0;c<r.size();++c){
        if(jacobian)Call(r[c].UpdateWithJacobian(u,range,g.areas[c]));
        else Call(r[c].Update(u,range,g.areas[c]));
    }
}

struct ExactValue {PetscReal u=0,ut=0;Point gradient{};};
struct Field {
    PetscInt kind=0,axis=0,orientation=0; // 0=sine,1=diffused step,2=quadratic,3=constant
    Point velocity{};PetscReal kappa=.02,time=.2;
    Point Direction() const {
        if(axis==1)return Point{{1,0}};
        if(axis==2)return Point{{0,1}};
        if(orientation==0)return Point{{1,0}};
        if(orientation==1)return Point{{0,1}};
        return Point{{.6,.8}};
    }
    ExactValue operator()(const Point& x)const {
        ExactValue v;PetscReal lap=0;
        if(kind==3){v.u=1.25;return v;}
        if(kind==2){v.u=1+x.p[0]+.5*x.p[1]+.2*x.p[0]*x.p[0]+.3*x.p[0]*x.p[1]+.4*x.p[1]*x.p[1];
            v.gradient=Point{{1+.4*x.p[0]+.3*x.p[1],.5+.3*x.p[0]+.8*x.p[1]}};lap=1.2;}
        else if(kind==0){const Point k=axis==1?Point{{1.2,0}}:axis==2?Point{{0,1.2}}:Point{{.8,.6}};
            const PetscReal a=.2*std::exp(-kappa*Dot(k,k)*time);
            const PetscReal phase=Dot(k,x)-Dot(k,velocity)*time;
            v.u=1+a*std::sin(phase);v.gradient=Point{{a*std::cos(phase)*k.p[0],a*std::cos(phase)*k.p[1]}};
            lap=-Dot(k,k)*a*std::sin(phase);}
        else {const Point n=Direction();const PetscReal c=.5*(n.p[0]+n.p[1]);
            const PetscReal s=Dot(n,x)-c-Dot(n,velocity)*time;
            const PetscReal kt=kappa*time,z=s/(2*std::sqrt(kt));
            v.u=.5*std::erfc(-z);
            const PetscReal d=std::exp(-z*z)/std::sqrt(4*std::acos(PetscReal(-1))*kt);
            v.gradient=Point{{d*n.p[0],d*n.p[1]}};lap=-s*d/(2*kt);}
        v.ut=kappa*lap-Dot(velocity,v.gradient);return v;
    }
};
inline Field MakeField(const Options& o,bool advection)
{
    Field f;f.kind=o.field;f.axis=o.axis;f.orientation=o.orientation;f.kappa=o.kappa;f.time=o.time;
    if(advection)f.velocity=o.axis==1?Point{{.7,0}}:o.axis==2?Point{{0,-.4}}:Point{{.7,-.4}};
    return f;
}

// Reference integrals are independent of the tested flux/reconstruction.
// Adaptive subdivision resolves thin erf layers, even on the coarsest mesh.
inline PetscReal ReferenceCell(const QuadVertices& c,const std::function<PetscReal(const Point&)>& f,
                               const GaussRule1D& low,const GaussRule1D& high,int depth=0)
{
    PetscReal a,b,area;Call(IntegrateCell(c,low,f,a));Call(IntegrateCell(c,high,f,b));
    Call(IntegrateCell(c,low,[](const Point&){return PetscReal(1);},area));
    if(std::abs(a-b)<=2e-12*(area+std::abs(b)))return b;
    Require(depth<9,"Reference cell quadrature did not resolve the exact field");
    PetscReal sum=0;
    for(int j=0;j<2;++j)for(int i=0;i<2;++i){const PetscReal x=i-1,y=j-1;
        const QuadVertices sub{{MapCellPoint(Point{{x,y}},c),MapCellPoint(Point{{x+1,y}},c),
                                MapCellPoint(Point{{x+1,y+1}},c),MapCellPoint(Point{{x,y+1}},c)}};
        sum+=ReferenceCell(sub,f,low,high,depth+1);}
    return sum;
}
inline PetscReal ReferenceEdge(const EdgeVertices& e,const std::function<PetscReal(const Point&)>& f,
                               const GaussRule1D& low,const GaussRule1D& high,int depth=0)
{
    PetscReal a,b,length;Point n;Call(IntegrateEdge(e,low,f,a));Call(IntegrateEdge(e,high,f,b));
    Call(GetEdgeGeometry(e,length,n));
    if(std::abs(a-b)<=2e-12*(length+std::abs(b)))return b;
    Require(depth<10,"Reference edge quadrature did not resolve the exact flux");
    const Point middle=MapEdgePoint(0,e);
    return ReferenceEdge({{e[0],middle}},f,low,high,depth+1)+
           ReferenceEdge({{middle,e[1]}},f,low,high,depth+1);
}
inline std::vector<PetscReal> Averages(const Grid& g,const Field& f,bool timeDerivative=false)
{
    const auto low=Rule(8),high=Rule(12);std::vector<PetscReal> u(g.cells.size());
    const auto function=[&](const Point& p){const auto v=f(p);return timeDerivative?v.ut:v.u;};
    for(std::size_t c=0;c<u.size();++c)
        u[c]=ReferenceCell(g.cells[c],function,low,high)/g.areas[c];
    return u;
}

struct Face {
    std::size_t left=0;std::ptrdiff_t right=-1;EdgeVertices edge{};DiffusiveSampling sampling;
};
inline std::vector<Face> Faces(const Grid& g,const GaussRule1D& q,PetscInt samples)
{
    std::vector<Face> faces;
    const auto add=[&](std::size_t c,int side,std::ptrdiff_t other){
        Face f;f.left=c;f.right=other;f.edge={{g.cells[c][side],g.cells[c][(side+1)%4]}};
        if(other>=0)Call(CreateInteriorDiffusiveSampling(f.edge,g.cells[c],g.cells[other],q,{samples,.9},f.sampling));
        else Call(CreateBoundaryDiffusiveSampling(f.edge,g.cells[c],q,{samples,.9},f.sampling));
        faces.push_back(std::move(f));
    };
    for(PetscInt j=0;j<g.size.j;++j)for(PetscInt i=0;i<g.size.i;++i){const auto c=Id(g.size,{i,j});
        if(i+1<g.size.i)add(c,1,static_cast<std::ptrdiff_t>(Id(g.size,{i+1,j})));else add(c,1,-1);
        if(j+1<g.size.j)add(c,2,static_cast<std::ptrdiff_t>(Id(g.size,{i,j+1})));else add(c,2,-1);
        if(i==0)add(c,3,-1);
        if(j==0)add(c,0,-1);
    }
    return faces;
}
using Sparse=std::map<std::size_t,PetscReal>;
struct FaceResult {PetscReal flux=0,advective=0,diffusive=0;Sparse jacobian;};
inline void AddRow(const Grid& g,const Reconstruction& r,const std::vector<PetscReal>& row,
                   PetscReal scale,Sparse& to)
{
    Require(row.size()==r.JacobianCells().size(),"Reconstruction Jacobian size mismatch");
    for(std::size_t j=0;j<row.size();++j)to[Id(g.size,r.JacobianCells()[j])]+=scale*row[j];
}
inline FaceResult EvaluateFace(const Grid& g,const Face& face,const GaussRule1D& q,
                               const std::vector<Reconstruction>& recon,const Field& f,
                               bool globalLF,bool advection,bool diffusion,bool jacobian=false)
{
    FaceResult result;const auto nq=q.points.size();
    const Reconstruction& left=recon[face.left];
    const bool boundary=face.right<0;
    const auto value=[&](const Reconstruction& r,const Point& p,PetscReal& v,std::vector<PetscReal>& row){
        if(jacobian)Call(r.EvaluateWithJacobian(p,v,row));else Call(r.Evaluate(p,v));};
    if(advection){
        std::vector<PetscReal> uL(nq),uR(nq);std::vector<std::vector<PetscReal>> jL(nq),jR(nq);
        const std::vector<Point> velocity(nq,f.velocity);
        for(std::size_t k=0;k<nq;++k){const Point p=MapEdgePoint(q.points[k],face.edge);
            value(left,p,uL[k],jL[k]);if(!boundary)value(recon[face.right],p,uR[k],jR[k]);}
        LaxFriedrichsOptions lf;lf.mode=globalLF?LaxFriedrichsMode::Global:LaxFriedrichsMode::Local;
        // A state-independent global bound, valid on every face of the mesh.
        lf.globalSpeed=std::hypot(f.velocity.p[0],f.velocity.p[1]);
        if(boundary){std::vector<AdvectiveBoundaryValue> data;
            Call(BuildLinearAdvectionBoundaryData(face.edge,q,velocity,f.time,
                [&](const AdvectiveFluxPoint& p,PetscReal& v)->PetscErrorCode{v=f(p.position).u;return PETSC_SUCCESS;},data));
            if(jacobian){AdvectiveBoundaryFluxResult r;
                Call(IntegrateAdvectiveBoundaryFluxWithDerivatives(face.edge,q,velocity,uL,data,f.time,{},r));
                result.advective=r.flux;for(std::size_t k=0;k<nq;++k)AddRow(g,left,jL[k],r.derivativeInterior[k],result.jacobian);
            }else Call(IntegrateAdvectiveBoundaryFlux(face.edge,q,velocity,uL,data,f.time,{},result.advective));
        }else if(jacobian){AdvectiveEdgeFluxResult r;
            Call(IntegrateAdvectiveFluxWithDerivatives(face.edge,q,velocity,uL,uR,f.time,{},lf,r));
            result.advective=r.flux;for(std::size_t k=0;k<nq;++k){AddRow(g,left,jL[k],r.derivativeLeft[k],result.jacobian);
                AddRow(g,recon[face.right],jR[k],r.derivativeRight[k],result.jacobian);}
        }else Call(IntegrateAdvectiveFlux(face.edge,q,velocity,uL,uR,f.time,{},lf,result.advective));
    }
    if(diffusion){
        const auto& s=face.sampling;std::vector<PetscReal> u(s.SamplePoints().size(),0),kappa(nq,f.kappa);
        std::vector<std::vector<PetscReal>> rows(u.size());std::vector<std::size_t> owner(u.size(),face.left);
        for(std::size_t a=0;a<u.size();++a){if(s.SampleSides()[a]==DiffusiveSampleSide::Boundary)continue;
            owner[a]=s.SampleSides()[a]==DiffusiveSampleSide::Left?face.left:static_cast<std::size_t>(face.right);
            value(recon[owner[a]],s.SamplePoints()[a],u[a],rows[a]);}
        DiffusiveEdgeFluxResult r;
        if(boundary){std::vector<DiffusiveBoundaryValue> data(nq);
            const bool inactive=f.axis && std::abs(s.Normal().p[f.axis-1])<1e-14;
            for(std::size_t k=0;k<nq;++k)data[k]={inactive?DiffusiveBoundaryType::ZeroFlux:
                DiffusiveBoundaryType::PrescribedState,f(s.QuadraturePoints()[k]).u};
            if(jacobian)Call(IntegrateDiffusiveBoundaryFluxWithDerivatives(s,u,kappa,data,f.time,{},r));
            else Call(IntegrateDiffusiveBoundaryFlux(s,u,kappa,data,f.time,{},result.diffusive));
        }else if(jacobian)Call(IntegrateDiffusiveFluxWithDerivatives(s,u,kappa,f.time,{},r));
        else Call(IntegrateDiffusiveFlux(s,u,kappa,f.time,{},result.diffusive));
        if(jacobian){result.diffusive=r.flux;for(std::size_t a=0;a<u.size();++a)
            if(!rows[a].empty())AddRow(g,recon[owner[a]],rows[a],r.derivativeSamples[a],result.jacobian);}
    }
    result.flux=result.advective+result.diffusive;return result;
}
struct OperatorResult {
    std::vector<PetscReal> rhs,flux;std::vector<Sparse> jacobian;PetscReal boundary=0;
};
inline OperatorResult Assemble(const Grid& g,const std::vector<Face>& faces,const GaussRule1D& q,
                                const std::vector<Reconstruction>& r,const Field& f,
                                bool globalLF,bool advection=true,bool diffusion=true,bool jacobian=false)
{
    OperatorResult result;result.rhs.assign(g.cells.size(),0);if(jacobian)result.jacobian.resize(g.cells.size());
    for(const auto& face:faces){const auto v=EvaluateFace(g,face,q,r,f,globalLF,advection,diffusion,jacobian);
        result.flux.push_back(v.flux);result.rhs[face.left]-=v.flux/g.areas[face.left];
        if(face.right>=0)result.rhs[face.right]+=v.flux/g.areas[face.right];else result.boundary+=v.flux;
        if(jacobian)for(const auto& e:v.jacobian){result.jacobian[face.left][e.first]-=e.second/g.areas[face.left];
            if(face.right>=0)result.jacobian[face.right][e.first]+=e.second/g.areas[face.right];}
    }
    long double mass=result.boundary,scale=1+std::abs(result.boundary);
    for(std::size_t c=0;c<g.cells.size();++c){mass+=g.areas[c]*result.rhs[c];scale+=std::abs(g.areas[c]*result.rhs[c]);}
    Require(std::abs(mass)<2e-11*scale,"Discrete global mass/boundary-flux balance");return result;
}
inline PetscReal ExactFlux(const Face& face,const Field& f,bool adv,bool diff)
{
    const Point normal=face.sampling.Normal();
    const auto fun=[&](const Point& p){const auto v=f(p);return (adv?Dot(f.velocity,normal)*v.u:0)-
                                                       (diff?f.kappa*Dot(v.gradient,normal):0);};
    return ReferenceEdge(face.edge,fun,Rule(8),Rule(12));
}

struct Norm {
    long double weight=0,l1=0,l2=0;PetscReal maximum=0;
    void Add(PetscReal error,PetscReal w){Require(std::isfinite(error)&&w>=0,"Invalid error sample");
        weight+=w;l1+=w*std::abs(error);l2+=w*error*error;maximum=std::max(maximum,std::abs(error));}
    PetscReal L1()const{return weight?static_cast<PetscReal>(l1/weight):0;}
    PetscReal L2()const{return weight?std::sqrt(static_cast<PetscReal>(l2/weight)):0;}
};
inline PetscReal Rate(PetscReal oldError,PetscReal newError,PetscReal oldH,PetscReal newH)
{return oldError>0&&newError>0?std::log(oldError/newError)/std::log(oldH/newH):
                               std::numeric_limits<PetscReal>::quiet_NaN();}
inline bool Interior(const Grid& g,std::size_t cell,PetscInt margin)
{
    const PetscInt i=static_cast<PetscInt>(cell%g.size.i),j=static_cast<PetscInt>(cell/g.size.i);
    return (g.size.i==1||(i>=margin&&i<g.size.i-margin))&&
           (g.size.j==1||(j>=margin&&j<g.size.j-margin));
}

// Convex polygon clipping gives exact step averages and separates the error
// integrals on either side of a jump. No Gauss rule is applied across a jump.
using Polygon=std::vector<Point>;
inline Polygon Clip(const Polygon& in,const Point& normal,PetscReal offset,bool positive)
{
    Polygon out;if(in.empty())return out;
    const auto distance=[&](const Point& p){return (positive?1:-1)*(Dot(normal,p)-offset);};
    Point a=in.back();PetscReal da=distance(a);
    for(const auto& b:in){const PetscReal db=distance(b);const bool ina=da>=0,inb=db>=0;
        if(ina!=inb){const PetscReal t=da/(da-db);out.push_back(Point{{a.p[0]+t*(b.p[0]-a.p[0]),a.p[1]+t*(b.p[1]-a.p[1])}});}
        if(inb)out.push_back(b);
        a=b;da=db;}
    return out;
}
inline std::pair<PetscReal,Point> AreaCentroid(const Polygon& p)
{
    if(p.size()<3)return {0,{}};
    long double area=0,mx=0,my=0;
    for(std::size_t k=1;k+1<p.size();++k){const long double ax=p[k].p[0]-p[0].p[0],ay=p[k].p[1]-p[0].p[1];
        const long double bx=p[k+1].p[0]-p[0].p[0],by=p[k+1].p[1]-p[0].p[1];const long double a=(ax*by-ay*bx)/2;
        area+=a;mx+=a*(p[0].p[0]+p[k].p[0]+p[k+1].p[0])/3;
        my+=a*(p[0].p[1]+p[k].p[1]+p[k+1].p[1])/3;}
    return {static_cast<PetscReal>(area),area>0?Point{{static_cast<PetscReal>(mx/area),static_cast<PetscReal>(my/area)}}:Point{}};
}
template<class F> void PolygonQuadrature(const Polygon& p,const GaussRule1D& q,F&& function)
{
    if(p.size()<3)return;
    for(std::size_t k=1;k+1<p.size();++k){const Point a=p[0],b=p[k],c=p[k+1];
        const PetscReal det=(b.p[0]-a.p[0])*(c.p[1]-a.p[1])-(b.p[1]-a.p[1])*(c.p[0]-a.p[0]);
        if(det<=0)continue;
        for(std::size_t i=0;i<q.points.size();++i)for(std::size_t j=0;j<q.points.size();++j){
            const PetscReal r=(q.points[i]+1)/2,s=(q.points[j]+1)/2;
            const Point x{{a.p[0]+r*(b.p[0]-a.p[0])+(1-r)*s*(c.p[0]-a.p[0]),
                           a.p[1]+r*(b.p[1]-a.p[1])+(1-r)*s*(c.p[1]-a.p[1])}};
            function(x,q.weights[i]*q.weights[j]*det*(1-r)/4);}
    }
}
inline std::ptrdiff_t Locate(const Grid& g,const Point& p)
{
    for(std::size_t c=0;c<g.cells.size();++c){bool inside=true;
        for(std::size_t k=0;k<4;++k){const auto&a=g.cells[c][k];const auto&b=g.cells[c][(k+1)%4];
            const PetscReal cross=(b.p[0]-a.p[0])*(p.p[1]-a.p[1])-(b.p[1]-a.p[1])*(p.p[0]-a.p[0]);
            if(cross < -1e-13*g.h){inside=false;break;}}
        if(inside)return static_cast<std::ptrdiff_t>(c);}
    return -1;
}
} // namespace transport_test
#endif
