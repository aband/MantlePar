// Serial numerical tests for Reconstruction. Place in src/mlweno/test/.
// Modes: 0=convergence, 1=properties, 2=discontinuity.
// The default convergence field is bounded with nonzero gradients.
// -recon_field 1 selects the strict critical-point stress test.
// ReconstructionOptions supplies weight defaults, including epsilon=1e-2.
// Rates are asserted only on interior cells.
#include "reconstruction.h"
#include <petscsys.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string>
#include <utility>
#include <vector>

namespace {
constexpr PetscReal L = 2, H = 1;
const PetscReal pi = std::acos(PetscReal(-1));
using Values = std::array<PetscReal,3>;

struct Options {
    PetscInt mode=0, mesh=0, order=3, constant=0, axis=0, field=0;
    PetscInt n0=16, levels=4, seed=7, targetSmoothness=1;
    PetscReal perturbation=.15, rateTolerance=.4;
    std::string output="reconstruction.csv";
};

// MeshInfo owns its snapshot; retain handles here for error-path cleanup.
struct Grid {
    DM dm=nullptr;
    Vec coordinates=nullptr;
    MeshInfo info;
    MeshIndex size{};
    std::vector<QuadVertices> cells;
    std::vector<PetscReal> areas;
    Grid()=default;
    Grid(const Grid&)=delete;
    Grid& operator=(const Grid&)=delete;
    ~Grid() { if(coordinates) (void)VecDestroy(&coordinates); if(dm) (void)DMDestroy(&dm); }
};

struct Norms {
    Values squared{}, maximum{};
    PetscReal area=0, constantIntegral=0, constantMax=0, conservationMax=0;
    PetscInt cells=0;
};
struct Row {
    PetscInt n=0;
    PetscReal h=0, polynomialH=0;
    std::array<Norms,3> regions; // interior, boundary, all
};
const char* const regionNames[]={"interior","boundary","all"};
const char* const fieldNames[]={"u","ux","uy"};
const char* MeshName(PetscInt mesh) { return mesh ? "quadrilateral" : "rectangular"; }

std::size_t Id(MeshIndex size, MeshIndex cell)
{ return static_cast<std::size_t>(cell.j)*size.i+cell.i; }

PetscErrorCode Near(PetscReal actual, PetscReal expected, const char* message,
                    PetscReal tolerance=2e-10)
{
    PetscFunctionBeginUser;
    PetscCheck(std::isfinite(actual) && std::isfinite(expected) &&
               std::abs(actual-expected)<=tolerance*(1+std::abs(expected)),
               PETSC_COMM_SELF,PETSC_ERR_PLIB,
               "%s: got %.17g, expected %.17g",message,
               static_cast<double>(actual),static_cast<double>(expected));
    PetscFunctionReturn(PETSC_SUCCESS);
}

template<class Function>
PetscErrorCode ExpectError(Function&& function, const char* message)
{
    PetscFunctionBeginUser;
    PetscCall(PetscPushErrorHandler(PetscReturnErrorHandler,nullptr));
    const PetscErrorCode error=function();
    PetscCall(PetscPopErrorHandler());
    PetscCheck(error!=PETSC_SUCCESS,PETSC_COMM_SELF,PETSC_ERR_PLIB,"%s",message);
    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode OpenCSV(const std::string& name, std::ofstream& stream)
{
    PetscFunctionBeginUser;
    const std::filesystem::path path(name);
    std::error_code error;
    if(!path.parent_path().empty()) std::filesystem::create_directories(path.parent_path(),error);
    PetscCheck(!error,PETSC_COMM_SELF,PETSC_ERR_FILE_OPEN,
               "Cannot create output directory: %s",error.message().c_str());
    stream.open(path);
    PetscCheck(stream.is_open(),PETSC_COMM_SELF,PETSC_ERR_FILE_OPEN,
               "Cannot open %s",name.c_str());
    stream<<std::scientific<<std::setprecision(17);
    PetscFunctionReturn(PETSC_SUCCESS);
}
PetscErrorCode CloseCSV(std::ofstream& stream)
{
    PetscFunctionBeginUser;
    stream.close();
    PetscCheck(!stream.fail(),PETSC_COMM_SELF,PETSC_ERR_FILE_WRITE,"Failed to write CSV");
    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode ReadOptions(Options& o)
{
    PetscFunctionBeginUser;
    PetscCall(PetscOptionsGetInt(nullptr,nullptr,"-recon_mode",&o.mode,nullptr));
    PetscCall(PetscOptionsGetInt(nullptr,nullptr,"-recon_mesh_type",&o.mesh,nullptr));
    PetscCall(PetscOptionsGetInt(nullptr,nullptr,"-recon_order",&o.order,nullptr));
    PetscCall(PetscOptionsGetInt(nullptr,nullptr,"-recon_constant",&o.constant,nullptr));
    PetscCall(PetscOptionsGetInt(nullptr,nullptr,"-recon_axis",&o.axis,nullptr));
    PetscCall(PetscOptionsGetInt(nullptr,nullptr,"-recon_field",&o.field,nullptr));
    PetscCall(PetscOptionsGetInt(nullptr,nullptr,"-recon_n0",&o.n0,nullptr));
    PetscCall(PetscOptionsGetInt(nullptr,nullptr,"-recon_levels",&o.levels,nullptr));
    PetscCall(PetscOptionsGetInt(nullptr,nullptr,"-recon_seed",&o.seed,nullptr));
    PetscCall(PetscOptionsGetInt(nullptr,nullptr,"-recon_target_smoothness",&o.targetSmoothness,nullptr));
    PetscCall(PetscOptionsGetReal(nullptr,nullptr,"-recon_perturbation",&o.perturbation,nullptr));
    PetscCall(PetscOptionsGetReal(nullptr,nullptr,"-recon_rate_tolerance",&o.rateTolerance,nullptr));
    char output[4096]="";
    PetscCall(PetscOptionsGetString(nullptr,nullptr,"-recon_output",output,sizeof(output),nullptr));
    if(output[0]) o.output=output;
    PetscCheck(o.mode>=0 && o.mode<=2 && o.mesh>=0 && o.mesh<=1 &&
               o.constant>=0 && o.constant<=1 && o.axis>=0 && o.axis<=2 && o.field>=0 && o.field<=1 &&
               o.targetSmoothness>=0 && o.targetSmoothness<=1,
               PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Invalid reconstruction test selector");
    PetscCheck((o.order==3 || o.order==4) && o.n0>=16 && o.levels>=3 && o.seed>=0,
               PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,
               "Require order 3 or 4, n0>=16, levels>=3, seed>=0; pseudo-1D uses (5,3)");
    PetscCheck(std::isfinite(o.perturbation) && o.perturbation>0 && o.perturbation<.25 &&
               std::isfinite(o.rateTolerance) && o.rateTolerance>=0 && o.rateTolerance<1,
               PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Invalid perturbation or rate tolerance");
    PetscCheck(o.axis==0 || o.mesh==0,PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,
               "Pseudo-1D cases use rectangular cells (the short direction has one cell)");
    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode MakeGrid(MeshIndex size, const Options& o, Grid& grid)
{
    PetscFunctionBeginUser;
    PetscCheck(size.i>0 && size.j>0 && size.i<std::numeric_limits<PetscInt>::max() &&
               size.j<std::numeric_limits<PetscInt>::max(),
               PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Invalid grid dimensions");
    // One rank owns all cells, so width 1 also supports pseudo-1D grids.
    PetscCall(DMDACreate2d(PETSC_COMM_WORLD,DM_BOUNDARY_NONE,DM_BOUNDARY_NONE,
                           DMDA_STENCIL_BOX,size.i+1,size.j+1,1,1,2,1,
                           nullptr,nullptr,&grid.dm));
    PetscCall(DMSetUp(grid.dm));
    PetscCall(DMCreateGlobalVector(grid.dm,&grid.coordinates));
    MeshParam parameters;
    parameters.L=L; parameters.H=H;
    parameters.perturbation=o.perturbation;
    parameters.seed=static_cast<std::uint64_t>(o.seed);
    if(o.mesh) PetscCall(LogicRectMesh(grid.dm,grid.coordinates,parameters));
    else PetscCall(CreateFullMesh(grid.dm,grid.coordinates,parameters));
    PetscCall(BuildMeshInfo(grid.dm,grid.coordinates,grid.info));
    grid.size=size;
    const auto nx=static_cast<std::size_t>(size.i), ny=static_cast<std::size_t>(size.j);
    PetscCheck(nx<=grid.cells.max_size()/ny,PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,
               "Grid allocation too large");
    grid.cells.resize(nx*ny); grid.areas.resize(nx*ny);
    GaussRule1D rule;
    PetscCall(CreateGaussRule(1,rule));
    long double total=0;
    bool slanted=false;
    for(PetscInt j=0;j<size.j;++j) for(PetscInt i=0;i<size.i;++i) {
        const auto id=Id(size,{i,j});
        auto& corners=grid.cells[id];
        PetscCall(grid.info.GetCellCorners({i,j},corners));
        PetscCall(ValidateQuad(corners));
        PetscCall(IntegrateCell(corners,rule,[](const Point&){return PetscReal(1);},grid.areas[id]));
        total+=grid.areas[id];
        for(std::size_t k=0;k<4;++k)
            slanted=slanted || (std::abs(corners[k].p[0]-corners[(k+1)%4].p[0])>1e-12*L/size.i &&
                                std::abs(corners[k].p[1]-corners[(k+1)%4].p[1])>1e-12*H/size.j);
    }
    PetscCall(Near(static_cast<PetscReal>(total),L*H,"Grid total area",2e-12));
    PetscCheck(!o.mesh || slanted,PETSC_COMM_SELF,PETSC_ERR_PLIB,
               "Quadrilateral test requires actual slanted edges");
    PetscFunctionReturn(PETSC_SUCCESS);
}

// Default field: 1+.1*x+.2*y+.02*sin(pi*x)*cos(2*pi*y)+.05*exp(x/4+y/2).
// Both first derivatives stay positive. Field 1 removes the linear background
// and uses sinusoid amplitude .1, retaining critical points as a stress case.
// 1-D default: 1+.2*z+.02*sin(k*z)+.05*exp(z/4), k=pi (x), 2*pi (y).
Values Exact(const Point& p, PetscInt axis, PetscInt field)
{
    const PetscReal x=p.p[0], y=p.p[1], amplitude=field ? .1 : .02;
    if(axis) {
        const PetscReal frequency=axis==1 ? pi : 2*pi;
        const PetscReal z=axis==1 ? x : y, linear=field ? 0 : .2;
        const PetscReal u=1+linear*z+amplitude*std::sin(frequency*z)+.05*std::exp(z/4);
        const PetscReal derivative=linear+amplitude*frequency*std::cos(frequency*z)+.0125*std::exp(z/4);
        return {{u,axis==1 ? derivative : 0,axis==2 ? derivative : 0}};
    }
    const PetscReal ax=field ? 0 : .1, ay=field ? 0 : .2;
    const PetscReal e=.05*std::exp(x/4+y/2);
    return {{1+ax*x+ay*y+amplitude*std::sin(pi*x)*std::cos(2*pi*y)+e,
              ax+amplitude*pi*std::cos(pi*x)*std::cos(2*pi*y)+e/4,
              ay-2*amplitude*pi*std::sin(pi*x)*std::sin(2*pi*y)+e/2}};
}
PetscReal Sinc(PetscReal z)
{ return std::abs(z)<1e-4 ? 1-z*z/6+z*z*z*z/120 : std::sin(z)/z; }
PetscReal Sinhc(PetscReal z)
{ return std::abs(z)<1e-4 ? 1+z*z/6+z*z*z*z/120 : std::sinh(z)/z; }

PetscReal RectangularAverage(const QuadVertices& c, PetscInt axis, PetscInt field)
{
    const PetscReal amplitude=field ? .1 : .02;
    const PetscReal dx=c[1].p[0]-c[0].p[0], dy=c[3].p[1]-c[0].p[1];
    const PetscReal x=c[0].p[0]+dx/2, y=c[0].p[1]+dy/2;
    if(axis) {
        const PetscReal z=axis==1 ? x : y, dz=axis==1 ? dx : dy;
        const PetscReal frequency=axis==1 ? pi : 2*pi;
        return 1+(field ? 0 : .2*z)+amplitude*std::sin(frequency*z)*Sinc(frequency*dz/2)
                +.05*std::exp(z/4)*Sinhc(dz/8);
    }
    return 1+(field ? 0 : .1*x+.2*y)+amplitude*std::sin(pi*x)*Sinc(pi*dx/2)*std::cos(2*pi*y)*Sinc(pi*dy)
            +.05*std::exp(x/4+y/2)*Sinhc(dx/8)*Sinhc(dy/4);
}

template<class Function>
PetscErrorCode Averages(const Grid& grid, Function&& field, std::vector<PetscReal>& result)
{
    PetscFunctionBeginUser;
    GaussRule1D rule;
    PetscCall(CreateGaussRule(10,rule));
    result.resize(grid.cells.size());
    for(std::size_t k=0;k<result.size();++k) {
        PetscCall(IntegrateCell(grid.cells[k],rule,field,result[k]));
        result[k]/=grid.areas[k];
    }
    PetscFunctionReturn(PETSC_SUCCESS);
}

ReconstructionOptions Configuration(const Options& o)
{
    ReconstructionOptions c;
    c.useConstant=o.constant!=0;
    if(o.axis==0) {
        c.large.size={o.order,o.order}; c.large.order=o.order-1;
        const PetscInt shift=(o.order-1)/2;
        c.large.offsets={{-shift,-shift}};
        c.small.size={o.order-1,o.order-1}; c.small.order=o.order-2;
        for(PetscInt j=0;j<c.small.size.j;++j)
            for(PetscInt i=0;i<c.small.size.i;++i) c.small.offsets.push_back({-i,-j});
    } else {
        c.large.size=o.axis==1 ? MeshIndex{5,1} : MeshIndex{1,5};
        c.large.order=4;
        c.large.offsets={o.axis==1 ? MeshIndex{-2,0} : MeshIndex{0,-2}};
        c.small.size=o.axis==1 ? MeshIndex{3,1} : MeshIndex{1,3};
        c.small.order=2;
        for(PetscInt i=0;i<3;++i)
            c.small.offsets.push_back(o.axis==1 ? MeshIndex{-i,0} : MeshIndex{0,-i});
    }
    if(o.targetSmoothness) c.small.smoothness=ReconstructionSmoothness::TargetCell;
    return c;
}

PetscErrorCode Reconstructed(const Reconstruction& reconstruction, const Point& p, Values& v)
{
    PetscFunctionBeginUser;
    PetscCall(reconstruction.Evaluate(p,v[0]));
    PetscCall(reconstruction.EvaluateDerivative(p,1,0,v[1]));
    PetscCall(reconstruction.EvaluateDerivative(p,0,1,v[2]));
    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode CheckWeights(const Reconstruction& r, PetscReal& w0)
{
    PetscFunctionBeginUser;
    PetscCall(r.ConstantWeight(w0));
    PetscCheck(std::isfinite(w0) && w0>=0 && w0<=1,
               PETSC_COMM_SELF,PETSC_ERR_PLIB,"Invalid constant weight");
    long double sum=w0;
    for(std::size_t k=0;k<r.CandidateCount();++k) {
        ReconstructionCandidateInfo info;
        PetscCall(r.GetCandidate(k,info));
        PetscCheck(std::isfinite(info.smoothness) && info.smoothness>=0 &&
                   std::isfinite(info.nonlinearWeight) && info.nonlinearWeight>=0 &&
                   info.nonlinearWeight<=1,PETSC_COMM_SELF,PETSC_ERR_PLIB,"Invalid candidate diagnostics");
        sum+=info.nonlinearWeight;
    }
    PetscCall(Near(static_cast<PetscReal>(sum),1,"Weight normalization",5e-14));
    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode Measure(PetscInt n, const Options& o, Row& row)
{
    PetscFunctionBeginUser;
    Grid grid;
    const MeshIndex size=o.axis==1 ? MeshIndex{n,1} : (o.axis==2 ? MeshIndex{1,n} : MeshIndex{n,n});
    PetscCall(MakeGrid(size,o,grid));
    row.n=n;
    row.polynomialH=std::sqrt(L*H/static_cast<PetscReal>(size.i)/size.j);
    row.h=o.axis==1 ? L/n : (o.axis==2 ? H/n : row.polynomialH);
    std::vector<PetscReal> averages;
    if(o.mesh) {
        PetscCall(Averages(grid,[&](const Point& p){return Exact(p,o.axis,o.field)[0];},averages));
    } else {
        averages.resize(grid.cells.size());
        for(std::size_t k=0;k<averages.size();++k) averages[k]=RectangularAverage(grid.cells[k],o.axis,o.field);
    }
    const auto config=Configuration(o);
    const auto completeCount=config.large.offsets.size()+config.small.offsets.size();
    const MeshRange full{{0,0},size};
    GaussRule1D rule;
    PetscCall(CreateGaussRule(8,rule));
    for(PetscInt j=0;j<size.j;++j) for(PetscInt i=0;i<size.i;++i) {
        const auto id=Id(size,{i,j});
        const auto& cell=grid.cells[id];
        Reconstruction r;
        PetscCall(r.Initialize(grid.info,{i,j},config,row.polynomialH));
        // Preserve the legacy NOMINAL area scale also on perturbed meshes.
        PetscCall(r.Update(averages,full,row.polynomialH*row.polynomialH));
        PetscReal w0;
        PetscCall(CheckWeights(r,w0));
        const std::size_t region=r.CandidateCount()==completeCount ? 0 : 1;
        Norms local;
        local.area=grid.areas[id]; local.cells=1;
        local.constantIntegral=w0*local.area; local.constantMax=w0;
        long double reconstructedIntegral=0;
        for(std::size_t qy=0;qy<rule.points.size();++qy)
            for(std::size_t qx=0;qx<rule.points.size();++qx) {
                const Point reference{{rule.points[qx],rule.points[qy]}};
                const Point p=MapCellPoint(reference,cell);
                const PetscReal weight=rule.weights[qx]*rule.weights[qy]*CellJacobian(reference,cell);
                Values value;
                PetscCall(Reconstructed(r,p,value));
                const auto exact=Exact(p,o.axis,o.field);
                reconstructedIntegral+=static_cast<long double>(weight)*value[0];
                for(std::size_t d=0;d<3;++d) {
                    const PetscReal error=std::abs(value[d]-exact[d]);
                    local.squared[d]+=weight*error*error;
                    local.maximum[d]=std::max(local.maximum[d],error);
                }
            }
        // Sample vertices, edge midpoints and center too: Linf is a sampled max.
        for(PetscReal sy : {PetscReal(-1),PetscReal(0),PetscReal(1)})
            for(PetscReal sx : {PetscReal(-1),PetscReal(0),PetscReal(1)}) {
                const Point p=MapCellPoint(Point{{sx,sy}},cell);
                Values value;
                PetscCall(Reconstructed(r,p,value));
                const auto exact=Exact(p,o.axis,o.field);
                for(std::size_t d=0;d<3;++d)
                    local.maximum[d]=std::max(local.maximum[d],std::abs(value[d]-exact[d]));
            }
        const PetscReal mean=static_cast<PetscReal>(reconstructedIntegral/local.area);
        local.conservationMax=std::abs(mean-averages[id]);
        PetscCall(Near(mean,averages[id],"Target-cell mean conservation",2e-10));
        for(std::size_t destination : {region,std::size_t(2)}) {
            auto& result=row.regions[destination];
            result.area+=local.area; result.cells+=1;
            result.constantIntegral+=local.constantIntegral;
            result.constantMax=std::max(result.constantMax,w0);
            result.conservationMax=std::max(result.conservationMax,local.conservationMax);
            for(std::size_t d=0;d<3;++d) {
                result.squared[d]+=local.squared[d];
                result.maximum[d]=std::max(result.maximum[d],local.maximum[d]);
            }
        }
    }
    PetscCheck(row.regions[0].cells>0 && row.regions[1].cells>0,
               PETSC_COMM_SELF,PETSC_ERR_PLIB,"Expected both interior and boundary cells");
    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscReal Error(const Norms& norm, std::size_t field, bool maximum)
{ return maximum ? norm.maximum[field] : std::sqrt(norm.squared[field]); }

PetscReal Rate(PetscReal oldError, PetscReal newError, PetscReal oldH, PetscReal newH)
{
    return oldError>0 && newError>0 ? std::log(oldError/newError)/std::log(oldH/newH)
                                  : std::numeric_limits<PetscReal>::quiet_NaN();
}

void WriteHeader(std::ostream& out)
{
    out<<"mesh,axis,field,order,constant,target_smoothness,N,h,polynomial_h,seed,perturbation,epsilon,constant_linear_weight,region,cells,area";
    for(const auto* name : fieldNames)
        out<<','<<name<<"_l2,"<<name<<"_l2_rate,"<<name<<"_linf,"<<name<<"_linf_rate";
    out<<",constant_mean,constant_max,conservation_max\n";
}
void WriteRows(std::ostream& out, const Options& o, const Row& row, const Row* previous)
{
    const auto config=Configuration(o);
    for(std::size_t region=0;region<3;++region) {
        const auto& norm=row.regions[region];
        out<<MeshName(o.mesh)<<','<<o.axis<<','<<o.field<<','<<(o.axis ? 5 : o.order)<<','<<o.constant<<','
           <<o.targetSmoothness<<','<<row.n<<','<<row.h<<','<<row.polynomialH<<','<<o.seed<<','
           <<(o.mesh ? o.perturbation : 0)<<','<<config.epsilon<<','
           <<config.constantLinearWeight<<','<<regionNames[region]<<','
           <<norm.cells<<','<<norm.area;
        for(std::size_t d=0;d<3;++d) for(bool maximum : {false,true}) {
            const PetscReal error=Error(norm,d,maximum);
            out<<','<<error<<',';
            if(previous) {
                const PetscReal rate=Rate(Error(previous->regions[region],d,maximum),
                                         error,previous->h,row.h);
                if(std::isfinite(rate)) out<<rate;
            }
        }
        out<<','<<norm.constantIntegral/norm.area<<','<<norm.constantMax<<','
           <<norm.conservationMax<<'\n';
    }
}

PetscErrorCode CheckRates(const std::vector<Row>& rows, const Options& o)
{
    PetscFunctionBeginUser;
    const std::size_t begin=rows.size()-3;
    bool passed=true;
    for(std::size_t d=0;d<3;++d) {
        if((o.axis==1 && d==2) || (o.axis==2 && d==1)) {
            for(const auto& row : rows)
                PetscCall(Near(row.regions[2].maximum[d],0,"Inactive derivative",1e-12));
            continue;
        }
        const PetscReal expected=(o.axis ? 5 : o.order)-(d ? 1 : 0);
        for(bool maximum : {false,true}) {
            PetscReal mx=0,my=0;
            for(std::size_t k=begin;k<rows.size();++k) {
                const PetscReal error=Error(rows[k].regions[0],d,maximum);
                PetscCheck(std::isfinite(error) && error>0,PETSC_COMM_SELF,PETSC_ERR_PLIB,
                           "Interior error must be finite and positive for rate fitting");
                mx+=std::log(rows[k].h)/3; my+=std::log(error)/3;
            }
            PetscReal numerator=0,denominator=0;
            bool decreasing=true;
            for(std::size_t k=begin;k<rows.size();++k) {
                const PetscReal x=std::log(rows[k].h)-mx;
                numerator+=x*(std::log(Error(rows[k].regions[0],d,maximum))-my);
                denominator+=x*x;
                if(k>begin) decreasing=decreasing &&
                    Error(rows[k].regions[0],d,maximum)<Error(rows[k-1].regions[0],d,maximum);
            }
            const PetscReal rate=numerator/denominator;
            const bool ok=decreasing && std::isfinite(rate) && rate>=expected-o.rateTolerance;
            passed=passed && ok;
            PetscCall(PetscPrintf(PETSC_COMM_SELF,"interior %-2s %-11s rate=%.4f expected=%.0f %s\n",
                fieldNames[d],maximum ? "sampled-max" : "L2",
                static_cast<double>(rate),static_cast<double>(expected),ok ? "PASS" : "FAIL"));
        }
    }
    PetscCheck(passed,PETSC_COMM_SELF,PETSC_ERR_PLIB,"Interior convergence rate check failed");
    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode Convergence(const Options& o)
{
    PetscFunctionBeginUser;
    std::ofstream csv;
    PetscCall(OpenCSV(o.output,csv));
    WriteHeader(csv);
    PetscCall(PetscPrintf(PETSC_COMM_SELF,
        "Reconstruction convergence: %s, axis=%lld, constant=%lld, field=%lld\n"
        "Field 0: bounded smooth function with nonzero gradients; field 1: critical-point stress test.\n"
        "Analytic formulas and 1D variants are documented in the source.\n"
        "L2 is a physical integral over each region; boundary strip area shrinks on refinement.\n"
        "Only complete-stencil INTERIOR rates are pass/fail gates; boundary/all errors are reported.\n",
        MeshName(o.mesh),static_cast<long long>(o.axis),static_cast<long long>(o.constant),
        static_cast<long long>(o.field)));
    std::vector<Row> rows;
    PetscInt n=o.n0;
    for(PetscInt level=0;level<o.levels;++level) {
        Row row;
        PetscCall(Measure(n,o,row));
        WriteRows(csv,o,row,rows.empty() ? nullptr : &rows.back());
        csv.flush();
        PetscCheck(csv.good(),PETSC_COMM_SELF,PETSC_ERR_FILE_WRITE,"CSV write failed");
        PetscCall(PetscPrintf(PETSC_COMM_SELF,
            "N=%lld interior L2: u=%.6e ux=%.6e uy=%.6e; max omega0=%.3e; max mean error=%.3e\n",
            static_cast<long long>(n),static_cast<double>(Error(row.regions[0],0,false)),
            static_cast<double>(Error(row.regions[0],1,false)),
            static_cast<double>(Error(row.regions[0],2,false)),
            static_cast<double>(row.regions[0].constantMax),
            static_cast<double>(row.regions[2].conservationMax)));
        rows.push_back(row);
        if(level+1<o.levels) {
            PetscCheck(n<=(std::numeric_limits<PetscInt>::max()-1)/2,
                       PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Refinement overflows PetscInt");
            n*=2;
        }
    }
    PetscCall(CloseCSV(csv));
    PetscCall(CheckRates(rows,o));
    PetscFunctionReturn(PETSC_SUCCESS);
}

// Common polynomial represented by both nonsquare families:
// p=(1+.2*x)*(1+.3*y+.1*y*y), including the mixed term x*y*y.
PetscReal Polynomial(const Point& p, PetscInt dx, PetscInt dy)
{
    PetscReal x=0,y=0;
    if(dx==0) x=1+.2*p.p[0]; else if(dx==1) x=.2;
    if(dy==0) y=1+.3*p.p[1]+.1*p.p[1]*p.p[1];
    else if(dy==1) y=.3+.2*p.p[1]; else if(dy==2) y=.2;
    return x*y;
}

PetscErrorCode PolynomialMean(const Reconstruction& r, const QuadVertices& corners,
                              PetscReal expected)
{
    PetscFunctionBeginUser;
    GaussRule1D rule;
    PetscCall(CreateGaussRule(6,rule));
    long double integral=0,area=0;
    for(std::size_t j=0;j<rule.points.size();++j)
        for(std::size_t i=0;i<rule.points.size();++i) {
            const Point ref{{rule.points[i],rule.points[j]}};
            const PetscReal w=rule.weights[i]*rule.weights[j]*CellJacobian(ref,corners);
            PetscReal value;
            PetscCall(r.Evaluate(MapCellPoint(ref,corners),value));
            integral+=w*value; area+=w;
        }
    PetscCall(Near(static_cast<PetscReal>(integral/area),expected,"Polynomial target mean"));
    PetscFunctionReturn(PETSC_SUCCESS);
}

// Independent analytic-derivative integral, rather than reusing the matrix
// smoothness routine to predict its own answer.
PetscErrorCode AnalyticIndicator(const Grid& grid, const ReconstructionCandidateInfo& info,
                                 PetscReal h, PetscReal& sigma)
{
    PetscFunctionBeginUser;
    TensorStencilPoly polynomial;
    PetscCall(polynomial.Initialize(grid.info,info.start,info.size,info.order,h));
    QuadVertices region;
    if(info.smoothnessRegion==ReconstructionSmoothness::TargetCell) {
        PetscCall(grid.info.GetCellCorners({info.start.i+info.targetOffset.i,
                                           info.start.j+info.targetOffset.j},region));
    } else {
        const auto center=polynomial.Center();
        region={{Point{{center.p[0]-h/2,center.p[1]-h/2}},
                 Point{{center.p[0]+h/2,center.p[1]-h/2}},
                 Point{{center.p[0]+h/2,center.p[1]+h/2}},
                 Point{{center.p[0]-h/2,center.p[1]+h/2}}}};
    }
    GaussRule1D rule;
    PetscCall(CreateGaussRule(6,rule));
    PetscReal area;
    PetscCall(IntegrateCell(region,rule,[](const Point&){return PetscReal(1);},area));
    sigma=0;
    for(PetscInt dy=0;dy<=2;++dy) for(PetscInt dx=0;dx<=1;++dx) {
        const PetscInt degree=dx+dy;
        if(degree==0 || degree>info.order) continue;
        PetscReal term;
        PetscCall(IntegrateCell(region,rule,[&](const Point& p){
            const auto derivative=Polynomial(p,dx,dy); return derivative*derivative;
        },term));
        sigma+=std::pow(h,2*degree)*term/area;
    }
    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode CheckNonsquare(const Grid& grid, bool targetRegion, bool constant)
{
    PetscFunctionBeginUser;
    ReconstructionOptions config;
    config.large.size={4,3}; config.large.order=3;
    config.large.offsets={{-2,-1},{-1,-1}}; config.large.linearWeights={.7,1.4};
    config.small.size={2,3}; config.small.order=2;
    config.small.offsets={{-1,0},{0,-1},{-1,-2}}; config.small.linearWeights={.2,.5,.8};
    config.large.smoothness=targetRegion ? ReconstructionSmoothness::TargetCell :
                                                   ReconstructionSmoothness::ReferenceSquare;
    config.small.smoothness=config.large.smoothness;
    config.useConstant=constant; config.constantLinearWeight=2; config.epsilon=.4; config.s=2;
    const MeshIndex target{5,5};
    const PetscReal h=std::sqrt(L*H/grid.size.i/grid.size.j), weightArea=.6;
    Reconstruction r;
    PetscCall(r.Initialize(grid.info,target,config,h));
    std::vector<PetscReal> values;
    PetscCall(Averages(grid,[](const Point& p){return Polynomial(p,0,0);},values));
    // Gather only the required patch to check nonzero global-to-local origins.
    const MeshRange patchRange=r.RequiredCells();
    std::vector<PetscReal> patch;
    for(PetscInt j=patchRange.begin.j;j<patchRange.end.j;++j)
        for(PetscInt i=patchRange.begin.i;i<patchRange.end.i;++i) patch.push_back(values[Id(grid.size,{i,j})]);
    PetscCall(r.Update(patch,patchRange,weightArea));
    PetscReal w0;
    PetscCall(CheckWeights(r,w0));
    const PetscReal mean=values[Id(grid.size,target)];
    const auto& corners=grid.cells[Id(grid.size,target)];
    for(const Point ref : {Point{{-.7,-.3}},Point{{.45,.8}},Point{{1,-1}}}) {
        const Point point=MapCellPoint(ref,corners);
        for(const MeshIndex derivative : {MeshIndex{0,0},MeshIndex{1,0},MeshIndex{0,1},
                                          MeshIndex{1,1},MeshIndex{0,2},MeshIndex{1,2}}) {
            PetscReal actual;
            PetscCall(r.EvaluateDerivative(point,derivative.i,derivative.j,actual));
            const PetscReal expected=(1-w0)*Polynomial(point,derivative.i,derivative.j)
                +((derivative.i==0 && derivative.j==0) ? w0*mean : 0);
            PetscCall(Near(actual,expected,"Nonsquare polynomial/derivative"));
        }
    }
    PetscCall(PolynomialMean(r,corners,mean));
    std::vector<long double> alpha;
    long double total=0, effective=0;
    for(std::size_t k=0;k<r.CandidateCount();++k) {
        ReconstructionCandidateInfo info;
        PetscCall(r.GetCandidate(k,info));
        PetscReal sigma=0;
        PetscCall(AnalyticIndicator(grid,info,h,sigma));
        PetscCall(Near(info.smoothness,sigma,"Analytic smoothness",2e-11));
        const PetscInt order=info.order+1;
        const PetscInt eta=order==1 ? 1 : (order==2 ? 3 : 4);
        const long double a=info.linearWeight/std::pow(static_cast<long double>(sigma)+
                                 static_cast<long double>(config.epsilon)*weightArea,config.s*order+eta);
        alpha.push_back(a); total+=a;
    }
    const long double a0=constant ? config.constantLinearWeight/
        std::pow(static_cast<long double>(config.epsilon)*weightArea,config.s+1) : 0;
    total+=a0; effective=a0/total;
    PetscCall(Near(w0,static_cast<PetscReal>(a0/total),"Direct-formula constant weight",2e-11));
    for(std::size_t k=0;k<alpha.size();++k) {
        ReconstructionCandidateInfo info;
        PetscCall(r.GetCandidate(k,info));
        PetscCall(Near(info.nonlinearWeight,static_cast<PetscReal>(alpha[k]/total),
                       "Direct-formula polynomial weight",2e-11));
        effective+=alpha[k]/total*(info.order+1);
    }
    PetscReal order;
    PetscCall(r.EffectiveOrder(order));
    PetscCall(Near(order,static_cast<PetscReal>(effective),"Effective order diagnostic",2e-11));
    // Constant fields and derivatives must be exact, with or without p0.
    std::fill(patch.begin(),patch.end(),1e8);
    PetscCall(r.Update(patch,patchRange,weightArea));
    PetscReal value;
    PetscCall(r.Evaluate(corners[0],value));
    PetscCheck(value==1e8,PETSC_COMM_SELF,PETSC_ERR_PLIB,"Constant field was not exact");
    PetscCall(r.EvaluateDerivative(corners[0],1,0,value));
    PetscCheck(value==0,PETSC_COMM_SELF,PETSC_ERR_PLIB,"Constant derivative was not zero");
    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode CheckBoundaryAndErrors(const Grid& grid)
{
    PetscFunctionBeginUser;
    Options o;
    auto config=Configuration(o);
    config.small.linearWeights={1,2,3,4};
    const std::array<MeshIndex,4> targets{{{0,0},{0,5},{5,5},{grid.size.i-1,grid.size.j-1}}};
    const std::array<std::size_t,4> counts{{1,2,5,1}};
    for(std::size_t k=0;k<targets.size();++k) {
        Reconstruction r;
        PetscCall(r.Initialize(grid.info,targets[k],config,.1));
        PetscCheck(r.CandidateCount()==counts[k],PETSC_COMM_SELF,PETSC_ERR_PLIB,"Boundary candidate count");
        if(k==0 || k==3) {
            ReconstructionCandidateInfo info;
            PetscCall(r.GetCandidate(0,info));
            PetscCheck(info.family==ReconstructionFamilyKind::Small &&
                       info.configuredIndex==(k==0 ? 0U : 3U) &&
                       info.linearWeight==(k==0 ? 1 : 4),
                       PETSC_COMM_SELF,PETSC_ERR_PLIB,"Boundary candidate/weight mapping");
        }
    }
    Reconstruction r;
    const MeshIndex target{5,5};
    PetscCall(r.Initialize(grid.info,target,config,.1));
    PetscReal value=42;
    PetscCall(ExpectError([&](){return r.Evaluate(Point{{1,.5}},value);},"Evaluation before Update should fail"));
    PetscCall(Near(value,42,"Failed output remains unchanged",0));
    const auto range=r.RequiredCells();
    std::vector<PetscReal> patch(static_cast<std::size_t>(range.Size().i)*range.Size().j,2);
    PetscCall(r.Update(patch,range,.01));
    const MeshRange oneCell{target,{target.i+1,target.j+1}};
    PetscCall(ExpectError([&](){return r.Update({2},oneCell,.01);},"Missing solution halo should fail"));
    PetscCall(ExpectError([&](){return r.Update(patch,range,0);},"Zero area should fail"));
    auto invalid=config; invalid.small.offsets[0]={1,0};
    PetscCall(ExpectError([&](){return r.Initialize(grid.info,target,invalid,.1);},"Target-excluding stencil"));
    invalid=config; invalid.small.linearWeights[0]=-1;
    PetscCall(ExpectError([&](){return r.Initialize(grid.info,target,invalid,.1);},"Negative linear weight"));
    invalid=config; invalid.small.linearWeights.pop_back();
    PetscCall(ExpectError([&](){return r.Initialize(grid.info,target,invalid,.1);},"Mismatched weight count"));
    PetscCall(r.Evaluate(Point{{1,.5}},value));
    PetscCall(Near(value,2,"Failed updates preserve old state",0));
    Reconstruction copy=r;
    std::fill(patch.begin(),patch.end(),3);
    PetscCall(r.Update(patch,range,.01));
    PetscCall(copy.Evaluate(Point{{1,.5}},value));
    PetscCall(Near(value,2,"Independent copy",0));
    Reconstruction moved=std::move(copy);
    PetscCheck(!copy.IsInitialized() && moved.HasWeights(),PETSC_COMM_SELF,PETSC_ERR_PLIB,"Move state");
    // Explicit constant-only reconstruction.
    ReconstructionOptions constantOnly;
    constantOnly.useConstant=true;
    PetscCall(r.Initialize(grid.info,target,constantOnly,.1));
    PetscCall(r.Update({7},oneCell,.01));
    PetscCall(r.ConstantWeight(value)); PetscCall(Near(value,1,"Constant-only weight",0));
    PetscCall(r.Evaluate(Point{{1,.5}},value)); PetscCall(Near(value,7,"Constant-only value",0));
    PetscCall(r.EvaluateDerivative(Point{{1,.5}},0,1,value));
    PetscCall(Near(value,0,"Constant-only derivative",0));
    // Tiny epsilon*A and powers that would underflow with direct arithmetic.
    config=Configuration(o); config.large.order=100;
    config.large.offsets={{-1,-1},{0,0}}; config.large.linearWeights={1,3};
    config.useConstant=true; config.epsilon=1e-300;
    PetscCall(r.Initialize(grid.info,target,config,.1));
    const auto extremeRange=r.RequiredCells();
    patch.assign(static_cast<std::size_t>(extremeRange.Size().i)*extremeRange.Size().j,1e8);
    PetscCall(r.Update(patch,extremeRange,1e-300));
    ReconstructionCandidateInfo info;
    PetscCall(r.GetCandidate(0,info)); PetscCall(Near(info.nonlinearWeight,.25,"Extreme weights",3e-13));
    PetscCall(r.GetCandidate(1,info)); PetscCall(Near(info.nonlinearWeight,.75,"Extreme weights",3e-13));
    PetscCall(CheckWeights(r,value));
    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode Properties(const Options& o)
{
    PetscFunctionBeginUser;
    Grid grid;
    PetscCall(MakeGrid({12,10},o,grid));
    for(bool target : {false,true}) for(bool constant : {false,true})
        PetscCall(CheckNonsquare(grid,target,constant));
    PetscCall(CheckBoundaryAndErrors(grid));
    PetscCall(PetscPrintf(PETSC_COMM_SELF,
        "PASS properties (%s): nonsquare polynomials/derivatives, analytic indicators, legacy weights,\n"
        "constants, conservation, boundary filtering, missing solution halo, state, and extreme weights.\n",
        MeshName(o.mesh)));
    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode Discontinuity(const Options& input)
{
    PetscFunctionBeginUser;
    Options o=input; o.mesh=0; o.axis=0; o.order=3;
    Grid grid;
    const MeshIndex size{64,8};
    PetscCall(MakeGrid(size,o,grid));
    std::vector<PetscReal> averages(grid.cells.size());
    for(PetscInt j=0;j<size.j;++j) for(PetscInt i=0;i<size.i;++i)
        averages[Id(size,{i,j})]=i<size.i/2 ? 1 : 0; // Exact averages; jump x=1.
    const MeshRange full{{0,0},size};
    const PetscReal h=std::sqrt(L*H/size.i/size.j);
    const std::filesystem::path base(o.output);
    std::ofstream profile,weights;
    PetscCall(OpenCSV(o.output,profile));
    PetscCall(OpenCSV((base.parent_path()/(base.stem().string()+"_weights.csv")).string(),weights));
    profile<<"constant,cell,x,y,u,exact,constant_weight\n";
    weights<<"constant,cell,x,family,candidate,start_i,start_j,size_i,size_j,sigma,weight,crosses_jump\n";
    for(PetscInt constant=0;constant<=1;++constant) {
        o.constant=constant;
        const auto config=Configuration(o);
        PetscReal minimum=1,maximum=0,maxCrossingWeight=0;
        PetscInt suppressed=0;
        for(PetscInt i=0;i<size.i;++i) {
            const MeshIndex target{i,size.j/2};
            const auto& cell=grid.cells[Id(size,target)];
            Reconstruction r;
            PetscCall(r.Initialize(grid.info,target,config,h));
            PetscCall(r.Update(averages,full,h*h));
            PetscReal w0;
            PetscCall(CheckWeights(r,w0));
            const Point center=MapCellPoint(Point{{0,0}},cell);
            PetscReal crossing=0;
            bool hasCrossing=false,hasSmooth=false;
            for(std::size_t k=0;k<r.CandidateCount();++k) {
                ReconstructionCandidateInfo info;
                PetscCall(r.GetCandidate(k,info));
                const bool crosses=info.start.i<size.i/2 && info.start.i+info.size.i>size.i/2;
                hasCrossing=hasCrossing || crosses; hasSmooth=hasSmooth || !crosses;
                if(crosses) crossing+=info.nonlinearWeight;
                weights<<constant<<','<<i<<','<<center.p[0]<<','
                       <<(info.family==ReconstructionFamilyKind::Large ? "large" : "small")<<','
                       <<info.configuredIndex<<','<<info.start.i<<','<<info.start.j<<','
                       <<info.size.i<<','<<info.size.j<<','<<info.smoothness<<','
                       <<info.nonlinearWeight<<','<<crosses<<'\n';
            }
            weights<<constant<<','<<i<<','<<center.p[0]<<",constant,0,"
                   <<i<<','<<target.j<<",1,1,0,"<<w0<<",0\n";
            if(hasCrossing && hasSmooth) {
                maxCrossingWeight=std::max(maxCrossingWeight,crossing);
                PetscCheck(crossing<1e-4,PETSC_COMM_SELF,PETSC_ERR_PLIB,
                           "A jump-crossing candidate was not suppressed");
                ++suppressed;
            }
            for(PetscInt q=0;q<=12;++q) {
                const Point p=MapCellPoint(Point{{-1+2.0*q/12,0}},cell);
                PetscReal value;
                PetscCall(r.Evaluate(p,value));
                minimum=std::min(minimum,value); maximum=std::max(maximum,value);
                profile<<constant<<','<<i<<','<<p.p[0]<<','<<p.p[1]<<','<<value<<','
                       <<(p.p[0]<1 ? 1 : 0)<<','<<w0<<'\n';
                if(!hasCrossing)
                    PetscCall(Near(value,averages[Id(size,target)],"Constant region of step",1e-12));
            }
            PetscCall(PolynomialMean(r,cell,averages[Id(size,target)]));
        }
        PetscCheck(suppressed>0,PETSC_COMM_SELF,PETSC_ERR_PLIB,"Jump test must exercise crossing stencils");
        PetscCall(PetscPrintf(PETSC_COMM_SELF,
            "Step, constant=%lld: overshoot=%.6e undershoot=%.6e max crossing weight=%.6e\n",
            static_cast<long long>(constant),static_cast<double>(std::max(PetscReal(0),maximum-1)),
            static_cast<double>(std::max(PetscReal(0),-minimum)),static_cast<double>(maxCrossingWeight)));
    }
    PetscCall(CloseCSV(profile)); PetscCall(CloseCSV(weights));
    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode Run()
{
    PetscFunctionBeginUser;
    PetscMPIInt size;
    PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD,&size));
    PetscCheck(size==1,PETSC_COMM_WORLD,PETSC_ERR_ARG_WRONG,
               "These tests require one MPI rank; distributed halo exchange is outside this suite");
    Options o;
    PetscCall(ReadOptions(o));
    if(o.mode==0) PetscCall(Convergence(o));
    else if(o.mode==1) PetscCall(Properties(o));
    else PetscCall(Discontinuity(o));
    PetscFunctionReturn(PETSC_SUCCESS);
}
} // namespace

int main(int argc, char** argv)
{
    PetscErrorCode error=PetscInitialize(&argc,&argv,nullptr,
        "ML-WENO reconstruction: -recon_mode 0 (convergence), 1 (properties), 2 (step).\n");
    if(error) return static_cast<int>(error);
    try {
        error=Run();
    } catch(const std::exception& exception) {
        std::cerr<<"Reconstruction test exception: "<<exception.what()<<'\n';
        error=PETSC_ERR_LIB;
    }
    const PetscErrorCode finalError=PetscFinalize();
    return static_cast<int>(error ? error : finalError);
}
