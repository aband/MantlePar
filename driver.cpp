// MantlePar shared steady-flow driver.
// Place this file at MantlePar/driver.cpp, alongside src/ and example/.
//
// Requires the previously generated schema-v2 src/core/input.{h,cpp}, the
// reconstructed core/MFEM modules, C++17, PETSc 3.23+, yaml-cpp, and parallel
// HDF5 built with the same MPICH as PETSc. Compile this translation unit once
// and link it against mantle_mfem (which links mantle_core) and parallel HDF5.
// All driver-local helpers are defined below; no example/common files are used.
//
// Invocation, after registering the executable with CMake:
//   /home/renpo/system/mpich-install/bin/mpiexec -n 2 <executable> -input example/<case>/input.yaml
// The YAML rank count/process grid must match the actual MPI communicator.
// Input/output paths in YAML are resolved relative to that YAML file.
//
// Steady Stokes, Darcy and coupled saddle-point systems share the same path:
// input -> mesh/quadrature -> coefficient samples -> assembled blocks ->
// boundary lifting -> linear system -> solver -> fields/reports.
// YAML can also request refinement studies and exact/finer-mesh references.
// Field output is parallel HDF5 + XDMF; profiles/errors/solver reports are CSV.
// PNG/PDF plotting is an external postprocessing step, not performed here.
//
// The prescribed-porosity and Gauss-point HDF5 modes are supported. The HDF5
// input contract under porosity.group is:
//   cell_ids[Nc], cell_values[Nc,q*q], edge_ids[Ne], edge_values[Ne,e]
//     (these four dataset names are configurable in YAML),
//   cell_corners[Nc,4,2], edge_vertices[Ne,2,2],
//   cell_rule[q,2], edge_rule[e,2], time[1].
// IDs are natural IDs; corners are CCW; edge samples follow increasing logical
// i/j; each rule row stores a point and weight on [-1,1]. Geometry, quadrature
// and time must match the current mesh. Shared edges have one canonical trace.
// At a prescribed jump aligned with an internal edge, that trace is harmonic.
//
// Darcy velocity is the assembled rescaled unknown. Physical segregation flux
// is phi^(1+theta)*darcy_velocity. With only Gauss-point porosity, cell-average
// phi is used for visualizing that derived field; physical-point flux errors
// require a porosity reconstruction and are rejected in that mode.
// FE/reference snapshots and porosity tables are replicated for postprocessing;
// assembly, solves and HDF5 writes remain distributed. This limits scalability
// of very large example studies.
//
// Active transport, phase, preCICE and restart execution require later drivers;
// this steady driver rejects them instead of silently ignoring their settings.

#include "src/core/input.h"
#include "src/mfem/boundary_conditions.h"
#include "src/mfem/linear_solver.h"
#include <hdf5.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <exception>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iomanip>
#include <limits>
#include <map>
#include <optional>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#ifndef H5_HAVE_PARALLEL
#error "driver.cpp requires parallel HDF5 built with PETSc's MPICH."
#endif

// Driver data and forward declarations.
namespace mantle::driver {
namespace in = mantle::input;

// All configuration stays in input.yaml. These adapters contain numerical
// implementation, not case selection by folder/name.
in::ReadInputOptions ReaderOptions();
void ValidateDriver(const in::InputConfig& input);
LinearSolverOptions SolverOptions(const in::SolverInput& input);
PetscReal Porosity(const in::InputConfig&, const Point&);
Point VectorFunction(const in::FunctionInput&, const in::InputConfig&, const Point&);
PetscReal ScalarFunction(const in::FunctionInput&, const in::InputConfig&,
                         const Point&, const Point& normal = Point{});
PetscErrorCode BoundarySpecifications(const in::InputConfig&, int level,
    StokesBoundarySpecification&, DarcyBoundarySpecification&);

struct Values {
    Point stokes{}, darcy{}, darcyFlux{};
    double ps = 0, pd = 0, porosity = 0;
};
Values Exact(const in::InputConfig&, const Point&);
void DeriveFields(const in::InputConfig&, const Point&, Values&);
std::vector<double> Field(const Values&, const std::string& name);

// Replicated FE snapshots are used ONLY by example postprocessing. Algebraic
// assembly/solves and HDF5 output remain distributed. This intentionally keeps
// interpolation across nonnested meshes simple; large production runs should
// replace the replicated reference with a distributed point locator.
struct Cell {
    QuadVertices corners{};
    std::array<double, 12> us{};
    std::array<double, 8> ud{};
    double ps = 0, pd = 0, phi = 0;
    BRMixed br;
    HDivMixed hd;
};
struct Snapshot {
    PetscInt nx = 0, ny = 0;
    bool stokes = false, darcy = false;
    std::array<double, 2> x{}, y{};
    std::vector<Cell> cells;
    PetscErrorCode Evaluate(PetscInt cell, const Point&, Values&) const;
    PetscErrorCode Locate(const Point&, PetscInt& cell) const;
    PetscErrorCode Evaluate(const Point&, Values&) const;
};
struct Metrics {
    std::string family, reference;
    int level = 0;
    PetscInt nx = 0, ny = 0;
    double h = 0, seconds = 0;
    std::map<std::string, double> errors;
    double stokesMass = 0, darcyMass = 0;
    LinearSolveReport solver;
};

PetscErrorCode Agree(MPI_Comm, PetscErrorCode localError, const char* stage);
PetscErrorCode RootWrite(MPI_Comm, const std::string& path, const std::string& text);
std::string OutputPath(const in::InputConfig&, const std::string& file,
                       const std::string& family, int level);
PetscErrorCode WriteFields(MPI_Comm, const in::InputConfig&, const MeshInfo&,
    const Snapshot&, const std::string& family, int level);
PetscErrorCode WriteSolverReport(MPI_Comm, const in::InputConfig&, const Metrics&);
PetscErrorCode WriteReports(MPI_Comm, const in::InputConfig&, const Snapshot&,
    const Metrics&, const std::string& family, int level);
PetscErrorCode Compare(MPI_Comm, const in::InputConfig&, const MeshInfo&,
    const Snapshot&, const Snapshot* reference, Metrics&);

// HDF5 table is replicated once, keyed by natural entity IDs. All ranks read
// the same rows; no MPI-ownership numbering is persisted in an input file.
struct GaussData {
    std::vector<double> cell, edge;
};
PetscErrorCode ReadGaussData(MPI_Comm, const in::InputConfig&, const MeshInfo&,
    const GaussRule1D&, const GaussRule1D&, const std::string& family,
    int level, GaussData&);
PetscErrorCode Run(const in::InputConfig&);
} // namespace mantle::driver

// Input adapters, prescribed functions and analytic references.
namespace mantle::driver {
namespace {
constexpr double pi = 3.141592653589793238462643383279502884;
const std::set<std::string> vectorNames{"mms_stokes_force", "mms_darcy_force"};
const std::set<std::string> scalarNames{
    "mms_stokes_source", "mms_darcy_source", "mms_stokes_velocity_x",
    "mms_stokes_velocity_y", "mms_stokes_traction_x", "mms_stokes_traction_y",
    "mms_darcy_normal_velocity", "mms_darcy_pressure"};
const std::set<std::string> fields{"porosity", "stokes_velocity", "darcy_velocity",
    "stokes_pressure_assembled", "darcy_pressure_assembled", "darcy_pressure_potential", "darcy_segregation_flux"};
void Require(bool ok, const std::string& message)
{ if (!ok) throw in::InputError(message); }
bool Stokes(const in::InputConfig& c) { return c.flow.system != "darcy"; }
bool Darcy(const in::InputConfig& c) { return c.flow.system != "stokes"; }
double Phi(const in::InputConfig& c)
{
    Require(c.porosity.source == "prescribed_function" &&
            c.porosity.prescribedFunction.name == "constant",
            "Manufactured functions require constant prescribed porosity.");
    const double phi = c.porosity.prescribedFunction.parameters.At("value").AsReal();
    Require(phi > c.flow.darcyCompactionAverageCutoff &&
            phi > c.flow.couplingAverageCutoff && phi < 1,
            "Manufactured porosity must be above the configured wet-cell cutoffs and below one.");
    return phi;
}
// Analytic solution in PHYSICAL coordinates. The constant-porosity equations
// used here satisfy, for constant phi and s=1-phi:
// -div(2*s*(eps(us)-div(us)*I/3)) + grad(ps) = s*fs,
// ud + phi^(theta+1/2)*grad(pd) = fd,
// -div(us) - phi/s*ps + sqrt(phi)/s*pd = gs,
// -phi^(theta+1/2)*div(ud) + sqrt(phi)/s*ps - pd/s = gd.
// Independent systems omit the cross-pressure terms. No discrete matrix is
// used to manufacture the RHS.
struct Trig {
    double sx, cx, sy, cy;
    explicit Trig(const Point& p) : sx(std::sin(pi*p.p[0])), cx(std::cos(pi*p.p[0])),
        sy(std::sin(pi*p.p[1])), cy(std::cos(pi*p.p[1])) {}
    Point us() const { return Point{{sx*sy, cx*sy}}; }
    Point ud() const { return Point{{cx*sy, sx*cy}}; }
    double ps() const { return sx*cy; }
    double pd() const { return cx*sy; }
    double divs() const { return pi*cx*(sy+cy); }
    double divd() const { return -2*pi*sx*sy; }
    Point gradps() const { return Point{{pi*cx*cy, -pi*sx*sy}}; }
    Point gradpd() const { return Point{{-pi*sx*sy, pi*cx*cy}}; }
};
CellSide Side(in::Side s)
{
    switch(s) {
    case in::Side::Left: return CellSide::Left;
    case in::Side::Right: return CellSide::Right;
    case in::Side::Bottom: return CellSide::Bottom;
    case in::Side::Top: return CellSide::Top;
    }
    throw in::InputError("Invalid side.");
}
BoundaryCondition Condition(const in::BoundaryConditionInput& value,
                            const in::InputConfig& c, double sign = 1)
{
    BoundaryCondition result;
    result.type = value.type == in::BoundaryKind::Dirichlet
        ? BoundaryType::Dirichlet : BoundaryType::Neumann;
    result.value.function = [value, &c, sign](const BoundaryPoint& p) {
        return sign * ScalarFunction(value.value, c, p.position, p.outwardNormal);
    };
    return result;
}
void Components(const in::InputConfig& c, CellSide side,
    const std::array<std::optional<in::BoundaryConditionInput>,2>& input,
    std::array<BoundaryCondition,2>& output)
{
    if (c.flow.boundary.stokesComponents == "cartesian") {
        for(int k=0;k<2;++k) if(input[k]) output[k]=Condition(*input[k],c);
        return;
    }
    // Tangent is the CCW tangent (-n_y,n_x), with fixed axis-aligned sides.
    const Point normals[4] = {{{0,-1}},{{1,0}},{{0,1}},{{-1,0}}};
    const auto n=normals[static_cast<int>(side)];
    const Point direction[2]={n,Point{{-n.p[1],n.p[0]}}};
    for(int k=0;k<2;++k) if(input[k])
        for(int d=0;d<2;++d) if(direction[k].p[d]!=0)
            output[d]=Condition(*input[k],c,direction[k].p[d]);
}
} // namespace

in::ReadInputOptions ReaderOptions()
{
    in::ReadInputOptions out;
    for(const auto& name:vectorNames)
        out.extensions.functions[name]=[](const in::FunctionInput& f,int components){
            Require(components==2 && f.parameters.mapping.empty(),
                    f.name+" takes no parameters and returns two components.");
        };
    for(const auto& name:scalarNames)
        out.extensions.functions[name]=[](const in::FunctionInput& f,int components){
            Require(components==1 && f.parameters.mapping.empty(),
                    f.name+" takes no parameters and returns a scalar.");
        };
    out.extensions.references["manufactured"] = [](const in::Value& value,
                                                   const in::InputConfig& c) {
        Require(value.At("solution").AsString()=="trigonometric",
                "This example driver implements the trigonometric manufactured reference.");
        for(const auto& pair:value.mapping)
            Require(pair.first=="source" || pair.first=="solution",
                    "Unknown manufactured-reference setting: "+pair.first);
        (void)Phi(c);
    };
    out.extensions.references["column_velocity"] = [](const in::Value& value,const in::InputConfig& c) {
        for(const auto& kv:value.mapping)
            Require(kv.first=="source" || kv.first=="fit_interface", "Unknown column reference setting: "+kv.first);
        Require(c.flow.system=="coupled_stokes_darcy" && c.flow.theta==0,
                "Column reference requires coupled flow and theta=0.");
        Require(c.porosity.source=="prescribed_function","Column reference requires a prescribed porosity.");
        const auto& f=c.porosity.prescribedFunction;
        Require(f.name=="constant" || f.name=="piecewise_constant" || f.name=="quadratic_below_interface",
                "Unsupported column porosity profile.");
        if(f.name!="constant") {
            Require(value.Find("fit_interface") && value.At("fit_interface").AsBool(),
                    "The wet/dry column reference requires fit_interface: true.");
            const double y0=f.parameters.At("interface_y").AsReal();
            const double row=(y0-c.mesh.y[0])/(c.mesh.y[1]-c.mesh.y[0])*c.mesh.cells[1];
            Require(std::abs(row-std::round(row))<1e-10,
                    "Choose the base y cell count so the column interface is on a mesh row.");
            Require(c.mesh.y[0]<y0 && c.mesh.y[1]>y0,"Column interface must lie inside the domain.");
            Require(f.parameters.At(f.name=="piecewise_constant"?"value_above":"value_above_interface").AsReal()==0,
                    "Column reference requires a dry region above the interface.");
        }
        const auto& force=c.flow.stokesForce;
        Require(force.name=="constant" && force.parameters.At("value").sequence[0].AsReal()==0,
                "Column reference requires constant vertical Stokes forcing [0,gy].");
        Require(c.flow.darcyForce.name=="constant" &&
                c.flow.darcyForce.parameters.At("value").sequence[0].AsReal()==0 &&
                c.flow.darcyForce.parameters.At("value").sequence[1].AsReal()==0,
                "Column reference requires zero Darcy forcing.");
        for(const auto* source:{&c.flow.stokesPressureSource,&c.flow.darcyPressureSource})
            Require(source->name=="constant" && source->parameters.At("value").AsReal()==0,
                    "Column reference requires zero pressure-row sources.");
        for(const auto& field:c.convergence.fields)
            Require(field=="stokes_velocity" || field=="darcy_velocity" || field=="darcy_segregation_flux",
                    "Column reference provides velocity/segregation-flux errors, not pressure errors.");
        const auto zero=[](const in::BoundaryConditionInput& bc){
            return bc.value.name=="constant" && bc.value.parameters.At("value").AsReal()==0;
        };
        const auto& b=c.flow.boundary;
        Require(b.stokesComponents=="cartesian","Column reference uses Cartesian wall components.");
        for(const auto& bc:b.stokesDefault)
            Require(bc.type==in::BoundaryKind::Dirichlet && zero(bc),"Column default Stokes velocity must be zero.");
        Require(b.darcyDefault.type==in::BoundaryKind::Dirichlet && zero(b.darcyDefault) && b.darcySegments.empty(),
                "Column reference requires zero Darcy normal velocity on the entire boundary.");
        std::set<in::Side> walls;
        for(const auto& seg:b.stokesSegments) {
            const auto& region=c.boundaryRegions.definitions.at(seg.region);
            Require(region.selection==in::Selection::WholeSide && (region.side==in::Side::Left || region.side==in::Side::Right),
                    "Column overrides must cover complete left/right walls.");
            Require(seg.component[1] && seg.component[1]->type==in::BoundaryKind::Neumann && zero(*seg.component[1]),
                    "Column side walls require zero tangential traction.");
            if(seg.component[0])Require(seg.component[0]->type==in::BoundaryKind::Dirichlet && zero(*seg.component[0]),
                                      "Column wall normal velocity must be zero.");
            Require(walls.insert(region.side).second,"Duplicate column wall rule.");
        }
        Require(walls.size()==2,"Column reference requires both free-slip side walls.");
        if(f.name=="constant")Require(f.parameters.At("value").AsReal()>0,"Column constant porosity must be positive.");
        if(f.name=="piecewise_constant")Require(f.parameters.At("value_below").AsReal()>0,"Column wet porosity must be positive.");
        if(f.name=="quadratic_below_interface") {
            const double a=f.parameters.At("coefficient").AsReal();
            const double H=f.parameters.At("interface_y").AsReal()-c.mesh.y[0];
            Require(a>0 && a*H*H<=0.1,"The 128-term column reference supports 0 < maximum quadratic porosity <= 0.1; use finer_mesh for other ranges.");
            for(int n=2;n<128;n+=2)Require(std::abs(a*n*(n-3)-1)>1e-12,
                "This Frobenius reference does not implement resonant logarithmic terms; choose finer_mesh.");
        }

    };
    return out;
}
void ValidateDriver(const in::InputConfig& c)
{
    Require(c.simulation.mode=="steady" && c.flow.enabled,
            "This flow example driver requires steady mode and flow.enabled: true.");
    Require(!c.transport.enabled && !c.phase.enabled && c.extensions.empty() &&
            c.coupling.backend=="internal" && c.coupling.internalScheme=="single_pass",
            "This driver implements a steady internal flow solve; enabled extra modules need their own driver.");
    Require(!c.restart.enabled && !c.restart.writeCheckpoints,
            "Restart/checkpoint execution is not implemented by the steady examples.");
    Require(!c.flow.solver.initialGuessNonzero,
            "These independent steady solves start from zero; no nonzero initial field is supplied.");
    Require(c.porosity.source!="gauss_point_data" || c.porosity.provider=="hdf5",
            "A runtime porosity field requires a registered physics provider; use prescribed_function or HDF5 here.");
    const auto checkFunction=[&](const in::FunctionInput& f){
        if(f.name.rfind("mms_",0)==0) (void)Phi(c);
    };
    checkFunction(c.flow.stokesForce); checkFunction(c.flow.darcyForce);
    checkFunction(c.flow.stokesPressureSource); checkFunction(c.flow.darcyPressureSource);
    for(const auto& bc:c.flow.boundary.stokesDefault) checkFunction(bc.value);
    checkFunction(c.flow.boundary.darcyDefault.value);
    for(const auto& s:c.flow.boundary.stokesSegments)
        for(const auto& bc:s.component) if(bc) checkFunction(bc->value);
    for(const auto& s:c.flow.boundary.darcySegments) checkFunction(s.condition.value);
    const auto checkField=[&](const std::string& field) {
        Require(fields.count(field),"Unsupported output/error field: "+field);
        Require(field.rfind("stokes_",0)!=0 || Stokes(c),"Stokes field requested in a Darcy-only solve.");
        Require(field.rfind("darcy_",0)!=0 || Darcy(c),"Darcy field requested in a Stokes-only solve.");
    };
    for(const auto& f:c.output.quantities) checkField(f);
    if(c.porosity.source=="gauss_point_data")
        for(const auto& f:c.convergence.fields)Require(f!="darcy_segregation_flux", "Physical-point flux errors require a porosity reconstruction; use assembled velocity errors with HDF5 samples.");
    for(const auto& f:c.output.profileFields) checkField(f);
    for(const auto& f:c.convergence.fields) {
        checkField(f);
        Require(f!="porosity" && f!="darcy_pressure_potential",
                "Convergence fields are velocities and assembled pressures; dry pressure potentials are undefined.");
    }
    for(const auto& d:c.output.diagnostics)
        Require(d=="true_residual" || d=="subsolver_convergence" || d=="discrete_mass_balance",
                "Unsupported diagnostic: "+d);
    if(c.flow.pressureNullspace=="provided") {
        const auto& p=c.flow.pressureModes;
        for(const auto& kv:p.mapping) {
            Require(kv.first=="stokes" || kv.first=="darcy", "Provided mode accepts scalar stokes/darcy entries.");
            if(kv.first=="darcy" && kv.second.scalar=="sqrt_cell_average_porosity")continue;
            (void)kv.second.AsReal();
        }
        if(Stokes(c)) (void)p.At("stokes").AsReal();
        if(Darcy(c) && p.At("darcy").scalar!="sqrt_cell_average_porosity") (void)p.At("darcy").AsReal();
    }
}
LinearSolverOptions SolverOptions(const in::SolverInput& i)
{
    LinearSolverOptions o;
    o.optionsPrefix=i.optionsPrefix; o.kspType=i.ksp;
    o.preconditioner=i.preconditioner=="schur"?LinearPreconditioner::Schur:LinearPreconditioner::None;
    o.relativeTolerance=i.relativeTolerance; o.absoluteTolerance=i.absoluteTolerance;
    o.divergenceTolerance=i.divergenceTolerance; o.maximumIterations=i.maximumIterations;
    o.initialGuessNonzero=i.initialGuessNonzero;
    const auto block=[](const in::BlockSolverInput& b){
        LinearBlockSolverOptions r;
        r.kspType=b.ksp; r.pcType=b.pc; r.relativeTolerance=b.relativeTolerance;
        r.absoluteTolerance=b.absoluteTolerance; r.divergenceTolerance=b.divergenceTolerance;
        r.maximumIterations=b.maximumIterations; return r;
    };
    o.velocity=block(i.velocity); o.pressure=block(i.pressure);
    o.errorIfNotConverged=i.errorIfNotConverged;
    o.requireSubsolverConvergence=i.requireSubsolverConvergence;
    o.requireTrueResidual=i.requireTrueResidual;
    o.removePressureNullspace=i.removePressureNullspace;
    return o;
}
PetscReal Porosity(const in::InputConfig& c, const Point& point)
{
    const auto& f=c.porosity.prescribedFunction;
    const auto& p=f.parameters;
    if(f.name=="constant") return p.At("value").AsReal();
    const auto y=point.p[1], y0=p.At("interface_y").AsReal();
    if(f.name=="piecewise_constant")
        return p.At(y<y0?"value_below":"value_above").AsReal();
    if(f.name=="quadratic_below_interface")
        return y<y0?p.At("coefficient").AsReal()*(y-y0)*(y-y0):p.At("value_above_interface").AsReal();
    throw in::InputError("Unsupported porosity function: "+f.name);
}
void DeriveFields(const in::InputConfig& c,const Point& p,Values& value)
{
    const double phi=c.porosity.source=="prescribed_function"?Porosity(c,p):value.porosity;
    const double weight=std::pow(phi,1+c.flow.theta);
    value.darcyFlux=Point{{weight*value.darcy.p[0],weight*value.darcy.p[1]}};
}
namespace {
// Independent solution of phi^2*(A*u')'-u=phi^2*(1-phi),
// A=(3+phi-4*phi^2)/(3*phi). Here u is physical segregation flux.
long double ColumnFlux(const in::InputConfig& c,long double y)
{
    const auto& f=c.porosity.prescribedFunction; const auto& p=f.parameters;
    const long double lower=c.mesh.y[0];
    if(f.name=="constant" || f.name=="piecewise_constant") {
        const long double upper=f.name=="constant"?c.mesh.y[1]:p.At("interface_y").AsReal();
        if(y>=upper && f.name!="constant")return 0;
        const long double phi=p.At(f.name=="constant"?"value":"value_below").AsReal();
        const long double R=std::sqrt(3/(phi*(3+phi-4*phi*phi)));
        const long double a=R*(y-(lower+upper)/2),b=R*(upper-lower)/2;
        // Stable cosh(a)/cosh(b), including long columns.
        const long double ratio=(std::exp(a-b)+std::exp(-a-b))/(1+std::exp(-2*b));
        return -phi*phi*(1-phi)*(1-ratio);
    }
    const long double y0=p.At("interface_y").AsReal();
    if(y>=y0)return 0;
    const long double a=p.At("coefficient").AsReal(),z=y0-y,H=y0-lower;
    const long double r=(3+std::sqrt(9+4/a))/2;
    constexpr int terms=128;
    long double part[terms]{},hom[terms]{};hom[0]=1;
    for(int n=2;n<terms;n+=2) {
        const auto recurrence=[&](long double exponent,const long double* co,long double source){
            const long double m=n+exponent;
            const long double rhs=source-a*a/3*(m-2)*(m-3)*co[n-2]
                +(n>=4?4*a*a*a/3*(m-4)*(m-3)*co[n-4]:0);
            return rhs/(a*m*(m-3)-1);
        };
        hom[n]=recurrence(r,hom,0);
        part[n]=recurrence(0,part,n==4?a*a:n==6?-a*a*a:0);
    }
    const auto polynomial=[](const long double* co,long double x){
        long double value=0;for(int n=terms-1;n>=0;--n)value=value*x+co[n];return value;
    };
    return polynomial(part,z)-polynomial(part,H)*std::pow(z/H,r)*polynomial(hom,z)/polynomial(hom,H);
}
} // namespace
Values Exact(const in::InputConfig& c,const Point& p)
{
    Values out;
    if(c.convergence.referenceSource=="column_velocity") {
        // ColumnFlux is the unit-upward-force reference. The linear problem
        // scales with the signed forcing, including legacy downward gravity.
        const double gravity=c.flow.stokesForce.parameters.At("value").sequence[1].AsReal();
        const double u=gravity*static_cast<double>(ColumnFlux(c,p.p[1]));
        const double phi=Porosity(c,p);
        out.stokes=Point{{0,-u}};out.darcy=Point{{0,phi>0?u/phi:0}};out.porosity=phi;
    } else {
        const Trig t(p);out.stokes=t.us();out.darcy=t.ud();out.ps=t.ps();out.pd=t.pd();out.porosity=Phi(c);
    }
    DeriveFields(c,p,out);return out;
}
Point VectorFunction(const in::FunctionInput& f,const in::InputConfig& c,const Point& p)
{
    if(f.name=="constant") {
        const auto& v=f.parameters.At("value").sequence;
        return Point{{v[0].AsReal(),v[1].AsReal()}};
    }
    const Trig t(p); const double phi=Phi(c), solid=1-phi;
    if(f.name=="mms_darcy_force") {
        auto u=t.ud(), g=t.gradpd(); const double a=std::pow(phi,c.flow.theta+0.5);
        return Point{{u.p[0]+a*g.p[0],u.p[1]+a*g.p[1]}};
    }
    if(f.name=="mms_stokes_force") {
        const auto u=t.us(), g=t.gradps();
        const double dx=-pi*pi*t.sx*(t.sy+t.cy), dy=pi*pi*t.cx*(t.cy-t.sy);
        return Point{{2*pi*pi*u.p[0]-dx/3+g.p[0]/solid,
                      2*pi*pi*u.p[1]-dy/3+g.p[1]/solid}};
    }
    throw in::InputError("Unsupported vector function: "+f.name);
}
PetscReal ScalarFunction(const in::FunctionInput& f,const in::InputConfig& c,
                         const Point& p,const Point& n)
{
    if(f.name=="constant") return f.parameters.At("value").AsReal();
    const Trig t(p); const double phi=Phi(c), solid=1-phi;
    const double coupling=c.flow.system=="coupled_stokes_darcy"?std::sqrt(phi)/solid:0;
    if(f.name=="mms_stokes_source") return -t.divs()-phi/solid*t.ps()+coupling*t.pd();
    if(f.name=="mms_darcy_source")
        return -std::pow(phi,c.flow.theta+0.5)*t.divd()-t.pd()/solid+coupling*t.ps();
    if(f.name=="mms_stokes_velocity_x") return t.us().p[0];
    if(f.name=="mms_stokes_velocity_y") return t.us().p[1];
    if(f.name=="mms_darcy_normal_velocity") return t.ud().p[0]*n.p[0]+t.ud().p[1]*n.p[1];
    if(f.name=="mms_darcy_pressure") return t.pd();
    const double xx=2*solid*(pi*t.cx*t.sy-t.divs()/3)-t.ps();
    const double yy=2*solid*(pi*t.cx*t.cy-t.divs()/3)-t.ps();
    const double xy=solid*pi*(t.sx*t.cy-t.sx*t.sy);
    if(f.name=="mms_stokes_traction_x") return xx*n.p[0]+xy*n.p[1];
    if(f.name=="mms_stokes_traction_y") return xy*n.p[0]+yy*n.p[1];
    throw in::InputError("Unsupported scalar function: "+f.name);
}
std::vector<double> Field(const Values& v,const std::string& name)
{
    if(name=="porosity") return {v.porosity};
    if(name=="stokes_velocity") return {v.stokes.p[0],v.stokes.p[1]};
    if(name=="darcy_velocity") return {v.darcy.p[0],v.darcy.p[1]};
    if(name=="darcy_segregation_flux") return {v.darcyFlux.p[0],v.darcyFlux.p[1]};
    if(name=="stokes_pressure_assembled") return {v.ps};
    if(name=="darcy_pressure_assembled") return {v.pd};
    if(name=="darcy_pressure_potential")
        return {v.porosity>0?v.pd/std::sqrt(v.porosity):std::numeric_limits<double>::quiet_NaN()};
    throw in::InputError("Unknown field: "+name);
}
PetscErrorCode BoundarySpecifications(const in::InputConfig& c,int level,
    StokesBoundarySpecification& s,DarcyBoundarySpecification& d)
{
    PetscFunctionBeginUser;
    s.absoluteTolerance=c.flow.boundary.absoluteTolerance;
    s.relativeTolerance=c.flow.boundary.relativeTolerance;
    const auto& variable=c.flow.boundary.darcyNeumannVariable;
    d.neumannVariable=variable=="pressure_potential"?DarcyNeumannVariable::PressurePotential:
        variable=="assembled_pressure"?DarcyNeumannVariable::AssembledPressure:DarcyNeumannVariable::WeightedNormalLoad;
    d.pressureSign=-1;
    for(int side=0;side<4;++side) {
        StokesBoundaryRule sr; sr.region.side=static_cast<CellSide>(side);
        std::array<std::optional<in::BoundaryConditionInput>,2> bc{{c.flow.boundary.stokesDefault[0],c.flow.boundary.stokesDefault[1]}};
        Components(c,sr.region.side,bc,sr.component); s.rules.push_back(sr);
        DarcyBoundaryRule dr; dr.region.side=sr.region.side;
        dr.condition=Condition(c.flow.boundary.darcyDefault,c); d.rules.push_back(dr);
    }
    for(const auto& seg:c.flow.boundary.stokesSegments) {
        const auto r=in::ResolveBoundaryRegion(c,seg.region,level);
        StokesBoundaryRule sr;
        sr.region={Side(r.side),static_cast<PetscInt>(r.first),static_cast<PetscInt>(r.end),seg.priority};
        Components(c,sr.region.side,seg.component,sr.component); s.rules.push_back(sr);
    }
    for(const auto& seg:c.flow.boundary.darcySegments) {
        const auto r=in::ResolveBoundaryRegion(c,seg.region,level);
        DarcyBoundaryRule dr;
        dr.region={Side(r.side),static_cast<PetscInt>(r.first),static_cast<PetscInt>(r.end),seg.priority};
        dr.condition=Condition(seg.condition,c); d.rules.push_back(dr);
    }
    PetscFunctionReturn(PETSC_SUCCESS);
}
} // namespace mantle::driver

// Finite-element evaluation and physical-point error integration.
namespace mantle::driver {
PetscErrorCode Snapshot::Evaluate(PetscInt id,const Point& p,Values& out) const
{
    PetscFunctionBeginUser;
    PetscCheck(id>=0 && static_cast<std::size_t>(id)<cells.size(),PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Snapshot cell is invalid.");
    const auto& c=cells[static_cast<std::size_t>(id)];
    Values value; value.ps=c.ps; value.pd=c.pd; value.porosity=c.phi;
    if(stokes) {
        BRMixed::Values basis{}; PetscCall(c.br.EvaluateAll(p,basis));
        for(int k=0;k<12;++k) for(int d=0;d<2;++d) value.stokes.p[d]+=c.us[k]*basis[k].value.p[d];
    }
    if(darcy) {
        HDivMixed::Values basis{}; PetscCall(c.hd.EvaluateAll(p,basis));
        for(int k=0;k<8;++k) for(int d=0;d<2;++d) value.darcy.p[d]+=c.ud[k]*basis[k].value.p[d];
    }
    out=value;
    PetscFunctionReturn(PETSC_SUCCESS);
}
PetscErrorCode Snapshot::Locate(const Point& p,PetscInt& result) const
{
    PetscFunctionBeginUser;
    const double dx=(x[1]-x[0])/nx,dy=(y[1]-y[0])/ny;
    const auto i0=static_cast<PetscInt>(std::floor((p.p[0]-x[0])/dx));
    const auto j0=static_cast<PetscInt>(std::floor((p.p[1]-y[0])/dy));
    const double tol=128*std::numeric_limits<double>::epsilon()*std::max({1.0,std::abs(x[0]),std::abs(x[1]),std::abs(y[0]),std::abs(y[1])});
    // Perturbations are < 1/4 cell spacing; the containing cell is nearby.
    // Increasing natural ID makes a deterministic one-sided choice on faces.
    for(PetscInt j=std::max<PetscInt>(0,j0-2);j<=std::min<PetscInt>(ny-1,j0+2);++j)
        for(PetscInt i=std::max<PetscInt>(0,i0-2);i<=std::min<PetscInt>(nx-1,i0+2);++i) {
            const auto id=j*nx+i; const auto& v=cells[static_cast<std::size_t>(id)].corners;
            bool inside=true;
            for(int e=0;e<4;++e) {
                const double ex=v[(e+1)%4].p[0]-v[e].p[0],ey=v[(e+1)%4].p[1]-v[e].p[1];
                const double cross=ex*(p.p[1]-v[e].p[1])-ey*(p.p[0]-v[e].p[0]);
                if(cross < -tol*std::hypot(ex,ey)) inside=false;
            }
            if(inside){result=id; PetscFunctionReturn(PETSC_SUCCESS);}
        }
    SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Point (%g,%g) is outside the reference mesh.",static_cast<double>(p.p[0]),static_cast<double>(p.p[1]));
}
PetscErrorCode Snapshot::Evaluate(const Point& p,Values& result) const
{
    PetscFunctionBeginUser;
    PetscInt id=0; PetscCall(Locate(p,id)); PetscCall(Evaluate(id,p,result));
    PetscFunctionReturn(PETSC_SUCCESS);
}
PetscErrorCode Compare(MPI_Comm comm,const in::InputConfig& c,const MeshInfo& mesh,
    const Snapshot& solution,const Snapshot* reference,Metrics& metrics)
{
    PetscFunctionBeginUser;
    GaussRule1D rule;
    PetscCall(Agree(comm,CreateGaussRule(c.convergence.errorPointsPerAxis,rule),"error quadrature"));
    const auto count=c.convergence.fields.size(); std::vector<double> local(count,0),global(count,0);
    double h=0;
    const auto evaluate=[&]() -> PetscErrorCode {
        PetscFunctionBeginUser;
        const auto range=mesh.OwnedCells();
        for(PetscInt j=range.begin.j;j<range.end.j;++j)
            for(PetscInt i=range.begin.i;i<range.end.i;++i) {
                const PetscInt id=j*solution.nx+i; const auto& vertices=solution.cells[id].corners;
                for(int a=0;a<4;++a) for(int b=a+1;b<4;++b)
                    h=std::max(h,std::hypot(vertices[a].p[0]-vertices[b].p[0],vertices[a].p[1]-vertices[b].p[1]));
                for(std::size_t q=0;q<rule.points.size();++q) for(std::size_t p=0;p<rule.points.size();++p) {
                    const Point param{{rule.points[p],rule.points[q]}};
                    const auto point=MapCellPoint(param,vertices);
                    const double w=rule.weights[p]*rule.weights[q]*CellJacobian(param,vertices);
                    Values value,exact;
                    PetscCall(solution.Evaluate(id,point,value));
                    DeriveFields(c,point,value);
                    if(reference) {PetscCall(reference->Evaluate(point,exact));DeriveFields(c,point,exact);} else exact=Exact(c,point);
                    for(std::size_t field=0;field<count;++field) {
                        const auto a=Field(value,c.convergence.fields[field]), b=Field(exact,c.convergence.fields[field]);
                        for(std::size_t k=0;k<a.size();++k) local[field]+=w*(a[k]-b[k])*(a[k]-b[k]);
                    }
                }
            }
        PetscFunctionReturn(PETSC_SUCCESS);
    };
    PetscCall(Agree(comm,evaluate(),"physical-point error evaluation"));
    PetscCallMPI(MPI_Allreduce(local.data(),global.data(),static_cast<int>(count),MPI_DOUBLE,MPI_SUM,comm));
    PetscCallMPI(MPI_Allreduce(&h,&metrics.h,1,MPI_DOUBLE,MPI_MAX,comm));
    for(std::size_t k=0;k<count;++k) metrics.errors[c.convergence.fields[k]]=std::sqrt(global[k]);
    metrics.reference=reference?"finer_mesh":c.convergence.referenceSource;
    PetscFunctionReturn(PETSC_SUCCESS);
}
} // namespace mantle::driver

// Collective HDF5 I/O, XDMF and text reports.
namespace mantle::driver {
namespace {
class Handle {
    hid_t value_; herr_t (*close_)(hid_t);
public:
    Handle(hid_t value,herr_t (*close)(hid_t)):value_(value),close_(close){}
    ~Handle(){if(value_>=0)close_(value_);}
    Handle(const Handle&)=delete;
    Handle& operator=(const Handle&)=delete;
    hid_t get()const{return value_;}
    herr_t Close(){if(value_<0)return 0; const auto s=close_(value_);value_=-1;return s;}
};
PetscErrorCode Check(MPI_Comm comm,bool ok,const char* operation)
{ return Agree(comm,ok?PETSC_SUCCESS:PETSC_ERR_FILE_WRITE,operation); }
std::string Xml(const std::string& s)
{
    std::string out;
    for(char c:s) switch(c){case '&':out+="&amp;";break;case '<':out+="&lt;";break;
        case '>':out+="&gt;";break;case '"':out+="&quot;";break;case '\'':out+="&apos;";break;default:out+=c;}
    return out;
}
PetscErrorCode Directories(MPI_Comm comm,const std::string& filename)
{
    PetscFunctionBeginUser;
    int rank=0; PetscCallMPI(MPI_Comm_rank(comm,&rank));
    std::error_code ec;
    if(rank==0)std::filesystem::create_directories(std::filesystem::path(filename).parent_path(),ec);
    PetscCall(Check(comm,!ec,"create output directory"));
    PetscFunctionReturn(PETSC_SUCCESS);
}
PetscErrorCode WriteRows(MPI_Comm comm,hid_t file,const std::string& name,
    hsize_t total,hsize_t components,const std::vector<hsize_t>& rows,
    const void* values,hid_t fileType,hid_t memoryType)
{
    PetscFunctionBeginUser;
    const hsize_t shape[2]={total,components};
    Handle space(H5Screate_simple(2,shape,nullptr),H5Sclose);
    PetscCall(Check(comm,space.get()>=0,"HDF5 file dataspace"));
    Handle dataset(H5Dcreate2(file,name.c_str(),fileType,space.get(),H5P_DEFAULT,H5P_DEFAULT,H5P_DEFAULT),H5Dclose);
    PetscCall(Check(comm,dataset.get()>=0,"HDF5 dataset creation"));
    const hsize_t n=std::max<hsize_t>(1,rows.size()*components);
    Handle memory(H5Screate_simple(1,&n,nullptr),H5Sclose);
    PetscCall(Check(comm,memory.get()>=0,"HDF5 memory dataspace"));
    bool ok=H5Sselect_none(space.get())>=0;
    for(std::size_t first=0;first<rows.size();) {
        std::size_t last=first+1;
        while(last<rows.size() && rows[last]==rows[last-1]+1)++last;
        const hsize_t begin[2]={rows[first],0}, count[2]={last-first,components};
        ok=(H5Sselect_hyperslab(space.get(),H5S_SELECT_OR,begin,nullptr,count,nullptr)>=0)&&ok;
        first=last;
    }
    if(rows.empty())ok=(H5Sselect_none(memory.get())>=0)&&ok;
    PetscCall(Check(comm,ok,"HDF5 owned-row selection"));
    Handle transfer(H5Pcreate(H5P_DATASET_XFER),H5Pclose);
    PetscCall(Check(comm,transfer.get()>=0,"HDF5 transfer properties"));
    PetscCall(Check(comm,H5Pset_dxpl_mpio(transfer.get(),H5FD_MPIO_COLLECTIVE)>=0,"HDF5 collective transfer"));
    double unused=0;
    PetscCall(Check(comm,H5Dwrite(dataset.get(),memoryType,memory.get(),space.get(),transfer.get(),rows.empty()?&unused:values)>=0,"HDF5 dataset write"));
    PetscCall(Check(comm,dataset.Close()>=0,"HDF5 dataset close"));
    PetscFunctionReturn(PETSC_SUCCESS);
}
PetscErrorCode ReadArray(MPI_Comm comm,hid_t file,const std::string& name,
                        const std::vector<hsize_t>& shape,std::vector<double>& data)
{
    PetscFunctionBeginUser;
    Handle dataset(H5Dopen2(file,name.c_str(),H5P_DEFAULT),H5Dclose);
    PetscCall(Check(comm,dataset.get()>=0,name.c_str()));
    Handle space(H5Dget_space(dataset.get()),H5Sclose);
    PetscCall(Check(comm,space.get()>=0,"input HDF5 dataspace"));
    const int nd=H5Sget_simple_extent_ndims(space.get());
    std::vector<hsize_t> actual(shape.size());
    bool ok=nd==static_cast<int>(shape.size());
    if(ok)ok=H5Sget_simple_extent_dims(space.get(),actual.data(),nullptr)>=0 && actual==shape;
    PetscCall(Check(comm,ok,("wrong shape: "+name).c_str()));
    std::size_t count=1;for(auto d:shape)count*=d;data.resize(count);
    Handle transfer(H5Pcreate(H5P_DATASET_XFER),H5Pclose);
    PetscCall(Check(comm,transfer.get()>=0,"input HDF5 transfer properties"));
    PetscCall(Check(comm,H5Pset_dxpl_mpio(transfer.get(),H5FD_MPIO_COLLECTIVE)>=0,"input HDF5 collective transfer"));
    PetscCall(Check(comm,H5Dread(dataset.get(),H5T_NATIVE_DOUBLE,H5S_ALL,H5S_ALL,transfer.get(),data.data())>=0,name.c_str()));
    PetscFunctionReturn(PETSC_SUCCESS);
}
} // namespace
PetscErrorCode Agree(MPI_Comm comm,PetscErrorCode local,const char* stage)
{
    PetscFunctionBeginUser;
    int code=static_cast<int>(local),global=0;
    PetscCallMPI(MPI_Allreduce(&code,&global,1,MPI_INT,MPI_MAX,comm));
    PetscCheck(global==0,comm,static_cast<PetscErrorCode>(global),"Example stage failed: %s",stage);
    PetscFunctionReturn(PETSC_SUCCESS);
}
PetscErrorCode RootWrite(MPI_Comm comm,const std::string& path,const std::string& text)
{
    PetscFunctionBeginUser;
    PetscCall(Directories(comm,path));
    int rank=0; PetscCallMPI(MPI_Comm_rank(comm,&rank)); bool ok=true;
    if(rank==0){std::ofstream out(path);out<<text;out.close();ok=!out.fail();}
    PetscCall(Check(comm,ok,path.c_str()));
    PetscFunctionReturn(PETSC_SUCCESS);
}
std::string OutputPath(const in::InputConfig& c,const std::string& file,
                       const std::string& family,int level)
{ return in::ResolveOutputPath(c,file,family,level,0); }
PetscErrorCode WriteFields(MPI_Comm comm,const in::InputConfig& c,const MeshInfo& mesh,
    const Snapshot& solution,const std::string& family,int level)
{
    PetscFunctionBeginUser;
    const auto filename=OutputPath(c,c.output.fieldFile,family,level);
    PetscCall(Directories(comm,filename));
    Handle access(H5Pcreate(H5P_FILE_ACCESS),H5Pclose);
    PetscCall(Check(comm,access.get()>=0,"HDF5 access properties"));
    PetscCall(Check(comm,H5Pset_fapl_mpio(access.get(),comm,MPI_INFO_NULL)>=0,"HDF5 MPI access"));
    Handle file(H5Fcreate(filename.c_str(),H5F_ACC_TRUNC,H5P_DEFAULT,access.get()),H5Fclose);
    PetscCall(Check(comm,file.get()>=0,"HDF5 file creation"));
    std::vector<hsize_t> cellRows,pointRows;
    std::vector<double> xyz,centers,ids;
    std::vector<long long> topology;
    std::map<std::string,std::vector<double>> data;
    for(const auto& name:c.output.quantities)data[name]={};
    if(data.count("darcy_pressure_potential"))data["darcy_pressure_potential_valid"]={};
    const auto evaluate=[&]() -> PetscErrorCode {
        PetscFunctionBeginUser;
        const auto range=mesh.OwnedCells();
        for(PetscInt j=range.begin.j;j<range.end.j;++j) for(PetscInt i=range.begin.i;i<range.end.i;++i) {
            const PetscInt id=j*solution.nx+i; cellRows.push_back(static_cast<hsize_t>(id));ids.push_back(static_cast<double>(id));
            const auto& cell=solution.cells[id];
            for(int k=0;k<4;++k){
                pointRows.push_back(static_cast<hsize_t>(id)*4+k);
                xyz.insert(xyz.end(),{cell.corners[k].p[0],cell.corners[k].p[1],0});
                topology.push_back(static_cast<long long>(id)*4+k);
            }
            const auto center=MapCellPoint(Point{{0,0}},cell.corners);
            centers.insert(centers.end(),{center.p[0],center.p[1],0});
            Values value; PetscCall(solution.Evaluate(id,center,value));
            DeriveFields(c,center,value);
            for(const auto& name:c.output.quantities) {
                auto result=Field(value,name);
                if(result.size()==2)result.push_back(0);
                data[name].insert(data[name].end(),result.begin(),result.end());
            }
            if(data.count("darcy_pressure_potential_valid"))
                data["darcy_pressure_potential_valid"].push_back(value.porosity>0?1:0);
        }
        PetscFunctionReturn(PETSC_SUCCESS);
    };
    PetscCall(Agree(comm,evaluate(),"field evaluation"));
    const hsize_t total=solution.cells.size();
    PetscCall(WriteRows(comm,file.get(),"coordinates",total*4,3,pointRows,xyz.data(),H5T_IEEE_F64LE,H5T_NATIVE_DOUBLE));
    PetscCall(WriteRows(comm,file.get(),"topology",total,4,cellRows,topology.data(),H5T_STD_I64LE,H5T_NATIVE_LLONG));
    PetscCall(WriteRows(comm,file.get(),"cell_centers",total,3,cellRows,centers.data(),H5T_IEEE_F64LE,H5T_NATIVE_DOUBLE));
    PetscCall(WriteRows(comm,file.get(),"global_cell_ids",total,1,cellRows,ids.data(),H5T_IEEE_F64LE,H5T_NATIVE_DOUBLE));
    for(const auto& item:data) {
        const hsize_t components=item.first=="stokes_velocity" || item.first=="darcy_velocity" || item.first=="darcy_segregation_flux"?3:1;
        PetscCall(WriteRows(comm,file.get(),item.first,total,components,cellRows,item.second.data(),H5T_IEEE_F64LE,H5T_NATIVE_DOUBLE));
    }
    PetscCall(Check(comm,file.Close()>=0,"close field HDF5 file"));
    const auto xdmfPath=OutputPath(c,c.output.xdmfSeries,family,level);
    const auto relative=std::filesystem::path(filename).lexically_relative(std::filesystem::path(xdmfPath).parent_path()).generic_string();
    std::ostringstream xml;
    xml<<"<?xml version=\"1.0\"?>\n<Xdmf Version=\"3.0\"><Domain><Grid Name=\""<<Xml(c.simulation.name)<<"\" GridType=\"Uniform\">\n"
       <<"<Time Value=\""<<std::setprecision(17)<<c.time.start<<"\"/>\n"
       <<"<Topology TopologyType=\"Quadrilateral\" NumberOfElements=\""<<total<<"\"><DataItem Dimensions=\""<<total<<" 4\" NumberType=\"Int\" Precision=\"8\" Format=\"HDF\">"<<Xml(relative)<<":/topology</DataItem></Topology>\n"
       <<"<Geometry GeometryType=\"XYZ\"><DataItem Dimensions=\""<<total*4<<" 3\" NumberType=\"Float\" Precision=\"8\" Format=\"HDF\">"<<Xml(relative)<<":/coordinates</DataItem></Geometry>\n";
    for(const auto& item:data) {
        const bool vector=item.first=="stokes_velocity" || item.first=="darcy_velocity" || item.first=="darcy_segregation_flux";
        xml<<"<Attribute Name=\""<<Xml(item.first)<<"\" AttributeType=\""<<(vector?"Vector":"Scalar")<<"\" Center=\"Cell\"><DataItem Dimensions=\""<<total<<' '<<(vector?3:1)<<"\" NumberType=\"Float\" Precision=\"8\" Format=\"HDF\">"<<Xml(relative)<<":/"<<Xml(item.first)<<"</DataItem></Attribute>\n";
    }
    xml<<"</Grid></Domain></Xdmf>\n";
    PetscCall(RootWrite(comm,xdmfPath,xml.str()));
    PetscFunctionReturn(PETSC_SUCCESS);
}
PetscErrorCode WriteSolverReport(MPI_Comm comm,const in::InputConfig& c,const Metrics& m)
{
    PetscFunctionBeginUser;
    if(c.output.solverReport=="none")PetscFunctionReturn(PETSC_SUCCESS);
    std::ostringstream s;s<<std::setprecision(17);
    s<<"family,level,converged,ksp,pc,reason,iterations,true_residual,relative_true_residual,true_residual_threshold,subsolvers_converged,seconds\n"
     <<m.family<<','<<m.level<<','<<m.solver.converged<<','<<m.solver.kspType<<','<<m.solver.pcType<<','<<static_cast<int>(m.solver.reason)<<','
     <<m.solver.iterations<<','<<m.solver.trueResidualNorm<<','<<m.solver.relativeTrueResidualNorm<<','<<m.solver.trueResidualThreshold<<','<<m.solver.subsolversConverged<<','<<m.seconds<<'\n';
    PetscCall(RootWrite(comm,OutputPath(c,"solver_report.csv",m.family,m.level),s.str()));
    s.str("");s.clear();
    s<<"prefix,solves,total_iterations,failures,first_failure,last_reason,last_iterations,last_residual\n";
    for(const auto& r:m.solver.subsolvers)s<<r.optionsPrefix<<','<<r.solves<<','<<r.totalIterations<<','<<r.failures<<','<<static_cast<int>(r.firstFailure)<<','<<static_cast<int>(r.lastReason)<<','<<r.lastIterations<<','<<r.lastResidualNorm<<'\n';
    PetscCall(RootWrite(comm,OutputPath(c,"subsolvers.csv",m.family,m.level),s.str()));
    PetscFunctionReturn(PETSC_SUCCESS);
}
PetscErrorCode WriteReports(MPI_Comm comm,const in::InputConfig& c,const Snapshot& solution,
                           const Metrics& m,const std::string& family,int level)
{
    PetscFunctionBeginUser;
    std::ostringstream s;s<<std::setprecision(17);
    s<<"family,level,nx,ny,h,reference,stokes_mass_residual_linf,darcy_mass_residual_linf";
    for(const auto& kv:m.errors)s<<",l2_"<<kv.first;
    s<<'\n'<<family<<','<<level<<','<<m.nx<<','<<m.ny<<','<<m.h<<','<<m.reference<<','<<m.stokesMass<<','<<m.darcyMass;
    for(const auto& kv:m.errors)s<<','<<kv.second;
    s<<'\n';
    PetscCall(RootWrite(comm,OutputPath(c,"metrics.csv",family,level),s.str()));
    if(c.output.saveResolvedInput) {
        PetscCall(RootWrite(comm,OutputPath(c,"input_used.yaml",family,level),c.originalYaml));
        s.str("");s.clear();
        s<<"source="<<c.sourceFile<<"\nsystem="<<c.flow.system<<"\nfamily="<<family<<"\nlevel="<<level
         <<"\ncells="<<m.nx<<','<<m.ny<<"\nranks="<<c.parallel.ranks<<"\nprocess_grid="<<c.parallel.processGrid[0]<<','<<c.parallel.processGrid[1]
         <<"\ncell_gauss_points_per_axis="<<c.quadrature.cellPointsPerAxis<<"\nedge_gauss_points="<<c.quadrature.edgePoints
         <<"\nreference="<<m.reference<<"\nfield_location=cell_center; porosity=physical_cell_average\n";
        for(const auto& kv:c.boundaryRegions.definitions) {
            const auto r=in::ResolveBoundaryRegion(c,kv.first,level);
            s<<"boundary_region."<<kv.first<<"=["<<r.first<<','<<r.end<<")\n";
        }
        PetscCall(RootWrite(comm,OutputPath(c,"resolved_run.txt",family,level),s.str()));
    }
    if(c.output.profilesEnabled) {
        s.str("");s.clear();s<<"distance,x,y";
        for(const auto& f:c.output.profileFields) {
            s<<','<<f;if(f=="stokes_velocity" || f=="darcy_velocity" || f=="darcy_segregation_flux")s<<"_x,"<<f<<"_y";
        }
        s<<'\n'; int rank=0;PetscCallMPI(MPI_Comm_rank(comm,&rank));
        PetscErrorCode status=0;
        if(rank==0) {
            const double dx=c.output.profileEnd[0]-c.output.profileStart[0],dy=c.output.profileEnd[1]-c.output.profileStart[1];
            for(int i=0;i<c.output.profilePoints && !status;++i) {
                const double t=static_cast<double>(i)/(c.output.profilePoints-1);
                const Point p{{c.output.profileStart[0]+t*dx,c.output.profileStart[1]+t*dy}};
                Values value;status=solution.Evaluate(p,value);
                if(!status)DeriveFields(c,p,value);
                s<<t*std::hypot(dx,dy)<<','<<p.p[0]<<','<<p.p[1];
                for(const auto& f:c.output.profileFields)for(double v:Field(value,f))s<<','<<v;
                s<<'\n';
            }
        }
        PetscCall(Agree(comm,status,"line profiles"));
        PetscCall(RootWrite(comm,OutputPath(c,c.output.profileFile,family,level),s.str()));
    }
    PetscFunctionReturn(PETSC_SUCCESS);
}
PetscErrorCode ReadGaussData(MPI_Comm comm,const in::InputConfig& c,const MeshInfo& mesh,
    const GaussRule1D& cellRule,const GaussRule1D& edgeRule,const std::string& family,
    int level,GaussData& data)
{
    PetscFunctionBeginUser;
    const auto path=in::ResolveInputPath(c,in::ExpandInputPath(c.porosity.file,family,level,0));
    Handle access(H5Pcreate(H5P_FILE_ACCESS),H5Pclose);
    PetscCall(Check(comm,access.get()>=0,"porosity HDF5 access"));
    PetscCall(Check(comm,H5Pset_fapl_mpio(access.get(),comm,MPI_INFO_NULL)>=0,"porosity HDF5 MPI access"));
    Handle file(H5Fopen(path.c_str(),H5F_ACC_RDONLY,access.get()),H5Fclose);
    PetscCall(Check(comm,file.get()>=0,"open porosity HDF5 file"));
    const auto group=c.porosity.group;
    const auto key=[&](const std::string& k){return group+"/"+c.porosity.datasets.at(k);};
    const hsize_t nc=mesh.CellCount(),ne=mesh.EdgeCount(),q=cellRule.points.size(),e=edgeRule.points.size();
    std::vector<double> cellIds,edgeIds,cellValues,edgeValues,corners,edges,cr,er,time;
    PetscCall(ReadArray(comm,file.get(),key("cell_ids"),{nc},cellIds));
    PetscCall(ReadArray(comm,file.get(),key("edge_ids"),{ne},edgeIds));
    PetscCall(ReadArray(comm,file.get(),key("cell_values"),{nc,q*q},cellValues));
    PetscCall(ReadArray(comm,file.get(),key("edge_values"),{ne,e},edgeValues));
    PetscCall(ReadArray(comm,file.get(),group+"/cell_corners",{nc,4,2},corners));
    PetscCall(ReadArray(comm,file.get(),group+"/edge_vertices",{ne,2,2},edges));
    PetscCall(ReadArray(comm,file.get(),group+"/cell_rule",{q,2},cr));
    PetscCall(ReadArray(comm,file.get(),group+"/edge_rule",{e,2},er));
    PetscCall(ReadArray(comm,file.get(),group+"/time",{1},time));
    const auto close=[](double a,double b){return std::isfinite(a) && std::abs(a-b)<=128*std::numeric_limits<double>::epsilon()*std::max({1.0,std::abs(a),std::abs(b)});};
    const auto validate=[&]() -> PetscErrorCode {
        PetscFunctionBeginUser;
        PetscCheck(close(time[0],c.time.start),PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Gauss data time mismatch.");
        for(std::size_t i=0;i<q;++i)PetscCheck(close(cr[2*i],cellRule.points[i]) && close(cr[2*i+1],cellRule.weights[i]),PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Cell Gauss rule mismatch.");
        for(std::size_t i=0;i<e;++i)PetscCheck(close(er[2*i],edgeRule.points[i]) && close(er[2*i+1],edgeRule.weights[i]),PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Edge Gauss rule mismatch.");
        std::vector<bool> seenC(nc,false),seenE(ne,false);
        data.cell.resize(nc*q*q);data.edge.resize(ne*e);
        for(std::size_t row=0;row<nc;++row) {
            const double x=cellIds[row];
            PetscCheck(std::isfinite(x) && x>=0 && x<nc && std::floor(x)==x,PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Invalid natural cell ID.");
            const auto id=static_cast<std::size_t>(x);
            PetscCheck(!seenC[id],PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Duplicate natural cell ID.");seenC[id]=true;
            std::copy_n(cellValues.data()+row*q*q,q*q,data.cell.data()+id*q*q);
            MeshIndex index{};PetscCall(mesh.CellIndex(static_cast<PetscInt>(id),index));
            if(mesh.OwnsCell(index)) {
                QuadVertices v{};PetscCall(mesh.GetCellCorners(index,v));
                for(int k=0;k<4;++k)for(int d=0;d<2;++d)
                    PetscCheck(close(corners[row*8+k*2+d],v[k].p[d]),PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Gauss file cell geometry mismatch.");
            }
        }
        for(std::size_t row=0;row<ne;++row) {
            const double x=edgeIds[row];
            PetscCheck(std::isfinite(x) && x>=0 && x<ne && std::floor(x)==x,PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Invalid natural edge ID.");
            const auto id=static_cast<std::size_t>(x);
            PetscCheck(!seenE[id],PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Duplicate natural edge ID.");seenE[id]=true;
            std::copy_n(edgeValues.data()+row*e,e,data.edge.data()+id*e);
            if(mesh.OwnsEdge(static_cast<PetscInt>(id))) {
                EdgeVertices v{};PetscCall(mesh.GetEdgeVertices(static_cast<PetscInt>(id),v));
                for(int k=0;k<2;++k)for(int d=0;d<2;++d)
                    PetscCheck(close(edges[row*4+k*2+d],v[k].p[d]),PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Gauss file edge geometry mismatch.");
            }
        }
        for(double phi:data.cell)PetscCheck(std::isfinite(phi) && phi>=0 && phi<1,PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Invalid cell porosity.");
        for(double phi:data.edge)PetscCheck(std::isfinite(phi) && phi>=0 && phi<=1,PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Invalid edge porosity.");
        PetscFunctionReturn(PETSC_SUCCESS);
    };
    PetscCall(Agree(comm,validate(),"porosity metadata validation"));
    PetscCall(Check(comm,file.Close()>=0,"close porosity input"));
    PetscFunctionReturn(PETSC_SUCCESS);
}
} // namespace mantle::driver

// Mesh construction, block assembly, boundary conditions and solves.
namespace mantle::driver {
namespace {
struct Resources {
    DM dm=nullptr;
    Vec vertices=nullptr, modeS=nullptr, modeD=nullptr;
    MeshInfo mesh;
    DofMap sv,dv,p;
    GaussRule1D cellRule,edgeRule;
    MixedBlocks rawS,rawD,s,d;
    BoundaryData bcS,bcD;
    Mat coupling=nullptr;
    LinearSystem system;
    PetscErrorCode Clear() {
        PetscErrorCode first=0;
        const auto keep=[&](PetscErrorCode code){if(!first)first=code;};
        keep(DestroyLinearSystem(system)); keep(MatDestroy(&coupling));
        keep(DestroyMixedBlocks(rawS)); keep(DestroyMixedBlocks(rawD));
        keep(DestroyMixedBlocks(s)); keep(DestroyMixedBlocks(d));
        keep(DestroyBoundaryData(bcS)); keep(DestroyBoundaryData(bcD));
        keep(VecDestroy(&modeS)); keep(VecDestroy(&modeD));
        keep(VecDestroy(&vertices)); keep(DMDestroy(&dm)); return first;
    }
};
PetscErrorCode AllValues(Vec vector,std::vector<PetscScalar>& out)
{
    VecScatter scatter=nullptr; Vec all=nullptr; const PetscScalar* values=nullptr;
    PetscInt n=0;
    PetscFunctionBeginUser;
    PetscCall(VecScatterCreateToAll(vector,&scatter,&all));
    PetscCall(VecScatterBegin(scatter,vector,all,INSERT_VALUES,SCATTER_FORWARD));
    PetscCall(VecScatterEnd(scatter,vector,all,INSERT_VALUES,SCATTER_FORWARD));
    PetscCall(VecGetSize(all,&n)); PetscCall(VecGetArrayRead(all,&values));
    out.assign(values,values+n);
    PetscCall(VecRestoreArrayRead(all,&values));
    PetscCall(VecScatterDestroy(&scatter)); PetscCall(VecDestroy(&all));
    PetscFunctionReturn(PETSC_SUCCESS);
}
PetscErrorCode MakeSnapshot(MPI_Comm comm,const in::InputConfig& c,Resources& r,
    const CellPorosityFunction& porosity,Snapshot& out)
{
    PetscFunctionBeginUser;
    out.stokes=c.flow.system!="darcy"; out.darcy=c.flow.system!="stokes";
    out.nx=r.mesh.CellDimensions().i; out.ny=r.mesh.CellDimensions().j;
    out.x=c.mesh.x; out.y=c.mesh.y;
    Vec us=nullptr,ud=nullptr,ps=nullptr,pd=nullptr;
    if(out.stokes && out.darcy)
        PetscCall(GetCoupledLinearSystemSolution(r.system,us,ps,ud,pd));
    else if(out.stokes) PetscCall(GetLinearSystemSolution(r.system,us,ps));
    else PetscCall(GetLinearSystemSolution(r.system,ud,pd));
    std::vector<PetscScalar> vs,vd,qs,qd;
    if(us) {PetscCall(AllValues(us,vs)); PetscCall(AllValues(ps,qs));}
    if(ud) {PetscCall(AllValues(ud,vd)); PetscCall(AllValues(pd,qd));}
    constexpr std::size_t stride=31;
    const auto n=static_cast<std::size_t>(r.mesh.CellCount());
    PetscCheck(n<=static_cast<std::size_t>(std::numeric_limits<int>::max())/stride,
               comm,PETSC_ERR_ARG_SIZ,"Example snapshot exceeds MPI count limits.");
    std::vector<double> packed(n*stride,0);
    const auto local=[&]() -> PetscErrorCode {
        PetscFunctionBeginUser;
        const auto range=r.mesh.OwnedCells();
        for(PetscInt j=range.begin.j;j<range.end.j;++j)
            for(PetscInt i=range.begin.i;i<range.end.i;++i) {
                const MeshIndex cell{i,j}; PetscInt id=0;
                PetscCall(r.mesh.CellId(cell,id));
                auto* row=packed.data()+stride*static_cast<std::size_t>(id);
                QuadVertices corners{}; PetscCall(r.mesh.GetCellCorners(cell,corners));
                for(int k=0;k<4;++k) for(int d=0;d<2;++d) row[2*k+d]=corners[k].p[d];
                std::vector<PetscInt> ids;
                if(us) {
                    PetscCall(r.sv.GetCellGlobalDofs(cell,ids));
                    for(int k=0;k<12;++k) row[8+k]=PetscRealPart(vs[ids[k]]);
                }
                if(ud) {
                    PetscCall(r.dv.GetCellGlobalDofs(cell,ids));
                    for(int k=0;k<8;++k) row[20+k]=PetscRealPart(vd[ids[k]]);
                }
                PetscCall(r.p.GetCellGlobalDofs(cell,ids));
                if(ps) row[28]=PetscRealPart(qs[ids[0]]);
                if(pd) row[29]=PetscRealPart(qd[ids[0]]);
                LocalPorositySamples phi; PetscCall(porosity(cell,phi)); row[30]=phi.average;
            }
        PetscFunctionReturn(PETSC_SUCCESS);
    };
    PetscCall(Agree(comm,local(),"snapshot extraction"));
    PetscCallMPI(MPI_Allreduce(MPI_IN_PLACE,packed.data(),static_cast<int>(packed.size()),MPI_DOUBLE,MPI_SUM,comm));
    out.cells.resize(n);
    const auto initialize=[&]() -> PetscErrorCode {
        PetscFunctionBeginUser;
        for(std::size_t id=0;id<n;++id) {
            const auto* row=packed.data()+stride*id; auto& cell=out.cells[id];
            for(int k=0;k<4;++k) for(int d=0;d<2;++d) cell.corners[k].p[d]=row[2*k+d];
            std::copy_n(row+8,12,cell.us.begin()); std::copy_n(row+20,8,cell.ud.begin());
            cell.ps=row[28]; cell.pd=row[29]; cell.phi=row[30];
            if(out.stokes) PetscCall(cell.br.Initialize(cell.corners));
            if(out.darcy) PetscCall(cell.hd.Initialize(cell.corners));
        }
        PetscFunctionReturn(PETSC_SUCCESS);
    };
    PetscCall(Agree(comm,initialize(),"reference geometry"));
    PetscFunctionReturn(PETSC_SUCCESS);
}
PetscErrorCode PressureLoad(MPI_Comm comm,const in::InputConfig& c,const MeshInfo& mesh,
    const DofMap& map,const GaussRule1D& rule,const in::FunctionInput& source,Vec rhs)
{
    PetscFunctionBeginUser;
    std::vector<PetscInt> ids;
    std::vector<PetscScalar> values;
    const auto local=[&]() -> PetscErrorCode {
        PetscFunctionBeginUser;
        const auto range=mesh.OwnedCells();
        for(PetscInt j=range.begin.j;j<range.end.j;++j)
            for(PetscInt i=range.begin.i;i<range.end.i;++i) {
                QuadVertices corners{}; PetscCall(mesh.GetCellCorners({i,j},corners));
                PetscReal integral=0;
                PetscCall(IntegrateCell(corners,rule,[&](const Point& p){return ScalarFunction(source,c,p);},integral));
                std::vector<PetscInt> cellIds; PetscCall(map.GetCellGlobalDofs({i,j},cellIds));
                ids.push_back(cellIds[0]); values.push_back(integral);
            }
        PetscFunctionReturn(PETSC_SUCCESS);
    };
    PetscCall(Agree(comm,local(),"pressure sources"));
    PetscCall(VecSetValues(rhs,static_cast<PetscInt>(ids.size()),ids.data(),values.data(),INSERT_VALUES));
    PetscCall(VecAssemblyBegin(rhs)); PetscCall(VecAssemblyEnd(rhs));
    PetscFunctionReturn(PETSC_SUCCESS);
}
PetscErrorCode MassResidual(Resources& r,Metrics& m)
{
    Vec residual=nullptr,p=nullptr,q=nullptr;
    PetscFunctionBeginUser;
    PetscCall(VecDuplicate(r.system.rhs,&residual));
    PetscCall(MatMult(r.system.matrix,r.system.solution,residual));
    PetscCall(VecAXPY(residual,-1,r.system.rhs));
    PetscCall(VecNestGetSubVec(residual,1,&p));
    PetscReal norm=0;
    if(r.system.kind==LinearSystemKind::Coupled) {
        PetscCall(VecNestGetSubVec(p,0,&q)); PetscCall(VecNorm(q,NORM_INFINITY,&norm)); m.stokesMass=norm;
        PetscCall(VecNestGetSubVec(p,1,&q)); PetscCall(VecNorm(q,NORM_INFINITY,&norm)); m.darcyMass=norm;
    } else {
        PetscCall(VecNorm(p,NORM_INFINITY,&norm));
        if(r.system.kind==LinearSystemKind::Stokes)m.stokesMass=norm; else m.darcyMass=norm;
    }
    PetscCall(VecDestroy(&residual));
    PetscFunctionReturn(PETSC_SUCCESS);
}
PetscErrorCode SolveOneImpl(const in::InputConfig& c,const std::string& family,int level,
    const Snapshot* reference,bool isReference,Resources& r,Snapshot& snapshot)
{
    const MPI_Comm comm=PETSC_COMM_WORLD;
    const bool stokes=c.flow.system!="darcy",darcy=c.flow.system!="stokes";
    const auto start=std::chrono::steady_clock::now();
    PetscFunctionBeginUser;
    const auto cells=in::CellsAtLevel(c,level);
    PetscCall(PetscPrintf(comm,"%s: %s level %d, %lld x %lld cells%s\n",c.simulation.name.c_str(),
        family.c_str(),level,static_cast<long long>(cells[0]),static_cast<long long>(cells[1]),
        isReference?" (numerical reference)":""));
    PetscCall(DMDACreate2d(comm,DM_BOUNDARY_NONE,DM_BOUNDARY_NONE,DMDA_STENCIL_BOX,
        static_cast<PetscInt>(cells[0]+1),static_cast<PetscInt>(cells[1]+1),
        c.parallel.processGrid[0],c.parallel.processGrid[1],2,1,nullptr,nullptr,&r.dm));
    PetscCall(DMSetUp(r.dm)); PetscCall(DMCreateGlobalVector(r.dm,&r.vertices));
    MeshParam parameters;
    parameters.xstart=c.mesh.x[0]; parameters.ystart=c.mesh.y[0];
    parameters.L=c.mesh.x[1]-c.mesh.x[0]; parameters.H=c.mesh.y[1]-c.mesh.y[0];
    parameters.perturbation=c.mesh.perturbation; parameters.seed=c.mesh.seed;
    if(family=="rectangular") PetscCall(CreateFullMesh(r.dm,r.vertices,parameters));
    else PetscCall(LogicRectMesh(r.dm,r.vertices,parameters));
    if(c.convergence.referenceSource=="column_velocity" && c.convergence.reference.Find("fit_interface") &&
       c.convergence.reference.At("fit_interface").AsBool() && c.porosity.prescribedFunction.name!="constant") {
        const double y0=c.porosity.prescribedFunction.parameters.At("interface_y").AsReal();
        const auto row=static_cast<PetscInt>(std::llround((y0-c.mesh.y[0])/(c.mesh.y[1]-c.mesh.y[0])*cells[1]));
        PetscInt xs=0,ys=0,xm=0,ym=0;
        PetscCall(DMDAGetCorners(r.dm,&xs,&ys,nullptr,&xm,&ym,nullptr));
        PetscScalar*** v=nullptr;PetscCall(DMDAVecGetArrayDOF(r.dm,r.vertices,&v));
        if(row>=ys && row<ys+ym)for(PetscInt i=xs;i<xs+xm;++i)v[row][i][1]=y0;
        PetscCall(DMDAVecRestoreArrayDOF(r.dm,r.vertices,&v));
    }
    PetscCall(BuildMeshInfo(r.dm,r.vertices,r.mesh));
    PetscCall(r.p.Initialize(r.dm,r.mesh,DofSpace::CellPressure));
    if(stokes) PetscCall(r.sv.Initialize(r.dm,r.mesh,DofSpace::BRVelocity));
    if(darcy) PetscCall(r.dv.Initialize(r.dm,r.mesh,DofSpace::HDivVelocity));
    PetscErrorCode local=CreateGaussRule(c.quadrature.cellPointsPerAxis,r.cellRule);
    if(!local)local=CreateGaussRule(c.quadrature.edgePoints,r.edgeRule);
    PetscCall(Agree(comm,local,"quadrature"));
    GaussData data;
    if(c.porosity.source=="gauss_point_data")
        PetscCall(ReadGaussData(comm,c,r.mesh,r.cellRule,r.edgeRule,family,level,data));
    const CellPorosityFunction porosity=[&](MeshIndex index,LocalPorositySamples& p) -> PetscErrorCode {
        PetscFunctionBeginUser;
        QuadVertices v{}; PetscCall(r.mesh.GetCellCorners(index,v));
        PetscInt id=0; PetscCall(r.mesh.CellId(index,id));
        const auto nc=r.cellRule.points.size(), ne=r.edgeRule.points.size();
        p.cell.resize(nc*nc); double sum=0,area=0;
        for(std::size_t j=0;j<nc;++j) for(std::size_t i=0;i<nc;++i) {
            const Point ref{{r.cellRule.points[i],r.cellRule.points[j]}};
            const auto g=j*nc+i;
            p.cell[g]=data.cell.empty()?Porosity(c,MapCellPoint(ref,v)):data.cell[static_cast<std::size_t>(id)*nc*nc+g];
            const double w=r.cellRule.weights[i]*r.cellRule.weights[j]*CellJacobian(ref,v);
            sum+=w*p.cell[g]; area+=w;
        }
        p.average=sum/area;
        std::array<OrientedEdge,4> edges{}; PetscCall(r.mesh.GetCellEdges(index,edges));
        for(int e=0;e<4;++e) {
            p.edge[e].resize(ne);
            for(std::size_t g=0;g<ne;++g) {
                const auto canonical=edges[e].direction==1?g:ne-1-g;
                p.edge[e][g]=data.edge.empty()?Porosity(c,MapEdgePoint(r.edgeRule.points[g],{v[e],v[(e+1)%4]})):
                    data.edge[static_cast<std::size_t>(edges[e].id)*ne+canonical];
                // Chapter 4, Eq. (4.1): use one harmonic trace on an edge
                // coinciding with a jump. A dry neighbour gives a zero trace.
                if(data.edge.empty() && c.porosity.prescribedFunction.name=="piecewise_constant") {
                    const auto& fp=c.porosity.prescribedFunction.parameters;
                    const double y0=fp.At("interface_y").AsReal();
                    const double tol=c.boundaryRegions.coordinateTolerance;
                    if(std::abs(v[e].p[1]-y0)<=tol && std::abs(v[(e+1)%4].p[1]-y0)<=tol) {
                        EdgeTopology topology;PetscCall(r.mesh.GetEdgeTopology(edges[e].id,topology));
                        if(!topology.IsBoundary()) {
                            const double a=fp.At("value_below").AsReal(),b=fp.At("value_above").AsReal();
                            p.edge[e][g]=(a+b)>0?2*a*b/(a+b):0;
                        } else {
                            const auto center=MapCellPoint(Point{{0,0}},v);
                            p.edge[e][g]=fp.At(center.p[1]<y0?"value_below":"value_above").AsReal();
                        }
                    }
                }
            }
        }
        PetscFunctionReturn(PETSC_SUCCESS);
    };
    LocalMatrixParameters model;
    model.theta=c.flow.theta; model.darcyCompactionAverageCutoff=c.flow.darcyCompactionAverageCutoff;
    model.couplingAverageCutoff=c.flow.couplingAverageCutoff;
    if(stokes) {
        PetscCall(AssembleStokesBlocks(comm,r.mesh,r.sv,r.p,r.cellRule,porosity,
            [&](const Point& p){return VectorFunction(c.flow.stokesForce,c,p);},r.rawS));
        PetscCall(PressureLoad(comm,c,r.mesh,r.p,r.cellRule,c.flow.stokesPressureSource,r.rawS.g));
    }
    if(darcy) {
        PetscCall(AssembleDarcyBlocks(comm,r.mesh,r.dv,r.p,r.cellRule,r.edgeRule,porosity,model,
            [&](const Point& p){return VectorFunction(c.flow.darcyForce,c,p);},r.rawD));
        PetscCall(PressureLoad(comm,c,r.mesh,r.p,r.cellRule,c.flow.darcyPressureSource,r.rawD.g));
    }
    StokesBoundarySpecification sb; DarcyBoundarySpecification db;
    PetscCall(BoundarySpecifications(c,level,sb,db));
    BoundaryApplicationOptions application; application.pressureRowSign=-1;
    if(stokes) {
        PetscCall(BuildStokesBoundaryData(comm,r.mesh,r.sv,r.edgeRule,sb,c.time.start,r.bcS));
        PetscCall(ApplyBoundaryConditions(comm,r.rawS,r.bcS,application,r.s));
    }
    if(darcy) {
        PetscCall(BuildDarcyBoundaryData(comm,r.mesh,r.dv,r.edgeRule,db,c.time.start,porosity,model,r.bcD));
        PetscCall(ApplyBoundaryConditions(comm,r.rawD,r.bcD,application,r.d));
    }
    const auto mode=c.flow.pressureNullspace=="none"?PressureNullspaceMode::None:
        c.flow.pressureNullspace=="constant"?PressureNullspaceMode::Constant:PressureNullspaceMode::Provided;
    if(mode==PressureNullspaceMode::Provided) {
        if(stokes){PetscCall(VecDuplicate(r.s.g,&r.modeS)); PetscCall(VecSet(r.modeS,c.flow.pressureModes.At("stokes").AsReal()));}
        if(darcy) {
            PetscCall(VecDuplicate(r.d.g,&r.modeD));
            if(c.flow.pressureModes.At("darcy").scalar=="sqrt_cell_average_porosity") {
                PetscScalar* values=nullptr;PetscCall(VecGetArray(r.modeD,&values));
                const auto fill=[&]() -> PetscErrorCode {
                    PetscFunctionBeginUser;
                    const auto range=r.mesh.OwnedCells();
                    for(PetscInt j=range.begin.j;j<range.end.j;++j)for(PetscInt i=range.begin.i;i<range.end.i;++i) {
                        LocalPorositySamples phi;PetscCall(porosity({i,j},phi));
                        std::vector<PetscInt> ids;PetscCall(r.p.GetCellGlobalDofs({i,j},ids));
                        values[ids[0]-r.p.OwnershipBegin()]=std::sqrt(phi.average);
                    }
                    PetscFunctionReturn(PETSC_SUCCESS);
                };
                const auto error=fill();PetscCall(VecRestoreArray(r.modeD,&values));
                PetscCall(Agree(comm,error,"provided pressure mode"));
            } else PetscCall(VecSet(r.modeD,c.flow.pressureModes.At("darcy").AsReal()));
        }
    }
    if(stokes && darcy) {
        PetscCall(AssemblePressureCoupling(comm,r.mesh,r.p,r.p,r.cellRule,porosity,model,r.coupling));
        CoupledLinearSystemOptions options;
        options.stokes.pressureRowSign=-1; options.darcy.pressureRowSign=-1;
        options.pressureNullspace.mode=mode; options.pressureNullspace.projectRhs=c.flow.projectRhs;
        options.pressureNullspace.stokesPressureMode=r.modeS; options.pressureNullspace.darcyPressureMode=r.modeD;
        PetscCall(BuildCoupledLinearSystem(comm,r.s,r.d,r.coupling,options,r.system));
    } else {
        LinearSystemOptions options; options.pressureRowSign=-1;
        options.pressureNullspace.mode=mode; options.pressureNullspace.projectRhs=c.flow.projectRhs;
        options.pressureNullspace.pressureMode=stokes?r.modeS:r.modeD;
        if(stokes) PetscCall(BuildStokesLinearSystem(comm,r.s,options,r.system));
        else PetscCall(BuildDarcyLinearSystem(comm,r.d,options,r.system));
    }
    Metrics metrics; metrics.family=family; metrics.level=level;
    metrics.nx=static_cast<PetscInt>(cells[0]); metrics.ny=static_cast<PetscInt>(cells[1]);
    auto options=SolverOptions(c.flow.solver);
    // The example writes the report before propagating ordinary nonconvergence.
    const PetscErrorCode solveError=SolveLinearSystem(r.system,options,metrics.solver);
    metrics.seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
    PetscCall(WriteSolverReport(comm,c,metrics));
    PetscCall(solveError);
    PetscCheck(metrics.solver.converged,comm,PETSC_ERR_NOT_CONVERGED,
               "Example solve failed its acceptance policy; inspect solver_report.csv. No convergence error is reported for this iterate.");
    PetscCall(MassResidual(r,metrics));
    PetscCall(MakeSnapshot(comm,c,r,porosity,snapshot));
    if(!isReference && c.convergence.enabled)
        PetscCall(Compare(comm,c,r.mesh,snapshot,reference,metrics));
    else {
        metrics.reference=isReference?"numerical_reference":"none";
        double h=0;
        for(const auto& cell:snapshot.cells) for(int a=0;a<4;++a) for(int b=a+1;b<4;++b)
            h=std::max(h,std::hypot(cell.corners[a].p[0]-cell.corners[b].p[0],cell.corners[a].p[1]-cell.corners[b].p[1]));
        metrics.h=h;
    }
    PetscCall(WriteFields(comm,c,r.mesh,snapshot,family,level));
    PetscCall(WriteReports(comm,c,snapshot,metrics,family,level));
    PetscCall(PetscPrintf(comm,"  iterations=%" PetscInt_FMT ", true relative residual=%.3e\n",
        metrics.solver.iterations,metrics.solver.relativeTrueResidualNorm));
    PetscFunctionReturn(PETSC_SUCCESS);
}
PetscErrorCode SolveOne(const in::InputConfig& c,const std::string& family,int level,
                       const Snapshot* reference,bool isReference,Snapshot& output)
{
    Resources resources;
    PetscErrorCode status=SolveOneImpl(c,family,level,reference,isReference,resources,output);
    const PetscErrorCode cleanup=resources.Clear();
    return status?status:cleanup;
}
} // namespace
PetscErrorCode Run(const in::InputConfig& c)
{
    PetscFunctionBeginUser;
    const auto families=c.convergence.enabled?c.convergence.meshFamilies:std::vector<std::string>{c.mesh.family};
    for(const auto& family:families) {
        Snapshot reference;
        const bool numerical=c.convergence.enabled && c.convergence.referenceSource=="finer_mesh";
        if(numerical)
            PetscCall(SolveOne(c,family,c.convergence.levels-1+c.convergence.additionalReferenceLevels,
                              nullptr,true,reference));
        const int levels=c.convergence.enabled?c.convergence.levels:1;
        for(int level=0;level<levels;++level) {
            Snapshot solution;
            PetscCall(SolveOne(c,family,level,numerical?&reference:nullptr,false,solution));
        }
    }
    PetscFunctionReturn(PETSC_SUCCESS);
}
} // namespace mantle::driver

// PETSc/MPI process lifetime and command-line entry point.
namespace {
PetscErrorCode RunFromCommandLine()
{
    PetscFunctionBeginUser;
    char filename[PETSC_MAX_PATH_LEN]{};
    PetscBool hasInput = PETSC_FALSE, help = PETSC_FALSE;
    PetscCall(PetscOptionsGetString(nullptr, nullptr, "-input", filename,
                                   sizeof(filename), &hasInput));
    PetscCall(PetscOptionsHasName(nullptr, nullptr, "-help", &help));
    if (help && !hasInput) PetscFunctionReturn(PETSC_SUCCESS);
    PetscCheck(hasInput && filename[0], PETSC_COMM_WORLD, PETSC_ERR_USER_INPUT,
               "Supply -input /path/to/example/input.yaml");

    mantle::input::InputConfig input;
    PetscCall(mantle::input::ReadInput(PETSC_COMM_WORLD, filename, input,
                                      mantle::driver::ReaderOptions()));
    try {
        mantle::driver::ValidateDriver(input);
    } catch (const std::exception& error) {
        SETERRQ(PETSC_COMM_WORLD, PETSC_ERR_USER_INPUT, "%s", error.what());
    }
    PetscCall(mantle::driver::Run(input));
    PetscFunctionReturn(PETSC_SUCCESS);
}
} // namespace

int main(int argc, char** argv)
{
    const char help[] =
        "MantlePar shared steady-flow driver.\n"
        "Usage: <executable> -input /path/to/example/input.yaml\n"
        "Select Stokes, Darcy or coupled flow and all case settings in YAML.\n";
    PetscErrorCode status = PetscInitialize(&argc, &argv, nullptr, help);
    if (status) return static_cast<int>(status);

    status = RunFromCommandLine();
    const PetscErrorCode finalStatus = PetscFinalize();
    return static_cast<int>(status ? status : finalStatus);
}
