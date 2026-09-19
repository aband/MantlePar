#include "initialization.h"

#include <petscao.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>
#include <utility>

namespace mantle::couple {
PetscErrorCode InitializeFlow(MPI_Comm, const Configuration&, InitialState&);
namespace {
using input::Value;
using input::InputError;

void Require(bool ok, const std::string& message)
{ if (!ok) throw InputError(message); }
void Keys(const Value& v, std::initializer_list<const char*> allowed)
{
    if (v.kind != Value::Kind::Mapping) v.Fail("expected a mapping");
    for (const auto& kv : v.mapping) {
        const bool known = std::any_of(allowed.begin(), allowed.end(),
            [&](const char* key) { return kv.first == key; });
        if (!known) kv.second.Fail("unknown setting");
    }
}
double Real(const Value& v, const char* key, double fallback)
{ return v.Find(key) ? v.At(key).AsReal() : fallback; }
std::array<double,2> Pair(const Value& v)
{
    if (v.kind != Value::Kind::Sequence || v.sequence.size() != 2)
        v.Fail("expected exactly two numbers");
    return {{v.sequence[0].AsReal(), v.sequence[1].AsReal()}};
}
PetscReal Finite(double x)
{
    const PetscReal value = static_cast<PetscReal>(x);
    Require(std::isfinite(x) && !PetscIsInfOrNanReal(value),
            "A configured profile produced a nonfinite/unrepresentable value.");
    return value;
}
input::FunctionInput Function(const Value& v)
{
    Keys(v, {"name", "parameters"});
    input::FunctionInput f;
    f.name = v.At("name").AsString();
    if (const auto* p = v.Find("parameters")) f.parameters = *p;
    else { f.parameters.kind = Value::Kind::Mapping; f.parameters.path = v.path + ".parameters"; }
    return f;
}
ScalarProfile Profile(const input::FunctionInput& f)
{
    const auto& p = f.parameters;
    if (f.name == "constant") {
        Keys(p, {"value"});
        const auto value = Finite(p.At("value").AsReal());
        return [value](const Point&, PetscReal) { return value; };
    }
    if (f.name == "affine") {
        Keys(p, {"value", "origin", "gradient", "time_origin", "time_slope"});
        const auto value = p.At("value").AsReal();
        const auto origin = p.Find("origin") ? Pair(p.At("origin")) : std::array<double,2>{{0,0}};
        const auto gradient = Pair(p.At("gradient"));
        const auto t0 = Real(p, "time_origin", 0), slope = Real(p, "time_slope", 0);
        return [=](const Point& x, PetscReal t) {
            return Finite(value + gradient[0]*(x.p[0]-origin[0])
                + gradient[1]*(x.p[1]-origin[1]) + slope*(t-t0));
        };
    }
    if (f.name == "gaussian") {
        Keys(p, {"background", "amplitude", "center", "width"});
        const auto background = p.At("background").AsReal();
        const auto amplitude = p.At("amplitude").AsReal();
        const auto center = Pair(p.At("center")), width = Pair(p.At("width"));
        if (width[0] <= 0 || width[1] <= 0) p.At("width").Fail("widths must be positive");
        return [=](const Point& x, PetscReal) {
            const double a = (x.p[0]-center[0])/width[0], b = (x.p[1]-center[1])/width[1];
            return Finite(background + amplitude*std::exp(-0.5*(a*a+b*b)));
        };
    }
    if (f.name == "erf_ramp") {
        Keys(p, {"amplitude", "origin_x", "cutoff_x", "width"});
        const auto amplitude=Finite(p.At("amplitude").AsReal());
        const auto origin=Real(p,"origin_x",0), cutoff=p.At("cutoff_x").AsReal();
        const auto width=Real(p,"width",1);
        Require(width>0 && cutoff>origin,"erf_ramp requires width>0 and cutoff_x>origin_x.");
        const auto denominator=std::erf((cutoff-origin)/width);
        Require(denominator>0,"erf_ramp normalization underflowed.");
        return [=](const Point& x,PetscReal) {
            return x.p[0]>=cutoff?amplitude:Finite(amplitude*std::erf((x.p[0]-origin)/width)/denominator);
        };
    }
    if (f.name == "piecewise_constant") {
        Keys(p, {"interface_y", "value_below", "value_above"});
        const auto y = p.At("interface_y").AsReal();
        const auto lo = Finite(p.At("value_below").AsReal()), hi = Finite(p.At("value_above").AsReal());
        return [=](const Point& x, PetscReal) { return x.p[1] < y ? lo : hi; };
    }
    if (f.name == "quadratic_below_interface") {
        Keys(p, {"coefficient", "interface_y", "value_above_interface"});
        const auto a = p.At("coefficient").AsReal(), y = p.At("interface_y").AsReal();
        const auto above = Finite(p.At("value_above_interface").AsReal());
        return [=](const Point& x, PetscReal) {
            return x.p[1] < y ? Finite(a*(x.p[1]-y)*(x.p[1]-y)) : above;
        };
    }
    p.Fail("unsupported scalar profile '" + f.name + "'");
}
ScalarProfile BoundaryProfile(const Value& v, const ScalarProfile& initial, PetscReal initialTime)
{
    const auto f = Function(v);
    if (f.name != "initial") return Profile(f);
    Keys(f.parameters, {});
    Require(static_cast<bool>(initial), "The initial profile is only available for H and C boundary data.");
    // A reservoir fixed to the INITIAL profile, even if that profile has a time slope.
    return [initial, initialTime](const Point& p, PetscReal) { return initial(p, initialTime); };
}
ScalarProfile CheckedComposition(ScalarProfile profile, double Xe)
{
    return [profile=std::move(profile), Xe](const Point& p, PetscReal t) {
        const auto c = profile(p,t);
        Require(c >= 0 && c <= Xe, "Composition must lie in [0, phase.parameters.Xe]; no clipping is applied.");
        return c;
    };
}

void PhaseConfiguration(const input::InputConfig& c, Configuration& result)
{
    const auto& v = c.phase.settings;
    Keys(v, {"enabled", "model", "parameters", "pressure"});
    if (v.At("model").AsString() != "eutectic_rescaled") v.At("model").Fail("expected eutectic_rescaled");
    phase::MaterialParameters p;
    if (const auto* m = v.Find("parameters")) {
        Keys(*m, {"Tm0","Te0","nu","L","cp","Xe","rho","rhor","mus","mul","k0","g","alpha0"});
        p.Tm0=Real(*m,"Tm0",p.Tm0); p.Te0=Real(*m,"Te0",p.Te0); p.nu=Real(*m,"nu",p.nu);
        p.L=Real(*m,"L",p.L); p.cp=Real(*m,"cp",p.cp); p.Xe=Real(*m,"Xe",p.Xe);
        p.rho=Real(*m,"rho",p.rho); p.rhor=Real(*m,"rhor",p.rhor);
        p.mus=Real(*m,"mus",p.mus); p.mul=Real(*m,"mul",p.mul); p.k0=Real(*m,"k0",p.k0);
        p.g=Real(*m,"g",p.g); p.alpha0=Real(*m,"alpha0",p.alpha0);
    }
    try { result.phase.setParameters(p); }
    catch (const std::exception& e) { v.Fail(e.what()); }
    const auto& pressure = v.At("pressure");
    const auto name = pressure.At("model").AsString();
    if (name == "constant") {
        Keys(pressure, {"model", "value_pa"});
        const auto value = Finite(pressure.At("value_pa").AsReal());
        if (value < 0) pressure.Fail("pressure must be nonnegative Pa");
        result.pressure = [value](const Point&, PetscReal) { return value; };
    } else if (name == "lithostatic") {
        Keys(pressure, {"model", "surface_y"});
        const auto surface = pressure.At("surface_y").AsReal();
        if (surface < c.mesh.y[1]) pressure.Fail("surface_y must be at or above the mesh top");
        const auto model = result.phase;
        result.pressure = [model, surface](const Point& x, PetscReal) {
            return Finite(model.staticPressure(surface-x.p[1]));
        };
    } else pressure.At("model").Fail("expected constant or lithostatic");
}

CellSide Side(input::Side side)
{
    switch (side) {
    case input::Side::Left: return CellSide::Left;
    case input::Side::Right: return CellSide::Right;
    case input::Side::Bottom: return CellSide::Bottom;
    case input::Side::Top: return CellSide::Top;
    }
    throw InputError("Invalid boundary side.");
}
BoundaryRegion Region(const input::InputConfig& c, const Value& segment)
{
    const auto range = input::ResolveBoundaryRegion(c, segment.At("region").AsString());
    const auto priority = segment.Find("priority") ? segment.At("priority").AsInteger() : 10;
    if (priority <= 0 || priority > std::numeric_limits<int>::max())
        segment.Fail("segment priority must be a positive int (default priority is zero)");
    return {Side(range.side),static_cast<PetscInt>(range.first),static_cast<PetscInt>(range.end),static_cast<int>(priority)};
}
template<class Condition, class Parser>
std::vector<BoundaryRule<Condition>> Rules(const input::InputConfig& c, const Value& v, Parser parse)
{
    Keys(v, {"default", "segments"});
    std::vector<BoundaryRule<Condition>> result;
    const auto fallback = parse(v.At("default"),false);
    for (int k=0;k<4;++k) result.push_back({{static_cast<CellSide>(k),0,-1,0},fallback});
    if (const auto* segments = v.Find("segments")) {
        if (segments->kind != Value::Kind::Sequence) segments->Fail("expected a sequence");
        for (const auto& s : segments->sequence) {
            const auto r = Region(c,s);
            for (const auto& old : result)
                if (old.region.side == r.side && old.region.priority == r.priority &&
                    old.region.firstEdge < r.endEdge && r.firstEdge < old.region.endEdge)
                    s.Fail("overlapping boundary segments have equal priority");
            result.push_back({r,parse(s,true)});
        }
    }
    return result;
}
template<class Condition>
const Condition& Select(const std::vector<BoundaryRule<Condition>>& rules, const BoundaryPoint& p)
{
    Require(p.sideEdge >= 0, "Boundary sideEdge must be nonnegative.");
    const BoundaryRule<Condition>* selected = nullptr;
    for (const auto& r : rules) {
        if (r.region.side != p.side || p.sideEdge < r.region.firstEdge ||
            (r.region.endEdge >= 0 && p.sideEdge >= r.region.endEdge)) continue;
        if (selected && r.region.priority == selected->region.priority)
            throw InputError("Ambiguous boundary rules at equal priority.");
        if (!selected || r.region.priority > selected->region.priority) selected = &r;
    }
    Require(selected != nullptr, "Boundary rules do not cover this edge.");
    return selected->condition;
}

void TransportConfiguration(const input::InputConfig& c, Configuration& result)
{
    const auto& v = c.transport.settings;
    Keys(v, {"enabled", "initial_conditions", "boundary_conditions", "stabilizer", "thermal_diffusion"});
    const auto& initial = v.At("initial_conditions");
    Keys(initial, {"enthalpy", "composition"});
    result.initialEnthalpy = Profile(Function(initial.At("enthalpy")));
    result.initialComposition = CheckedComposition(Profile(Function(initial.At("composition"))), result.phase.parameters().Xe);
    const auto& bc = v.At("boundary_conditions");
    Keys(bc, {"enthalpy", "composition", "temperature"});
    const auto advection = [&](const Value& item, ScalarProfile initialProfile, bool composition) {
        return Rules<AdvectionCondition>(c,item,[&](const Value& a, bool segment) {
            if (segment) Keys(a,{"region","priority","type","value"});
            else Keys(a,{"type","value"});
            AdvectionCondition condition;
            const auto type = a.At("type").AsString();
            if (type == "inflow_outflow") {
                condition.policy = AdvectionPolicy::InflowOutflow;
                condition.inflow = BoundaryProfile(a.At("value"),initialProfile,c.time.start);
                if (composition) condition.inflow = CheckedComposition(condition.inflow,result.phase.parameters().Xe);
            } else {
                if (a.Find("value")) a.At("value").Fail("this condition takes no value");
                if (type == "outflow") condition.policy = AdvectionPolicy::Outflow;
                else if (type == "zero_flux") condition.policy = AdvectionPolicy::ZeroFlux;
                else a.At("type").Fail("expected inflow_outflow, outflow or zero_flux");
            }
            return condition;
        });
    };
    result.boundary.enthalpy = advection(bc.At("enthalpy"),result.initialEnthalpy,false);
    result.boundary.composition = advection(bc.At("composition"),result.initialComposition,true);
    result.boundary.temperature = Rules<ThermalCondition>(c,bc.At("temperature"),[](const Value& a, bool segment) {
        if (segment) Keys(a,{"region","priority","type","value"});
        else Keys(a,{"type","value"});
        ThermalCondition condition;
        const auto type = a.At("type").AsString();
        if (type == "zero_flux") {
            if (a.Find("value")) a.At("value").Fail("zero_flux takes no value");
        } else {
            if (type == "prescribed_temperature") condition.policy = ThermalPolicy::Temperature;
            else if (type == "prescribed_flux") condition.policy = ThermalPolicy::OutwardFlux;
            else a.At("type").Fail("expected prescribed_temperature, prescribed_flux or zero_flux");
            condition.value = Profile(Function(a.At("value")));
        }
        return condition;
    });
    if (const auto* s = v.Find("stabilizer")) {
        Keys(*s,{"type","linearization","global_speed"});
        const auto type = s->At("type").AsString();
        if (type == "local_lax_friedrichs") {
            if (s->Find("global_speed")) s->At("global_speed").Fail("global_speed requires global_lax_friedrichs");
        } else if (type == "global_lax_friedrichs") {
            result.stabilizer.mode = LaxFriedrichsMode::Global;
            result.stabilizer.globalSpeed = Finite(s->At("global_speed").AsReal());
            if (result.stabilizer.globalSpeed < 0) s->Fail("global_speed must be nonnegative");
        } else s->At("type").Fail("expected local_lax_friedrichs or global_lax_friedrichs");
        if (const auto* l = s->Find("linearization")) {
            const auto name = l->AsString();
            if (name == "frozen_speed") result.stabilizer.linearization = LaxFriedrichsLinearization::FrozenSpeed;
            else if (name != "full") l->Fail("expected full or frozen_speed");
        }
    }
    if (const auto* d = v.Find("thermal_diffusion")) {
        Keys(*d,{"diffusivity","number_of_samples","extent_fraction"});
        result.thermalDiffusivity = Finite(d->At("diffusivity").AsReal());
        if (result.thermalDiffusivity < 0) d->Fail("diffusivity must be nonnegative");
        if (const auto* n = d->Find("number_of_samples")) {
            const auto count = n->AsInteger();
            if (count < 2 || count%2 || count >= std::numeric_limits<PetscInt>::max())
                n->Fail("expected an even sample count >= 2 representable by PetscInt, with room for the boundary node");
            result.sampling.numberOfSamples = static_cast<PetscInt>(count);
        }
        result.sampling.extentFraction = Finite(Real(*d,"extent_fraction",0.9));
        if (!(result.sampling.extentFraction > 0 && result.sampling.extentFraction <= 1))
            d->Fail("extent_fraction must belong to (0,1]");
    }
}

BoundaryCondition FlowCondition(const input::BoundaryConditionInput& in, double sign=1)
{
    BoundaryCondition result;
    result.type = in.type == input::BoundaryKind::Dirichlet ? BoundaryType::Dirichlet : BoundaryType::Neumann;
    const auto profile = Profile(in.value);
    result.value.function = [profile,sign](const BoundaryPoint& p) { return Finite(sign*profile(p.position,p.time)); };
    return result;
}
void Components(const input::InputConfig& c, CellSide side,
                const std::array<std::optional<input::BoundaryConditionInput>,2>& in,
                std::array<BoundaryCondition,2>& out)
{
    if (c.flow.boundary.stokesComponents == "cartesian") {
        for (int k=0;k<2;++k) if (in[k]) out[k]=FlowCondition(*in[k]);
    } else {
        const Point normals[4] = {{{0,-1}},{{1,0}},{{0,1}},{{-1,0}}};
        const auto n = normals[static_cast<int>(side)];
        const Point directions[2] = {n,Point{{-n.p[1],n.p[0]}}};
        for (int k=0;k<2;++k) if (in[k])
            for (int d=0;d<2;++d) if (directions[k].p[d] != 0)
                out[d]=FlowCondition(*in[k],directions[k].p[d]);
    }
}
LocalForceFunction Force(const input::FunctionInput& f)
{
    if (f.name != "constant") f.parameters.Fail("initialization currently supports constant vector forcing");
    Keys(f.parameters,{"value"});
    const auto v = Pair(f.parameters.At("value"));
    const Point point{{Finite(v[0]),Finite(v[1])}};
    return [point](const Point&) { return point; };
}
FlowSetup FlowConfiguration(const input::InputConfig& c)
{
    FlowSetup out;
    out.dryPorosity=c.porosity.source=="prescribed_function";
    const auto& b = c.flow.boundary;
    out.stokes.absoluteTolerance=b.absoluteTolerance; out.stokes.relativeTolerance=b.relativeTolerance;
    out.darcy.neumannVariable = b.darcyNeumannVariable=="pressure_potential" ? DarcyNeumannVariable::PressurePotential
        : b.darcyNeumannVariable=="assembled_pressure" ? DarcyNeumannVariable::AssembledPressure
        : DarcyNeumannVariable::WeightedNormalLoad;
    out.darcy.pressureSign=-1; // legacy A*u - B^T*p momentum equation
    const std::array<std::optional<input::BoundaryConditionInput>,2> defaults{{b.stokesDefault[0],b.stokesDefault[1]}};
    for (int k=0;k<4;++k) {
        StokesBoundaryRule s; s.region.side=static_cast<CellSide>(k);
        Components(c,s.region.side,defaults,s.component); out.stokes.rules.push_back(s);
        DarcyBoundaryRule d; d.region.side=s.region.side; d.condition=FlowCondition(b.darcyDefault);
        out.darcy.rules.push_back(d);
    }
    for (const auto& s : b.stokesSegments) {
        const auto r=input::ResolveBoundaryRegion(c,s.region);
        StokesBoundaryRule rule;
        rule.region={Side(r.side),static_cast<PetscInt>(r.first),static_cast<PetscInt>(r.end),s.priority};
        Components(c,rule.region.side,s.component,rule.component); out.stokes.rules.push_back(rule);
    }
    for (const auto& s : b.darcySegments) {
        const auto r=input::ResolveBoundaryRegion(c,s.region);
        DarcyBoundaryRule rule;
        rule.region={Side(r.side),static_cast<PetscInt>(r.first),static_cast<PetscInt>(r.end),s.priority};
        rule.condition=FlowCondition(s.condition); out.darcy.rules.push_back(rule);
    }
    out.material.theta=c.flow.theta;
    out.material.darcyCompactionAverageCutoff=c.flow.darcyCompactionAverageCutoff;
    out.material.couplingAverageCutoff=c.flow.couplingAverageCutoff;
    const auto& in=c.flow.solver; auto& solver=out.solver;
    solver.optionsPrefix=in.optionsPrefix; solver.kspType=in.ksp;
    solver.preconditioner=in.preconditioner=="schur" ? LinearPreconditioner::Schur :
        in.preconditioner=="sparse_lu" ? LinearPreconditioner::SparseLU : LinearPreconditioner::None;
    solver.relativeTolerance=in.relativeTolerance; solver.absoluteTolerance=in.absoluteTolerance;
    solver.divergenceTolerance=in.divergenceTolerance; solver.maximumIterations=in.maximumIterations;
    solver.initialGuessNonzero=in.initialGuessNonzero;
    const auto block=[](const input::BlockSolverInput& i) {
        LinearBlockSolverOptions o; o.kspType=i.ksp; o.pcType=i.pc;
        o.relativeTolerance=i.relativeTolerance; o.absoluteTolerance=i.absoluteTolerance;
        o.divergenceTolerance=i.divergenceTolerance; o.maximumIterations=i.maximumIterations;
        return o;
    };
    solver.velocity=block(in.velocity); solver.pressure=block(in.pressure);
    solver.errorIfNotConverged=in.errorIfNotConverged;
    solver.requireSubsolverConvergence=in.requireSubsolverConvergence;
    solver.requireTrueResidual=in.requireTrueResidual;
    solver.removePressureNullspace=in.removePressureNullspace;
    out.stokesForce=Force(c.flow.stokesForce); out.darcyForce=Force(c.flow.darcyForce);
    out.stokesPressureSource=Profile(c.flow.stokesPressureSource);
    out.darcyPressureSource=Profile(c.flow.darcyPressureSource);
    return out;
}

void ValidateStage(const input::InputConfig& c)
{
    Require(c.flow.enabled && c.phase.enabled && c.transport.enabled,
            "Coupling initialization requires flow, phase and transport enabled.");
    Require(c.flow.system=="coupled_stokes_darcy", "flow.system must be coupled_stokes_darcy.");
    const bool phasePorosity=c.porosity.source=="gauss_point_data" && c.porosity.provider=="runtime_field" &&
            c.porosity.runtimeField=="phase_porosity";
    const bool dryPorosity=c.porosity.source=="prescribed_function" && c.porosity.prescribedFunction.name=="constant" &&
            c.porosity.prescribedFunction.parameters.At("value").AsReal()==0;
    Require(phasePorosity || dryPorosity,
            "porosity must select runtime phase_porosity or prescribed constant zero for legacy dry preheat.");
    Require(c.coupling.backend=="internal" && c.coupling.internalScheme=="single_pass" &&
            c.coupling.updateOrder==std::vector<std::string>{"phase","flow","transport"},
            "This starter uses internal/single_pass and update_order: [phase, flow, transport].");
    Require(!c.restart.enabled && !c.restart.writeCheckpoints, "Restart/checkpoints are not implemented by the initializer.");
    Require(!c.convergence.enabled, "The initializer runs one mesh; disable studies.convergence.");
    Require(c.output.quantities.empty() && !c.output.profilesEnabled && !c.output.plotProfiles && !c.output.plotConvergence,
            "The initializer writes initial CSV/setup output; omit flow-driver fields/profiles/plots requests.");
    for (const auto& m : c.extensions) Require(!m.second.enabled,"The initializer does not support extra physics modules.");
    for (int d=0;d<2;++d) {
        Require(c.mesh.cells[d]>0 && c.mesh.cells[d]<std::numeric_limits<PetscInt>::max(),"Mesh dimensions exceed PetscInt.");
        Require(c.parallel.processGrid[d]<=c.mesh.cells[d],"Each field-DM rank must own at least one cell in each direction.");
    }
    Require(c.quadrature.edgePoints>=2,"Stokes boundary projection requires at least two edge quadrature points.");
    if (c.flow.pressureNullspace=="provided") {
        Keys(c.flow.pressureModes,{"stokes","darcy"});
        (void)c.flow.pressureModes.At("stokes").AsReal();
        const auto& d=c.flow.pressureModes.At("darcy");
        if (d.AsString()!="sqrt_cell_average_porosity") (void)d.AsReal();
    }
}

// Execute only local work here, then agree before any subsequent collective.
template<class Function> PetscErrorCode Local(Function&& f)
{
    PetscFunctionBeginUser;
    try { const auto error=f(); PetscFunctionReturn(error); }
    catch (const std::bad_alloc&) { SETERRQ(PETSC_COMM_SELF,PETSC_ERR_MEM,"Coupling initialization ran out of memory"); }
    catch (const std::exception& e) { SETERRQ(PETSC_COMM_SELF,PETSC_ERR_USER,"%s",e.what()); }
    catch (...) { SETERRQ(PETSC_COMM_SELF,PETSC_ERR_USER,"Unknown coupling initialization exception"); }
}
PetscErrorCode Agree(MPI_Comm comm, PetscErrorCode error)
{
    PetscFunctionBeginUser;
    const int local=static_cast<int>(error); int global=0;
    PetscCallMPI(MPI_Allreduce(&local,&global,1,MPI_INT,MPI_MAX,comm));
    PetscCheck(!global,comm,static_cast<PetscErrorCode>(global),"Coupling initialization failed on at least one rank");
    PetscFunctionReturn(PETSC_SUCCESS);
}
} // namespace

Configuration MakeInitializationConfiguration(const input::InputConfig& c)
{
    ValidateStage(c);
    Configuration result; result.input=c;
    PhaseConfiguration(c,result); TransportConfiguration(c,result);
    result.flow=FlowConfiguration(c);
    return result;
}
input::ReadInputOptions InitializationReaderOptions()
{
    input::ReadInputOptions result;
    // Module validators are deliberately pure/noncollective. The full build
    // checks cross-module references while core::ReadInput coordinates failures.
    const auto validate=[](const Value&, const input::InputConfig& c) { (void)MakeInitializationConfiguration(c); };
    result.extensions.modules["phase"]=validate;
    result.extensions.modules["transport"]=validate;
    for (const auto* name : {"affine","gaussian","erf_ramp"})
        result.extensions.functions[name]=[](const input::FunctionInput& f,int components) {
            if (components!=1) f.parameters.Fail("this profile is scalar");
            (void)Profile(f);
        };
    return result;
}
PetscErrorCode ReadInitializationInput(MPI_Comm comm, const std::string& filename, Configuration& output)
{
    PetscFunctionBeginUser;
    input::InputConfig common;
    PetscCall(input::ReadInput(comm,filename,common,InitializationReaderOptions()));
    Configuration candidate;
    PetscCall(Agree(comm,Local([&]() { candidate=MakeInitializationConfiguration(common); return PETSC_SUCCESS; })));
    output=std::move(candidate);
    PetscFunctionReturn(PETSC_SUCCESS);
}
PetscErrorCode EvaluateAdvectionBoundary(
    const std::vector<BoundaryRule<AdvectionCondition>>& rules, const BoundaryPoint& point,
    PetscReal normalSpeed, AdvectiveBoundaryValue& value)
{
    PetscFunctionBeginUser;
    PetscCall(Local([&]() {
        const auto& condition=Select(rules,point);
        const auto speed=Finite(normalSpeed);
        AdvectiveBoundaryValue candidate;
        if (condition.policy==AdvectionPolicy::ZeroFlux || speed==0) candidate.type=AdvectiveBoundaryType::ZeroFlux;
        else if (speed<0) {
            Require(condition.policy==AdvectionPolicy::InflowOutflow,"Backflow encountered on an outflow-only boundary.");
            Require(static_cast<bool>(condition.inflow),"Missing inflow profile.");
            candidate.type=AdvectiveBoundaryType::PrescribedState;
            candidate.value=Finite(condition.inflow(point.position,point.time));
        } else candidate.type=AdvectiveBoundaryType::ExtrapolatedState;
        value=candidate;
        return PETSC_SUCCESS;
    }));
    PetscFunctionReturn(PETSC_SUCCESS);
}
PetscErrorCode EvaluateThermalBoundary(const std::vector<BoundaryRule<ThermalCondition>>& rules,
                                     const BoundaryPoint& point, DiffusiveBoundaryValue& value)
{
    PetscFunctionBeginUser;
    PetscCall(Local([&]() {
        const auto& condition=Select(rules,point);
        DiffusiveBoundaryValue candidate;
        if (condition.policy!=ThermalPolicy::ZeroFlux) {
            Require(static_cast<bool>(condition.value),"Missing thermal boundary profile.");
            candidate.type=condition.policy==ThermalPolicy::Temperature ? DiffusiveBoundaryType::PrescribedQuantity : DiffusiveBoundaryType::PrescribedFlux;
            candidate.value=Finite(condition.value(point.position,point.time));
        }
        value=candidate;
        return PETSC_SUCCESS;
    }));
    PetscFunctionReturn(PETSC_SUCCESS);
}

namespace {
PetscErrorCode BuildInitialState(MPI_Comm comm, const Configuration& c, InitialState& s, bool solveFlow)
{
    PetscFunctionBeginUser;
    const auto& m=c.input.mesh; const auto& grid=c.input.parallel.processGrid;
    const auto nx=static_cast<PetscInt>(m.cells[0]), ny=static_cast<PetscInt>(m.cells[1]);
    PetscMPIInt size=0; PetscCallMPI(MPI_Comm_size(comm,&size));
    PetscCheck(size==c.input.parallel.ranks,comm,PETSC_ERR_ARG_SIZ,"MPI size differs from parallel.ranks");
    PetscCall(DMDACreate2d(comm,DM_BOUNDARY_NONE,DM_BOUNDARY_NONE,DMDA_STENCIL_BOX,
        nx+1,ny+1,grid[0],grid[1],2,1,nullptr,nullptr,&s.vertexDM));
    PetscCall(DMSetUp(s.vertexDM));
    PetscCall(DMCreateGlobalVector(s.vertexDM,&s.vertices));
    MeshParam parameters;
    parameters.xstart=m.x[0]; parameters.ystart=m.y[0];
    parameters.L=m.x[1]-m.x[0]; parameters.H=m.y[1]-m.y[0];
    parameters.perturbation=m.perturbation; parameters.seed=m.seed;
    if (m.family=="rectangular") PetscCall(CreateFullMesh(s.vertexDM,s.vertices,parameters));
    else PetscCall(LogicRectMesh(s.vertexDM,s.vertices,parameters));
    PetscCall(BuildMeshInfo(s.vertexDM,s.vertices,s.mesh));
    PetscCall(DMDACreate2d(comm,DM_BOUNDARY_NONE,DM_BOUNDARY_NONE,DMDA_STENCIL_BOX,
        nx,ny,grid[0],grid[1],1,1,nullptr,nullptr,&s.cellDM));
    PetscCall(DMSetUp(s.cellDM));
    s.time=Finite(c.input.time.start);
    PetscCall(DMCreateGlobalVector(s.cellDM,&s.enthalpy));
    PetscCall(VecDuplicate(s.enthalpy,&s.composition));
    PetscCall(VecDuplicate(s.enthalpy,&s.temperature));
    PetscCall(VecDuplicate(s.enthalpy,&s.porosity));
    PetscCall(PetscObjectSetName(reinterpret_cast<PetscObject>(s.enthalpy),"enthalpy"));
    PetscCall(PetscObjectSetName(reinterpret_cast<PetscObject>(s.composition),"composition"));
    PetscCall(PetscObjectSetName(reinterpret_cast<PetscObject>(s.temperature),"temperature"));
    PetscCall(PetscObjectSetName(reinterpret_cast<PetscObject>(s.porosity),"porosity"));
    const auto time=s.time;
    const auto H=c.initialEnthalpy, C=c.initialComposition, pressure=c.pressure;
    const auto model=c.phase;
    const auto points=c.input.quadrature.cellPointsPerAxis;
    PetscCall(InitializeCellAverages(s.cellDM,s.enthalpy,s.mesh,
        [H,time](const Point& p) { return H(p,time); },points));
    PetscCall(InitializeCellAverages(s.cellDM,s.composition,s.mesh,
        [C,time](const Point& p) { return C(p,time); },points));
    const auto phase=[=](const Point& p) { return model.evaluate(H(p,time),C(p,time),pressure(p,time)); };
    PetscCall(InitializeCellAverages(s.cellDM,s.temperature,s.mesh,
        [phase](const Point& p) { return Finite(phase(p).TDp); },points));
    PetscCall(InitializeCellAverages(s.cellDM,s.porosity,s.mesh,
        [phase](const Point& p) { return Finite(phase(p).phil); },points));
    if (solveFlow) PetscCall(InitializeFlow(comm,c,s));
    PetscFunctionReturn(PETSC_SUCCESS);
}
} // namespace

PetscErrorCode DestroyInitialState(InitialState& s)
{
    PetscFunctionBeginUser;
    PetscErrorCode first=PETSC_SUCCESS;
    const auto record=[&](PetscErrorCode e) { if (!first) first=e; };
    record(DestroyLinearSystem(s.flowSystem));
    s.stokesVelocityMap=DofMap{}; s.darcyVelocityMap=DofMap{}; s.pressureMap=DofMap{};
    s.flowReport=LinearSolveReport{}; s.flowPorosity.clear();
    record(VecDestroy(&s.porosity)); record(VecDestroy(&s.temperature));
    record(VecDestroy(&s.composition)); record(VecDestroy(&s.enthalpy));
    record(VecDestroy(&s.vertices)); record(DMDestroy(&s.cellDM)); record(DMDestroy(&s.vertexDM));
    s.mesh=MeshInfo{}; s.time=0; s.acceptedSteps=0; s.phaseCoupled=false; s.sourceState.clear(); s.sourceTime=0;
    PetscFunctionReturn(first);
}
PetscErrorCode Initialize(MPI_Comm comm, const Configuration& c, InitialState& result, bool solveFlow)
{
    PetscFunctionBeginUser;
    PetscCall(Agree(comm,result.IsEmpty() ? PETSC_SUCCESS : PETSC_ERR_ARG_WRONGSTATE));
    // Revalidate common input before allocating collective objects. Configuration
    // callbacks/options should be those produced by MakeInitializationConfiguration.
    PetscCall(Agree(comm,Local([&]() { ValidateStage(c.input); return PETSC_SUCCESS; })));
    InitialState temporary;
    const auto error=BuildInitialState(comm,c,temporary,solveFlow);
    if (error) { (void)DestroyInitialState(temporary); PetscFunctionReturn(error); }
    std::swap(result.vertexDM,temporary.vertexDM); std::swap(result.cellDM,temporary.cellDM);
    std::swap(result.vertices,temporary.vertices); std::swap(result.enthalpy,temporary.enthalpy);
    std::swap(result.composition,temporary.composition); std::swap(result.temperature,temporary.temperature);
    std::swap(result.porosity,temporary.porosity);
    result.mesh=std::move(temporary.mesh); result.time=temporary.time; result.acceptedSteps=temporary.acceptedSteps;
    result.stokesVelocityMap=std::move(temporary.stokesVelocityMap);
    result.darcyVelocityMap=std::move(temporary.darcyVelocityMap);
    result.pressureMap=std::move(temporary.pressureMap);
    result.flowPorosity=std::move(temporary.flowPorosity);
    result.flowReport=std::move(temporary.flowReport);
    auto& out=result.flowSystem; auto& in=temporary.flowSystem;
    std::swap(out.matrix,in.matrix); std::swap(out.rhs,in.rhs);
    std::swap(out.solution,in.solution); std::swap(out.pressureNullspace,in.pressureNullspace);
    for (int k=0;k<4;++k) std::swap(out.coupledFieldIS[k],in.coupledFieldIS[k]);
    out.kind=in.kind; out.removedRhsComponent=in.removedRhsComponent;
    PetscFunctionReturn(PETSC_SUCCESS);
}

namespace {
// Bring field values to geometry owners without gathering a global vector.
PetscErrorCode OwnedValues(MPI_Comm comm, const InitialState& s, std::array<std::vector<PetscReal>,4>& values)
{
    PetscFunctionBeginUser;
    const auto range=s.mesh.OwnedCells(); std::vector<PetscInt> ids;
    PetscCall(Agree(comm,Local([&]() {
        for (PetscInt j=range.begin.j;j<range.end.j;++j)
            for (PetscInt i=range.begin.i;i<range.end.i;++i) {
                PetscInt id=0; const auto error=s.mesh.CellId({i,j},id); if (error) return error;
                ids.push_back(id);
            }
        for (auto& v : values) v.resize(ids.size());
        return PETSC_SUCCESS;
    })));
    AO ao=nullptr; PetscCall(DMDAGetAO(s.cellDM,&ao));
    PetscCall(AOApplicationToPetsc(ao,static_cast<PetscInt>(ids.size()),ids.data()));
    IS selected=nullptr; Vec local=nullptr; VecScatter scatter=nullptr;
    // Keep cleanup outside the operation, including on a PETSc error.
    const auto operation=[&]() -> PetscErrorCode {
        PetscFunctionBeginUser;
        PetscCall(ISCreateGeneral(comm,static_cast<PetscInt>(ids.size()),ids.data(),PETSC_COPY_VALUES,&selected));
        PetscCall(VecCreateSeq(PETSC_COMM_SELF,static_cast<PetscInt>(ids.size()),&local));
        PetscCall(VecScatterCreate(s.enthalpy,selected,local,nullptr,&scatter));
        const Vec fields[4]={s.enthalpy,s.composition,s.temperature,s.porosity};
        for (int k=0;k<4;++k) {
            PetscCall(VecScatterBegin(scatter,fields[k],local,INSERT_VALUES,SCATTER_FORWARD));
            PetscCall(VecScatterEnd(scatter,fields[k],local,INSERT_VALUES,SCATTER_FORWARD));
            const PetscScalar* data=nullptr; PetscCall(VecGetArrayRead(local,&data));
            for (std::size_t i=0;i<ids.size();++i) values[k][i]=PetscRealPart(data[i]);
            PetscCall(VecRestoreArrayRead(local,&data));
        }
        PetscFunctionReturn(PETSC_SUCCESS);
    };
    const auto error=operation();
    const auto e1=VecScatterDestroy(&scatter), e2=VecDestroy(&local), e3=ISDestroy(&selected);
    PetscCall(error); PetscCall(e1); PetscCall(e2); PetscCall(e3);
    PetscFunctionReturn(PETSC_SUCCESS);
}
void CheckStream(const std::ofstream& file, const std::string& name)
{ Require(static_cast<bool>(file),"Could not write "+name); }
// Legacy ReadVectorTransport consumes x-fast rows of nondimensional cell
// averages. PETSc's distributed ordering is generally NOT that row ordering.
PetscErrorCode WriteTransportFields(MPI_Comm comm, const InitialState& s,
                                   const std::string& directory, PetscMPIInt rank)
{
    PetscFunctionBeginUser;
    Vec natural=nullptr, gathered=nullptr;
    VecScatter scatter=nullptr;
    const auto operation=[&]() -> PetscErrorCode {
        PetscFunctionBeginUser;
        PetscCall(DMDACreateNaturalVector(s.cellDM,&natural));
        PetscCall(VecScatterCreateToZero(natural,&scatter,&gathered));
        const Vec fields[2]={s.enthalpy,s.composition};
        const char* names[2]={"cellH1.dat","cellC1.dat"};
        const auto shape=s.mesh.CellDimensions();
        for (int k=0;k<2;++k) {
            PetscCall(DMDAGlobalToNaturalBegin(s.cellDM,fields[k],INSERT_VALUES,natural));
            PetscCall(DMDAGlobalToNaturalEnd(s.cellDM,fields[k],INSERT_VALUES,natural));
            PetscCall(VecScatterBegin(scatter,natural,gathered,INSERT_VALUES,SCATTER_FORWARD));
            PetscCall(VecScatterEnd(scatter,natural,gathered,INSERT_VALUES,SCATTER_FORWARD));
            PetscCall(Agree(comm,Local([&]() -> PetscErrorCode {
                if (rank) return PETSC_SUCCESS;
                const auto name=directory+"/"+names[k], temporary=name+".tmp";
                std::ofstream file(temporary); CheckStream(file,temporary);
                file<<std::setprecision(std::numeric_limits<PetscReal>::max_digits10);
                const PetscScalar* data=nullptr;
                PetscCall(VecGetArrayRead(gathered,&data));
                for (PetscInt j=0;j<shape.j;++j) {
                    for (PetscInt i=0;i<shape.i;++i) {
                        if (i) file<<' ';
                        file<<PetscRealPart(data[j*shape.i+i]);
                    }
                    file<<'\n';
                }
                PetscCall(VecRestoreArrayRead(gathered,&data));
                file.close(); CheckStream(file,temporary);
                std::filesystem::rename(temporary,name);
                return PETSC_SUCCESS;
            })));
        }
        PetscFunctionReturn(PETSC_SUCCESS);
    };
    const auto error=operation();
    const auto e1=VecScatterDestroy(&scatter), e2=VecDestroy(&gathered), e3=VecDestroy(&natural);
    PetscCall(error); PetscCall(e1); PetscCall(e2); PetscCall(e3);
    PetscFunctionReturn(PETSC_SUCCESS);
}
PetscErrorCode WriteFlowDofs(MPI_Comm comm, const InitialState& s, const std::string& directory, PetscMPIInt rank)
{
    PetscFunctionBeginUser;
    Vec fields[4]{};
    PetscCall(GetCoupledLinearSystemSolution(s.flowSystem,fields[0],fields[1],fields[2],fields[3]));
    const DofMap* maps[4]={&s.stokesVelocityMap,&s.pressureMap,&s.darcyVelocityMap,&s.pressureMap};
    const char* names[4]={"stokes_velocity","stokes_pressure","darcy_velocity","darcy_pressure"};
    std::array<std::vector<PetscInt>,4> ids;
    std::array<std::vector<PetscReal>,4> values;
    PetscCall(Agree(comm,Local([&]() -> PetscErrorCode {
        for (int k=0;k<4;++k) {
            const auto& map=*maps[k]; ids[k].resize(map.OwnedDofs()); values[k].resize(map.OwnedDofs());
            for (PetscInt i=0;i<map.OwnedDofs();++i) {
                const auto e=map.GlobalToNatural(i+map.OwnershipBegin(),ids[k][i]); if (e) return e;
            }
            const PetscScalar* data=nullptr; auto e=VecGetArrayRead(fields[k],&data); if (e) return e;
            for (PetscInt i=0;i<map.OwnedDofs();++i) values[k][i]=PetscRealPart(data[i]);
            e=VecRestoreArrayRead(fields[k],&data); if (e) return e;
        }
        std::ostringstream name; name<<directory<<"/flow_dofs_rank_"<<std::setw(6)<<std::setfill('0')<<rank<<".csv";
        std::ofstream file(name.str()); CheckStream(file,name.str()); file<<std::setprecision(17);
        file<<"field,natural_dof,value\n";
        for (int k=0;k<4;++k) for (std::size_t i=0;i<ids[k].size();++i)
            file<<names[k]<<','<<ids[k][i]<<','<<values[k][i]<<'\n';
        file.close(); CheckStream(file,name.str());
        return PETSC_SUCCESS;
    })));
    PetscFunctionReturn(PETSC_SUCCESS);
}
} // namespace

PetscErrorCode WriteInitialState(MPI_Comm comm, const Configuration& c, const InitialState& s)
{
    PetscFunctionBeginUser;
    PetscCall(Agree(comm,s.IsEmpty() || !s.flowReport.converged ? PETSC_ERR_ARG_WRONGSTATE : PETSC_SUCCESS));
    PetscMPIInt rank=0; PetscCallMPI(MPI_Comm_rank(comm,&rank));
    std::string directory;
    PetscCall(Agree(comm,Local([&]() {
        directory=std::filesystem::path(input::ResolveOutputPath(
            c.input,"setup.txt",c.input.mesh.family)).parent_path().string();
        if (!rank) {
            std::filesystem::create_directories(directory);
            const auto name=directory+"/transport_state.json";
            std::ofstream file(name);
            file<<"{\"schema_version\":1,\"status\":\"incomplete\"}\n";
            file.close(); CheckStream(file,name);
        }
        return PETSC_SUCCESS;
    })));
    std::array<std::vector<PetscReal>,4> values;
    PetscCall(OwnedValues(comm,s,values));
    PetscCall(WriteFlowDofs(comm,s,directory,rank));
    PetscCall(WriteTransportFields(comm,s,directory,rank));
    GaussRule1D rule; PetscCall(Agree(comm,CreateGaussRule(2,rule)));
    PetscCall(Agree(comm,Local([&]() -> PetscErrorCode {
        std::ostringstream name; name<<directory<<"/initial_rank_"<<std::setw(6)<<std::setfill('0')<<rank<<".csv";
        std::ofstream file(name.str()); CheckStream(file,name.str()); file<<std::setprecision(17);
        file<<"cell_id,i,j,x_centroid,y_centroid,area,H,C,T,phi,"
            <<"x_centroid_m,y_centroid_m,area_m2,h_J_kg,temperature_K\n";
        const auto& scale=c.phase.derived();
        const auto r=s.mesh.OwnedCells(); std::size_t index=0;
        for (PetscInt j=r.begin.j;j<r.end.j;++j) for (PetscInt i=r.begin.i;i<r.end.i;++i,++index) {
            PetscInt id; PetscReal area,x,y; QuadVertices corners;
            // No exceptions are thrown by these simple integrands.
            auto e=s.mesh.CellId({i,j},id); if (e) return e;
            e=s.mesh.GetCellArea({i,j},area); if (e) return e;
            e=s.mesh.GetCellCorners({i,j},corners); if (e) return e;
            e=IntegrateCell(corners,rule,[](const Point& p) { return p.p[0]; },x); if (e) return e;
            e=IntegrateCell(corners,rule,[](const Point& p) { return p.p[1]; },y); if (e) return e;
            file<<id<<','<<i<<','<<j<<','<<x/area<<','<<y/area<<','<<area;
            for (const auto& v : values) file<<','<<v[index];
            file<<','<<x/area*scale.l0<<','<<y/area*scale.l0<<','<<area*scale.l0*scale.l0
                <<','<<values[0][index]*scale.h0<<','<<values[2][index]*scale.dT<<'\n';
        }
        file.close(); CheckStream(file,name.str());
        return PETSC_SUCCESS;
    })));
    PetscReal minimum[4]{},maximum[4]{};
    const Vec fields[4]={s.enthalpy,s.composition,s.temperature,s.porosity};
    for (int k=0;k<4;++k) {
        PetscCall(VecMin(fields[k],nullptr,&minimum[k])); PetscCall(VecMax(fields[k],nullptr,&maximum[k]));
    }
    PetscCall(Agree(comm,Local([&]() {
        if (rank) return PETSC_SUCCESS;
        const auto name=directory+"/setup.txt";
        std::ofstream file(name); CheckStream(file,name); file<<std::setprecision(17);
        file<<(s.phaseCoupled?"Full phase coupling completed; SSPRK2 H/C, phase and current Darcy-Stokes flow.\n":s.acceptedSteps?"Legacy dry preheat completed; initial Darcy-Stokes flow held fixed.\n":
            "Initialization includes an accepted coupled Darcy-Stokes solve; no time step performed.\n")
            <<"simulation: "<<c.input.simulation.name<<"\nsource: "<<c.input.sourceFile
            <<"\nmesh: "<<c.input.mesh.family<<" "<<c.input.mesh.cells[0]<<" x "<<c.input.mesh.cells[1]
            <<"\nranks: "<<c.input.parallel.ranks<<"\ntime: "<<s.time<<"\naccepted steps: "<<s.acceptedSteps
            <<"\nplanned time end / initial step: "<<c.input.time.end<<" / "<<c.input.time.initialStep<<'\n';
        const char* names[4]={"H","C","T","phi"};
        for (int k=0;k<4;++k) file<<names[k]<<" cell-average min/max: "<<minimum[k]<<" "<<maximum[k]<<'\n';
        file<<(s.phaseCoupled?"T,phi: quadrature averages of bounded evolved H/C phase; coupled flow at current time.\n"
            "Thermal model: H_t + div(v*T - L*(1-phi)*us - kappa*grad(T)) = adiabatic source.\n":s.acceptedSteps?"T,phi: diagnostic phase averages from evolved ML-WENO H/C.\n"
            "Legacy thermal model: H_t + div(us*H - kappa*grad(H)) = 0; C and flow fixed.\n":
            "T,phi: averages of phase applied to initial profiles at quadrature points.\n")
            <<"H/C handoff: cellH1.dat and cellC1.dat contain actual Vec cell averages.\n"
            <<"Layout: ny rows of nx values, x fastest, y increasing; no header.\n"
            <<"Units: H nondimensional h/(cp*dT), C unscaled; metadata: transport_state.json.\n"
            <<"Reuse requires matching cell geometry and reference scales; no interpolation is performed.\n"
            <<"Flow porosity: "<<(c.flow.dryPorosity?"prescribed zero (legacy dry preheat)":"equilibrium phase porosity")<<'\n'
            <<"Phase pressure uses Pa; H=h/(cp*dT), T=T_K/dT, x=x_m/l0, C is unscaled.\n";
        c.phase.printInfo(file);
        file<<"flow solver (configured defaults): "<<c.flow.solver.kspType
            <<" / "<<c.input.flow.solver.preconditioner<<"\noptions prefix: "<<c.flow.solver.optionsPrefix
            <<"\nrtol / atol / max_it: "<<c.flow.solver.relativeTolerance<<" / "<<c.flow.solver.absoluteTolerance
            <<" / "<<c.flow.solver.maximumIterations
            <<"\nvelocity block: "<<c.flow.solver.velocity.kspType<<" / "<<c.flow.solver.velocity.pcType
            <<"\npressure block: "<<c.flow.solver.pressure.kspType<<" / "<<c.flow.solver.pressure.pcType
            <<"\npressure nullspace: "<<c.input.flow.pressureNullspace
            <<"\nrequire true residual: "<<std::boolalpha<<c.flow.solver.requireTrueResidual
            <<"\nStokes boundary values/corners checked at initial time.\n"
            <<"Darcy loads assembled with phase porosity and coupled pressure blocks.\n"
            <<"LF mode: "<<(c.stabilizer.mode==LaxFriedrichsMode::Local ? "local" : "global (supplied fixed bound)")
            <<"\nLF global speed: "<<c.stabilizer.globalSpeed
            <<"\nthermal diffusivity: "<<c.thermalDiffusivity
            <<"\ndiffusion samples / extent: "<<c.sampling.numberOfSamples<<" / "<<c.sampling.extentFraction<<'\n';
        file<<"flow solver (effective): "<<s.flowReport.kspType<<" / "<<s.flowReport.pcType
            <<"\nconverged: "<<s.flowReport.converged<<"\niterations: "<<s.flowReport.iterations
            <<"\nKSP reason: "<<static_cast<int>(s.flowReport.reason)
            <<"\ntrue residual: "<<s.flowReport.trueResidualNorm
            <<"\ntrue relative residual: "<<s.flowReport.relativeTrueResidualNorm
            <<"\ntrue residual threshold: "<<s.flowReport.trueResidualThreshold
            <<"\nsubsolvers converged: "<<s.flowReport.subsolversConverged
            <<"\npressure gauge removed: "<<s.flowReport.pressureGaugeRemoved
            <<"\nremoved RHS component norm: "<<s.flowSystem.removedRhsComponent<<'\n';
        for (const auto& block:s.flowReport.subsolvers)
            file<<"subsolver "<<block.optionsPrefix<<": "<<block.solves<<" solves, "
                <<block.totalIterations<<" iterations, "<<block.failures<<" failures\n";
        file.close(); CheckStream(file,name);
        if (c.input.output.saveResolvedInput) {
            const auto yamlName=directory+"/input_used.yaml";
            std::ofstream yaml(yamlName); yaml<<c.input.originalYaml; yaml.close(); CheckStream(yaml,yamlName);
        }
        const auto nameState=directory+"/transport_state.json", temporary=nameState+".tmp";
        std::ofstream metadata(temporary); CheckStream(metadata,temporary);
        const auto& scale=c.phase.derived();
        const auto& mesh=c.input.mesh;
        metadata<<std::setprecision(17)
            <<"{\n\"schema_version\":1,\"status\":\"complete\","
            <<"\"representation\":\"cell averages\",\"ordering\":\"j*nx+i\",\n"
            <<"\"H\":{\"file\":\"cellH1.dat\",\"units\":\"h/(cp*dT)\"},"
            <<"\"C\":{\"file\":\"cellC1.dat\",\"units\":\"fraction\"},\n"
            <<"\"time\":"<<s.time<<",\"time_s\":"<<s.time*scale.t0<<",\n"
            <<"\"accepted_steps\":"<<s.acceptedSteps<<",\n"
            <<"\"mesh\":{\"family\":\""<<mesh.family<<"\",\"nx\":"<<s.mesh.CellDimensions().i
            <<",\"ny\":"<<s.mesh.CellDimensions().j
            <<",\"domain\":["<<mesh.x[0]<<','<<mesh.x[1]<<','<<mesh.y[0]<<','<<mesh.y[1]<<']'
            <<",\"perturbation\":"<<mesh.perturbation<<",\"seed\":"<<mesh.seed<<"},\n"
            <<"\"scales\":{\"enthalpy_Jkg\":"<<scale.h0<<",\"length_m\":"<<scale.l0
            <<",\"time_s\":"<<scale.t0<<"}\n}\n";
        metadata.close(); CheckStream(metadata,temporary);
        std::filesystem::rename(temporary,nameState);
        return PETSC_SUCCESS;
    })));
    PetscCall(PetscPrintf(comm,"%s %s: %" PetscInt_FMT " x %" PetscInt_FMT " cells, t=%g\n"
        "  H=[%g,%g], C=[%g,%g], T=[%g,%g], phi=[%g,%g]\n"
        "  Output: %s\n  H/C handoff: cellH1.dat, cellC1.dat (cell averages; legacy units).\n"
        "  Accepted time steps: %" PetscInt_FMT "\n",
        s.phaseCoupled?"Evolved":s.acceptedSteps?"Preheated":"Initialized",c.input.simulation.name.c_str(),s.mesh.CellDimensions().i,s.mesh.CellDimensions().j,static_cast<double>(s.time),
        static_cast<double>(minimum[0]),static_cast<double>(maximum[0]),static_cast<double>(minimum[1]),static_cast<double>(maximum[1]),
        static_cast<double>(minimum[2]),static_cast<double>(maximum[2]),static_cast<double>(minimum[3]),static_cast<double>(maximum[3]),directory.c_str(),s.acceptedSteps));
    PetscFunctionReturn(PETSC_SUCCESS);
}
} // namespace mantle::couple
