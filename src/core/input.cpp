#include "input.h"

#include <yaml-cpp/yaml.h>

#include <algorithm>
#include <charconv>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <initializer_list>
#include <limits>
#include <set>
#include <sstream>
#include <type_traits>
#include <utility>

namespace mantle::input {
namespace {
constexpr std::size_t MaximumInputBytes = 16 * 1024 * 1024;
constexpr std::size_t MaximumNodes = 100000;
constexpr unsigned MaximumDepth = 64;

std::string Where(const Value& value)
{
    std::string result = value.path.empty() ? "input" : value.path;
    if (value.line > 0)
        result += " (line " + std::to_string(value.line) + ", column "
                  + std::to_string(value.column) + ")";
    return result;
}

Value EmptyMap(const std::string& path)
{
    Value result;
    result.kind = Value::Kind::Mapping;
    result.path = path;
    return result;
}

void Map(const Value& value)
{
    if (value.kind != Value::Kind::Mapping) value.Fail("expected a mapping");
}
void List(const Value& value)
{
    if (value.kind != Value::Kind::Sequence) value.Fail("expected a sequence");
}
void Keys(const Value& value, std::initializer_list<const char*> allowed)
{
    Map(value);
    for (const auto& entry : value.mapping) {
        bool found = false;
        for (const auto* key : allowed) if (entry.first == key) found = true;
        if (!found) entry.second.Fail("unknown input key");
    }
}
void Require(bool condition, const Value& value, const std::string& message)
{
    if (!condition) value.Fail(message);
}
std::string Nonempty(const Value& value)
{
    const auto text = value.AsString();
    Require(!text.empty(), value, "must not be empty");
    Require(text.find('\0') == std::string::npos, value, "contains a NUL character");
    return text;
}
std::string Choice(const Value& value, std::initializer_list<const char*> choices)
{
    const auto text = Nonempty(value);
    for (const auto* choice : choices) if (text == choice) return text;
    std::string message = "expected one of: ";
    bool first = true;
    for (const auto* choice : choices) {
        if (!first) message += ", ";
        message += choice;
        first = false;
    }
    value.Fail(message);
}
bool Boolean(const Value& map, const char* key, bool fallback)
{
    const auto* value = map.Find(key);
    return value ? value->AsBool() : fallback;
}
double Real(const Value& map, const char* key, double fallback)
{
    const auto* value = map.Find(key);
    return value ? value->AsReal() : fallback;
}
std::string Text(const Value& map, const char* key, const std::string& fallback)
{
    const auto* value = map.Find(key);
    return value ? Nonempty(*value) : fallback;
}
int Integer(const Value& value, int minimum = 0)
{
    const auto result = value.AsInteger();
    Require(result >= minimum && result <= std::numeric_limits<int>::max(), value,
            "integer must be in [" + std::to_string(minimum) + ", INT_MAX]");
    return static_cast<int>(result);
}
int Integer(const Value& map, const char* key, int fallback, int minimum = 0)
{
    const auto* value = map.Find(key);
    return value ? Integer(*value, minimum) : fallback;
}
std::int64_t PositiveCount(const Value& value)
{
    const auto result = value.AsInteger();
    Require(result > 0, value, "must be a positive integer");
    return result;
}
std::array<double, 2> Pair(const Value& value)
{
    List(value);
    Require(value.sequence.size() == 2, value, "expected exactly two values");
    return {{value.sequence[0].AsReal(), value.sequence[1].AsReal()}};
}
std::vector<std::string> Strings(const Value& value, bool unique = true)
{
    List(value);
    std::vector<std::string> result;
    std::set<std::string> seen;
    for (const auto& item : value.sequence) {
        const auto text = Nonempty(item);
        if (unique && !seen.insert(text).second) item.Fail("duplicate entry '" + text + "'");
        result.push_back(text);
    }
    return result;
}

Value Import(const YAML::Node& node, const std::string& path,
             unsigned depth, std::size_t& count)
{
    if (depth > MaximumDepth || ++count > MaximumNodes)
        throw InputError(path + ": input nesting/size limit exceeded (possibly cyclic aliases)");
    Value result;
    result.path = path;
    const auto mark = node.Mark();
    if (!mark.is_null()) { result.line = mark.line + 1; result.column = mark.column + 1; }
    const auto tag = node.Tag();
    if (!tag.empty() && tag != "?" && tag != "!" &&
        tag != "tag:yaml.org,2002:str" && tag != "tag:yaml.org,2002:int" &&
        tag != "tag:yaml.org,2002:float" && tag != "tag:yaml.org,2002:bool" &&
        tag != "tag:yaml.org,2002:null" && tag != "tag:yaml.org,2002:map" &&
        tag != "tag:yaml.org,2002:seq") result.Fail("custom YAML tags are unsupported");
    if (!node || node.IsNull()) return result;
    if (node.IsScalar()) {
        result.kind = Value::Kind::Scalar;
        result.scalar = node.Scalar();
    } else if (node.IsSequence()) {
        result.kind = Value::Kind::Sequence;
        for (std::size_t i = 0; i < node.size(); ++i)
            result.sequence.push_back(Import(node[i], path + "[" + std::to_string(i) + "]",
                                             depth + 1, count));
    } else if (node.IsMap()) {
        result.kind = Value::Kind::Mapping;
        for (const auto& item : node) {
            if (!item.first.IsScalar()) result.Fail("mapping keys must be scalar names");
            const std::string key = item.first.Scalar();
            if (key.empty() || key == "<<") result.Fail("empty keys and YAML merge keys are unsupported");
            const auto childPath = path.empty() ? key : path + "." + key;
            if (result.mapping.count(key))
                throw InputError(childPath + ": duplicate YAML key at line "
                                 + std::to_string(item.first.Mark().line + 1));
            result.mapping.emplace(key, Import(item.second, childPath, depth + 1, count));
        }
    } else result.Fail("unsupported YAML node");
    return result;
}

FunctionInput Function(const Value& value)
{
    Keys(value, {"name", "parameters"});
    FunctionInput result;
    result.name = Nonempty(value.At("name"));
    result.parameters = value.Find("parameters") ? value.At("parameters")
        : EmptyMap(value.path + ".parameters");
    Map(result.parameters);
    return result;
}
FunctionInput Constant(double value, bool vector = false)
{
    FunctionInput result;
    result.name = "constant";
    result.parameters = EmptyMap("default.parameters");
    Value v;
    v.kind = Value::Kind::Scalar;
    v.scalar = std::to_string(value);
    v.path = "default.parameters.value";
    if (vector) {
        Value list;
        list.kind = Value::Kind::Sequence;
        list.sequence = {v, v};
        result.parameters.mapping.emplace("value", std::move(list));
    } else result.parameters.mapping.emplace("value", std::move(v));
    return result;
}
void CheckFunction(const FunctionInput& function, int components,
                   const InputExtensions& extensions)
{
    const auto& p = function.parameters;
    if (function.name == "constant") {
        Keys(p, {"value"});
        if (components == 1) (void)p.At("value").AsReal();
        else (void)Pair(p.At("value"));
    } else if (function.name == "quadratic_below_interface") {
        Require(components == 1, p, "quadratic_below_interface is a scalar function");
        Keys(p, {"coefficient", "interface_y", "value_above_interface"});
        (void)p.At("coefficient").AsReal();
        (void)p.At("interface_y").AsReal();
        (void)p.At("value_above_interface").AsReal();
    } else if (function.name == "piecewise_constant") {
        Require(components == 1, p, "piecewise_constant is a scalar function");
        Keys(p, {"interface_y", "value_below", "value_above"});
        (void)p.At("interface_y").AsReal();
        (void)p.At("value_below").AsReal();
        (void)p.At("value_above").AsReal();
    } else {
        const auto it = extensions.functions.find(function.name);
        if (it == extensions.functions.end() || !it->second)
            p.Fail("unregistered function '" + function.name + "'");
        it->second(function, components);
    }
}
BoundaryConditionInput Condition(const Value& value, const InputExtensions& extensions,
                                 bool segment = false)
{
    if (segment) Keys(value, {"region", "priority", "type", "value"});
    else Keys(value, {"type", "value"});
    BoundaryConditionInput result;
    result.type = Choice(value.At("type"), {"dirichlet", "neumann"}) == "dirichlet"
                  ? BoundaryKind::Dirichlet : BoundaryKind::Neumann;
    result.value = Function(value.At("value"));
    CheckFunction(result.value, 1, extensions);
    return result;
}

void ReadMesh(const Value& root, InputConfig& input)
{
    const auto& sim = root.At("simulation");
    Keys(sim, {"name", "mode"});
    input.simulation.name = Nonempty(sim.At("name"));
    input.simulation.mode = Choice(sim.At("mode"), {"steady", "transient"});
    const auto& parallel = root.At("parallel");
    Keys(parallel, {"launcher", "ranks", "process_grid"});
    input.parallel.launcher = Nonempty(parallel.At("launcher"));
    input.parallel.ranks = Integer(parallel.At("ranks"), 1);
    const auto& grid = parallel.At("process_grid");
    List(grid);
    Require(grid.sequence.size() == 2, grid, "expected [px, py]");
    for (int d = 0; d < 2; ++d)
        input.parallel.processGrid[d] = Integer(grid.sequence[d], 1);
    Require(static_cast<std::int64_t>(input.parallel.processGrid[0])
            * input.parallel.processGrid[1] == input.parallel.ranks,
            grid, "px * py must equal parallel.ranks");
    const auto& mesh = root.At("mesh");
    Keys(mesh, {"family", "domain", "cells", "perturbation"});
    input.mesh.family = Choice(mesh.At("family"), {"rectangular", "perturbed_quadrilateral"});
    const auto& domain = mesh.At("domain");
    Keys(domain, {"x", "y"});
    input.mesh.x = Pair(domain.At("x"));
    input.mesh.y = Pair(domain.At("y"));
    for (const auto& p : {input.mesh.x, input.mesh.y})
        Require(p[1] > p[0] && std::isfinite(p[1] - p[0]), domain,
                "domain bounds must be increasing with finite extent");
    const auto& cells = mesh.At("cells");
    List(cells);
    Require(cells.sequence.size() == 2, cells, "expected two CELL counts");
    for (int d = 0; d < 2; ++d) {
        input.mesh.cells[d] = PositiveCount(cells.sequence[d]);
        Require(input.mesh.cells[d] >= input.parallel.processGrid[d], cells,
                "each process-grid direction must have at least one cell per rank");
    }
    if (const auto* p = mesh.Find("perturbation")) {
        Keys(*p, {"amplitude", "seed", "boundary_vertices"});
        input.mesh.perturbation = Real(*p, "amplitude", 0.15);
        if (p->Find("seed")) input.mesh.seed = p->At("seed").AsUnsigned();
        if (p->Find("boundary_vertices"))
            input.mesh.boundaryVertices = Choice(p->At("boundary_vertices"), {"fixed"});
        Require(input.mesh.perturbation >= 0 && input.mesh.perturbation < 0.25,
                *p, "amplitude must satisfy 0 <= amplitude < 0.25");
    }
    if (const auto* q = root.Find("quadrature")) {
        Keys(*q, {"cell_points_per_axis", "edge_points"});
        input.quadrature.cellPointsPerAxis = Integer(*q, "cell_points_per_axis", 5, 1);
        input.quadrature.edgePoints = Integer(*q, "edge_points", 5, 2);
    }
}

void ReadPorosity(const Value& root, InputConfig& input, const InputExtensions& extensions)
{
    const auto& p = root.At("porosity");
    Keys(p, {"source", "prescribed_function", "gauss_point_data", "cell_average"});
    input.porosity.source = Choice(p.At("source"), {"prescribed_function", "gauss_point_data"});
    if (p.Find("cell_average"))
        input.porosity.cellAverage = Choice(p.At("cell_average"), {"physical_quadrature"});
    if (const auto* fn = p.Find("prescribed_function"))
        input.porosity.prescribedFunction = Function(*fn);
    if (const auto* data = p.Find("gauss_point_data")) {
        Keys(*data, {"provider", "hdf5", "runtime_field"});
        if (data->Find("provider"))
            input.porosity.provider = Choice(data->At("provider"), {"hdf5", "runtime_field"});
        if (const auto* hdf5 = data->Find("hdf5")) {
            Keys(*hdf5, {"file", "group", "datasets"});
            input.porosity.file = Text(*hdf5, "file", "");
            input.porosity.group = Text(*hdf5, "group", "/porosity");
            if (const auto* datasets = hdf5->Find("datasets")) {
                Keys(*datasets, {"cell_ids", "cell_values", "edge_ids", "edge_values"});
                for (const auto& entry : datasets->mapping)
                    input.porosity.datasets.emplace(entry.first, Nonempty(entry.second));
            }
        }
        if (const auto* runtime = data->Find("runtime_field")) {
            Keys(*runtime, {"field"});
            input.porosity.runtimeField = Text(*runtime, "field", "");
        }
    }
    if (!input.flow.enabled) return;
    if (input.porosity.source == "prescribed_function") {
        (void)p.At("prescribed_function");
        CheckFunction(input.porosity.prescribedFunction, 1, extensions);
        const auto& fn = input.porosity.prescribedFunction;
        const auto& par = fn.parameters;
        const auto check = [&](double phi) {
            Require(std::isfinite(phi) && phi >= 0 && phi < 1, par,
                    "porosity must satisfy 0 <= phi < 1 throughout the domain");
        };
        if (fn.name == "constant") check(par.At("value").AsReal());
        else if (fn.name == "piecewise_constant") {
            const double y0 = par.At("interface_y").AsReal();
            if (input.mesh.y[0] < y0) check(par.At("value_below").AsReal());
            if (input.mesh.y[1] >= y0) check(par.At("value_above").AsReal());
        } else if (fn.name == "quadratic_below_interface") {
            const double y0 = par.At("interface_y").AsReal();
            const double a = par.At("coefficient").AsReal();
            if (input.mesh.y[0] < y0) {
                Require(a >= 0, par, "porosity coefficient must be nonnegative");
                const double dy = input.mesh.y[0] - y0;
                check(a * dy * dy);
            }
            if (input.mesh.y[1] >= y0) check(par.At("value_above_interface").AsReal());
        }
    } else {
        const auto& data = p.At("gauss_point_data");
        if (input.porosity.provider == "hdf5") {
            Require(!input.porosity.file.empty(), data, "hdf5.file is required");
            Require(!input.porosity.group.empty() && input.porosity.group.front() == '/',
                    data, "HDF5 group must be an absolute dataset-group path");
            Require(input.porosity.datasets.size() == 4, data,
                    "HDF5 input requires cell_ids, cell_values, edge_ids and edge_values");
        } else Require(!input.porosity.runtimeField.empty(), data,
                       "runtime_field.field is required");
    }
}

void ReadRegions(const Value& root, InputConfig& input)
{
    const auto* regions = root.Find("boundary_regions");
    if (!regions) return;
    Keys(*regions, {"physical_endpoint_policy", "coordinate_tolerance", "definitions"});
    if (regions->Find("physical_endpoint_policy"))
        (void)Choice(regions->At("physical_endpoint_policy"), {"require_boundary_vertex"});
    input.boundaryRegions.coordinateTolerance = Real(*regions, "coordinate_tolerance", 1e-12);
    Require(input.boundaryRegions.coordinateTolerance >= 0, *regions,
            "coordinate_tolerance must be nonnegative");
    const auto& definitions = regions->At("definitions");
    Map(definitions);
    for (const auto& entry : definitions.mapping) {
        const auto& v = entry.second;
        Keys(v, {"side", "selector"});
        BoundaryRegionInput region;
        const auto side = Choice(v.At("side"), {"left", "right", "bottom", "top"});
        region.side = side == "left" ? Side::Left : side == "right" ? Side::Right
                      : side == "bottom" ? Side::Bottom : Side::Top;
        const auto& selector = v.At("selector");
        Map(selector);
        const auto mode = Choice(selector.At("mode"),
                                {"whole_side", "physical_interval", "boundary_cells"});
        if (mode == "whole_side") Keys(selector, {"mode"});
        else if (mode == "physical_interval") {
            Keys(selector, {"mode", "coordinate", "interval"});
            region.selection = Selection::PhysicalInterval;
            region.coordinate = Choice(selector.At("coordinate"), {"x", "y"});
            region.interval = Pair(selector.At("interval"));
            Require(region.interval[1] > region.interval[0], selector,
                    "physical interval must have positive length");
            const bool horizontal = region.side == Side::Bottom || region.side == Side::Top;
            Require(region.coordinate == (horizontal ? "x" : "y"), selector,
                    "coordinate must run along the chosen side");
        } else {
            Keys(selector, {"mode", "start", "count", "index_mesh"});
            region.selection = Selection::BoundaryCells;
            region.start = selector.At("start").AsInteger();
            region.count = PositiveCount(selector.At("count"));
            Require(region.start >= 0, selector, "start must be nonnegative");
            if (selector.Find("index_mesh"))
                region.indicesOnBaseMesh = Choice(selector.At("index_mesh"), {"base", "current"}) == "base";
        }
        input.boundaryRegions.definitions.emplace(entry.first, std::move(region));
    }
}

void ReadBoundary(const Value& v, InputConfig& input, const InputExtensions& extensions)
{
    Keys(v, {"value_absolute_tolerance", "value_relative_tolerance", "stokes", "darcy"});
    auto& b = input.flow.boundary;
    b.absoluteTolerance = Real(v, "value_absolute_tolerance", 1e-12);
    b.relativeTolerance = Real(v, "value_relative_tolerance", 1e-10);
    Require(b.absoluteTolerance >= 0 && b.relativeTolerance >= 0, v,
            "boundary value tolerances must be nonnegative");
    if (input.flow.system != "darcy") {
        const auto& stokes = v.At("stokes");
        Keys(stokes, {"components", "default", "segments"});
        if (stokes.Find("components"))
            b.stokesComponents = Choice(stokes.At("components"), {"cartesian", "normal_tangent"});
        const std::array<const char*, 2> names = b.stokesComponents == "cartesian"
            ? std::array<const char*, 2>{{"x", "y"}}
            : std::array<const char*, 2>{{"normal", "tangent"}};
        const auto& defaults = stokes.At("default");
        Keys(defaults, {names[0], names[1]});
        for (int d = 0; d < 2; ++d)
            b.stokesDefault[d] = Condition(defaults.At(names[d]), extensions);
        if (const auto* segments = stokes.Find("segments")) {
            List(*segments);
            for (const auto& segment : segments->sequence) {
                Keys(segment, {"region", "priority", names[0], names[1]});
                StokesSegmentInput item;
                item.region = Nonempty(segment.At("region"));
                item.priority = Integer(segment, "priority", 10, 1);
                for (int d = 0; d < 2; ++d)
                    if (segment.Find(names[d]))
                        item.component[d] = Condition(segment.At(names[d]), extensions);
                Require(item.component[0] || item.component[1], segment,
                        "segment must override at least one component");
                b.stokesSegments.push_back(std::move(item));
            }
        }
    }
    if (input.flow.system != "stokes") {
        const auto& darcy = v.At("darcy");
        Keys(darcy, {"dirichlet_variable", "neumann_variable", "default", "segments"});
        if (darcy.Find("dirichlet_variable"))
            b.darcyDirichletVariable = Choice(darcy.At("dirichlet_variable"), {"assembled_normal_velocity"});
        if (darcy.Find("neumann_variable"))
            b.darcyNeumannVariable = Choice(darcy.At("neumann_variable"),
                                           {"assembled_pressure", "pressure_potential", "weighted_normal_load"});
        b.darcyDefault = Condition(darcy.At("default"), extensions);
        if (const auto* segments = darcy.Find("segments")) {
            List(*segments);
            for (const auto& segment : segments->sequence) {
                DarcySegmentInput item;
                item.region = Nonempty(segment.At("region"));
                item.priority = Integer(segment, "priority", 10, 1);
                item.condition = Condition(segment, extensions, true);
                b.darcySegments.push_back(std::move(item));
            }
        }
    }
}

void Tolerances(const Value& v, double& rtol, double& atol, double& dtol, int& maxit)
{
    rtol = Real(v, "relative_tolerance", rtol);
    atol = Real(v, "absolute_tolerance", atol);
    dtol = Real(v, "divergence_tolerance", dtol);
    maxit = Integer(v, "maximum_iterations", maxit, 1);
    Require(rtol >= 0 && rtol < 1 && atol >= 0 && (rtol > 0 || atol > 0) && dtol > 0,
            v, "require 0 <= rtol < 1, atol >= 0, at least one positive tolerance, dtol > 0");
}
BlockSolverInput BlockSolver(const Value& v, BlockSolverInput out)
{
    Keys(v, {"ksp", "pc", "relative_tolerance", "absolute_tolerance",
             "divergence_tolerance", "maximum_iterations"});
    out.ksp = Text(v, "ksp", out.ksp);
    out.pc = Text(v, "pc", out.pc);
    Tolerances(v, out.relativeTolerance, out.absoluteTolerance, out.divergenceTolerance,
               out.maximumIterations);
    return out;
}
void ReadFlow(const Value& root, InputConfig& input, const InputExtensions& extensions)
{
    const auto* ptr = root.Find("flow");
    if (!ptr) return;
    const auto& v = *ptr;
    Keys(v, {"enabled", "system", "porosity", "parameters", "forcing",
             "boundary_conditions", "linear_system", "solver"});
    auto& f = input.flow;
    f.enabled = Boolean(v, "enabled", false);
    if (!f.enabled) return;
    f.system = Choice(v.At("system"), {"stokes", "darcy", "coupled_stokes_darcy"});
    f.porosity = Choice(v.At("porosity"), {"porosity"});
    const auto& p = v.At("parameters");
    Keys(p, {"formulation", "theta", "darcy_compaction_average_cutoff", "coupling_average_cutoff"});
    f.formulation = Choice(p.At("formulation"), {"legacy_rescaled"});
    f.theta = Real(p, "theta", 0);
    Require(std::isfinite(1 + f.theta), p, "1 + theta must be finite");
    // Whether a negative exponent is permitted depends on the sampled edge
    // porosity. The coefficient adapter/local kernel checks zero edge values.
    f.darcyCompactionAverageCutoff = Real(p, "darcy_compaction_average_cutoff", 1e-15);
    f.couplingAverageCutoff = Real(p, "coupling_average_cutoff", 1e-16);
    Require(f.darcyCompactionAverageCutoff >= 0 && f.darcyCompactionAverageCutoff < 1 &&
            f.couplingAverageCutoff >= 0 && f.couplingAverageCutoff < 1, p,
            "average cutoffs must belong to [0, 1)");
    f.stokesForce = Constant(0, true);
    f.darcyForce = Constant(0, true);
    f.stokesPressureSource = Constant(0);
    f.darcyPressureSource = Constant(0);
    if (const auto* loads = v.Find("forcing")) {
        Keys(*loads, {"stokes", "darcy", "stokes_pressure_source", "darcy_pressure_source"});
        const std::array<const char*, 4> keys{{"stokes", "darcy", "stokes_pressure_source", "darcy_pressure_source"}};
        const std::array<FunctionInput*, 4> fields{{&f.stokesForce, &f.darcyForce,
                                                  &f.stokesPressureSource, &f.darcyPressureSource}};
        for (int i = 0; i < 4; ++i) {
            if (loads->Find(keys[i])) *fields[i] = Function(loads->At(keys[i]));
            CheckFunction(*fields[i], i < 2 ? 2 : 1, extensions);
        }
    }
    ReadBoundary(v.At("boundary_conditions"), input, extensions);
    if (const auto* system = v.Find("linear_system")) {
        Keys(*system, {"sign_convention", "pressure_nullspace", "project_rhs", "pressure_modes"});
        if (system->Find("sign_convention"))
            f.signConvention = Choice(system->At("sign_convention"), {"legacy"});
        if (system->Find("pressure_nullspace"))
            f.pressureNullspace = Choice(system->At("pressure_nullspace"), {"none", "constant", "provided"});
        f.projectRhs = Boolean(*system, "project_rhs", false);
        if (system->Find("pressure_modes")) f.pressureModes = system->At("pressure_modes");
        if (f.pressureNullspace == "provided") {
            Map(f.pressureModes);
            Require(!f.pressureModes.mapping.empty(), *system, "provided nullspace requires pressure_modes");
        }
        Require(f.pressureModes.IsNull() || f.pressureNullspace == "provided", *system,
                "pressure_modes requires pressure_nullspace: provided");
        Require(!f.projectRhs || f.pressureNullspace != "none", *system,
                "project_rhs requires a pressure nullspace");
    }
    f.solver.velocity.pc = "jacobi";
    f.solver.optionsPrefix = f.system == "coupled_stokes_darcy" ? "coupled_" : f.system + "_";
    if (const auto* solver = v.Find("solver")) {
        Keys(*solver, {"options_prefix", "ksp", "preconditioner", "relative_tolerance",
                      "absolute_tolerance", "divergence_tolerance", "maximum_iterations",
                      "initial_guess_nonzero", "velocity", "pressure", "error_if_not_converged",
                      "require_subsolver_convergence", "require_true_residual", "remove_pressure_nullspace"});
        auto& s = f.solver;
        s.optionsPrefix = Text(*solver, "options_prefix", f.system == "coupled_stokes_darcy" ? "coupled_" : f.system + "_");
        Require(s.optionsPrefix.size() <= 64 &&
                s.optionsPrefix.find_first_not_of("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_") == std::string::npos,
                *solver, "options_prefix must have 1-64 ASCII letters, digits or underscores");
        s.ksp = Text(*solver, "ksp", "fgmres");
        if (solver->Find("preconditioner"))
            s.preconditioner = Choice(solver->At("preconditioner"), {"schur", "none"});
        Tolerances(*solver, s.relativeTolerance, s.absoluteTolerance, s.divergenceTolerance, s.maximumIterations);
        s.initialGuessNonzero = Boolean(*solver, "initial_guess_nonzero", false);
        if (solver->Find("velocity")) s.velocity = BlockSolver(solver->At("velocity"), s.velocity);
        if (solver->Find("pressure")) s.pressure = BlockSolver(solver->At("pressure"), s.pressure);
        s.errorIfNotConverged = Boolean(*solver, "error_if_not_converged", true);
        s.requireSubsolverConvergence = Boolean(*solver, "require_subsolver_convergence", true);
        s.requireTrueResidual = Boolean(*solver, "require_true_residual", true);
        s.removePressureNullspace = Boolean(*solver, "remove_pressure_nullspace", false);
        Require(!s.removePressureNullspace || f.pressureNullspace != "none", *solver,
                "remove_pressure_nullspace requires a configured nullspace");
    }
}

} // namespace (parsing helpers)

[[noreturn]] void Value::Fail(const std::string& message) const
{ throw InputError(Where(*this) + ": " + message); }
const Value* Value::Find(const std::string& key) const
{
    Map(*this);
    const auto it = mapping.find(key);
    return it == mapping.end() ? nullptr : &it->second;
}
const Value& Value::At(const std::string& key) const
{
    const auto* value = Find(key);
    if (!value) Fail("missing required key '" + key + "'");
    return *value;
}
std::string Value::AsString() const
{
    if (kind != Kind::Scalar) Fail("expected a scalar");
    return scalar;
}
bool Value::AsBool() const
{
    const auto text = AsString();
    if (text == "true") return true;
    if (text == "false") return false;
    Fail("expected true or false");
}
double Value::AsReal() const
{
    const auto text = AsString();
    double result = 0;
    const char* first = text.data();
    const char* end = first + text.size();
    if (first != end && *first == '+') {
        ++first;
        if (first != end && *first == '-') Fail("expected a finite decimal number");
    }
    const auto parsed = std::from_chars(first, end, result, std::chars_format::general);
    if (parsed.ec != std::errc{} || parsed.ptr != end || !std::isfinite(result))
        Fail("expected a finite decimal number");
    return result;
}
std::int64_t Value::AsInteger() const
{
    const auto text = AsString();
    std::int64_t result = 0;
    const char* first = text.data();
    const char* end = first + text.size();
    if (first != end && *first == '+') {
        ++first;
        if (first != end && *first == '-') Fail("expected an exact signed 64-bit decimal integer");
    }
    const auto parsed = std::from_chars(first, end, result, 10);
    if (parsed.ec != std::errc{} || parsed.ptr != end) Fail("expected an exact signed 64-bit decimal integer");
    return result;
}
std::uint64_t Value::AsUnsigned() const
{
    const auto text = AsString();
    std::uint64_t result = 0;
    const char* first = text.data();
    const char* end = first + text.size();
    if (first != end && *first == '+') ++first;
    const auto parsed = std::from_chars(first, end, result, 10);
    if (parsed.ec != std::errc{} || parsed.ptr != end) Fail("expected an unsigned 64-bit decimal integer");
    return result;
}

} // namespace mantle::input

namespace mantle::input {
namespace {

ModuleInput Module(const Value* value, const std::string& path)
{
    ModuleInput result;
    result.settings = value ? *value : EmptyMap(path);
    Map(result.settings);
    result.enabled = Boolean(result.settings, "enabled", false);
    return result;
}
void ReadModules(const Value& root, InputConfig& input)
{
    input.transport = Module(root.Find("transport"), "transport");
    input.phase = Module(root.Find("phase"), "phase");
    if (const auto* extra = root.Find("extensions")) {
        Map(*extra);
        for (const auto& entry : extra->mapping) {
            Require(entry.first != "flow" && entry.first != "transport" &&
                    entry.first != "phase" && entry.first != "precice", entry.second,
                    "extension name is reserved");
            input.extensions.emplace(entry.first, Module(&entry.second, entry.second.path));
        }
    }
}
void ReadTime(const Value& root, InputConfig& input)
{
    if (const auto* v = root.Find("time")) {
        Keys(*v, {"start", "end", "maximum_steps", "step"});
        auto& t = input.time;
        t.start = Real(*v, "start", 0);
        t.end = Real(*v, "end", 1);
        if (v->Find("maximum_steps")) t.maximumSteps = PositiveCount(v->At("maximum_steps"));
        Require(t.end > t.start && std::isfinite(t.end - t.start), *v,
                "end must exceed start by a finite interval");
        if (const auto* step = v->Find("step")) {
            Keys(*step, {"control", "initial", "minimum", "maximum", "cfl"});
            if (step->Find("control")) t.control = Choice(step->At("control"), {"fixed", "cfl"});
            t.initialStep = Real(*step, "initial", t.initialStep);
            t.minimumStep = Real(*step, "minimum", t.minimumStep);
            t.maximumStep = Real(*step, "maximum", t.maximumStep);
            t.cfl = Real(*step, "cfl", t.cfl);
            Require(t.minimumStep > 0 && t.initialStep >= t.minimumStep &&
                    t.maximumStep >= t.initialStep && t.cfl > 0, *step,
                    "require 0 < minimum <= initial <= maximum and cfl > 0");
        }
    } else if (input.simulation.mode == "transient") {
        root.Fail("transient simulation requires time settings");
    }
}
void ReadCoupling(const Value& root, InputConfig& input)
{
    auto& c = input.coupling;
    if (const auto* v = root.Find("coupling")) {
        Keys(*v, {"backend", "internal", "iteration_checkpoint", "precice"});
        if (v->Find("backend")) c.backend = Choice(v->At("backend"), {"internal", "precice"});
        if (const auto* internal = v->Find("internal")) {
            Keys(*internal, {"scheme", "update_order", "fixed_point"});
            if (internal->Find("scheme"))
                c.internalScheme = Choice(internal->At("scheme"), {"single_pass", "fixed_point"});
            if (internal->Find("update_order")) c.updateOrder = Strings(internal->At("update_order"), false);
            if (const auto* fp = internal->Find("fixed_point")) {
                Keys(*fp, {"maximum_iterations", "relative_tolerance", "absolute_tolerance",
                           "relaxation", "monitored_fields"});
                c.maximumIterations = Integer(*fp, "maximum_iterations", 30, 1);
                c.relativeTolerance = Real(*fp, "relative_tolerance", 1e-8);
                c.absoluteTolerance = Real(*fp, "absolute_tolerance", 1e-12);
                c.relaxation = Real(*fp, "relaxation", 1);
                if (fp->Find("monitored_fields")) c.monitoredFields = Strings(fp->At("monitored_fields"));
                Require(c.relativeTolerance >= 0 && c.relativeTolerance < 1 &&
                        c.absoluteTolerance >= 0 && (c.relativeTolerance > 0 || c.absoluteTolerance > 0)
                        && c.relaxation > 0 && c.relaxation <= 1, *fp,
                        "invalid fixed-point tolerances or relaxation (require 0 < relaxation <= 1)");
            }
        }
        if (const auto* checkpoint = v->Find("iteration_checkpoint")) {
            Keys(*checkpoint, {"storage", "scope"});
            c.checkpointStorage = Choice(checkpoint->At("storage"), {"memory"});
            c.checkpointScope = Choice(checkpoint->At("scope"), {"complete_state"});
        }
        c.precice = v->Find("precice") ? v->At("precice") : EmptyMap("coupling.precice");
        Map(c.precice);
    } else c.precice = EmptyMap("coupling.precice");

    std::set<std::string> enabled;
    if (input.flow.enabled) enabled.insert("flow");
    if (input.transport.enabled) enabled.insert("transport");
    if (input.phase.enabled) enabled.insert("phase");
    for (const auto& module : input.extensions)
        if (module.second.enabled) enabled.insert(module.first);
    if (enabled.empty()) throw InputError("input: at least one physics module must be enabled");
    if (c.backend == "internal") {
        if (c.updateOrder.empty() && enabled.size() == 1) c.updateOrder.push_back(*enabled.begin());
        std::set<std::string> visited;
        for (const auto& name : c.updateOrder) {
            if (!enabled.count(name))
                throw InputError("coupling.internal.update_order: module '" + name + "' is unknown or disabled");
            visited.insert(name);
        }
        if (visited != enabled)
            throw InputError("coupling.internal.update_order: every enabled module must be included");
        if (c.internalScheme == "fixed_point" && c.monitoredFields.empty())
            throw InputError("coupling.internal.fixed_point.monitored_fields: a nonempty list is required");
    }
    if (const auto* v = root.Find("restart")) {
        Keys(*v, {"enabled", "file", "checkpoints"});
        auto& r = input.restart;
        r.enabled = Boolean(*v, "enabled", false);
        if (const auto* file = v->Find("file")) if (!file->IsNull()) r.file = Nonempty(*file);
        Require(!r.enabled || !r.file.empty(), *v, "enabled restart requires file");
        if (const auto* cp = v->Find("checkpoints")) {
            Keys(*cp, {"enabled", "every_accepted_steps", "file"});
            r.writeCheckpoints = Boolean(*cp, "enabled", false);
            if (cp->Find("every_accepted_steps"))
                r.everyAcceptedSteps = PositiveCount(cp->At("every_accepted_steps"));
            r.checkpointFile = Text(*cp, "file", r.checkpointFile);
        }
    }
}

void ReadConvergence(const Value& root, InputConfig& input)
{
    const auto* studies = root.Find("studies");
    if (!studies) return;
    Keys(*studies, {"convergence"});
    const auto* v = studies->Find("convergence");
    if (!v) return;
    Keys(*v, {"enabled", "mesh_families", "levels", "refinement_factor", "h_measure", "reference", "errors"});
    auto& c = input.convergence;
    c.enabled = Boolean(*v, "enabled", false);
    c.levels = Integer(*v, "levels", 1, 1);
    if (v->Find("mesh_families")) c.meshFamilies = Strings(v->At("mesh_families"));
    else c.meshFamilies = {input.mesh.family};
    for (const auto& family : c.meshFamilies)
        Require(family == "rectangular" || family == "perturbed_quadrilateral", *v,
                "unknown convergence mesh family '" + family + "'");
    if (c.enabled) Require(!c.meshFamilies.empty(), *v, "mesh_families must not be empty");
    if (const auto* factor = v->Find("refinement_factor")) {
        List(*factor);
        Require(factor->sequence.size() == 2, *factor, "expected two integer refinement factors");
        for (int d = 0; d < 2; ++d) c.refinementFactor[d] = Integer(factor->sequence[d], 1);
        if (c.enabled) Require(c.refinementFactor[0] > 1 || c.refinementFactor[1] > 1,
                               *factor, "a refinement study must refine at least one direction");
    }
    if (v->Find("h_measure")) c.hMeasure = Choice(v->At("h_measure"), {"maximum_cell_diameter"});
    if (const auto* reference = v->Find("reference")) {
        Map(*reference);
        c.reference = *reference;
        c.referenceSource = Nonempty(reference->At("source"));
        if (c.referenceSource == "finer_mesh") {
            Keys(*reference, {"source", "additional_levels"});
            c.additionalReferenceLevels = Integer(*reference, "additional_levels", 1, 1);
        }
    } else {
        c.reference = EmptyMap("studies.convergence.reference");
        if (c.enabled) v->Fail("enabled convergence requires reference");
    }
    if (const auto* errors = v->Find("errors")) {
        Keys(*errors, {"fields", "norm", "cell_points_per_axis", "evaluation"});
        if (errors->Find("fields")) c.fields = Strings(errors->At("fields"));
        if (errors->Find("norm")) c.norm = Choice(errors->At("norm"), {"l2"});
        c.errorPointsPerAxis = Integer(*errors, "cell_points_per_axis", 7, 1);
        if (errors->Find("evaluation"))
            c.evaluation = Choice(errors->At("evaluation"), {"common_physical_points"});
    }
    if (c.enabled) Require(!c.fields.empty(), *v, "errors.fields must not be empty");
    Require(static_cast<std::int64_t>(c.levels) + c.additionalReferenceLevels <= 64,
            *v, "at most 64 mesh levels, including reference levels, are supported");
}

void ReadOutput(const Value& root, InputConfig& input)
{
    const auto& v = root.At("output");
    Keys(v, {"directory", "fields", "profiles", "convergence_table", "solver_report",
             "diagnostics", "plots", "schedule", "save_resolved_input"});
    auto& o = input.output;
    o.directory = Nonempty(v.At("directory"));
    if (const auto* fields = v.Find("fields")) {
        Keys(*fields, {"format", "file", "xdmf_series", "parallel_hdf5", "quantities"});
        if (fields->Find("format")) o.format = Choice(fields->At("format"), {"hdf5_xdmf"});
        o.fieldFile = Text(*fields, "file", o.fieldFile);
        o.xdmfSeries = Text(*fields, "xdmf_series", o.xdmfSeries);
        o.parallelHdf5 = Boolean(*fields, "parallel_hdf5", true);
        if (fields->Find("quantities")) o.quantities = Strings(fields->At("quantities"));
        Require(!o.quantities.empty(), *fields, "quantities must not be empty");
        Require(input.parallel.ranks == 1 || o.parallelHdf5, *fields,
                "parallel HDF5 must be enabled for multi-rank output");
    }
    if (const auto* profiles = v.Find("profiles")) {
        Keys(*profiles, {"enabled", "file", "start", "end", "points", "fields"});
        o.profilesEnabled = Boolean(*profiles, "enabled", false);
        o.profileFile = Text(*profiles, "file", o.profileFile);
        if (profiles->Find("start")) o.profileStart = Pair(profiles->At("start"));
        if (profiles->Find("end")) o.profileEnd = Pair(profiles->At("end"));
        o.profilePoints = Integer(*profiles, "points", 201, 2);
        if (profiles->Find("fields")) o.profileFields = Strings(profiles->At("fields"));
        if (o.profilesEnabled) {
            Require(profiles->Find("start") && profiles->Find("end") && !o.profileFields.empty(),
                    *profiles, "enabled profiles require start, end and fields");
            Require(o.profileStart != o.profileEnd, *profiles, "profile line must have nonzero length");
            const double tol = input.boundaryRegions.coordinateTolerance;
            for (const auto& point : {o.profileStart, o.profileEnd})
                Require(point[0] >= input.mesh.x[0] - tol && point[0] <= input.mesh.x[1] + tol &&
                        point[1] >= input.mesh.y[0] - tol && point[1] <= input.mesh.y[1] + tol,
                        *profiles, "profile endpoints must belong to the physical domain");
        }
    }
    if (v.Find("convergence_table")) o.convergenceTable = Choice(v.At("convergence_table"), {"csv", "none"});
    if (v.Find("solver_report")) o.solverReport = Choice(v.At("solver_report"), {"csv", "none"});
    if (v.Find("diagnostics")) o.diagnostics = Strings(v.At("diagnostics"));
    if (const auto* plots = v.Find("plots")) {
        Keys(*plots, {"convergence", "profiles", "formats"});
        o.plotConvergence = Boolean(*plots, "convergence", false);
        o.plotProfiles = Boolean(*plots, "profiles", false);
        if (plots->Find("formats")) o.plotFormats = Strings(plots->At("formats"));
        for (const auto& format : o.plotFormats)
            Require(format == "png" || format == "pdf" || format == "svg", *plots,
                    "unsupported plot format '" + format + "'");
        Require(!o.plotConvergence || input.convergence.enabled, *plots,
                "convergence plotting requires studies.convergence.enabled");
        Require(!o.plotProfiles || o.profilesEnabled, *plots,
                "profile plotting requires output.profiles.enabled");
        Require(!(o.plotConvergence || o.plotProfiles) || !o.plotFormats.empty(), *plots,
                "plot formats must not be empty when plots are enabled");
    }
    if (const auto* schedule = v.Find("schedule")) {
        Keys(*schedule, {"steady", "transient", "every_accepted_steps"});
        if (schedule->Find("steady")) o.steadySchedule = Choice(schedule->At("steady"), {"after_solve"});
        if (schedule->Find("transient")) o.transientSchedule = Choice(schedule->At("transient"), {"accepted_steps"});
        if (schedule->Find("every_accepted_steps")) o.everyAcceptedSteps = PositiveCount(schedule->At("every_accepted_steps"));
    }
    o.saveResolvedInput = Boolean(v, "save_resolved_input", true);
}

std::int64_t Multiply(std::int64_t a, std::int64_t b, const std::string& where)
{
    if (a < 0 || b < 0 || (b && a > std::numeric_limits<std::int64_t>::max() / b))
        throw InputError(where + ": integer overflow");
    return a * b;
}
std::int64_t Add(std::int64_t a, std::int64_t b, const std::string& where)
{
    if (a < 0 || b < 0 || a > std::numeric_limits<std::int64_t>::max() - b)
        throw InputError(where + ": integer overflow");
    return a + b;
}

void ValidateRanges(const InputConfig& input)
{
    const int last = input.convergence.enabled ? input.convergence.levels - 1 +
        (input.convergence.referenceSource == "finer_mesh" ? input.convergence.additionalReferenceLevels : 0) : 0;
    // Check actual referenced BCs, including their levels, before a driver
    // allocates matrices. Unused region definitions are validated on level 0.
    for (const auto& entry : input.boundaryRegions.definitions)
        (void)ResolveBoundaryRegion(input, entry.first, 0);
    std::set<std::string> used;
    for (const auto& segment : input.flow.boundary.stokesSegments) used.insert(segment.region);
    for (const auto& segment : input.flow.boundary.darcySegments) used.insert(segment.region);
    const auto physical = [&](const BoundaryEdgeRange& r, int level) {
        const bool horizontal = r.side == Side::Bottom || r.side == Side::Top;
        const auto domain = horizontal ? input.mesh.x : input.mesh.y;
        const auto cells = CellsAtLevel(input, level)[horizontal ? 0 : 1];
        const double dx = (domain[1] - domain[0]) / static_cast<double>(cells);
        return std::array<double, 2>{{domain[0] + dx * static_cast<double>(r.first),
                                       domain[0] + dx * static_cast<double>(r.end)}};
    };
    for (int level = 0; level <= last; ++level) {
        (void)CellsAtLevel(input, level);
        std::map<std::string, BoundaryEdgeRange> ranges;
        for (const auto& name : used) {
            const auto r = ResolveBoundaryRegion(input, name, level);
            ranges.emplace(name, r);
            if (input.convergence.enabled && level > 0) {
                const auto initial = physical(ResolveBoundaryRegion(input, name, 0), 0);
                const auto refined = physical(r, level);
                for (int d = 0; d < 2; ++d)
                    if (std::abs(initial[d] - refined[d]) > input.boundaryRegions.coordinateTolerance)
                        throw InputError("boundary_regions.definitions." + name +
                            ": refinement changes the physical boundary condition; use index_mesh: base for convergence");
            }
        }
        const auto overlaps = [&](const std::string& a, const std::string& b) {
            const auto& ra = ranges.at(a);
            const auto& rb = ranges.at(b);
            return ra.side == rb.side && ra.first < rb.end && rb.first < ra.end;
        };
        const auto& stokes = input.flow.boundary.stokesSegments;
        for (std::size_t i = 0; i < stokes.size(); ++i)
            for (std::size_t j = i + 1; j < stokes.size(); ++j)
                if (stokes[i].priority == stokes[j].priority && overlaps(stokes[i].region, stokes[j].region))
                    for (int d = 0; d < 2; ++d)
                        if (stokes[i].component[d] && stokes[j].component[d])
                            throw InputError("flow.boundary_conditions.stokes.segments: equal-priority overlap between '" +
                                             stokes[i].region + "' and '" + stokes[j].region + "'");
        const auto& darcy = input.flow.boundary.darcySegments;
        for (std::size_t i = 0; i < darcy.size(); ++i)
            for (std::size_t j = i + 1; j < darcy.size(); ++j)
                if (darcy[i].priority == darcy[j].priority && overlaps(darcy[i].region, darcy[j].region))
                    throw InputError("flow.boundary_conditions.darcy.segments: equal-priority overlap between '" +
                                     darcy[i].region + "' and '" + darcy[j].region + "'");
    }
}

void ValidateExtensions(const InputConfig& input, const InputExtensions& extensions)
{
    const auto validate = [&](const std::string& name, const ModuleInput& module) {
        if (!module.enabled) return;
        const auto it = extensions.modules.find(name);
        if (it == extensions.modules.end() || !it->second)
            module.settings.Fail("enabled module '" + name + "' requires a registered module validator");
        it->second(module.settings, input);
    };
    validate("transport", input.transport);
    validate("phase", input.phase);
    for (const auto& entry : input.extensions) validate(entry.first, entry.second);
    if (input.coupling.backend == "precice") {
        if (!extensions.precice)
            input.coupling.precice.Fail("preCICE backend requires a registered adapter validator");
        extensions.precice(input.coupling.precice, input);
    }
    if (input.convergence.enabled && input.convergence.referenceSource != "finer_mesh") {
        const auto it = extensions.references.find(input.convergence.referenceSource);
        if (it == extensions.references.end() || !it->second)
            input.convergence.reference.Fail("unregistered reference provider '" + input.convergence.referenceSource + "'");
        it->second(input.convergence.reference, input);
    }
}

std::string ReadBytes(const std::string& filename)
{
    std::ifstream stream(filename, std::ios::binary);
    if (!stream) throw InputError("cannot open input file '" + filename + "'");
    std::string result;
    char buffer[8192];
    while (stream) {
        stream.read(buffer, sizeof(buffer));
        const auto count = static_cast<std::size_t>(stream.gcount());
        if (result.size() > MaximumInputBytes - count)
            throw InputError("input file exceeds 16 MiB");
        result.append(buffer, count);
    }
    if (!stream.eof()) throw InputError("failed reading input file '" + filename + "'");
    return result;
}

} // namespace (validation helpers)

std::array<std::int64_t, 2> CellsAtLevel(const InputConfig& input, int level)
{
    if (level < 0 || level > 63) throw InputError("mesh refinement level must be in [0,63]");
    auto result = input.mesh.cells;
    for (int d = 0; d < 2; ++d) {
        if (result[d] <= 0 || input.convergence.refinementFactor[d] < 1)
            throw InputError("mesh/refinement counts must be positive");
        for (int l = 0; l < level; ++l)
            result[d] = Multiply(result[d], input.convergence.refinementFactor[d], "mesh refinement");
    }
    // Bound the largest supported scalar index, including full BR/H(div)
    // numbering and the nested coupled global size, for this PETSc build.
    const auto nx = result[0], ny = result[1];
    const auto vx = Add(nx, 1, "mesh vertices"), vy = Add(ny, 1, "mesh vertices");
    const auto vertices = Multiply(vx, vy, "mesh vertices");
    const auto edges = Add(Multiply(nx, vy, "mesh edges"), Multiply(vx, ny, "mesh edges"), "mesh edges");
    const auto cells = Multiply(nx, ny, "mesh cells");
    const auto stokes = Add(Multiply(2, vertices, "Stokes DOFs"), edges, "Stokes DOFs");
    const auto darcy = Multiply(2, edges, "Darcy DOFs");
    auto required = Multiply(2, vertices, "coordinate DOFs");
    if (input.flow.enabled) {
        const auto flowDofs = input.flow.system == "stokes" ? Add(stokes, cells, "Stokes system")
            : input.flow.system == "darcy" ? Add(darcy, cells, "Darcy system")
            : Add(Add(stokes, darcy, "coupled DOFs"), Multiply(2, cells, "pressure DOFs"), "coupled DOFs");
        required = std::max(required, flowDofs);
    }
    if (required > static_cast<std::int64_t>(std::numeric_limits<PetscInt>::max()))
        throw InputError("mesh: requested level exceeds this PETSc build's integer index range");
    return result;
}

BoundaryEdgeRange ResolveBoundaryRegion(const InputConfig& input, const std::string& name, int level)
{
    const auto found = input.boundaryRegions.definitions.find(name);
    if (found == input.boundaryRegions.definitions.end())
        throw InputError("boundary_regions.definitions: unknown region '" + name + "'");
    const auto& region = found->second;
    const std::string where = "boundary_regions.definitions." + name;
    const bool horizontal = region.side == Side::Bottom || region.side == Side::Top;
    const int axis = horizontal ? 0 : 1;
    const auto n = CellsAtLevel(input, level)[axis];
    BoundaryEdgeRange result{region.side, 0, n};
    if (region.selection == Selection::BoundaryCells) {
        if (region.start < 0 || region.count <= 0) throw InputError(where + ": invalid start/count");
        auto start = region.start, count = region.count;
        if (region.indicesOnBaseMesh)
            for (int l = 0; l < level; ++l) {
                start = Multiply(start, input.convergence.refinementFactor[axis], where);
                count = Multiply(count, input.convergence.refinementFactor[axis], where);
            }
        result.first = start;
        result.end = Add(start, count, where);
    } else if (region.selection == Selection::PhysicalInterval) {
        const auto domain = horizontal ? input.mesh.x : input.mesh.y;
        const double tol = input.boundaryRegions.coordinateTolerance;
        const double length = domain[1] - domain[0];
        std::array<std::int64_t, 2> indices{};
        for (int d = 0; d < 2; ++d) {
            const double x = region.interval[d];
            if (x < domain[0] - tol || x > domain[1] + tol)
                throw InputError(where + ": physical interval lies outside the side");
            const long double ratio = (static_cast<long double>(x) - domain[0]) / length;
            const long double index = std::round(ratio * n);
            if (index < 0 || index > n)
                throw InputError(where + ": physical endpoint has no boundary vertex");
            const long double vertex = domain[0] + index * static_cast<long double>(length) / n;
            if (std::abs(vertex - x) > tol)
                throw InputError(where + ": physical endpoint cuts a boundary edge at refinement level "
                                 + std::to_string(level));
            indices[d] = static_cast<std::int64_t>(index);
        }
        result.first = indices[0];
        result.end = indices[1];
    }
    if (result.first < 0 || result.end > n || result.first >= result.end)
        throw InputError(where + ": region is empty or outside the side at refinement level " + std::to_string(level));
    return result;
}

std::string ExpandInputPath(const std::string& pattern, const std::string& meshFamily,
                            int level, std::int64_t step)
{
    if (level < 0 || step < 0) throw InputError("path expansion: level and step must be nonnegative");
    std::string result;
    for (std::size_t i = 0; i < pattern.size();) {
        if (pattern[i] == '}') throw InputError("path expansion: unmatched closing brace");
        if (pattern[i] != '{') { result += pattern[i++]; continue; }
        const auto end = pattern.find('}', i + 1);
        if (end == std::string::npos) throw InputError("path expansion: missing closing brace");
        const auto token = pattern.substr(i + 1, end - i - 1);
        if (token == "mesh_family") result += meshFamily;
        else if (token == "level") result += std::to_string(level);
        else if (token == "step") result += std::to_string(step);
        else throw InputError("path expansion: unknown token '{" + token + "}'");
        i = end + 1;
    }
    if (result.find('\0') != std::string::npos) throw InputError("path contains NUL");
    return result;
}
std::string ResolveInputPath(const InputConfig& input, const std::string& path)
{
    if (path.empty() || path.find('\0') != std::string::npos) throw InputError("input path must be nonempty and contain no NUL");
    const std::filesystem::path p(path);
    return (p.is_absolute() ? p : std::filesystem::path(input.baseDirectory) / p).lexically_normal().string();
}
std::string ResolveOutputPath(const InputConfig& input, const std::string& filename,
                             const std::string& meshFamily, int level, std::int64_t step)
{
    const auto directory = ResolveInputPath(input, ExpandInputPath(input.output.directory, meshFamily, level, step));
    const auto expanded = ExpandInputPath(filename, meshFamily, level, step);
    if (expanded.empty()) throw InputError("output filename must not be empty");
    const std::filesystem::path file(expanded);
    return (file.is_absolute() ? file : std::filesystem::path(directory) / file).lexically_normal().string();
}

InputConfig ParseInput(const std::string& yaml, const std::string& sourceFile,
                       const InputExtensions& extensions)
{
    try {
        if (yaml.size() > MaximumInputBytes) throw InputError("input exceeds 16 MiB");
        if (yaml.find('\0') != std::string::npos) throw InputError("input contains a NUL byte");
        const auto documents = YAML::LoadAll(yaml);
        if (documents.size() != 1) throw InputError("expected exactly one YAML document");
        std::size_t nodes = 0;
        const auto root = Import(documents[0], "", 0, nodes);
        Keys(root, {"schema_version", "simulation", "parallel", "mesh", "quadrature", "porosity",
                    "boundary_regions", "flow", "transport", "phase", "time", "coupling",
                    "restart", "studies", "output", "extensions"});
        InputConfig candidate;
        candidate.schemaVersion = Integer(root.At("schema_version"), 1);
        Require(candidate.schemaVersion == 2, root.At("schema_version"), "this reader supports schema_version: 2");
        if (sourceFile.empty() || sourceFile.find('\0') != std::string::npos)
            throw InputError("source filename must be nonempty and contain no NUL");
        candidate.sourceFile = std::filesystem::absolute(sourceFile).lexically_normal().string();
        candidate.baseDirectory = std::filesystem::path(candidate.sourceFile).parent_path().string();
        candidate.originalYaml = yaml;
        ReadMesh(root, candidate);
        ReadRegions(root, candidate);
        ReadFlow(root, candidate, extensions);
        if (root.Find("porosity")) ReadPorosity(root, candidate, extensions);
        else if (candidate.flow.enabled) root.Fail("enabled flow requires porosity");
        ReadModules(root, candidate);
        ReadTime(root, candidate);
        ReadCoupling(root, candidate);
        ReadConvergence(root, candidate);
        ReadOutput(root, candidate);
        ValidateRanges(candidate);
        ValidateExtensions(candidate, extensions);
        // Validate placeholders without opening data/output files.
        (void)ResolveOutputPath(candidate, candidate.output.fieldFile, candidate.mesh.family);
        (void)ResolveOutputPath(candidate, candidate.output.xdmfSeries, candidate.mesh.family);
        if (candidate.output.profilesEnabled)
            (void)ResolveOutputPath(candidate, candidate.output.profileFile, candidate.mesh.family);
        if (candidate.flow.enabled && candidate.porosity.source == "gauss_point_data"
            && candidate.porosity.provider == "hdf5")
            (void)ResolveInputPath(candidate, ExpandInputPath(candidate.porosity.file, candidate.mesh.family, 0, 0));
        if (candidate.restart.enabled)
            (void)ResolveInputPath(candidate, ExpandInputPath(candidate.restart.file, candidate.mesh.family, 0, 0));
        if (candidate.restart.writeCheckpoints)
            (void)ResolveInputPath(candidate, ExpandInputPath(candidate.restart.checkpointFile, candidate.mesh.family, 0, 0));
        return candidate;
    } catch (const InputError&) { throw; }
    catch (const YAML::Exception& e) { throw InputError(sourceFile + ": YAML error: " + e.what()); }
    catch (const std::exception& e) { throw InputError(sourceFile + ": " + e.what()); }
}

InputConfig ReadInputFile(const std::string& filename, const InputExtensions& extensions)
{
    try {
        if (filename.empty() || filename.find('\0') != std::string::npos)
            throw InputError("input filename must be nonempty and contain no NUL");
        const auto absolute = std::filesystem::absolute(filename).lexically_normal().string();
        return ParseInput(ReadBytes(absolute), absolute, extensions);
    } catch (const InputError&) { throw; }
    catch (const std::exception& e) { throw InputError(filename + ": " + e.what()); }
}

namespace {
// Fixed-size diagnostics avoid allocating after an allocation failure.
struct Failure {
    bool failed = false;
    char message[4096]{};
    void Set(const char* text) noexcept
    {
        failed = true;
        std::snprintf(message, sizeof(message), "%s", text);
    }
};
PetscErrorCode AgreeFailure(MPI_Comm comm, const Failure& failure)
{
    PetscFunctionBegin;
    int rank = 0;
    PetscCallMPI(MPI_Comm_rank(comm, &rank));
    int local = failure.failed ? rank : std::numeric_limits<int>::max();
    int owner = std::numeric_limits<int>::max();
    PetscCallMPI(MPI_Allreduce(&local, &owner, 1, MPI_INT, MPI_MIN, comm));
    if (owner != std::numeric_limits<int>::max()) {
        char message[4096]{};
        if (rank == owner) std::snprintf(message, sizeof(message), "%s", failure.message);
        PetscCallMPI(MPI_Bcast(message, static_cast<int>(sizeof(message)), MPI_CHAR, owner, comm));
        SETERRQ(comm, PETSC_ERR_USER_INPUT, "Input reader (rank %d): %s", owner, message);
    }
    PetscFunctionReturn(PETSC_SUCCESS);
}
PetscErrorCode BroadcastString(MPI_Comm comm, int rank, std::string& value)
{
    PetscFunctionBegin;
    std::uint64_t length = rank == 0 ? static_cast<std::uint64_t>(value.size()) : 0;
    PetscCallMPI(MPI_Bcast(&length, 1, MPI_UINT64_T, 0, comm));
    Failure failure;
    try {
        if (length > MaximumInputBytes || length > static_cast<std::uint64_t>(std::numeric_limits<int>::max()))
            throw InputError("broadcast input/path exceeds size limit");
        value.resize(static_cast<std::size_t>(length));
    } catch (const std::exception& e) { failure.Set(e.what()); }
    catch (...) { failure.Set("failed allocating broadcast input buffer"); }
    PetscCall(AgreeFailure(comm, failure));
    if (length) PetscCallMPI(MPI_Bcast(value.data(), static_cast<int>(length), MPI_CHAR, 0, comm));
    PetscFunctionReturn(PETSC_SUCCESS);
}
} // namespace (collective helpers)

PetscErrorCode ReadInput(MPI_Comm comm, const std::string& filename,
                        InputConfig& output, const ReadInputOptions& options)
{
    PetscFunctionBeginUser;
    int rank = 0, size = 0;
    PetscCallMPI(MPI_Comm_rank(comm, &rank));
    PetscCallMPI(MPI_Comm_size(comm, &size));
    std::string yaml, source;
    Failure readFailure;
    if (rank == 0) {
        try {
            if (filename.empty() || filename.find('\0') != std::string::npos)
                throw InputError("input filename must be nonempty and contain no NUL");
            source = std::filesystem::absolute(filename).lexically_normal().string();
            yaml = ReadBytes(source);
        } catch (const std::exception& e) { readFailure.Set(e.what()); }
        catch (...) { readFailure.Set("unexpected exception while reading input"); }
    }
    PetscCall(AgreeFailure(comm, readFailure));
    PetscCall(BroadcastString(comm, rank, source));
    PetscCall(BroadcastString(comm, rank, yaml));
    std::optional<InputConfig> candidate;
    Failure parseFailure;
    try {
        candidate.emplace(ParseInput(yaml, source, options.extensions));
        if (options.requireMatchingMpiSize && candidate->parallel.ranks != size)
            throw InputError("parallel.ranks=" + std::to_string(candidate->parallel.ranks)
                + " differs from communicator size=" + std::to_string(size)
                + "; launch with the MPICH installation matching PETSc");
    } catch (const std::exception& e) { parseFailure.Set(e.what()); }
    catch (...) { parseFailure.Set("unexpected exception while parsing input"); }
    PetscCall(AgreeFailure(comm, parseFailure));
    static_assert(std::is_nothrow_move_assignable<InputConfig>::value,
                  "collective publication must not throw");
    output = std::move(*candidate);
    PetscFunctionReturn(PETSC_SUCCESS);
}

} // namespace mantle::input
