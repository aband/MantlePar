#include "boundary_conditions.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <exception>
#include <limits>
#include <map>
#include <new>
#include <stdexcept>
#include <utility>

namespace {

PetscErrorCode AgreeError(MPI_Comm comm, PetscErrorCode local, const char* stage)
{
    PetscFunctionBeginUser;
    const int own = static_cast<int>(local);
    int all = 0;
    PetscCallMPI(MPI_Allreduce(&own, &all, 1, MPI_INT, MPI_MAX, comm));
    PetscCheck(!all, comm, static_cast<PetscErrorCode>(all),
               "%s failed on at least one rank; see the originating error", stage);
    PetscFunctionReturn(PETSC_SUCCESS);
}

template<class T>
PetscErrorCode Push(std::vector<T>& values, const T& value)
{
    PetscFunctionBeginUser;
    try { values.push_back(value); }
    catch (const std::bad_alloc&) {
        SETERRQ(PETSC_COMM_SELF, PETSC_ERR_MEM, "Cannot allocate boundary storage");
    } catch (const std::length_error&) {
        SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_SIZ, "Boundary array is too large");
    }
    PetscFunctionReturn(PETSC_SUCCESS);
}

template<class T>
PetscErrorCode Resize(std::vector<T>& values, std::size_t size)
{
    PetscFunctionBeginUser;
    try { values.resize(size); }
    catch (const std::bad_alloc&) {
        SETERRQ(PETSC_COMM_SELF, PETSC_ERR_MEM, "Cannot allocate boundary storage");
    } catch (const std::length_error&) {
        SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_SIZ, "Boundary array is too large");
    }
    PetscFunctionReturn(PETSC_SUCCESS);
}

bool Finite(PetscReal value) { return !PetscIsInfOrNanReal(value); }
PetscReal Dot(const Point& a, const Point& b)
{ return a.p[0]*b.p[0] + a.p[1]*b.p[1]; }

struct Entry { PetscInt id; PetscScalar value; };
struct WorkData {
    std::vector<Entry> constraints;
    std::vector<PetscInt> loadRows;
    std::vector<PetscScalar> loadValues;
};

PetscErrorCode AddConstraint(const DofMap& map, PetscInt id, PetscReal value,
                             WorkData& work)
{
    PetscFunctionBeginUser;
    PetscCheck(Finite(value), PETSC_COMM_SELF, PETSC_ERR_FP,
               "Non-finite prescribed value at DOF %" PetscInt_FMT, id);
    if (map.OwnsGlobal(id)) PetscCall(Push(work.constraints, Entry{id, value}));
    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode AddLoad(PetscInt id, PetscReal value, WorkData& work)
{
    PetscFunctionBeginUser;
    PetscCheck(Finite(value), PETSC_COMM_SELF, PETSC_ERR_FP,
               "Non-finite boundary integral at DOF %" PetscInt_FMT, id);
    PetscCall(Push(work.loadRows, id));
    PetscCall(Push(work.loadValues, PetscScalar(value)));
    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode ResolveConstraints(WorkData& work, PetscReal absolute, PetscReal relative)
{
    PetscFunctionBeginUser;
    std::sort(work.constraints.begin(), work.constraints.end(),
              [](const Entry& a, const Entry& b) { return a.id < b.id; });
    std::size_t count = 0;
    for (const auto& entry : work.constraints) {
        if (count && work.constraints[count-1].id == entry.id) {
            const auto a = work.constraints[count-1].value;
            const PetscReal scale = std::max(PetscAbsScalar(a), PetscAbsScalar(entry.value));
            PetscCheck(PetscAbsScalar(a-entry.value) <= absolute + relative*scale,
                       PETSC_COMM_SELF, PETSC_ERR_ARG_INCOMP,
                       "Conflicting Dirichlet values at vertex/DOF %" PetscInt_FMT
                       "; make incident boundary rules agree at their common endpoint", entry.id);
        } else {
            work.constraints[count++] = entry;
        }
    }
    work.constraints.resize(count);
    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode CheckMap(const MeshInfo& mesh, const DofMap& map, DofSpace space)
{
    PetscFunctionBeginUser;
    PetscCheck(mesh.IsInitialized() && map.IsInitialized(), PETSC_COMM_SELF,
               PETSC_ERR_ARG_WRONG, "Initialize MeshInfo and DofMap first");
    PetscCheck(map.Space() == space, PETSC_COMM_SELF, PETSC_ERR_ARG_INCOMP,
               "Incorrect velocity space for boundary data");
    const PetscInt nv = mesh.VertexCount(), ne = mesh.EdgeCount();
    const PetscInt limit = std::numeric_limits<PetscInt>::max();
    const auto owned = mesh.OwnedVertices().Size();
    PetscInt global = 0, local = 0;
    if (space == DofSpace::BRVelocity) {
        PetscCheck(nv <= (limit-ne)/2, PETSC_COMM_SELF, PETSC_ERR_ARG_SIZ,
                   "BR DOF count overflows PetscInt");
        global = 2*nv + ne;
        local = 2*owned.i*owned.j + static_cast<PetscInt>(mesh.OwnedEdgeIds().size());
    } else {
        PetscCheck(ne <= limit/2, PETSC_COMM_SELF, PETSC_ERR_ARG_SIZ,
                   "Darcy DOF count overflows PetscInt");
        global = 2*ne;
        local = 2*static_cast<PetscInt>(mesh.OwnedEdgeIds().size());
    }
    PetscCheck(map.GlobalDofs() == global && map.OwnedDofs() == local,
               PETSC_COMM_SELF, PETSC_ERR_ARG_SIZ, "Mesh and map DOF counts differ");
    const auto cells = mesh.CellDimensions();
    std::vector<PetscInt> ids;
    PetscCall(map.GetCellNaturalDofs({cells.i-1,cells.j-1}, ids));
    for (PetscInt id = map.OwnershipBegin(); id < map.OwnershipEnd(); ++id) {
        DofInfo info;
        PetscCall(map.GetDofInfo(id, info));
        bool owns = false;
        if (info.entity == DofEntity::Vertex) {
            MeshIndex index;
            PetscCall(mesh.VertexIndex(info.entityId, index));
            owns = mesh.OwnedVertices().Contains(index);
        } else if (info.entity == DofEntity::Edge) owns = mesh.OwnsEdge(info.entityId);
        PetscCheck(owns, PETSC_COMM_SELF, PETSC_ERR_ARG_INCOMP,
                   "Mesh and map partition differ; rebuild the map after repartitioning");
    }
    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode CheckLayout(MPI_Comm comm, const DofMap& map)
{
    PetscFunctionBeginUser;
    PetscMPIInt ranks = 0;
    PetscCallMPI(MPI_Comm_size(comm, &ranks));
    std::vector<PetscInt> all;
    PetscCall(AgreeError(comm, Resize(all, static_cast<std::size_t>(ranks)*3),
                         "Boundary layout allocation"));
    const PetscInt own[3] = {map.OwnershipBegin(), map.OwnershipEnd(), map.GlobalDofs()};
    PetscCallMPI(MPI_Allgather(own, 3, MPIU_INT, all.data(), 3, MPIU_INT, comm));
    PetscInt end = 0;
    for (PetscMPIInt rank = 0; rank < ranks; ++rank) {
        const auto offset = static_cast<std::size_t>(rank)*3;
        PetscCheck(all[offset] == end && all[offset+1] >= end &&
                   all[offset+2] == all[2] && all[offset+1] <= all[2],
                   comm, PETSC_ERR_ARG_INCOMP, "Map communicator/rank ordering differs");
        end = all[offset+1];
    }
    PetscCheck(end == all[2], comm, PETSC_ERR_ARG_SIZ, "Map ownership is incomplete");
    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode CheckInput(const MeshInfo& mesh, const DofMap& map, DofSpace space,
                          const GaussRule1D& rule, PetscReal time, const BoundaryData& result)
{
    PetscFunctionBeginUser;
    PetscCheck(result.IsEmpty(), PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG,
               "Boundary output must be empty; call DestroyBoundaryData first");
    PetscCall(CheckMap(mesh, map, space));
    PetscCall(ValidateGaussRule(rule));
    PetscCheck(rule.points.size() >= 2, PETSC_COMM_SELF, PETSC_ERR_ARG_SIZ,
               "Boundary interpolation/projection needs at least two Gauss points");
    PetscCheck(Finite(time), PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE,
               "Boundary time must be finite");
    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscInt SideCount(const MeshInfo& mesh, CellSide side)
{
    return (side == CellSide::Bottom || side == CellSide::Top)
        ? mesh.CellDimensions().i : mesh.CellDimensions().j;
}

PetscErrorCode CheckRegion(const MeshInfo& mesh, const BoundaryRegion& region)
{
    PetscFunctionBeginUser;
    const int side = static_cast<int>(region.side);
    PetscCheck(side >= 0 && side < 4, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE,
               "Invalid boundary side");
    const auto size = SideCount(mesh, region.side);
    const auto end = region.endEdge == -1 ? size : region.endEdge;
    PetscCheck(region.firstEdge >= 0 && region.firstEdge < end && end <= size,
               PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE,
               "Boundary region must be a nonempty, valid half-open edge interval");
    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode CheckCondition(const BoundaryCondition& condition)
{
    PetscFunctionBeginUser;
    PetscCheck(condition.type == BoundaryType::Unspecified ||
               condition.type == BoundaryType::Dirichlet || condition.type == BoundaryType::Neumann,
               PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Unsupported boundary type");
    if (condition.type != BoundaryType::Unspecified && !condition.value.function)
        PetscCheck(Finite(condition.value.constant), PETSC_COMM_SELF, PETSC_ERR_FP,
                   "Boundary constant must be finite");
    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode CheckSpecification(const MeshInfo& mesh, const StokesBoundarySpecification& spec)
{
    PetscFunctionBeginUser;
    PetscCheck(Finite(spec.absoluteTolerance) && spec.absoluteTolerance >= 0 &&
               Finite(spec.relativeTolerance) && spec.relativeTolerance >= 0,
               PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Invalid corner comparison tolerances");
    for (const auto& rule : spec.rules) {
        PetscCall(CheckRegion(mesh, rule.region));
        for (const auto& condition : rule.component) PetscCall(CheckCondition(condition));
    }
    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode CheckSpecification(const MeshInfo& mesh, const DarcyBoundarySpecification& spec)
{
    PetscFunctionBeginUser;
    PetscCheck(spec.pressureSign == 1 || spec.pressureSign == -1, PETSC_COMM_SELF,
               PETSC_ERR_ARG_OUTOFRANGE, "Darcy pressureSign must be +1 or -1");
    PetscCheck(spec.neumannVariable == DarcyNeumannVariable::AssembledPressure ||
               spec.neumannVariable == DarcyNeumannVariable::PressurePotential ||
               spec.neumannVariable == DarcyNeumannVariable::WeightedNormalLoad,
               PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Invalid Darcy Neumann variable");
    for (const auto& rule : spec.rules) {
        PetscCall(CheckRegion(mesh, rule.region));
        PetscCall(CheckCondition(rule.condition));
    }
    PetscFunctionReturn(PETSC_SUCCESS);
}

struct Edge {
    MeshIndex cell{};
    CellSide side{};
    PetscInt id = 0, index = 0;
    EdgeVertices vertices{};
    Point normal{};
    PetscReal length = 0;
    bool owned = false;
};

// Visit only physical boundary edges of locally available cells. A DOF owner
// can own a top/right edge without owning its incident cell; its ghost geometry
// is still available. Natural contributions below explicitly require owned.
template<class Function>
PetscErrorCode VisitEdges(const MeshInfo& mesh, const Function& function)
{
    PetscFunctionBeginUser;
    const auto range = mesh.AvailableCells();
    const auto cells = mesh.CellDimensions();
    for (int side = 0; side < 4; ++side) {
        const bool horizontal = side == 0 || side == 2;
        const PetscInt fixed = side == 0 || side == 3 ? 0
                              : side == 1 ? cells.i-1 : cells.j-1;
        const PetscInt first = horizontal ? range.begin.i : range.begin.j;
        const PetscInt end = horizontal ? range.end.i : range.end.j;
        for (PetscInt index = first; index < end; ++index) {
            Edge edge;
            edge.cell = horizontal ? MeshIndex{index,fixed} : MeshIndex{fixed,index};
            if (!range.Contains(edge.cell)) continue;
            edge.side = static_cast<CellSide>(side);
            edge.index = index;
            edge.owned = mesh.OwnsCell(edge.cell);
            std::array<OrientedEdge,4> ids;
            PetscCall(mesh.GetCellEdges(edge.cell, ids));
            edge.id = ids[static_cast<std::size_t>(side)].id;
            PetscCall(mesh.GetCellEdgeVertices(edge.cell, edge.side, edge.vertices));
            PetscCall(GetEdgeGeometry(edge.vertices, edge.length, edge.normal));
            PetscCall(function(edge));
        }
    }
    PetscFunctionReturn(PETSC_SUCCESS);
}

bool Matches(const BoundaryRegion& region, const Edge& edge)
{
    return region.side == edge.side && edge.index >= region.firstEdge &&
           (region.endEdge == -1 || edge.index < region.endEdge);
}

template<class Specification, class GetCondition>
PetscErrorCode Select(const Specification& spec, const Edge& edge,
                      const GetCondition& get, const BoundaryCondition*& selected)
{
    PetscFunctionBeginUser;
    selected = nullptr;
    int priority = 0;
    bool ambiguous = false;
    for (const auto& rule : spec.rules) {
        const auto& condition = get(rule);
        if (!Matches(rule.region, edge) || condition.type == BoundaryType::Unspecified) continue;
        if (!selected || rule.region.priority > priority) {
            selected = &condition;
            priority = rule.region.priority;
            ambiguous = false;
        } else if (rule.region.priority == priority) ambiguous = true;
    }
    PetscCheck(selected && !ambiguous, PETSC_COMM_SELF, PETSC_ERR_ARG_INCOMP,
               "Boundary side %d edge %" PetscInt_FMT
               " is unspecified or has equal-priority overlapping rules",
               static_cast<int>(edge.side), edge.index);
    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode Evaluate(const BoundaryValue& value, const Edge& edge,
                        const Point& position, PetscReal time, PetscReal& result)
{
    PetscFunctionBeginUser;
    const BoundaryPoint point{position, edge.normal, edge.side, edge.cell,
                              edge.id, edge.index, time};
    PetscReal evaluated = value.constant;
    try { if (value.function) evaluated = value.function(point); }
    catch (const std::bad_alloc&) {
        SETERRQ(PETSC_COMM_SELF, PETSC_ERR_MEM, "Boundary callback allocation failed");
    } catch (const std::exception& exception) {
        SETERRQ(PETSC_COMM_SELF, PETSC_ERR_USER, "Boundary callback failed: %s", exception.what());
    } catch (...) {
        SETERRQ(PETSC_COMM_SELF, PETSC_ERR_USER, "Boundary callback threw an unknown exception");
    }
    PetscCheck(Finite(evaluated), PETSC_COMM_SELF, PETSC_ERR_FP,
               "Boundary callback returned a non-finite value at side %d edge %" PetscInt_FMT,
               static_cast<int>(edge.side), edge.index);
    result = evaluated;
    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode StokesEdge(const MeshInfo& mesh, const DofMap& map,
                          const GaussRule1D& rule, const StokesBoundarySpecification& spec,
                          PetscReal time, const Edge& edge, WorkData& work)
{
    PetscFunctionBeginUser;
    std::array<const BoundaryCondition*,2> condition{};
    for (int d = 0; d < 2; ++d) {
        const auto get = [d](const StokesBoundaryRule& r) -> const BoundaryCondition&
        { return r.component[static_cast<std::size_t>(d)]; };
        PetscCall(Select(spec, edge, get, condition[static_cast<std::size_t>(d)]));
    }
    BRMixed basis;
    PetscCall(basis.Initialize(mesh, edge.cell));
    std::vector<PetscInt> ids;
    PetscCall(map.GetCellGlobalDofs(edge.cell, ids));
    const auto e = static_cast<std::size_t>(edge.side);
    Point sharedNormal;
    PetscCall(basis.GetEdgeNormal(edge.side, sharedNormal));
    PetscReal endpoint[2][2]{}; // endpoint, Cartesian component.
    std::array<bool,2> essential{};
    for (int d = 0; d < 2; ++d) {
        essential[d] = condition[d]->type == BoundaryType::Dirichlet;
        if (!essential[d]) continue;
        for (std::size_t v = 0; v < 2; ++v) {
            PetscCall(Evaluate(condition[d]->value, edge, edge.vertices[v], time, endpoint[v][d]));
            PetscCall(AddConstraint(map, ids[(e+v)%4 + static_cast<std::size_t>(4*d)],
                                     endpoint[v][d], work));
        }
    }
    PetscReal mean[2]{}, bubbleMean = 0;
    std::array<PetscReal,12> load{};
    for (std::size_t q = 0; q < rule.points.size(); ++q) {
        const auto point = MapEdgePoint(rule.points[q], edge.vertices);
        const PetscReal weight = PetscReal(0.5)*rule.weights[q];
        BRMixed::Values values;
        PetscCall(basis.EvaluateAll(point, values));
        ScalarBasisValue bubble;
        PetscCall(basis.EvaluateBubble(edge.side, point, bubble));
        bubbleMean += weight*bubble.value;
        for (int d = 0; d < 2; ++d) {
            if (!essential[d] && !edge.owned) continue;
            PetscReal value = 0;
            PetscCall(Evaluate(condition[d]->value, edge, point, time, value));
            if (essential[d]) mean[d] += weight*value;
            else for (std::size_t k = 0; k < values.size(); ++k)
                load[k] += edge.length*weight*value*values[k].value.p[d];
        }
    }
    bool constrainBubble = false;
    PetscReal numerator = 0, denominator = bubbleMean;
    if (essential[0] && essential[1]) {
        constrainBubble = true;
        for (int d = 0; d < 2; ++d)
            numerator += sharedNormal.p[d]*(mean[d] - PetscReal(0.5)*(endpoint[0][d]+endpoint[1][d]));
    } else for (int d = 0; d < 2; ++d) {
        // A Cartesian tangential condition on an axis-aligned edge does not
        // constrain the normal bubble. On a slanted edge this one coefficient
        // is fixed by the specified component's moment.
        if (essential[d] && sharedNormal.p[d] != 0) {
            constrainBubble = true;
            numerator = mean[d] - PetscReal(0.5)*(endpoint[0][d]+endpoint[1][d]);
            denominator *= sharedNormal.p[d];
        }
    }
    if (constrainBubble) {
        PetscCheck(Finite(denominator) && denominator != 0, PETSC_COMM_SELF,
                   PETSC_ERR_FP, "Invalid BR boundary bubble normalization");
        PetscCall(AddConstraint(map, ids[8+e], numerator/denominator, work));
    }
    if (edge.owned) for (std::size_t k = 0; k < load.size(); ++k)
        PetscCall(AddLoad(ids[k], load[k], work));
    PetscFunctionReturn(PETSC_SUCCESS);
}

using PorosityCache = std::map<PetscInt, LocalPorositySamples>;

PetscErrorCode GetPorosity(const MeshInfo& mesh, const Edge& edge,
                           const CellPorosityFunction& provider, PorosityCache& cache,
                           const LocalPorositySamples*& samples)
{
    PetscFunctionBeginUser;
    PetscCheck(static_cast<bool>(provider), PETSC_COMM_SELF, PETSC_ERR_ARG_NULL,
               "Weighted Darcy pressure boundary data need a porosity callback");
    PetscInt id = 0;
    PetscCall(mesh.CellId(edge.cell, id));
    auto found = cache.find(id);
    if (found == cache.end()) {
        try { found = cache.emplace(id, LocalPorositySamples{}).first; }
        catch (const std::bad_alloc&) {
            SETERRQ(PETSC_COMM_SELF, PETSC_ERR_MEM, "Cannot allocate boundary porosity cache");
        }
        PetscErrorCode error = PETSC_SUCCESS;
        try { error = provider(edge.cell, found->second); }
        catch (const std::bad_alloc&) {
            SETERRQ(PETSC_COMM_SELF, PETSC_ERR_MEM, "Boundary porosity callback allocation failed");
        } catch (const std::exception& exception) {
            SETERRQ(PETSC_COMM_SELF, PETSC_ERR_USER, "Boundary porosity callback failed: %s", exception.what());
        } catch (...) {
            SETERRQ(PETSC_COMM_SELF, PETSC_ERR_USER, "Boundary porosity callback threw an unknown exception");
        }
        PetscCall(error);
    }
    samples = &found->second;
    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode DarcyWeight(const LocalPorositySamples& samples, std::size_t edge,
                           std::size_t q, std::size_t count, DarcyNeumannVariable variable,
                           const LocalMatrixParameters& parameters, PetscReal& weight)
{
    PetscFunctionBeginUser;
    PetscCheck(samples.edge[edge].size() == count, PETSC_COMM_SELF, PETSC_ERR_ARG_SIZ,
               "Darcy boundary porosity samples do not match edge quadrature");
    const PetscReal phi = samples.edge[edge][q];
    const PetscReal exponent = 1 + parameters.theta;
    PetscCheck(Finite(phi) && phi >= 0 && phi <= 1 && Finite(exponent),
               PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Invalid Darcy edge porosity or theta");
    PetscCheck(phi != 0 || exponent >= 0, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE,
               "Negative porosity power at a dry boundary");
    weight = phi == 0 ? (exponent == 0 ? PetscReal(1) : PetscReal(0)) : std::pow(phi, exponent);
    if (variable == DarcyNeumannVariable::AssembledPressure) {
        const PetscReal average = samples.average;
        PetscCheck(Finite(average) && average >= 0 && average < 1, PETSC_COMM_SELF,
                   PETSC_ERR_ARG_OUTOFRANGE, "Supply the physical cell-average porosity");
        if (average != 0) weight /= PetscSqrtReal(average);
    }
    PetscCheck(Finite(weight), PETSC_COMM_SELF, PETSC_ERR_FP,
               "Darcy natural-boundary weight is not finite");
    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode DarcyEdge(const MeshInfo& mesh, const DofMap& map,
                         const GaussRule1D& rule, const DarcyBoundarySpecification& spec,
                         PetscReal time, const CellPorosityFunction& provider,
                         const LocalMatrixParameters& parameters, const Edge& edge,
                         PorosityCache& cache, WorkData& work)
{
    PetscFunctionBeginUser;
    const BoundaryCondition* condition = nullptr;
    const auto get = [](const DarcyBoundaryRule& r) -> const BoundaryCondition& { return r.condition; };
    PetscCall(Select(spec, edge, get, condition));
    std::vector<PetscInt> ids;
    PetscCall(map.GetCellGlobalDofs(edge.cell, ids));
    const auto e = static_cast<std::size_t>(edge.side);
    const bool essential = condition->type == BoundaryType::Dirichlet;
    if (essential ? (!map.OwnsGlobal(ids[e]) && !map.OwnsGlobal(ids[e+4])) : !edge.owned)
        PetscFunctionReturn(PETSC_SUCCESS);
    HDivMixed basis;
    PetscCall(basis.Initialize(mesh, edge.cell));
    const LocalPorositySamples* samples = nullptr;
    if (!essential && spec.neumannVariable != DarcyNeumannVariable::WeightedNormalLoad)
        PetscCall(GetPorosity(mesh, edge, provider, cache, samples));
    PetscReal gram00 = 0, gram01 = 0, gram11 = 0, rhs0 = 0, rhs1 = 0;
    for (std::size_t q = 0; q < rule.points.size(); ++q) {
        const auto point = MapEdgePoint(rule.points[q], edge.vertices);
        HDivMixed::EdgeValues values;
        PetscCall(basis.EvaluateEdge(point, edge.side, values));
        const PetscReal trace0 = Dot(values[0].value, edge.normal);
        const PetscReal trace1 = Dot(values[1].value, edge.normal);
        PetscReal value = 0;
        PetscCall(Evaluate(condition->value, edge, point, time, value));
        const PetscReal weight = PetscReal(0.5)*rule.weights[q];
        if (essential) {
            // Edge length cancels from the 2x2 projection. Using averages
            // avoids an unnecessarily small determinant on short edges.
            gram00 += weight*trace0*trace0;
            gram01 += weight*trace0*trace1;
            gram11 += weight*trace1*trace1;
        } else {
            PetscReal coefficient = 1;
            if (samples) PetscCall(DarcyWeight(*samples, e, q, rule.points.size(),
                                               spec.neumannVariable, parameters, coefficient));
            value *= spec.pressureSign*edge.length*coefficient;
        }
        rhs0 += weight*value*trace0;
        rhs1 += weight*value*trace1;
    }
    if (essential) {
        const PetscReal determinant = gram00*gram11 - gram01*gram01;
        PetscCheck(Finite(determinant) && gram00 > 0 && gram11 > 0 &&
                   determinant > PetscReal(64)*PETSC_MACHINE_EPSILON*gram00*gram11,
                   PETSC_COMM_SELF, PETSC_ERR_FP, "Singular Darcy boundary projection");
        PetscCall(AddConstraint(map, ids[e], (gram11*rhs0-gram01*rhs1)/determinant, work));
        PetscCall(AddConstraint(map, ids[e+4], (gram00*rhs1-gram01*rhs0)/determinant, work));
    } else {
        PetscCall(AddLoad(ids[e], rhs0, work));
        PetscCall(AddLoad(ids[e+4], rhs1, work));
    }
    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode CreateData(MPI_Comm comm, const DofMap& map, const WorkData& work,
                          BoundaryData& data)
{
    PetscFunctionBeginUser;
    PetscCall(AgreeError(comm, Resize(data.essentialDofs, work.constraints.size()), "Boundary ID allocation"));
    PetscCall(AgreeError(comm, Resize(data.essentialValues, work.constraints.size()), "Boundary value allocation"));
    for (std::size_t i = 0; i < work.constraints.size(); ++i) {
        data.essentialDofs[i] = work.constraints[i].id;
        data.essentialValues[i] = work.constraints[i].value;
    }
    const bool fits = static_cast<std::uintmax_t>(work.loadRows.size()) <=
                      static_cast<std::uintmax_t>(std::numeric_limits<PetscCount>::max());
    PetscCall(AgreeError(comm, fits ? PETSC_SUCCESS : PETSC_ERR_ARG_SIZ, "Boundary COO count"));
    PetscCall(AgreeError(comm, VecCreateMPI(comm, map.OwnedDofs(), map.GlobalDofs(), &data.naturalLoad),
                         "Boundary Vec creation"));
    PetscCall(AgreeError(comm, VecSet(data.naturalLoad, 0), "Initialize boundary load"));
    PetscCall(AgreeError(comm, VecSetPreallocationCOO(data.naturalLoad,
                         static_cast<PetscCount>(work.loadRows.size()), work.loadRows.data()),
                         "Boundary Vec preallocation"));
    PetscCall(AgreeError(comm, VecSetValuesCOO(data.naturalLoad, work.loadValues.data(), INSERT_VALUES),
                         "Boundary load assembly"));
    data.space = map.Space();
    data.globalDofs = map.GlobalDofs();
    data.ownedBegin = map.OwnershipBegin();
    data.ownedEnd = map.OwnershipEnd();
    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode CommitData(MPI_Comm comm, const DofMap& map, const WorkData& work,
                          BoundaryData& result)
{
    PetscFunctionBeginUser;
    BoundaryData data;
    const PetscErrorCode error = CreateData(comm, map, work, data);
    if (error) {
        const PetscErrorCode cleanup = DestroyBoundaryData(data);
        (void)cleanup;
        PetscCall(error);
    }
    result.essentialDofs.swap(data.essentialDofs);
    result.essentialValues.swap(data.essentialValues);
    std::swap(result.naturalLoad, data.naturalLoad);
    result.space = data.space;
    result.globalDofs = data.globalDofs;
    result.ownedBegin = data.ownedBegin;
    result.ownedEnd = data.ownedEnd;
    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode CheckComm(MPI_Comm comm, PetscObject object)
{
    PetscFunctionBeginUser;
    int comparison = MPI_UNEQUAL;
    PetscCallMPI(MPI_Comm_compare(comm, PetscObjectComm(object), &comparison));
    PetscCheck(comparison == MPI_IDENT || comparison == MPI_CONGRUENT,
               PETSC_COMM_SELF, PETSC_ERR_ARG_INCOMP, "Boundary object communicator differs");
    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode CheckVector(MPI_Comm comm, Vec vector, PetscInt size, PetscInt begin, PetscInt end)
{
    PetscFunctionBeginUser;
    PetscCheck(vector, PETSC_COMM_SELF, PETSC_ERR_ARG_NULL, "Missing boundary/system Vec");
    PetscCall(CheckComm(comm, reinterpret_cast<PetscObject>(vector)));
    PetscInt actual = 0, first = 0, last = 0;
    PetscCall(VecGetSize(vector, &actual));
    PetscCall(VecGetOwnershipRange(vector, &first, &last));
    PetscCheck(actual == size && first == begin && last == end,
               PETSC_COMM_SELF, PETSC_ERR_ARG_SIZ, "Boundary/system Vec layout differs");
    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode CheckMatrix(MPI_Comm comm, Mat matrix, PetscInt rows, PetscInt columns,
                           PetscInt rowBegin, PetscInt rowEnd, PetscInt colBegin, PetscInt colEnd)
{
    PetscFunctionBeginUser;
    PetscCheck(matrix, PETSC_COMM_SELF, PETSC_ERR_ARG_NULL, "Missing assembled matrix");
    PetscCall(CheckComm(comm, reinterpret_cast<PetscObject>(matrix)));
    PetscInt m = 0, n = 0, begin = 0, end = 0, cb = 0, ce = 0;
    PetscBool assembled = PETSC_FALSE;
    PetscCall(MatGetSize(matrix, &m, &n));
    PetscCall(MatGetOwnershipRange(matrix, &begin, &end));
    PetscCall(MatGetOwnershipRangeColumn(matrix, &cb, &ce));
    PetscCall(MatAssembled(matrix, &assembled));
    PetscCheck(assembled && m == rows && n == columns && begin == rowBegin && end == rowEnd &&
               cb == colBegin && ce == colEnd, PETSC_COMM_SELF, PETSC_ERR_ARG_SIZ,
               "Boundary/system Mat is unassembled or its layout differs");
    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode CheckApplication(MPI_Comm comm, const MixedBlocks& original,
                               const BoundaryData& data, const BoundaryApplicationOptions& options,
                               const MixedBlocks& result)
{
    PetscFunctionBeginUser;
    PetscCheck(result.IsEmpty(), PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG,
               "Constrained output must be empty; use an independent MixedBlocks");
    PetscCheck(options.pressureRowSign == 1 || options.pressureRowSign == -1,
               PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE,
               "Set pressureRowSign explicitly to the eventual pressure equation's B sign (+1 or -1)");
    PetscCheck(Finite(options.diagonal) && options.diagonal > 0, PETSC_COMM_SELF,
               PETSC_ERR_ARG_OUTOFRANGE, "Boundary diagonal must be finite and positive");
    PetscCheck(data.space == DofSpace::BRVelocity || data.space == DofSpace::HDivVelocity,
               PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Build velocity boundary data first");
    PetscCheck(data.globalDofs > 0 && data.ownedBegin >= 0 && data.ownedBegin <= data.ownedEnd &&
               data.ownedEnd <= data.globalDofs && data.essentialDofs.size() == data.essentialValues.size(),
               PETSC_COMM_SELF, PETSC_ERR_ARG_SIZ, "Invalid boundary data dimensions");
    PetscInt previous = -1;
    for (std::size_t k = 0; k < data.essentialDofs.size(); ++k) {
        const auto id = data.essentialDofs[k];
        PetscCheck(id > previous && id >= data.ownedBegin && id < data.ownedEnd &&
                   Finite(PetscRealPart(data.essentialValues[k])) &&
                   PetscImaginaryPart(data.essentialValues[k]) == 0 &&
                   Finite(options.diagonal*PetscRealPart(data.essentialValues[k])),
                   PETSC_COMM_SELF, PETSC_ERR_ARG_INCOMP, "Invalid, repeated, or nonowned essential DOF/value");
        previous = id;
    }
    PetscCheck(original.B, PETSC_COMM_SELF, PETSC_ERR_ARG_NULL, "Missing assembled B block");
    PetscInt pressures = 0, velocities = 0, first = 0, last = 0;
    PetscCall(MatGetSize(original.B, &pressures, &velocities));
    PetscCall(MatGetOwnershipRange(original.B, &first, &last));
    PetscCheck(velocities == data.globalDofs, PETSC_COMM_SELF, PETSC_ERR_ARG_SIZ,
               "B must have pressure rows and velocity columns");
    PetscCall(CheckMatrix(comm, original.A, velocities, velocities,
                          data.ownedBegin, data.ownedEnd, data.ownedBegin, data.ownedEnd));
    PetscCall(CheckMatrix(comm, original.B, pressures, velocities,
                          first, last, data.ownedBegin, data.ownedEnd));
    PetscCall(CheckMatrix(comm, original.C, pressures, pressures, first, last, first, last));
    PetscCall(CheckVector(comm, original.f, velocities, data.ownedBegin, data.ownedEnd));
    PetscCall(CheckVector(comm, original.g, pressures, first, last));
    PetscCall(CheckVector(comm, data.naturalLoad, velocities, data.ownedBegin, data.ownedEnd));
    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode FillEssential(Vec vector, const BoundaryData& data, bool mask)
{
    PetscFunctionBeginUser;
    PetscScalar* values = nullptr;
    PetscCall(VecGetArray(vector, &values));
    const auto size = data.ownedEnd-data.ownedBegin;
    for (PetscInt k = 0; k < size; ++k) values[k] = mask ? PetscScalar(1) : PetscScalar(0);
    for (std::size_t k = 0; k < data.essentialDofs.size(); ++k)
        values[data.essentialDofs[k]-data.ownedBegin] = mask ? PetscScalar(0) : data.essentialValues[k];
    PetscCall(VecRestoreArray(vector, &values));
    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode ApplyCopy(MPI_Comm comm, const MixedBlocks& original, const BoundaryData& data,
                         const BoundaryApplicationOptions& options, MixedBlocks& work,
                         Vec& prescribed, Vec& mask, Vec& lift)
{
    PetscFunctionBeginUser;
    PetscCall(AgreeError(comm, MatDuplicate(original.A, MAT_COPY_VALUES, &work.A), "Copy A"));
    PetscCall(AgreeError(comm, MatDuplicate(original.B, MAT_COPY_VALUES, &work.B), "Copy B"));
    PetscCall(AgreeError(comm, MatDuplicate(original.C, MAT_COPY_VALUES, &work.C), "Copy C"));
    PetscCall(AgreeError(comm, VecDuplicate(original.f, &work.f), "Create constrained f"));
    PetscCall(AgreeError(comm, VecCopy(original.f, work.f), "Copy f"));
    PetscCall(AgreeError(comm, VecDuplicate(original.g, &work.g), "Create constrained g"));
    PetscCall(AgreeError(comm, VecCopy(original.g, work.g), "Copy g"));
    PetscCall(AgreeError(comm, VecDuplicate(original.f, &prescribed), "Create prescribed velocity"));
    PetscCall(AgreeError(comm, VecDuplicate(original.f, &mask), "Create free-DOF mask"));
    PetscCall(AgreeError(comm, VecDuplicate(original.g, &lift), "Create pressure lift"));
    PetscCall(AgreeError(comm, FillEssential(prescribed, data, false), "Fill prescribed velocity"));
    PetscCall(AgreeError(comm, FillEssential(mask, data, true), "Fill free-DOF mask"));
    PetscCall(AgreeError(comm, VecAXPY(work.f, 1, data.naturalLoad), "Add natural boundary load"));
    PetscCall(AgreeError(comm, MatMult(original.B, prescribed, lift), "Compute pressure-row lift"));
    PetscCall(AgreeError(comm, VecAXPY(work.g, -options.pressureRowSign, lift), "Apply pressure-row lift"));
    PetscCall(AgreeError(comm, MatZeroRowsColumns(work.A,
                         static_cast<PetscInt>(data.essentialDofs.size()), data.essentialDofs.data(),
                         options.diagonal, prescribed, work.f), "Eliminate velocity Dirichlet DOFs"));
    PetscCall(AgreeError(comm, MatDiagonalScale(work.B, nullptr, mask), "Zero constrained B columns"));
    PetscFunctionReturn(PETSC_SUCCESS);
}

} // namespace

PetscErrorCode DestroyBoundaryData(BoundaryData& data)
{
    PetscFunctionBeginUser;
    PetscCall(VecDestroy(&data.naturalLoad));
    data.essentialDofs.clear();
    data.essentialValues.clear();
    data.space = DofSpace::CellPressure;
    data.globalDofs = data.ownedBegin = data.ownedEnd = 0;
    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode BuildStokesBoundaryData(
    MPI_Comm comm, const MeshInfo& mesh, const DofMap& velocityMap,
    const GaussRule1D& edgeRule, const StokesBoundarySpecification& specification,
    PetscReal time, BoundaryData& result)
{
    PetscFunctionBeginUser;
    PetscCall(AgreeError(comm, CheckInput(mesh, velocityMap, DofSpace::BRVelocity,
                                         edgeRule, time, result), "Stokes boundary inputs"));
    PetscCall(AgreeError(comm, CheckSpecification(mesh, specification), "Stokes boundary specification"));
    PetscCall(CheckLayout(comm, velocityMap));
    WorkData work;
    const auto visit = [&](const Edge& edge) -> PetscErrorCode {
        return StokesEdge(mesh, velocityMap, edgeRule, specification, time, edge, work);
    };
    PetscCall(AgreeError(comm, VisitEdges(mesh, visit), "Stokes boundary evaluation"));
    PetscCall(AgreeError(comm, ResolveConstraints(work, specification.absoluteTolerance,
                                                 specification.relativeTolerance), "Stokes corner consistency"));
    PetscCall(CommitData(comm, velocityMap, work, result));
    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode BuildDarcyBoundaryData(
    MPI_Comm comm, const MeshInfo& mesh, const DofMap& velocityMap,
    const GaussRule1D& edgeRule, const DarcyBoundarySpecification& specification,
    PetscReal time, const CellPorosityFunction& porosity,
    const LocalMatrixParameters& parameters, BoundaryData& result)
{
    PetscFunctionBeginUser;
    PetscCall(AgreeError(comm, CheckInput(mesh, velocityMap, DofSpace::HDivVelocity,
                                         edgeRule, time, result), "Darcy boundary inputs"));
    PetscCall(AgreeError(comm, CheckSpecification(mesh, specification), "Darcy boundary specification"));
    PetscCall(CheckLayout(comm, velocityMap));
    WorkData work;
    PorosityCache cache;
    const auto visit = [&](const Edge& edge) -> PetscErrorCode {
        return DarcyEdge(mesh, velocityMap, edgeRule, specification, time, porosity,
                          parameters, edge, cache, work);
    };
    PetscCall(AgreeError(comm, VisitEdges(mesh, visit), "Darcy boundary evaluation"));
    PetscCall(AgreeError(comm, ResolveConstraints(work, 0, 0), "Darcy constraint consistency"));
    PetscCall(CommitData(comm, velocityMap, work, result));
    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode ApplyBoundaryConditions(
    MPI_Comm comm, const MixedBlocks& original, const BoundaryData& data,
    const BoundaryApplicationOptions& options, MixedBlocks& result)
{
    PetscFunctionBeginUser;
    PetscCall(AgreeError(comm, CheckApplication(comm, original, data, options, result),
                         "Boundary application inputs"));
    // All ranks must agree on the algebraic signs and diagonal.
    const PetscReal own[2] = {options.pressureRowSign, options.diagonal};
    PetscReal minimum[2]{}, maximum[2]{};
    PetscCallMPI(MPI_Allreduce(own, minimum, 2, MPIU_REAL, MPI_MIN, comm));
    PetscCallMPI(MPI_Allreduce(own, maximum, 2, MPIU_REAL, MPI_MAX, comm));
    PetscCheck(minimum[0] == maximum[0] && minimum[1] == maximum[1], comm,
               PETSC_ERR_ARG_INCOMP, "Boundary application options differ across ranks");
    MixedBlocks work;
    Vec prescribed = nullptr, mask = nullptr, lift = nullptr;
    const PetscErrorCode error = ApplyCopy(comm, original, data, options, work, prescribed, mask, lift);
    PetscErrorCode cleanup = VecDestroy(&prescribed);
    const PetscErrorCode second = VecDestroy(&mask), third = VecDestroy(&lift);
    if (!cleanup) cleanup = second;
    if (!cleanup) cleanup = third;
    if (error || cleanup) {
        const PetscErrorCode destroy = DestroyMixedBlocks(work);
        (void)destroy;
        PetscCall(error ? error : cleanup);
    }
    std::swap(result.A, work.A);
    std::swap(result.B, work.B);
    std::swap(result.C, work.C);
    std::swap(result.f, work.f);
    std::swap(result.g, work.g);
    PetscFunctionReturn(PETSC_SUCCESS);
}
