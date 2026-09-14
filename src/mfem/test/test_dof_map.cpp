#include "dof_map.h"
#include "brmixed.h"
#include "hdivmixed.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <utility>
#include <vector>

namespace {

// All checks remain active in Release builds. The reference catalogue is built
// from MeshInfo entities/incidence, independently of DofMap's partition blocks.
PetscErrorCode Require(bool condition, const char* message)
{
    PetscFunctionBeginUser;
    PetscCheck(condition, PETSC_COMM_SELF, PETSC_ERR_PLIB, "%s", message);
    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode Equal(PetscInt actual, PetscInt expected, const char* message)
{
    PetscFunctionBeginUser;
    PetscCheck(actual == expected, PETSC_COMM_SELF, PETSC_ERR_PLIB,
               "%s: got %" PetscInt_FMT ", expected %" PetscInt_FMT,
               message, actual, expected);
    PetscFunctionReturn(PETSC_SUCCESS);
}

template<class Function>
PetscErrorCode ExpectError(Function&& function, PetscErrorCode expected)
{
    PetscFunctionBeginUser;
    PetscCall(PetscPushErrorHandler(PetscReturnErrorHandler, nullptr));
    const PetscErrorCode error = function();
    PetscCall(PetscPopErrorHandler());
    PetscCheck(error == expected, PETSC_COMM_SELF, PETSC_ERR_PLIB,
               "Expected error %d, got %d", static_cast<int>(expected), static_cast<int>(error));
    PetscFunctionReturn(PETSC_SUCCESS);
}

bool SameInfo(const DofInfo& a, const DofInfo& b)
{
    return a.entity == b.entity && a.entityId == b.entityId &&
           a.component == b.component && a.boundarySides == b.boundarySides;
}

struct MeshFixture {
    DM dm = nullptr;
    Vec coordinates = nullptr;
    MeshInfo mesh;
    MeshFixture() = default;
    MeshFixture(const MeshFixture&) = delete;
    MeshFixture& operator=(const MeshFixture&) = delete;
    ~MeshFixture()
    {
        if (coordinates) (void)VecDestroy(&coordinates);
        if (dm) (void)DMDestroy(&dm);
    }
    PetscErrorCode Close()
    {
        PetscFunctionBeginUser;
        PetscCall(VecDestroy(&coordinates));
        PetscCall(DMDestroy(&dm));
        mesh = MeshInfo{};
        PetscFunctionReturn(PETSC_SUCCESS);
    }
};

PetscErrorCode SetGeometry(MeshFixture& fixture, PetscInt kind)
{
    PetscFunctionBeginUser;
    MeshParam parameters;
    parameters.xstart = -0.75;
    parameters.ystart = 0.25;
    parameters.L = 2.7;
    parameters.H = 1.3;
    parameters.seed = 79;
    parameters.perturbation = 0.22;
    if (kind == 0) PetscCall(CreateFullMesh(fixture.dm, fixture.coordinates, parameters));
    else if (kind == 1) PetscCall(LogicRectMesh(fixture.dm, fixture.coordinates, parameters));
    else PetscCall(RefineMesh(fixture.dm, fixture.coordinates, parameters));
    PetscCall(BuildMeshInfo(fixture.dm, fixture.coordinates, fixture.mesh));
    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode MakeMesh(PetscInt nx, PetscInt ny, PetscInt px, PetscInt py,
                        PetscInt kind, MeshFixture& fixture)
{
    PetscFunctionBeginUser;
    PetscCall(DMDACreate2d(PETSC_COMM_WORLD, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE,
                          DMDA_STENCIL_BOX, nx, ny, px, py, 2, 1,
                          nullptr, nullptr, &fixture.dm));
    PetscCall(DMSetUp(fixture.dm));
    PetscCall(DMCreateGlobalVector(fixture.dm, &fixture.coordinates));
    PetscCall(SetGeometry(fixture, kind));
    PetscFunctionReturn(PETSC_SUCCESS);
}

struct ReferenceDof {
    DofInfo info;
    bool owned = false;
    bool present = false;
    // Sum of (cell ID + 1) over incident cells, for assembly verification.
    PetscInt assembledLoad = 0;
};

PetscErrorCode CellLoad(const MeshInfo& mesh, MeshIndex cell, PetscInt& load)
{
    PetscFunctionBeginUser;
    PetscInt id;
    PetscCall(mesh.CellId(cell, id));
    load = id + 1;
    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode MakeCatalogue(const MeshInfo& mesh, DofSpace space,
                             std::vector<ReferenceDof>& catalogue)
{
    PetscFunctionBeginUser;
    const auto cells = mesh.CellDimensions();
    catalogue.clear();
    if (space == DofSpace::BRVelocity) {
        for (PetscInt component = 0; component < 2; ++component)
            for (PetscInt v = 0; v < mesh.VertexCount(); ++v) {
                MeshIndex p;
                PetscCall(mesh.VertexIndex(v, p));
                ReferenceDof item;
                item.info.entity = DofEntity::Vertex;
                item.info.entityId = v;
                item.info.component = component;
                item.info.boundarySides = {{p.j == 0, p.i == cells.i,
                                             p.j == cells.j, p.i == 0}};
                item.owned = mesh.OwnedVertices().Contains(p);
                item.present = mesh.GhostVertices().Contains(p);
                for (PetscInt dj : {-1, 0}) for (PetscInt di : {-1, 0}) {
                    const MeshIndex cell{p.i+di, p.j+dj};
                    if (mesh.ContainsCell(cell)) {
                        PetscInt load;
                        PetscCall(CellLoad(mesh, cell, load));
                        item.assembledLoad += load;
                    }
                }
                catalogue.push_back(item);
            }
    }
    if (space != DofSpace::CellPressure) {
        const PetscInt modes = space == DofSpace::HDivVelocity ? 2 : 1;
        for (PetscInt component = 0; component < modes; ++component)
            for (PetscInt e = 0; e < mesh.EdgeCount(); ++e) {
                EdgeTopology edge;
                PetscCall(mesh.GetEdgeTopology(e, edge));
                ReferenceDof item;
                item.info.entity = DofEntity::Edge;
                item.info.entityId = e;
                item.info.component = component;
                const auto start = edge.vertices[0];
                if (edge.axis == EdgeAxis::AlongI) {
                    item.info.boundarySides[0] = start.j == 0;
                    item.info.boundarySides[2] = start.j == cells.j;
                } else {
                    item.info.boundarySides[1] = start.i == cells.i;
                    item.info.boundarySides[3] = start.i == 0;
                }
                item.owned = mesh.OwnsEdge(e);
                item.present = mesh.GhostVertices().Contains(edge.vertices[0]) &&
                               mesh.GhostVertices().Contains(edge.vertices[1]);
                for (const auto& cell : {edge.leftCell, edge.rightCell}) if (cell) {
                    PetscInt load;
                    PetscCall(CellLoad(mesh, *cell, load));
                    item.assembledLoad += load;
                }
                catalogue.push_back(item);
            }
    } else {
        for (PetscInt c = 0; c < mesh.CellCount(); ++c) {
            MeshIndex cell;
            PetscCall(mesh.CellIndex(c, cell));
            ReferenceDof item;
            item.info.entity = DofEntity::Cell;
            item.info.entityId = c;
            item.owned = mesh.OwnsCell(cell);
            item.present = mesh.HasCell(cell);
            item.assembledLoad = c+1;
            // A cell touching the boundary still has an interior P0 DOF.
            catalogue.push_back(item);
        }
    }
    PetscFunctionReturn(PETSC_SUCCESS);
}

struct ReferenceNumbering {
    std::vector<PetscInt> naturalToGlobal, globalToNatural, offsets;
    std::vector<int> owner;
};

PetscErrorCode MakeReference(const std::vector<ReferenceDof>& catalogue,
                             ReferenceNumbering& reference)
{
    PetscFunctionBeginUser;
    PetscMPIInt rank, ranks;
    PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD, &rank));
    PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD, &ranks));
    const auto size = catalogue.size();
    std::vector<int> localOwner(size, 0), localCount(size, 0), count(size);
    reference.owner.resize(size);
    PetscInt owned = 0;
    for (std::size_t n = 0; n < size; ++n) if (catalogue[n].owned) {
        localOwner[n] = rank+1;
        localCount[n] = 1;
        ++owned;
    }
    PetscCallMPI(MPI_Allreduce(localCount.data(), count.data(), static_cast<int>(size),
                               MPI_INT, MPI_SUM, PETSC_COMM_WORLD));
    PetscCallMPI(MPI_Allreduce(localOwner.data(), reference.owner.data(), static_cast<int>(size),
                               MPI_INT, MPI_SUM, PETSC_COMM_WORLD));
    std::vector<PetscInt> counts(static_cast<std::size_t>(ranks));
    PetscCallMPI(MPI_Allgather(&owned, 1, MPIU_INT, counts.data(), 1, MPIU_INT, PETSC_COMM_WORLD));
    reference.offsets.assign(static_cast<std::size_t>(ranks)+1, 0);
    for (PetscMPIInt r = 0; r < ranks; ++r)
        reference.offsets[static_cast<std::size_t>(r)+1] =
            reference.offsets[static_cast<std::size_t>(r)] + counts[static_cast<std::size_t>(r)];
    reference.naturalToGlobal.resize(size);
    reference.globalToNatural.assign(size, -1);
    auto next = reference.offsets;
    for (std::size_t n = 0; n < size; ++n) {
        PetscCall(Equal(count[n], 1, "Each entity DOF must have exactly one MPI owner"));
        --reference.owner[n];
        const auto owner = static_cast<std::size_t>(reference.owner[n]);
        PetscCall(Require(owner < counts.size(), "Invalid reference owner"));
        const PetscInt global = next[owner]++;
        reference.naturalToGlobal[n] = global;
        reference.globalToNatural[static_cast<std::size_t>(global)] = static_cast<PetscInt>(n);
    }
    PetscCall(Equal(reference.offsets.back(), static_cast<PetscInt>(size), "Total reference ownership"));
    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode CheckNumbering(const DofMap& map, const std::vector<ReferenceDof>& catalogue,
                              const ReferenceNumbering& reference, bool smallCase)
{
    PetscFunctionBeginUser;
    PetscMPIInt rank, ranks;
    PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD, &rank));
    PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD, &ranks));
    PetscCall(Require(map.IsInitialized(), "DofMap was not initialized"));
    PetscCall(Equal(map.GlobalDofs(), static_cast<PetscInt>(catalogue.size()), "Global DOF count"));
    PetscCall(Equal(map.OwnershipBegin(), reference.offsets[static_cast<std::size_t>(rank)], "Ownership begin"));
    PetscCall(Equal(map.OwnershipEnd(), reference.offsets[static_cast<std::size_t>(rank)+1], "Ownership end"));
    std::vector<PetscInt> ghosts;
    for (std::size_t n = 0; n < catalogue.size(); ++n) {
        const auto& expected = catalogue[n];
        const PetscInt natural = static_cast<PetscInt>(n);
        PetscInt global = -1, inverse = -1;
        PetscMPIInt owner = -1;
        PetscCall(map.NaturalToGlobal(natural, global));
        PetscCall(Equal(global, reference.naturalToGlobal[n], "Natural-to-global independent reference"));
        PetscCall(map.GlobalToNatural(global, inverse));
        PetscCall(Equal(inverse, natural, "Global-to-natural inverse"));
        PetscCall(map.GetOwnerRank(global, owner));
        PetscCall(Equal(owner, reference.owner[n], "DOF owner rank"));
        PetscCall(Require(map.OwnsGlobal(global) == expected.owned, "Owned DOF disagrees with mesh entity owner"));
        DofInfo info;
        PetscCall(map.GetDofInfo(global, info));
        PetscCall(Require(SameInfo(info, expected.info), "DOF entity/component/boundary description"));
        PetscCall(Require(info.IsBoundary() == expected.info.IsBoundary(), "Boundary predicate"));
        if (expected.present && !expected.owned) ghosts.push_back(global);
        if (expected.present || expected.owned) {
            PetscInt local, back;
            PetscCall(map.GlobalToLocal(global, local));
            PetscCall(map.LocalToGlobal(local, back));
            PetscCall(Equal(back, global, "Local/global round trip"));
        } else {
            PetscInt local = -91;
            PetscCall(ExpectError([&] { return map.GlobalToLocal(global, local); }, PETSC_ERR_ARG_OUTOFRANGE));
            PetscCall(Equal(local, -91, "Unavailable DOF changed output"));
        }
    }
    std::sort(ghosts.begin(), ghosts.end());
    PetscCall(Require(ghosts == map.GhostGlobalIds(), "Ghost list differs from available mesh entities"));
    PetscCall(Equal(map.GhostDofs(), static_cast<PetscInt>(ghosts.size()), "Ghost count"));
    PetscCall(Equal(map.LocalDofs(), map.OwnedDofs()+map.GhostDofs(), "Local count"));
    for (PetscInt local = 0; local < map.LocalDofs(); ++local) {
        PetscInt global;
        PetscCall(map.LocalToGlobal(local, global));
        const PetscInt expected = local < map.OwnedDofs() ? map.OwnershipBegin()+local :
            ghosts[static_cast<std::size_t>(local-map.OwnedDofs())];
        PetscCall(Equal(global, expected, "Owned-first then sorted-ghost local ordering"));
    }
    const int localEmpty = map.OwnedDofs() == 0 ? 1 : 0;
    int emptyRanks = 0;
    PetscInt ghostTotal = 0, localGhosts = map.GhostDofs();
    PetscCallMPI(MPI_Allreduce(&localEmpty, &emptyRanks, 1, MPI_INT, MPI_SUM, PETSC_COMM_WORLD));
    PetscCallMPI(MPI_Allreduce(&localGhosts, &ghostTotal, 1, MPIU_INT, MPI_SUM, PETSC_COMM_WORLD));
    if (ranks > 1) PetscCall(Require(ghostTotal > 0, "MPI case did not exercise ghost DOFs"));
    if (smallCase) {
        const int expected = map.Space() == DofSpace::CellPressure ? ranks-1 :
                             map.Space() == DofSpace::HDivVelocity && ranks == 4 ? 1 : 0;
        PetscCall(Equal(emptyRanks, expected, "Expected zero-DOF ranks on the single-cell mesh"));
    }
    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode ExpectedCellDofs(const MeshInfo& mesh, DofSpace space, MeshIndex cell,
                               std::vector<PetscInt>& ids)
{
    PetscFunctionBeginUser;
    ids.clear();
    if (space == DofSpace::CellPressure) {
        PetscInt id;
        PetscCall(mesh.CellId(cell, id));
        ids.push_back(id);
        PetscFunctionReturn(PETSC_SUCCESS);
    }
    if (space == DofSpace::BRVelocity) {
        const std::array<MeshIndex,4> vertices{{cell, {cell.i+1,cell.j},
                                                {cell.i+1,cell.j+1}, {cell.i,cell.j+1}}};
        for (PetscInt component = 0; component < 2; ++component)
            for (const auto& vertex : vertices) {
                PetscInt id;
                PetscCall(mesh.VertexId(vertex, id));
                ids.push_back(component*mesh.VertexCount()+id);
            }
    }
    std::array<OrientedEdge,4> edges;
    PetscCall(mesh.GetCellEdges(cell, edges));
    const PetscInt modes = space == DofSpace::HDivVelocity ? 2 : 1;
    for (PetscInt mode = 0; mode < modes; ++mode) for (const auto& edge : edges)
        ids.push_back((space == DofSpace::BRVelocity ? 2*mesh.VertexCount() : mode*mesh.EdgeCount()) + edge.id);
    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode CheckCells(const MeshInfo& mesh, const DofMap& map,
                          const ReferenceNumbering& reference)
{
    PetscFunctionBeginUser;
    const PetscInt count = map.Space() == DofSpace::BRVelocity ? BRMixed::ElementDofs :
                           map.Space() == DofSpace::HDivVelocity ? HDivMixed::ElementDofs : 1;
    PetscCall(Equal(map.ElementDofs(), count, "Element DOF count matches basis"));
    for (PetscInt c = 0; c < mesh.CellCount(); ++c) {
        MeshIndex cell;
        PetscCall(mesh.CellIndex(c, cell));
        std::vector<PetscInt> expected, natural, global, local;
        PetscCall(ExpectedCellDofs(mesh, map.Space(), cell, expected));
        PetscCall(map.GetCellNaturalDofs(cell, natural));
        PetscCall(Require(natural == expected, "Element natural DOFs disagree with mesh topology/basis ordering"));
        PetscCall(map.GetCellGlobalDofs(cell, global));
        PetscCall(Equal(static_cast<PetscInt>(global.size()), count, "Element global DOF count"));
        for (std::size_t k = 0; k < global.size(); ++k)
            PetscCall(Equal(global[k], reference.naturalToGlobal[static_cast<std::size_t>(expected[k])], "Element global DOF"));
        if (mesh.HasCell(cell)) {
            PetscCall(map.GetCellLocalDofs(cell, local));
            PetscCall(Equal(static_cast<PetscInt>(local.size()), count, "Element local DOF count"));
            for (std::size_t k = 0; k < local.size(); ++k) {
                PetscInt back;
                PetscCall(map.LocalToGlobal(local[k], back));
                PetscCall(Equal(back, global[k], "Element local/global consistency"));
            }
        } else {
            const std::vector<PetscInt> before{-17, 31};
            local = before;
            PetscCall(ExpectError([&] { return map.GetCellLocalDofs(cell, local); }, PETSC_ERR_ARG_OUTOFRANGE));
            PetscCall(Require(local == before, "Unavailable cell changed output"));
        }
    }
    // On the serial 1-cell grid this hand-written numbering fixes the contract
    // independently of all topology helper formulas above.
    if (mesh.VertexDimensions() == MeshIndex{2,2}) {
        std::vector<PetscInt> actual;
        PetscCall(map.GetCellNaturalDofs({0,0}, actual));
        const std::vector<PetscInt> expected = map.Space() == DofSpace::BRVelocity
            ? std::vector<PetscInt>{0,1,3,2,4,5,7,6,8,11,9,10}
            : map.Space() == DofSpace::HDivVelocity
                ? std::vector<PetscInt>{0,3,1,2,4,7,5,6} : std::vector<PetscInt>{0};
        PetscCall(Require(actual == expected, "Hand-calculated single-cell DOF ordering"));
    }
    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscScalar Tag(PetscInt natural)
{
    const PetscReal value = static_cast<PetscReal>(natural+1);
#if defined(PETSC_USE_COMPLEX)
    return PetscCMPLX(value, -value/2);
#else
    return value;
#endif
}

PetscErrorCode ScalarEqual(PetscScalar actual, PetscScalar expected, const char* message)
{
    PetscFunctionBeginUser;
    // Tags and accumulated loads are small integers/dyadic fractions, so exact
    // comparison is appropriate in both real and complex PETSc configurations.
    PetscCall(Require(!PetscIsInfOrNanScalar(actual) && actual == expected, message));
    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode CheckGhostVector(const MeshInfo& mesh, const DofMap& map,
                                const std::vector<ReferenceDof>& catalogue,
                                const ReferenceNumbering& reference)
{
    PetscFunctionBeginUser;
    Vec vector = nullptr, local = nullptr;
    const auto& ghosts = map.GhostGlobalIds();
    PetscCall(VecCreateGhost(PETSC_COMM_WORLD, map.OwnedDofs(), map.GlobalDofs(),
                             map.GhostDofs(), ghosts.data(), &vector));
    PetscInt begin, end, size;
    PetscCall(VecGetOwnershipRange(vector, &begin, &end));
    PetscCall(Equal(begin, map.OwnershipBegin(), "PETSc vector ownership begin"));
    PetscCall(Equal(end, map.OwnershipEnd(), "PETSc vector ownership end"));
    PetscCall(VecGetSize(vector, &size));
    PetscCall(Equal(size, map.GlobalDofs(), "PETSc global vector size"));
    PetscScalar* values = nullptr;
    PetscCall(VecGetArray(vector, &values));
    for (PetscInt g = begin; g < end; ++g)
        values[g-begin] = Tag(reference.globalToNatural[static_cast<std::size_t>(g)]);
    PetscCall(VecRestoreArray(vector, &values));
    PetscCall(VecGhostUpdateBegin(vector, INSERT_VALUES, SCATTER_FORWARD));
    PetscCall(VecGhostUpdateEnd(vector, INSERT_VALUES, SCATTER_FORWARD));
    PetscCall(VecGhostGetLocalForm(vector, &local));
    PetscCall(Require(local != nullptr, "Missing ghosted local vector"));
    PetscCall(VecGetLocalSize(local, &size));
    PetscCall(Equal(size, map.LocalDofs(), "PETSc ghosted local vector size"));
    const PetscScalar* read = nullptr;
    PetscCall(VecGetArrayRead(local, &read));
    for (PetscInt l = 0; l < map.LocalDofs(); ++l) {
        PetscInt global;
        PetscCall(map.LocalToGlobal(l, global));
        PetscCall(ScalarEqual(read[l], Tag(reference.globalToNatural[static_cast<std::size_t>(global)]),
                              "Forward ghost update retrieved the wrong DOF"));
    }
    const auto available = mesh.AvailableCells();
    for (PetscInt j = available.begin.j; j < available.end.j; ++j)
        for (PetscInt i = available.begin.i; i < available.end.i; ++i) {
            std::vector<PetscInt> localDofs, expected;
            PetscCall(map.GetCellLocalDofs({i,j}, localDofs));
            PetscCall(ExpectedCellDofs(mesh, map.Space(), {i,j}, expected));
            for (std::size_t k = 0; k < localDofs.size(); ++k)
                PetscCall(ScalarEqual(read[localDofs[k]], Tag(expected[k]), "Ghosted element coefficient lookup"));
        }
    PetscCall(VecRestoreArrayRead(local, &read));
    // Clear BOTH owned and ghost entries before a reverse accumulation.
    PetscCall(VecSet(local, 0));
    PetscCall(VecGetArray(local, &values));
    const auto owned = mesh.OwnedCells();
    for (PetscInt j = owned.begin.j; j < owned.end.j; ++j)
        for (PetscInt i = owned.begin.i; i < owned.end.i; ++i) {
            PetscInt load;
            std::vector<PetscInt> localDofs;
            PetscCall(CellLoad(mesh, {i,j}, load));
            PetscCall(map.GetCellLocalDofs({i,j}, localDofs));
            for (PetscInt dof : localDofs) values[dof] += static_cast<PetscScalar>(load);
        }
    PetscCall(VecRestoreArray(local, &values));
    PetscCall(VecGhostRestoreLocalForm(vector, &local));
    PetscCall(VecGhostUpdateBegin(vector, ADD_VALUES, SCATTER_REVERSE));
    PetscCall(VecGhostUpdateEnd(vector, ADD_VALUES, SCATTER_REVERSE));
    PetscCall(VecGetArrayRead(vector, &read));
    for (PetscInt g = begin; g < end; ++g) {
        const auto natural = static_cast<std::size_t>(reference.globalToNatural[static_cast<std::size_t>(g)]);
        PetscCall(ScalarEqual(read[g-begin], static_cast<PetscScalar>(catalogue[natural].assembledLoad),
                              "Reverse assembly disagrees with independent entity incidence"));
    }
    PetscCall(VecRestoreArrayRead(vector, &read));
    PetscCall(VecDestroy(&vector));
    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscReal Coefficient(PetscInt natural) { return static_cast<PetscReal>(natural%17-8)/8; }

PetscErrorCode CheckConformingField(const MeshInfo& mesh, const DofMap& map,
                                    const ReferenceNumbering& reference)
{
    PetscFunctionBeginUser;
    if (map.Space() == DofSpace::CellPressure) PetscFunctionReturn(PETSC_SUCCESS);
    constexpr std::array<PetscReal,2> samples{{PetscReal(0.21), PetscReal(0.63)}};
    constexpr std::size_t stride = 4; // Two points, two vector components (BR) or normal trace (HDiv).
    std::vector<PetscReal> trace(stride*static_cast<std::size_t>(mesh.EdgeCount()), 0), sum(trace.size());
    PetscReal magnitude = 1, globalMagnitude;
    const auto owned = mesh.OwnedCells();
    for (PetscInt j = owned.begin.j; j < owned.end.j; ++j)
        for (PetscInt i = owned.begin.i; i < owned.end.i; ++i) {
            BRMixed br;
            HDivMixed hdiv;
            if (map.Space() == DofSpace::BRVelocity) PetscCall(br.Initialize(mesh, {i,j}));
            else PetscCall(hdiv.Initialize(mesh, {i,j}));
            std::vector<PetscInt> global;
            PetscCall(map.GetCellGlobalDofs({i,j}, global));
            std::array<OrientedEdge,4> edges;
            PetscCall(mesh.GetCellEdges({i,j}, edges));
            for (const auto& edge : edges) {
                EdgeVertices vertices;
                EdgeTopology topology;
                PetscCall(mesh.GetEdgeVertices(edge.id, vertices));
                PetscCall(mesh.GetEdgeTopology(edge.id, topology));
                const PetscReal dx = vertices[1].p[0]-vertices[0].p[0];
                const PetscReal dy = vertices[1].p[1]-vertices[0].p[1];
                const PetscReal length = std::hypot(dx,dy);
                const Point normal = topology.axis == EdgeAxis::AlongI
                    ? Point{{-dy/length,dx/length}} : Point{{dy/length,-dx/length}};
                for (std::size_t s = 0; s < samples.size(); ++s) {
                    const Point point{{vertices[0].p[0]+samples[s]*dx,
                                        vertices[0].p[1]+samples[s]*dy}};
                    Point value{};
                    if (map.Space() == DofSpace::BRVelocity) {
                        BRMixed::Values basis;
                        PetscCall(br.EvaluateAll(point, basis));
                        for (std::size_t k = 0; k < global.size(); ++k) {
                            const PetscReal coefficient = Coefficient(reference.globalToNatural[static_cast<std::size_t>(global[k])]);
                            for (int d = 0; d < 2; ++d) value.p[d] += coefficient*basis[k].value.p[d];
                        }
                    } else {
                        HDivMixed::Values basis;
                        PetscCall(hdiv.EvaluateAll(point, basis));
                        for (std::size_t k = 0; k < global.size(); ++k) {
                            const PetscReal coefficient = Coefficient(reference.globalToNatural[static_cast<std::size_t>(global[k])]);
                            for (int d = 0; d < 2; ++d) value.p[d] += coefficient*basis[k].value.p[d];
                        }
                        value = {{value.p[0]*normal.p[0]+value.p[1]*normal.p[1], 0}};
                    }
                    for (int d = 0; d < 2; ++d) {
                        PetscCall(Require(!PetscIsInfOrNanReal(value.p[d]), "Nonfinite reconstructed field"));
                        magnitude = std::max(magnitude, PetscAbsReal(value.p[d]));
                        trace[stride*static_cast<std::size_t>(edge.id)+2*s+static_cast<std::size_t>(d)] +=
                            edge.direction*value.p[d];
                    }
                }
            }
        }
    PetscCallMPI(MPI_Allreduce(trace.data(), sum.data(), static_cast<int>(trace.size()),
                               MPIU_REAL, MPI_SUM, PETSC_COMM_WORLD));
    PetscCallMPI(MPI_Allreduce(&magnitude, &globalMagnitude, 1, MPIU_REAL, MPI_MAX, PETSC_COMM_WORLD));
    for (PetscInt e = 0; e < mesh.EdgeCount(); ++e) {
        EdgeTopology topology;
        PetscCall(mesh.GetEdgeTopology(e, topology));
        if (topology.IsBoundary()) continue;
        for (std::size_t k = 0; k < stride; ++k)
            PetscCall(Require(PetscAbsReal(sum[stride*static_cast<std::size_t>(e)+k]) <=
                               32768*PETSC_MACHINE_EPSILON*globalMagnitude,
                               "Mapped coefficients break shared-edge conformity"));
    }
    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode SameMap(const DofMap& a, const DofMap& b)
{
    PetscFunctionBeginUser;
    PetscCall(Require(a.IsInitialized() && b.IsInitialized() && a.Space() == b.Space(), "Map state changed"));
    PetscCall(Equal(a.ElementDofs(), b.ElementDofs(), "Map element size changed"));
    PetscCall(Equal(a.GlobalDofs(), b.GlobalDofs(), "Map global size changed"));
    PetscCall(Equal(a.OwnershipBegin(), b.OwnershipBegin(), "Map ownership begin changed"));
    PetscCall(Equal(a.OwnershipEnd(), b.OwnershipEnd(), "Map ownership end changed"));
    PetscCall(Require(a.GhostGlobalIds() == b.GhostGlobalIds(), "Map ghosts changed"));
    for (PetscInt n = 0; n < a.GlobalDofs(); ++n) {
        PetscInt x,y;
        PetscCall(a.NaturalToGlobal(n,x));
        PetscCall(b.NaturalToGlobal(n,y));
        PetscCall(Equal(x,y,"Map numbering changed"));
        DofInfo p,q;
        PetscCall(a.GetDofInfo(x,p));
        PetscCall(b.GetDofInfo(y,q));
        PetscCall(Require(SameInfo(p,q),"Map description changed"));
    }
    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode CheckRejectedQueries(const MeshInfo& mesh, const DofMap& map)
{
    PetscFunctionBeginUser;
    DofMap empty;
    PetscInt output = -19;
    PetscCall(ExpectError([&] { return empty.NaturalToGlobal(0,output); }, PETSC_ERR_ARG_WRONG));
    PetscCall(Equal(output,-19,"Uninitialized query changed output"));
    for (PetscInt bad : {PetscInt(-1), map.GlobalDofs()}) {
        PetscCall(ExpectError([&] { return map.NaturalToGlobal(bad,output); }, PETSC_ERR_ARG_OUTOFRANGE));
        PetscCall(Equal(output,-19,"Bad natural ID changed output"));
        PetscCall(ExpectError([&] { return map.GlobalToNatural(bad,output); }, PETSC_ERR_ARG_OUTOFRANGE));
        PetscCall(Equal(output,-19,"Bad global ID changed output"));
        PetscCall(ExpectError([&] { return map.GlobalToLocal(bad,output); }, PETSC_ERR_ARG_OUTOFRANGE));
        PetscCall(Equal(output,-19,"Bad local lookup changed output"));
        PetscMPIInt owner = -11;
        PetscCall(ExpectError([&] { return map.GetOwnerRank(bad,owner); }, PETSC_ERR_ARG_OUTOFRANGE));
        PetscCall(Equal(owner,-11,"Bad owner query changed output"));
        DofInfo info;
        info.entityId = -27;
        info.component = 5;
        info.boundarySides = {{true,false,true,false}};
        const DofInfo before = info;
        PetscCall(ExpectError([&] { return map.GetDofInfo(bad,info); }, PETSC_ERR_ARG_OUTOFRANGE));
        PetscCall(Require(SameInfo(info,before),"Bad description query changed output"));
    }
    for (PetscInt bad : {PetscInt(-1), map.LocalDofs()}) {
        PetscCall(ExpectError([&] { return map.LocalToGlobal(bad,output); }, PETSC_ERR_ARG_OUTOFRANGE));
        PetscCall(Equal(output,-19,"Bad local ID changed output"));
    }
    const auto cells = mesh.CellDimensions();
    for (const MeshIndex cell : {MeshIndex{-1,0}, {0,-1}, {cells.i,0}, {0,cells.j}}) {
        const std::vector<PetscInt> before{31,-7};
        auto ids = before;
        PetscCall(ExpectError([&] { return map.GetCellNaturalDofs(cell,ids); }, PETSC_ERR_ARG_OUTOFRANGE));
        PetscCall(Require(ids == before,"Bad cell natural query changed output"));
        PetscCall(ExpectError([&] { return map.GetCellGlobalDofs(cell,ids); }, PETSC_ERR_ARG_OUTOFRANGE));
        PetscCall(Require(ids == before,"Bad cell global query changed output"));
        PetscCall(ExpectError([&] { return map.GetCellLocalDofs(cell,ids); }, PETSC_ERR_ARG_OUTOFRANGE));
        PetscCall(Require(ids == before,"Bad cell local query changed output"));
    }
    PetscCall(Require(!map.OwnsGlobal(-1) && !map.OwnsGlobal(map.GlobalDofs()), "Invalid ID reported as owned"));
    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode CheckState(MeshFixture& fixture, const DofMap& original,
                          PetscInt nx, PetscInt ny, PetscInt px, PetscInt py)
{
    PetscFunctionBeginUser;
    DofMap copy(original), assigned;
    assigned = original;
    PetscCall(SameMap(copy, original));
    PetscCall(SameMap(assigned, original));
    DofMap moved(std::move(copy));
    PetscCall(Require(!copy.IsInitialized() && copy.GlobalDofs() == 0 && copy.LocalDofs() == 0,
                       "Move constructor did not reset source"));
    assigned = std::move(moved);
    PetscCall(Require(!moved.IsInitialized() && moved.GlobalDofs() == 0 && moved.LocalDofs() == 0,
                       "Move assignment did not reset source"));
    PetscCall(SameMap(assigned, original));

    PetscMPIInt rank, ranks;
    PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD, &rank));
    PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD, &ranks));
    MeshInfo invalid;
    // Only one rank rejects its snapshot. Every rank must return, without
    // entering a mismatched collective or replacing the previous map.
    const MeshInfo& supplied = rank == ranks-1 ? invalid : fixture.mesh;
    PetscCall(ExpectError([&] { return assigned.Initialize(fixture.dm,supplied,original.Space()); }, PETSC_ERR_ARG_WRONG));
    PetscCall(SameMap(assigned, original));
    const DofSpace badSpace = rank == 0 ? static_cast<DofSpace>(-1) : original.Space();
    PetscCall(ExpectError([&] { return assigned.Initialize(fixture.dm,fixture.mesh,badSpace); }, PETSC_ERR_ARG_OUTOFRANGE));
    PetscCall(SameMap(assigned, original));
    if (ranks > 1) {
        const DofSpace inconsistent = rank == 0 ? DofSpace::BRVelocity : DofSpace::HDivVelocity;
        PetscCall(ExpectError([&] { return assigned.Initialize(fixture.dm,fixture.mesh,inconsistent); }, PETSC_ERR_ARG_WRONG));
        PetscCall(SameMap(assigned, original));
    }
    DM wrong = nullptr;
    PetscCall(DMDACreate2d(PETSC_COMM_WORLD,DM_BOUNDARY_NONE,DM_BOUNDARY_NONE,DMDA_STENCIL_BOX,
                          nx,ny,px,py,1,1,nullptr,nullptr,&wrong));
    PetscCall(DMSetUp(wrong));
    PetscCall(ExpectError([&] { return assigned.Initialize(wrong,fixture.mesh,original.Space()); }, PETSC_ERR_ARG_WRONG));
    PetscCall(DMDestroy(&wrong));
    PetscCall(SameMap(assigned, original));
    // Successful reinitialization switches space; copies of the old map stay independent.
    const DofSpace next = original.Space() == DofSpace::CellPressure
        ? DofSpace::BRVelocity : DofSpace::CellPressure;
    PetscCall(assigned.Initialize(fixture.dm,fixture.mesh,next));
    PetscCall(Require(assigned.Space() == next,"Successful reinitialization did not switch space"));
    PetscCall(Require(original.Space() != assigned.Space(),"Reinitializing copy changed original"));
    PetscCall(assigned.Initialize(fixture.dm,fixture.mesh,original.Space()));
    PetscCall(SameMap(assigned, original));
    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode Run()
{
    PetscFunctionBeginUser;
    PetscInt kind = 0, nx = 9, ny = 7, px = 1, py = 1, expectedRanks = 1, smallCase = 0;
    PetscCall(PetscOptionsGetInt(nullptr,nullptr,"-dof_mesh_type",&kind,nullptr));
    PetscCall(PetscOptionsGetInt(nullptr,nullptr,"-mesh_nx",&nx,nullptr));
    PetscCall(PetscOptionsGetInt(nullptr,nullptr,"-mesh_ny",&ny,nullptr));
    PetscCall(PetscOptionsGetInt(nullptr,nullptr,"-mesh_px",&px,nullptr));
    PetscCall(PetscOptionsGetInt(nullptr,nullptr,"-mesh_py",&py,nullptr));
    PetscCall(PetscOptionsGetInt(nullptr,nullptr,"-expected_ranks",&expectedRanks,nullptr));
    PetscCall(PetscOptionsGetInt(nullptr,nullptr,"-dof_small_case",&smallCase,nullptr));
    PetscMPIInt ranks;
    PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD,&ranks));
    PetscCheck(expectedRanks == ranks,PETSC_COMM_WORLD,PETSC_ERR_ARG_WRONG,
               "Expected %" PetscInt_FMT " MPI ranks, but PETSc sees %d; use /home/renpo/system/mpich-install/bin/mpiexec",
               expectedRanks,static_cast<int>(ranks));
    PetscCall(Require(kind >= 0 && kind <= 2,"dof_mesh_type must be 0, 1, or 2"));
    PetscCall(Require(px > 0 && py > 0 && px <= ranks && py <= ranks && px == ranks/py && ranks%py == 0,
                       "Process grid does not match MPI rank count"));
    // Keep this exhaustive test small and MPI count conversions safely in int.
    PetscCall(Require(nx >= 2 && ny >= 2 && nx <= 100 && ny <= 100 && nx >= px && ny >= py,
                       "Use 2..100 vertices in each direction, at least as many as MPI partitions"));
    PetscCall(Require(!smallCase || (nx == 2 && ny == 2),"dof_small_case requires a 2x2 vertex mesh"));
    MeshFixture fixture;
    PetscCall(MakeMesh(nx,ny,px,py,kind,fixture));
    const auto owned = fixture.mesh.OwnedCells();
    const int noCells = owned.Size().i == 0 || owned.Size().j == 0 ? 1 : 0;
    int emptyRanks = 0;
    PetscCallMPI(MPI_Allreduce(&noCells,&emptyRanks,1,MPI_INT,MPI_SUM,PETSC_COMM_WORLD));
    if (smallCase) PetscCall(Equal(emptyRanks,ranks-1,"Single-cell mesh must exercise empty cell owners"));

    const std::array<DofSpace,3> spaces{{DofSpace::BRVelocity,DofSpace::HDivVelocity,DofSpace::CellPressure}};
    std::array<DofMap,3> maps;
    std::array<std::vector<PetscInt>,3> firstCell;
    for (std::size_t s = 0; s < spaces.size(); ++s) {
        PetscCall(maps[s].Initialize(fixture.dm,fixture.mesh,spaces[s]));
        std::vector<ReferenceDof> catalogue;
        ReferenceNumbering reference;
        PetscCall(MakeCatalogue(fixture.mesh,spaces[s],catalogue));
        PetscCall(MakeReference(catalogue,reference));
        PetscCall(CheckNumbering(maps[s],catalogue,reference,smallCase != 0));
        PetscCall(CheckCells(fixture.mesh,maps[s],reference));
        PetscCall(CheckGhostVector(fixture.mesh,maps[s],catalogue,reference));
        PetscCall(CheckConformingField(fixture.mesh,maps[s],reference));
        PetscCall(CheckRejectedQueries(fixture.mesh,maps[s]));
        PetscCall(CheckState(fixture,maps[s],nx,ny,px,py));
        PetscCall(maps[s].GetCellGlobalDofs({0,0},firstCell[s]));
    }
    // Change physical geometry, preserving topology and MPI partition.
    PetscCall(SetGeometry(fixture, kind == 1 ? 0 : 1));
    for (std::size_t s = 0; s < spaces.size(); ++s) {
        DofMap other;
        PetscCall(other.Initialize(fixture.dm,fixture.mesh,spaces[s]));
        PetscCall(SameMap(other,maps[s]));
    }
    // Retained maps must remain usable after releasing ALL input objects.
    PetscCall(fixture.Close());
    for (std::size_t s = 0; s < spaces.size(); ++s) {
        std::vector<PetscInt> result;
        PetscCall(maps[s].GetCellGlobalDofs({0,0},result));
        PetscCall(Require(result == firstCell[s],"DofMap retained references to destroyed mesh objects"));
    }
    PetscCall(PetscPrintf(PETSC_COMM_WORLD,
        "DofMap: all three spaces passed on %" PetscInt_FMT "x%" PetscInt_FMT
        " vertices, mesh type %" PetscInt_FMT ", %d MPI rank(s)\n",
        nx,ny,kind,static_cast<int>(ranks)));
    PetscFunctionReturn(PETSC_SUCCESS);
}

} // namespace

int main(int argc, char** argv)
{
    const PetscErrorCode error = PetscInitialize(&argc,&argv,nullptr,
        "DofMap tests: -dof_mesh_type 0/1/2, -mesh_px/-mesh_py, -mesh_nx/-mesh_ny (vertices).\n");
    if (error) return static_cast<int>(error);
    // An unexpected rank-local failure must stop the whole job rather than
    // leaving another rank waiting inside a later collective.
    PetscCallAbort(PETSC_COMM_WORLD,Run());
    return static_cast<int>(PetscFinalize());
}
