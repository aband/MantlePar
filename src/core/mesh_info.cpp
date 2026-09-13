#include "mesh_info.h"

#include <algorithm>
#include <limits>
#include <utility>

namespace {

PetscErrorCode GridCount(MeshIndex size, PetscInt& count)
{
    PetscFunctionBeginUser;
    PetscCheck(size.i > 0 && size.j > 0, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE,
               "Grid dimensions must be positive");
    PetscCheck(size.i <= std::numeric_limits<PetscInt>::max() / size.j,
               PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE,
               "Grid size exceeds the PetscInt index range");
    count = size.i * size.j;
    PetscFunctionReturn(PETSC_SUCCESS);
}

bool Inside(MeshIndex size, MeshIndex p)
{ return p.i >= 0 && p.i < size.i && p.j >= 0 && p.j < size.j; }

std::size_t Offset(const MeshRange& range, MeshIndex p)
{
    return static_cast<std::size_t>(p.j - range.begin.j)
           * static_cast<std::size_t>(range.Size().i)
           + static_cast<std::size_t>(p.i - range.begin.i);
}

std::size_t RangeCount(const MeshRange& range)
{
    const MeshIndex size = range.Size();
    return static_cast<std::size_t>(size.i) * static_cast<std::size_t>(size.j);
}

// Restore borrowed PETSc resources even if a PetscCall exits early.
struct LocalCoordinates {
    DM dm;
    Vec local = nullptr;
    const PetscScalar*** array = nullptr;
    explicit LocalCoordinates(DM owner) : dm(owner) {}
    LocalCoordinates(const LocalCoordinates&) = delete;
    LocalCoordinates& operator=(const LocalCoordinates&) = delete;
    ~LocalCoordinates()
    {
        if (array) (void)DMDAVecRestoreArrayDOFRead(dm, local, &array);
        if (local) (void)DMRestoreLocalVector(dm, &local);
    }
    PetscErrorCode Read(Vec global)
    {
        PetscFunctionBeginUser;
        PetscCall(DMGetLocalVector(dm, &local));
        PetscCall(DMGlobalToLocalBegin(dm, global, INSERT_VALUES, local));
        PetscCall(DMGlobalToLocalEnd(dm, global, INSERT_VALUES, local));
        PetscCall(DMDAVecGetArrayDOFRead(dm, local, &array));
        PetscFunctionReturn(PETSC_SUCCESS);
    }
    PetscErrorCode Close()
    {
        PetscFunctionBeginUser;
        PetscCall(DMDAVecRestoreArrayDOFRead(dm, local, &array));
        array = nullptr;
        PetscCall(DMRestoreLocalVector(dm, &local));
        local = nullptr;
        PetscFunctionReturn(PETSC_SUCCESS);
    }
};

} // namespace

void MeshInfo::Swap(MeshInfo& other) noexcept
{
    using std::swap;
    swap(initialized_, other.initialized_);
    swap(vertices_, other.vertices_);
    swap(cells_, other.cells_);
    swap(vertexCount_, other.vertexCount_);
    swap(cellCount_, other.cellCount_);
    swap(iEdges_, other.iEdges_);
    swap(jEdges_, other.jEdges_);
    swap(width_, other.width_);
    swap(ownedVertices_, other.ownedVertices_);
    swap(ghostVertices_, other.ghostVertices_);
    swap(ownedCells_, other.ownedCells_);
    swap(availableCells_, other.availableCells_);
    swap(coordinates_, other.coordinates_);
    swap(areas_, other.areas_);
    swap(ownedEdgeIds_, other.ownedEdgeIds_);
}

MeshInfo::MeshInfo(MeshInfo&& other) noexcept { Swap(other); }
MeshInfo& MeshInfo::operator=(MeshInfo&& other) noexcept
{
    if (this != &other) {
        MeshInfo work(std::move(other));
        Swap(work);
    }
    return *this;
}

PetscErrorCode FlattenIndex(MeshIndex size, MeshIndex index, PetscInt& id)
{
    PetscFunctionBeginUser;
    PetscInt count;
    PetscCall(GridCount(size, count));
    PetscCheck(Inside(size, index), PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE,
               "Logical index lies outside the grid");
    id = index.j * size.i + index.i;
    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode UnflattenIndex(MeshIndex size, PetscInt id, MeshIndex& index)
{
    PetscFunctionBeginUser;
    PetscInt count;
    PetscCall(GridCount(size, count));
    PetscCheck(id >= 0 && id < count, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE,
               "Flat index lies outside the grid");
    index = {id % size.i, id / size.i};
    PetscFunctionReturn(PETSC_SUCCESS);
}

bool MeshInfo::ContainsCell(MeshIndex cell) const noexcept { return Inside(cells_, cell); }

PetscErrorCode MeshInfo::CellId(MeshIndex cell, PetscInt& id) const
{ return FlattenIndex(cells_, cell, id); }
PetscErrorCode MeshInfo::CellIndex(PetscInt id, MeshIndex& cell) const
{ return UnflattenIndex(cells_, id, cell); }
PetscErrorCode MeshInfo::VertexId(MeshIndex vertex, PetscInt& id) const
{ return FlattenIndex(vertices_, vertex, id); }
PetscErrorCode MeshInfo::VertexIndex(PetscInt id, MeshIndex& vertex) const
{ return UnflattenIndex(vertices_, id, vertex); }

PetscErrorCode MeshInfo::CellGlobalToLocal(MeshIndex global, MeshIndex& local) const
{
    PetscFunctionBeginUser;
    PetscCheck(OwnsCell(global), PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE,
               "Cell is not owned by this snapshot's rank");
    local = {global.i - ownedCells_.begin.i, global.j - ownedCells_.begin.j};
    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode MeshInfo::CellLocalToGlobal(MeshIndex local, MeshIndex& global) const
{
    PetscFunctionBeginUser;
    PetscCheck(Inside(ownedCells_.Size(), local), PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE,
               "Local cell offset lies outside the owned cells");
    global = {local.i + ownedCells_.begin.i, local.j + ownedCells_.begin.j};
    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode MeshInfo::EdgeId(EdgeAxis axis, MeshIndex start, PetscInt& id) const
{
    PetscFunctionBeginUser;
    PetscCheck(initialized_, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "MeshInfo is not initialized");
    PetscInt flat;
    if (axis == EdgeAxis::AlongI) {
        PetscCall(FlattenIndex({cells_.i, vertices_.j}, start, flat));
    } else {
        PetscCheck(axis == EdgeAxis::AlongJ, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE,
                   "Unknown logical edge axis");
        PetscCall(FlattenIndex({vertices_.i, cells_.j}, start, flat));
        flat += iEdges_;
    }
    id = flat;
    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode MeshInfo::GetEdgeTopology(PetscInt id, EdgeTopology& topology) const
{
    PetscFunctionBeginUser;
    PetscCheck(id >= 0 && id < EdgeCount(), PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE,
               "Global edge ID lies outside the mesh");
    EdgeTopology work;
    MeshIndex start, left, right;
    if (id < iEdges_) {
        work.axis = EdgeAxis::AlongI;
        PetscCall(UnflattenIndex({cells_.i, vertices_.j}, id, start));
        work.vertices = {{start, {start.i + 1, start.j}}};
        left = start;
        right = {start.i, start.j - 1};
    } else {
        work.axis = EdgeAxis::AlongJ;
        PetscCall(UnflattenIndex({vertices_.i, cells_.j}, id - iEdges_, start));
        work.vertices = {{start, {start.i, start.j + 1}}};
        left = {start.i - 1, start.j};
        right = start;
    }
    if (ContainsCell(left)) work.leftCell = left;
    if (ContainsCell(right)) work.rightCell = right;
    topology = work;
    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode MeshInfo::GetCellEdges(MeshIndex cell, std::array<OrientedEdge,4>& edges) const
{
    PetscFunctionBeginUser;
    PetscCheck(ContainsCell(cell), PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE,
               "Cell lies outside the physical mesh");
    std::array<OrientedEdge,4> work{};
    PetscCall(EdgeId(EdgeAxis::AlongI, cell, work[0].id));
    PetscCall(EdgeId(EdgeAxis::AlongJ, {cell.i + 1, cell.j}, work[1].id));
    PetscCall(EdgeId(EdgeAxis::AlongI, {cell.i, cell.j + 1}, work[2].id));
    PetscCall(EdgeId(EdgeAxis::AlongJ, cell, work[3].id));
    work[2].direction = work[3].direction = -1;
    edges = work;
    PetscFunctionReturn(PETSC_SUCCESS);
}

bool MeshInfo::OwnsEdge(PetscInt id) const noexcept
{ return std::binary_search(ownedEdgeIds_.begin(), ownedEdgeIds_.end(), id); }

PetscErrorCode MeshInfo::EdgeGlobalToLocal(PetscInt global, PetscInt& local) const
{
    PetscFunctionBeginUser;
    const auto found = std::lower_bound(ownedEdgeIds_.begin(), ownedEdgeIds_.end(), global);
    PetscCheck(found != ownedEdgeIds_.end() && *found == global,
               PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Edge is not locally owned");
    local = static_cast<PetscInt>(found - ownedEdgeIds_.begin());
    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode MeshInfo::EdgeLocalToGlobal(PetscInt local, PetscInt& global) const
{
    PetscFunctionBeginUser;
    PetscCheck(local >= 0 && static_cast<std::size_t>(local) < ownedEdgeIds_.size(),
               PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Local edge ID is out of range");
    global = ownedEdgeIds_[static_cast<std::size_t>(local)];
    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode MeshInfo::GetVertex(MeshIndex vertex, Point& point) const
{
    PetscFunctionBeginUser;
    PetscCheck(ghostVertices_.Contains(vertex), PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE,
               "Vertex is not present in this rank's coordinate snapshot");
    point = coordinates_[Offset(ghostVertices_, vertex)];
    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode MeshInfo::GetCellCorners(MeshIndex cell, QuadVertices& corners) const
{
    PetscFunctionBeginUser;
    PetscCheck(HasCell(cell), PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE,
               "All four cell corners must be present in the local ghost region");
    corners = {{coordinates_[Offset(ghostVertices_, cell)],
                coordinates_[Offset(ghostVertices_, {cell.i + 1, cell.j})],
                coordinates_[Offset(ghostVertices_, {cell.i + 1, cell.j + 1})],
                coordinates_[Offset(ghostVertices_, {cell.i, cell.j + 1})]}};
    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode MeshInfo::GetCellArea(MeshIndex cell, PetscReal& area) const
{
    PetscFunctionBeginUser;
    PetscCheck(HasCell(cell), PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE,
               "Cell area is not available in this snapshot");
    area = areas_[Offset(availableCells_, cell)];
    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode MeshInfo::GetEdgeVertices(PetscInt id, EdgeVertices& edge) const
{
    PetscFunctionBeginUser;
    EdgeTopology topology;
    EdgeVertices work;
    PetscCall(GetEdgeTopology(id, topology));
    PetscCall(GetVertex(topology.vertices[0], work[0]));
    PetscCall(GetVertex(topology.vertices[1], work[1]));
    edge = work;
    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode MeshInfo::GetCellEdgeVertices(MeshIndex cell, CellSide side, EdgeVertices& edge) const
{
    PetscFunctionBeginUser;
    const int sideIndex = static_cast<int>(side);
    PetscCheck(sideIndex >= 0 && sideIndex < 4, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE,
               "Unknown cell side");
    QuadVertices corners;
    PetscCall(GetCellCorners(cell, corners));
    edge = {{corners[sideIndex], corners[(sideIndex + 1) % 4]}};
    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode MeshInfo::CacheAreas()
{
    PetscFunctionBeginUser;
    GaussRule1D rule;
    // A Q1 cell's Jacobian is affine; a 1x1 rule integrates its area exactly.
    PetscCall(CreateGaussRule(1, rule));
    for (PetscInt j = availableCells_.begin.j; j < availableCells_.end.j; ++j) {
        for (PetscInt i = availableCells_.begin.i; i < availableCells_.end.i; ++i) {
            QuadVertices corners;
            PetscReal area;
            PetscCall(GetCellCorners({i,j}, corners));
            PetscCall(IntegrateCell(corners, rule, [](const Point&) { return PetscReal(1); }, area));
            PetscCheck(area > 0, PETSC_COMM_SELF, PETSC_ERR_FP, "Cell area underflowed to zero");
            areas_[Offset(availableCells_, {i,j})] = area;
        }
    }
    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode BuildMeshInfo(DM dm, Vec vertices, MeshInfo& info)
{
    PetscFunctionBeginUser;
    PetscCall(ValidateMesh(dm, vertices));
    MeshInfo work;
    PetscInt xs, ys, xm, ym, gx, gy, gxm, gym;
    PetscCall(DMDAGetInfo(dm, nullptr, &work.vertices_.i, &work.vertices_.j,
                          nullptr, nullptr, nullptr, nullptr, nullptr, &work.width_,
                          nullptr, nullptr, nullptr, nullptr));
    PetscCall(DMDAGetCorners(dm, &xs, &ys, nullptr, &xm, &ym, nullptr));
    PetscCall(DMDAGetGhostCorners(dm, &gx, &gy, nullptr, &gxm, &gym, nullptr));
    const auto size = work.vertices_;
    work.cells_ = {size.i - 1, size.j - 1};
    PetscCall(GridCount(size, work.vertexCount_));
    PetscCall(GridCount(work.cells_, work.cellCount_));
    PetscCall(GridCount({size.i - 1, size.j}, work.iEdges_));
    PetscCall(GridCount({size.i, size.j - 1}, work.jEdges_));
    PetscCheck(work.iEdges_ <= std::numeric_limits<PetscInt>::max() - work.jEdges_,
               PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Edge count exceeds PetscInt");
    work.ownedVertices_ = {{xs,ys}, {xs+xm,ys+ym}};
    work.ghostVertices_ = {{gx,gy}, {gx+gxm,gy+gym}};
    work.ownedCells_ = {{std::min(xs,size.i-1), std::min(ys,size.j-1)},
                        {std::min(xs+xm,size.i-1), std::min(ys+ym,size.j-1)}};
    work.availableCells_ = {{gx,gy}, {gx+gxm-1,gy+gym-1}};
    work.coordinates_.resize(RangeCount(work.ghostVertices_));
    work.areas_.resize(RangeCount(work.availableCells_));
    LocalCoordinates local(dm);
    PetscCall(local.Read(vertices));
    for (PetscInt j = gy; j < gy+gym; ++j)
        for (PetscInt i = gx; i < gx+gxm; ++i)
            work.coordinates_[Offset(work.ghostVertices_, {i,j})] =
                Point{{PetscRealPart(local.array[j][i][0]), PetscRealPart(local.array[j][i][1])}};
    PetscCall(local.Close());
    work.initialized_ = true;
    // Horizontal IDs first, then vertical IDs: globally sorted local ownership list.
    for (PetscInt j = ys; j < ys+ym; ++j)
        for (PetscInt i = xs; i < std::min(xs+xm,size.i-1); ++i)
            work.ownedEdgeIds_.push_back(j*(size.i-1)+i);
    for (PetscInt j = ys; j < std::min(ys+ym,size.j-1); ++j)
        for (PetscInt i = xs; i < xs+xm; ++i)
            work.ownedEdgeIds_.push_back(work.iEdges_+j*size.i+i);
    // A locally unrepresentable area must fail construction on ALL ranks before
    // any rank replaces its previous snapshot. No collectives occur in cell loops.
    const PetscErrorCode areaError = work.CacheAreas();
    const int localError = static_cast<int>(areaError);
    int globalError = 0;
    const MPI_Comm comm = PetscObjectComm(reinterpret_cast<PetscObject>(dm));
    PetscCallMPI(MPI_Allreduce(&localError, &globalError, 1, MPI_INT, MPI_MAX, comm));
    PetscCheck(globalError == 0, comm, static_cast<PetscErrorCode>(globalError),
               "Mesh area cache construction failed on at least one rank");
    info = std::move(work);
    PetscFunctionReturn(PETSC_SUCCESS);
}
