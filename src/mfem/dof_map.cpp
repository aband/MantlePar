#include "dof_map.h"

#include <algorithm>
#include <limits>
#include <new>
#include <stdexcept>
#include <utility>

namespace {

PetscErrorCode Product(PetscInt a, PetscInt b, PetscInt& result)
{
    PetscFunctionBeginUser;
    PetscCheck(a >= 0 && b >= 0 && (b == 0 || a <= std::numeric_limits<PetscInt>::max()/b),
               PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "DOF count exceeds PetscInt");
    result = a*b;
    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode Sum(PetscInt a, PetscInt b, PetscInt& result)
{
    PetscFunctionBeginUser;
    PetscCheck(a >= 0 && b >= 0 && a <= std::numeric_limits<PetscInt>::max()-b,
               PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "DOF count exceeds PetscInt");
    result = a+b;
    PetscFunctionReturn(PETSC_SUCCESS);
}

bool SameRange(MeshRange a, MeshRange b)
{ return a.begin == b.begin && a.end == b.end; }

MeshRange Clip(MeshRange range, MeshIndex dimensions)
{
    return {{std::min(range.begin.i, dimensions.i), std::min(range.begin.j, dimensions.j)},
            {std::min(range.end.i, dimensions.i), std::min(range.end.j, dimensions.j)}};
}

// Ranges below are validated subrectangles of a grid whose count fits PetscInt.
PetscInt Count(MeshRange range)
{ return range.Size().i * range.Size().j; }

PetscInt Offset(MeshRange range, MeshIndex p)
{ return (p.j-range.begin.j)*range.Size().i + p.i-range.begin.i; }

MeshIndex Index(MeshRange range, PetscInt offset)
{
    return {range.begin.i + offset%range.Size().i,
            range.begin.j + offset/range.Size().i};
}

// Complete each local phase on all ranks before another MPI collective or
// committing the snapshot. Allocation errors also participate in this check.
template<class Function>
PetscErrorCode CollectiveStage(MPI_Comm comm, Function&& function)
{
    PetscFunctionBeginUser;
    PetscErrorCode error = PETSC_SUCCESS;
    try { error = function(); }
    catch (const std::bad_alloc&) { error = PETSC_ERR_MEM; }
    catch (const std::length_error&) { error = PETSC_ERR_ARG_SIZ; }
    const int local = static_cast<int>(error);
    int global = 0;
    PetscCallMPI(MPI_Allreduce(&local, &global, 1, MPI_INT, MPI_MAX, comm));
    PetscCheck(global == 0, comm, static_cast<PetscErrorCode>(global),
               "DofMap initialization failed on at least one rank");
    PetscFunctionReturn(PETSC_SUCCESS);
}

} // namespace

void DofMap::Swap(DofMap& other) noexcept
{
    using std::swap;
    swap(initialized_, other.initialized_);
    swap(space_, other.space_);
    swap(rank_, other.rank_);
    swap(ranks_, other.ranks_);
    swap(vertices_, other.vertices_);
    swap(cells_, other.cells_);
    swap(ownedVertices_, other.ownedVertices_);
    swap(ghostVertices_, other.ghostVertices_);
    swap(availableCells_, other.availableCells_);
    swap(vertexCount_, other.vertexCount_);
    swap(iEdgeCount_, other.iEdgeCount_);
    swap(edgeCount_, other.edgeCount_);
    swap(elementDofs_, other.elementDofs_);
    swap(globalDofs_, other.globalDofs_);
    swap(ownedBegin_, other.ownedBegin_);
    swap(ownedEnd_, other.ownedEnd_);
    swap(blocks_, other.blocks_);
    swap(partitions_, other.partitions_);
    swap(xCuts_, other.xCuts_);
    swap(yCuts_, other.yCuts_);
    swap(rankOffsets_, other.rankOffsets_);
    swap(ghosts_, other.ghosts_);
    swap(rankAtBlock_, other.rankAtBlock_);
}

DofMap::DofMap(DofMap&& other) noexcept { Swap(other); }
DofMap& DofMap::operator=(DofMap&& other) noexcept
{
    if (this != &other) {
        DofMap work(std::move(other));
        Swap(work);
    }
    return *this;
}

PetscErrorCode DofMap::CheckInitialized() const
{
    PetscFunctionBeginUser;
    PetscCheck(initialized_, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG,
               "DofMap is not initialized");
    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode DofMap::ReadLocal(DM dm, const MeshInfo& mesh, DofSpace space)
{
    PetscFunctionBeginUser;
    PetscCheck(mesh.IsInitialized(), PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG,
               "MeshInfo is not initialized");
    PetscCheck(space == DofSpace::BRVelocity || space == DofSpace::HDivVelocity ||
               space == DofSpace::CellPressure, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE,
               "Unknown finite-element space");
    PetscBool isDMDA = PETSC_FALSE;
    PetscCall(PetscObjectTypeCompare(reinterpret_cast<PetscObject>(dm), DMDA, &isDMDA));
    PetscCheck(isDMDA, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Expected a vertex DMDA");
    PetscInt dim, dof, width;
    DMBoundaryType bx, by;
    DMDAStencilType stencil;
    PetscCall(DMDAGetInfo(dm, &dim, &vertices_.i, &vertices_.j, nullptr, nullptr,
                          nullptr, nullptr, &dof, &width, &bx, &by, nullptr, &stencil));
    PetscCheck(dim == 2 && dof == 2 && width >= 1 && stencil == DMDA_STENCIL_BOX &&
               bx == DM_BOUNDARY_NONE && by == DM_BOUNDARY_NONE &&
               vertices_.i >= 2 && vertices_.j >= 2,
               PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Vertex DMDA must satisfy mesh.h");
    cells_ = {vertices_.i-1, vertices_.j-1};
    PetscInt xs, ys, xm, ym, gx, gy, gxm, gym;
    PetscCall(DMDAGetCorners(dm, &xs, &ys, nullptr, &xm, &ym, nullptr));
    PetscCall(DMDAGetGhostCorners(dm, &gx, &gy, nullptr, &gxm, &gym, nullptr));
    PetscCheck(xs >= 0 && ys >= 0 && xm > 0 && ym > 0 &&
               xs <= vertices_.i && xm <= vertices_.i-xs &&
               ys <= vertices_.j && ym <= vertices_.j-ys &&
               gx >= 0 && gy >= 0 && gxm > 0 && gym > 0 &&
               gx <= vertices_.i && gxm <= vertices_.i-gx &&
               gy <= vertices_.j && gym <= vertices_.j-gy,
               PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Invalid vertex ownership/ghost range");
    ownedVertices_ = {{xs,ys}, {xs+xm,ys+ym}};
    ghostVertices_ = {{gx,gy}, {gx+gxm,gy+gym}};
    availableCells_ = {{gx,gy}, {gx+gxm-1,gy+gym-1}};
    PetscCheck(mesh.VertexDimensions() == vertices_ && mesh.CellDimensions() == cells_ &&
               mesh.StencilWidth() == width && SameRange(mesh.OwnedVertices(), ownedVertices_) &&
               SameRange(mesh.GhostVertices(), ghostVertices_) &&
               SameRange(mesh.OwnedCells(), Clip(ownedVertices_, cells_)) &&
               SameRange(mesh.AvailableCells(), availableCells_),
               PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "MeshInfo does not match this vertex DMDA");
    PetscInt jEdges;
    PetscCall(Product(vertices_.i, vertices_.j, vertexCount_));
    PetscCall(Product(cells_.i, vertices_.j, iEdgeCount_));
    PetscCall(Product(vertices_.i, cells_.j, jEdges));
    PetscCall(Sum(iEdgeCount_, jEdges, edgeCount_));
    space_ = space;
    elementDofs_ = space == DofSpace::BRVelocity ? 12 :
                  space == DofSpace::HDivVelocity ? 8 : 1;
    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode DofMap::Initialize(DM dm, const MeshInfo& mesh, DofSpace space)
{
    PetscFunctionBeginUser;
    PetscCheck(dm, PETSC_COMM_SELF, PETSC_ERR_ARG_NULL, "Null vertex DMDA");
    const MPI_Comm comm = PetscObjectComm(reinterpret_cast<PetscObject>(dm));
    DofMap work;
    PetscCallMPI(MPI_Comm_rank(comm, &work.rank_));
    PetscCallMPI(MPI_Comm_size(comm, &work.ranks_));
    std::vector<PetscInt> rectangles;
    // PetscCall can return an error from either lambda. Each lambda therefore
    // needs its own PETSc stack frame, including a matching normal return.
    PetscCall(CollectiveStage(comm, [&]() -> PetscErrorCode {
        PetscFunctionBeginUser;
        PetscCall(work.ReadLocal(dm, mesh, space));
        const auto count = static_cast<std::size_t>(work.ranks_);
        PetscCheck(count <= rectangles.max_size()/4, PETSC_COMM_SELF,
                   PETSC_ERR_ARG_SIZ, "Too many MPI partitions");
        rectangles.resize(4*count);
        PetscFunctionReturn(PETSC_SUCCESS);
    }));
    const PetscInt configuration[3] = {work.vertices_.i, work.vertices_.j,
                                        static_cast<PetscInt>(space)};
    PetscInt minimum[3], maximum[3];
    PetscCallMPI(MPI_Allreduce(configuration, minimum, 3, MPIU_INT, MPI_MIN, comm));
    PetscCallMPI(MPI_Allreduce(configuration, maximum, 3, MPIU_INT, MPI_MAX, comm));
    PetscCheck(std::equal(minimum, minimum+3, maximum), comm, PETSC_ERR_ARG_WRONG,
               "Mesh dimensions and DofSpace must agree on every rank");
    const auto range = work.ownedVertices_;
    const PetscInt rectangle[4] = {range.begin.i, range.begin.j, range.end.i, range.end.j};
    PetscCallMPI(MPI_Allgather(rectangle, 4, MPIU_INT, rectangles.data(), 4, MPIU_INT, comm));
    PetscCall(CollectiveStage(comm, [&]() -> PetscErrorCode {
        PetscFunctionBeginUser;
        PetscCall(work.BuildNumbering(rectangles));
        work.initialized_ = true;
        PetscCall(work.BuildGhosts());
        PetscFunctionReturn(PETSC_SUCCESS);
    }));
    Swap(work);
    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode DofMap::AddBlock(DofEntity entity, PetscInt component,
                                EdgeAxis axis, MeshIndex dimensions)
{
    PetscFunctionBeginUser;
    PetscInt count, end;
    PetscCall(Product(dimensions.i, dimensions.j, count));
    PetscCall(Sum(globalDofs_, count, end));
    blocks_.push_back({entity, component, axis, dimensions, globalDofs_, end});
    globalDofs_ = end;
    PetscFunctionReturn(PETSC_SUCCESS);
}

MeshRange DofMap::BlockRange(const FieldBlock& block, MeshRange vertices) const
{ return Clip(vertices, block.dimensions); }

PetscErrorCode DofMap::BuildNumbering(const std::vector<PetscInt>& rectangles)
{
    PetscFunctionBeginUser;
    const MeshIndex iEdges{cells_.i, vertices_.j}, jEdges{vertices_.i, cells_.j};
    if (space_ == DofSpace::BRVelocity) {
        PetscCall(AddBlock(DofEntity::Vertex, 0, EdgeAxis::AlongI, vertices_));
        PetscCall(AddBlock(DofEntity::Vertex, 1, EdgeAxis::AlongI, vertices_));
        PetscCall(AddBlock(DofEntity::Edge, 0, EdgeAxis::AlongI, iEdges));
        PetscCall(AddBlock(DofEntity::Edge, 0, EdgeAxis::AlongJ, jEdges));
    } else if (space_ == DofSpace::HDivVelocity) {
        for (PetscInt mode = 0; mode < 2; ++mode) {
            PetscCall(AddBlock(DofEntity::Edge, mode, EdgeAxis::AlongI, iEdges));
            PetscCall(AddBlock(DofEntity::Edge, mode, EdgeAxis::AlongJ, jEdges));
        }
    } else {
        PetscCall(AddBlock(DofEntity::Cell, 0, EdgeAxis::AlongI, cells_));
    }
    partitions_.resize(static_cast<std::size_t>(ranks_));
    rankOffsets_.assign(static_cast<std::size_t>(ranks_)+1, 0);
    for (PetscMPIInt rank = 0; rank < ranks_; ++rank) {
        const auto r = static_cast<std::size_t>(rank);
        const MeshRange range{{rectangles[4*r],rectangles[4*r+1]},
                               {rectangles[4*r+2],rectangles[4*r+3]}};
        partitions_[r] = range;
        xCuts_.push_back(range.begin.i);
        xCuts_.push_back(range.end.i);
        yCuts_.push_back(range.begin.j);
        yCuts_.push_back(range.end.j);
        PetscInt count = 0;
        for (const auto& block : blocks_)
            PetscCall(Sum(count, Count(BlockRange(block, range)), count));
        PetscCall(Sum(rankOffsets_[r], count, rankOffsets_[r+1]));
    }
    PetscCheck(rankOffsets_.back() == globalDofs_, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG,
               "MPI ownership does not cover the finite-element space");
    for (auto* cuts : {&xCuts_, &yCuts_}) {
        std::sort(cuts->begin(), cuts->end());
        cuts->erase(std::unique(cuts->begin(), cuts->end()), cuts->end());
    }
    PetscCheck(xCuts_.front() == 0 && xCuts_.back() == vertices_.i &&
               yCuts_.front() == 0 && yCuts_.back() == vertices_.j,
               PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "MPI partitions do not span the mesh");
    const auto nx = xCuts_.size()-1, ny = yCuts_.size()-1;
    const auto total = static_cast<std::size_t>(ranks_);
    PetscCheck(nx > 0 && ny > 0 && total%nx == 0 && total/nx == ny,
               PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Expected tensor-product DMDA partitions");
    rankAtBlock_.assign(total, -1);
    for (PetscMPIInt rank = 0; rank < ranks_; ++rank) {
        const auto range = partitions_[static_cast<std::size_t>(rank)];
        const auto x = static_cast<std::size_t>(
            std::lower_bound(xCuts_.begin(), xCuts_.end(), range.begin.i)-xCuts_.begin());
        const auto y = static_cast<std::size_t>(
            std::lower_bound(yCuts_.begin(), yCuts_.end(), range.begin.j)-yCuts_.begin());
        PetscCheck(x < nx && y < ny && xCuts_[x+1] == range.end.i &&
                   yCuts_[y+1] == range.end.j && rankAtBlock_[y*nx+x] == -1,
                   PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Invalid/overlapping DMDA partitions");
        rankAtBlock_[y*nx+x] = rank;
    }
    ownedBegin_ = rankOffsets_[static_cast<std::size_t>(rank_)];
    ownedEnd_ = rankOffsets_[static_cast<std::size_t>(rank_)+1];
    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscMPIInt DofMap::VertexOwner(MeshIndex vertex) const
{
    const auto x = static_cast<std::size_t>(
        std::upper_bound(xCuts_.begin(), xCuts_.end(), vertex.i)-xCuts_.begin()-1);
    const auto y = static_cast<std::size_t>(
        std::upper_bound(yCuts_.begin(), yCuts_.end(), vertex.j)-yCuts_.begin()-1);
    return rankAtBlock_[y*(xCuts_.size()-1)+x];
}

PetscErrorCode DofMap::NaturalToGlobal(PetscInt natural, PetscInt& global) const
{
    PetscFunctionBeginUser;
    PetscCall(CheckInitialized());
    PetscCheck(natural >= 0 && natural < globalDofs_, PETSC_COMM_SELF,
               PETSC_ERR_ARG_OUTOFRANGE, "Natural DOF ID is out of range");
    for (const auto& block : blocks_) {
        if (natural >= block.end) continue;
        const PetscInt flat = natural-block.begin;
        const MeshIndex index{flat%block.dimensions.i, flat/block.dimensions.i};
        const auto rank = static_cast<std::size_t>(VertexOwner(index));
        PetscInt result = rankOffsets_[rank];
        for (const auto& preceding : blocks_) {
            const auto range = BlockRange(preceding, partitions_[rank]);
            if (&preceding == &block) {
                global = result+Offset(range, index);
                PetscFunctionReturn(PETSC_SUCCESS);
            }
            result += Count(range);
        }
    }
    SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Missing natural DOF block");
}

PetscErrorCode DofMap::GetOwnerRank(PetscInt global, PetscMPIInt& rank) const
{
    PetscFunctionBeginUser;
    PetscCall(CheckInitialized());
    PetscCheck(global >= 0 && global < globalDofs_, PETSC_COMM_SELF,
               PETSC_ERR_ARG_OUTOFRANGE, "Global DOF ID is out of range");
    // upper_bound skips repeated offsets from ranks owning zero DOFs.
    rank = static_cast<PetscMPIInt>(std::upper_bound(rankOffsets_.begin(),
                                      rankOffsets_.end(), global)-rankOffsets_.begin()-1);
    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode DofMap::GlobalToNatural(PetscInt global, PetscInt& natural) const
{
    PetscFunctionBeginUser;
    PetscMPIInt owner;
    PetscCall(GetOwnerRank(global, owner));
    const auto rank = static_cast<std::size_t>(owner);
    PetscInt offset = global-rankOffsets_[rank];
    for (const auto& block : blocks_) {
        const auto range = BlockRange(block, partitions_[rank]);
        const PetscInt count = Count(range);
        if (offset < count) {
            const auto index = Index(range, offset);
            natural = block.begin + index.j*block.dimensions.i + index.i;
            PetscFunctionReturn(PETSC_SUCCESS);
        }
        offset -= count;
    }
    SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Missing global DOF block");
}

PetscErrorCode DofMap::BuildGhosts()
{
    PetscFunctionBeginUser;
    for (const auto& block : blocks_) {
        MeshRange range = ghostVertices_;
        // A ghost edge needs both endpoints, not just its canonical start.
        if (block.entity == DofEntity::Cell) range = availableCells_;
        else if (block.entity == DofEntity::Edge) {
            if (block.axis == EdgeAxis::AlongI) --range.end.i;
            else --range.end.j;
        }
        for (PetscInt j = range.begin.j; j < range.end.j; ++j)
            for (PetscInt i = range.begin.i; i < range.end.i; ++i) {
                PetscInt global;
                PetscCall(NaturalToGlobal(block.begin+j*block.dimensions.i+i, global));
                if (!OwnsGlobal(global)) ghosts_.push_back(global);
            }
    }
    std::sort(ghosts_.begin(), ghosts_.end());
    // Blocks/entities are disjoint; no duplicates are generated.
    PetscCheck(ghosts_.size() <= static_cast<std::size_t>(globalDofs_-OwnedDofs()),
               PETSC_COMM_SELF, PETSC_ERR_PLIB, "Invalid number of ghost DOFs");
    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode DofMap::GetCellNaturalDofs(MeshIndex cell, std::vector<PetscInt>& ids) const
{
    PetscFunctionBeginUser;
    PetscCall(CheckInitialized());
    PetscInt cellId;
    PetscCall(FlattenIndex(cells_, cell, cellId));
    std::vector<PetscInt> result(static_cast<std::size_t>(elementDofs_));
    if (space_ == DofSpace::CellPressure) {
        result[0] = cellId;
    } else {
        // Bottom,Right,Top,Left in MeshInfo edge IDs. No orientation factors.
        const std::array<PetscInt,4> edges{{
            cell.j*cells_.i+cell.i,
            iEdgeCount_+cell.j*vertices_.i+cell.i+1,
            (cell.j+1)*cells_.i+cell.i,
            iEdgeCount_+cell.j*vertices_.i+cell.i}};
        if (space_ == DofSpace::BRVelocity) {
            const PetscInt v = cell.j*vertices_.i+cell.i;
            const std::array<PetscInt,4> vertices{{v,v+1,v+vertices_.i+1,v+vertices_.i}};
            for (std::size_t k = 0; k < 4; ++k) {
                result[k] = vertices[k];
                result[k+4] = vertexCount_+vertices[k];
                result[k+8] = 2*vertexCount_+edges[k];
            }
        } else {
            for (std::size_t k = 0; k < 4; ++k) {
                result[k] = edges[k];
                result[k+4] = edgeCount_+edges[k];
            }
        }
    }
    ids.swap(result);
    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode DofMap::GetCellGlobalDofs(MeshIndex cell, std::vector<PetscInt>& ids) const
{
    PetscFunctionBeginUser;
    std::vector<PetscInt> result;
    PetscCall(GetCellNaturalDofs(cell, result));
    for (auto& id : result) PetscCall(NaturalToGlobal(id, id));
    ids.swap(result);
    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode DofMap::GetCellLocalDofs(MeshIndex cell, std::vector<PetscInt>& ids) const
{
    PetscFunctionBeginUser;
    PetscCall(CheckInitialized());
    PetscCheck(availableCells_.Contains(cell), PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE,
               "Cell is not available in this rank's MeshInfo snapshot");
    std::vector<PetscInt> result;
    PetscCall(GetCellGlobalDofs(cell, result));
    for (auto& id : result) PetscCall(GlobalToLocal(id, id));
    ids.swap(result);
    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode DofMap::LocalToGlobal(PetscInt local, PetscInt& global) const
{
    PetscFunctionBeginUser;
    PetscCall(CheckInitialized());
    PetscCheck(local >= 0 && local < LocalDofs(), PETSC_COMM_SELF,
               PETSC_ERR_ARG_OUTOFRANGE, "Local DOF ID is out of range");
    global = local < OwnedDofs() ? ownedBegin_+local :
                                   ghosts_[static_cast<std::size_t>(local-OwnedDofs())];
    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode DofMap::GlobalToLocal(PetscInt global, PetscInt& local) const
{
    PetscFunctionBeginUser;
    PetscCall(CheckInitialized());
    if (OwnsGlobal(global)) {
        local = global-ownedBegin_;
    } else {
        const auto found = std::lower_bound(ghosts_.begin(), ghosts_.end(), global);
        PetscCheck(found != ghosts_.end() && *found == global, PETSC_COMM_SELF,
                   PETSC_ERR_ARG_OUTOFRANGE, "DOF is neither owned nor ghosted on this rank");
        local = OwnedDofs()+static_cast<PetscInt>(found-ghosts_.begin());
    }
    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode DofMap::GetDofInfo(PetscInt global, DofInfo& info) const
{
    PetscFunctionBeginUser;
    PetscInt natural;
    PetscCall(GlobalToNatural(global, natural));
    for (const auto& block : blocks_) {
        if (natural >= block.end) continue;
        const PetscInt flat = natural-block.begin;
        const MeshIndex p{flat%block.dimensions.i, flat/block.dimensions.i};
        DofInfo result;
        result.entity = block.entity;
        result.component = block.component;
        result.entityId = flat;
        if (block.entity == DofEntity::Vertex) {
            result.boundarySides = {{p.j == 0, p.i == cells_.i,
                                      p.j == cells_.j, p.i == 0}};
        } else if (block.entity == DofEntity::Edge) {
            if (block.axis == EdgeAxis::AlongI) {
                result.boundarySides[0] = p.j == 0;
                result.boundarySides[2] = p.j == cells_.j;
            } else {
                result.entityId += iEdgeCount_;
                result.boundarySides[1] = p.i == cells_.i;
                result.boundarySides[3] = p.i == 0;
            }
        }
        info = result;
        PetscFunctionReturn(PETSC_SUCCESS);
    }
    SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Missing DOF description");
}
