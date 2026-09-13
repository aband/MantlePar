#ifndef MANTLE_CORE_MESH_INFO_H
#define MANTLE_CORE_MESH_INFO_H

#include "integral.h"

#include <array>
#include <optional>
#include <vector>

struct MeshIndex {
    PetscInt i = 0;
    PetscInt j = 0;
};
inline bool operator==(MeshIndex a, MeshIndex b) noexcept
{ return a.i == b.i && a.j == b.j; }
inline bool operator!=(MeshIndex a, MeshIndex b) noexcept { return !(a == b); }

// Half-open ranges of GLOBAL logical indices: [begin.i,end.i) x [begin.j,end.j).
struct MeshRange {
    MeshIndex begin{}, end{};
    MeshIndex Size() const noexcept { return {end.i - begin.i, end.j - begin.j}; }
    bool Contains(MeshIndex p) const noexcept
    { return p.i >= begin.i && p.i < end.i && p.j >= begin.j && p.j < end.j; }
};

// Checked row-major indexing. These are natural logical IDs, NOT PETSc Vec
// global algebraic indices. Both dimensions must be positive; overflow is rejected.
PetscErrorCode FlattenIndex(MeshIndex dimensions, MeshIndex index, PetscInt& id);
PetscErrorCode UnflattenIndex(MeshIndex dimensions, PetscInt id, MeshIndex& index);

// Logical edge families; an AlongI edge may be physically slanted.
enum class EdgeAxis { AlongI, AlongJ };
enum class CellSide { Bottom = 0, Right = 1, Top = 2, Left = 3 };

struct EdgeTopology {
    EdgeAxis axis = EdgeAxis::AlongI;
    std::array<MeshIndex, 2> vertices{}; // Canonical direction: increasing i or j.
    std::optional<MeshIndex> leftCell;  // Left of the canonically directed edge.
    std::optional<MeshIndex> rightCell; // Right of that edge; its right normal points here.
    bool IsBoundary() const noexcept { return !leftCell || !rightCell; }
};

struct OrientedEdge {
    PetscInt id = 0;
    int direction = 1; // +1 canonical direction; -1 reversed for the cell's CCW boundary.
};

// An owning, local geometry snapshot. No DM, Vec, communicator, or solution-array
// pointers are retained. Rebuild after changing coordinates or repartitioning.
// Getters and queries are noncollective. Checked outputs are assigned on success.
class MeshInfo {
public:
    MeshInfo() = default;
    MeshInfo(const MeshInfo&) = default;
    MeshInfo& operator=(const MeshInfo&) = default;
    MeshInfo(MeshInfo&& other) noexcept;
    MeshInfo& operator=(MeshInfo&& other) noexcept;
    bool IsInitialized() const noexcept { return initialized_; }
    MeshIndex VertexDimensions() const noexcept { return vertices_; }
    MeshIndex CellDimensions() const noexcept { return cells_; }
    PetscInt VertexCount() const noexcept { return vertexCount_; }
    PetscInt CellCount() const noexcept { return cellCount_; }
    PetscInt AlongIEdgeCount() const noexcept { return iEdges_; }
    PetscInt AlongJEdgeCount() const noexcept { return jEdges_; }
    PetscInt EdgeCount() const noexcept { return iEdges_ + jEdges_; }
    PetscInt StencilWidth() const noexcept { return width_; }
    const MeshRange& OwnedVertices() const noexcept { return ownedVertices_; }
    const MeshRange& GhostVertices() const noexcept { return ghostVertices_; }
    const MeshRange& OwnedCells() const noexcept { return ownedCells_; }
    // All cells whose FOUR corners are present locally, including MPI neighbors.
    const MeshRange& AvailableCells() const noexcept { return availableCells_; }
    bool ContainsCell(MeshIndex cell) const noexcept;
    bool OwnsCell(MeshIndex cell) const noexcept { return ownedCells_.Contains(cell); }
    bool HasCell(MeshIndex cell) const noexcept { return availableCells_.Contains(cell); }

    PetscErrorCode CellId(MeshIndex cell, PetscInt& id) const;
    PetscErrorCode CellIndex(PetscInt id, MeshIndex& cell) const;
    PetscErrorCode VertexId(MeshIndex vertex, PetscInt& id) const;
    PetscErrorCode VertexIndex(PetscInt id, MeshIndex& vertex) const;
    // Owned-cell offsets, not ghost-array offsets. Empty cell owners are valid.
    PetscErrorCode CellGlobalToLocal(MeshIndex global, MeshIndex& local) const;
    PetscErrorCode CellLocalToGlobal(MeshIndex local, MeshIndex& global) const;

    // AlongI IDs first: j*(M-1)+i. AlongJ IDs: (M-1)*N+j*M+i.
    PetscErrorCode EdgeId(EdgeAxis axis, MeshIndex start, PetscInt& id) const;
    PetscErrorCode GetEdgeTopology(PetscInt id, EdgeTopology& topology) const;
    // Order matches CellSide: bottom, right, top, left; directions are +,+,-,-.
    PetscErrorCode GetCellEdges(MeshIndex cell, std::array<OrientedEdge,4>& edges) const;

    // Each edge belongs to the owner of its canonical START vertex. This includes
    // top/right boundary edges on ranks that may own no cells. IDs are sorted.
    const std::vector<PetscInt>& OwnedEdgeIds() const noexcept { return ownedEdgeIds_; }
    bool OwnsEdge(PetscInt globalId) const noexcept;
    PetscErrorCode EdgeGlobalToLocal(PetscInt globalId, PetscInt& localId) const;
    PetscErrorCode EdgeLocalToGlobal(PetscInt localId, PetscInt& globalId) const;

    PetscErrorCode GetVertex(MeshIndex vertex, Point& point) const;
    PetscErrorCode GetCellCorners(MeshIndex cell, QuadVertices& corners) const;
    PetscErrorCode GetCellArea(MeshIndex cell, PetscReal& area) const;
    PetscErrorCode GetEdgeVertices(PetscInt id, EdgeVertices& edge) const;
    // Endpoints follow the cell's CCW boundary. GetEdgeGeometry then gives its
    // OUTWARD normal, including on physically perturbed quadrilaterals.
    PetscErrorCode GetCellEdgeVertices(MeshIndex cell, CellSide side, EdgeVertices& edge) const;

private:
    friend PetscErrorCode BuildMeshInfo(DM, Vec, MeshInfo&);
    void Swap(MeshInfo& other) noexcept;
    PetscErrorCode CacheAreas();
    bool initialized_ = false;
    MeshIndex vertices_{}, cells_{};
    PetscInt vertexCount_ = 0, cellCount_ = 0, iEdges_ = 0, jEdges_ = 0, width_ = 0;
    MeshRange ownedVertices_{}, ghostVertices_{}, ownedCells_{}, availableCells_{};
    std::vector<Point> coordinates_;
    std::vector<PetscReal> areas_;
    std::vector<PetscInt> ownedEdgeIds_;
};

// Collective on vertexDM's communicator; same DM/Vec contract as mesh.h.
// Updates ghosts itself, validates the mesh, and replaces info only on success.
// Cell ownership follows the owner of the lower-left vertex. A cell DM is not
// required, and no assumption is made about a future solution DM's partition.
// Cached areas include AvailableCells(); reduce ONLY OwnedCells() for totals.
PetscErrorCode BuildMeshInfo(DM vertexDM, Vec vertices, MeshInfo& info);

#endif
