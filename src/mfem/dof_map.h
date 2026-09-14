#ifndef MANTLE_MFEM_DOF_MAP_H
#define MANTLE_MFEM_DOF_MAP_H

#include "mesh_info.h"

#include <array>
#include <vector>

enum class DofSpace { BRVelocity = 0, HDivVelocity = 1, CellPressure = 2 };
enum class DofEntity { Vertex, Edge, Cell };

struct DofInfo {
    DofEntity entity = DofEntity::Cell;
    PetscInt entityId = 0; // MeshInfo's natural vertex, edge, or cell ID.
    // Vertex: 0=x, 1=y. HDiv edge: 0=linear, 1=constant.
    // BR edge and P0 cell: 0.
    PetscInt component = 0;
    // Geometric boundary membership in CellSide order: Bottom,Right,Top,Left.
    // A corner vertex has two true entries. P0 cell DOFs have none.
    // These flags do not select essential/natural boundary conditions.
    std::array<bool, 4> boundarySides{};
    bool IsBoundary() const noexcept
    { return boundarySides[0] || boundarySides[1] || boundarySides[2] || boundarySides[3]; }
};

// Numbering for ONE finite-element space on a nonperiodic logical rectangle.
// Supports rectangular and perturbed quadrilateral geometry identically.
//
// Three distinct index systems, each starting at zero:
//   Natural: partition-independent IDs, useful for output/restarts.
//     BR   = [vertex x, vertex y, edges], count 2*Nv+Ne.
//     HDiv = [linear edges, constant edges], count 2*Ne.
//     P0   = [cells], count Nc.
//     Entity IDs use MeshInfo ordering (AlongI edges before AlongJ).
//   Global: PETSc algebraic IDs, contiguous within each MPI rank. Each rank
//     lists its owned DOFs in increasing NATURAL order. Changes on repartition.
//   Local: owned global IDs in increasing order, followed by GhostGlobalIds().
//     Includes DOFs on all locally available vertices, complete edges and cells.
//
// Ownership follows MeshInfo: vertices belong to the vertex DMDA owner; an
// edge to its canonical START vertex owner; a cell to its lower-left vertex
// owner. Ranks with no cells and ranks with no DOFs are valid.
//
// Element ordering matches BRMixed/HDivMixed exactly. Their shared-edge signs
// are already in the basis values: DO NOT apply another orientation factor.
// Each velocity/pressure map has its own independent global numbering; combine
// spaces later using an explicit block layout. These indices are not the
// coordinate DM's vector indices or an unrelated solution DM's indices.
//
// Owns topology/partition metadata, no DM, Vec, communicator, or mesh pointers.
// Copies are independent. Initialization is collective; all subsequent queries
// are noncollective. Rebuild after repartitioning/topology changes. Moving only
// coordinates, with topology and partition unchanged, does not change the map.
// Failed initialization preserves the previous map on every rank; checked
// query outputs are assigned only on success. Use after PetscInitialize.
class DofMap {
public:
    DofMap() = default;
    DofMap(const DofMap&) = default;
    DofMap& operator=(const DofMap&) = default;
    DofMap(DofMap&& other) noexcept;
    DofMap& operator=(DofMap&& other) noexcept;

    // Collective on vertexDM's communicator. Pass a valid, set-up vertex DMDA
    // on every rank, satisfying mesh.h, and the matching BuildMeshInfo snapshot.
    // All ranks must select the same space. Only partition metadata is gathered;
    // no array of all global DOFs is constructed. Storage O(ranks + local ghosts).
    PetscErrorCode Initialize(DM vertexDM, const MeshInfo& mesh, DofSpace space);

    bool IsInitialized() const noexcept { return initialized_; }
    DofSpace Space() const noexcept { return space_; }
    PetscInt ElementDofs() const noexcept { return elementDofs_; }
    PetscInt GlobalDofs() const noexcept { return globalDofs_; }
    PetscInt OwnedDofs() const noexcept { return ownedEnd_ - ownedBegin_; }
    PetscInt GhostDofs() const noexcept { return static_cast<PetscInt>(ghosts_.size()); }
    PetscInt LocalDofs() const noexcept { return OwnedDofs() + GhostDofs(); }
    PetscInt OwnershipBegin() const noexcept { return ownedBegin_; }
    PetscInt OwnershipEnd() const noexcept { return ownedEnd_; } // Exclusive.
    bool OwnsGlobal(PetscInt id) const noexcept
    { return initialized_ && id >= ownedBegin_ && id < ownedEnd_; }
    const std::vector<PetscInt>& GhostGlobalIds() const noexcept { return ghosts_; }

    // Natural/global queries can address ANY valid cell or DOF in the mesh.
    // Cell-local queries require mesh.AvailableCells() from initialization.
    PetscErrorCode GetCellNaturalDofs(MeshIndex cell, std::vector<PetscInt>& ids) const;
    PetscErrorCode GetCellGlobalDofs(MeshIndex cell, std::vector<PetscInt>& ids) const;
    PetscErrorCode GetCellLocalDofs(MeshIndex cell, std::vector<PetscInt>& ids) const;
    PetscErrorCode NaturalToGlobal(PetscInt naturalId, PetscInt& globalId) const;
    PetscErrorCode GlobalToNatural(PetscInt globalId, PetscInt& naturalId) const;
    PetscErrorCode LocalToGlobal(PetscInt localId, PetscInt& globalId) const;
    // Fails if the global DOF is neither owned nor ghosted on this rank.
    PetscErrorCode GlobalToLocal(PetscInt globalId, PetscInt& localId) const;
    PetscErrorCode GetOwnerRank(PetscInt globalId, PetscMPIInt& rank) const;
    PetscErrorCode GetDofInfo(PetscInt globalId, DofInfo& info) const;

    // For later vector construction, on the SAME communicator/rank order:
    // VecCreateGhost(comm, map.OwnedDofs(), map.GlobalDofs(), map.GhostDofs(),
    //                map.GhostGlobalIds().data(), &v);
    // LocalToGlobal/GetCellLocalDofs then index VecGhostGetLocalForm(v).
    // Refresh ghost values with VecGhostUpdateBegin/End before reading them.
    // Use GetCellGlobalDofs with MatSetValues/VecSetValues. Set matrix local
    // row/column sizes from the corresponding maps' OwnedDofs().

private:
    struct FieldBlock {
        DofEntity entity = DofEntity::Cell;
        PetscInt component = 0;
        EdgeAxis axis = EdgeAxis::AlongI; // Only meaningful for edges.
        MeshIndex dimensions{};
        PetscInt begin = 0, end = 0; // Natural ID interval.
    };

    void Swap(DofMap& other) noexcept;
    PetscErrorCode CheckInitialized() const;
    PetscErrorCode ReadLocal(DM dm, const MeshInfo& mesh, DofSpace space);
    PetscErrorCode BuildNumbering(const std::vector<PetscInt>& rectangles);
    PetscErrorCode BuildGhosts();
    PetscErrorCode AddBlock(DofEntity entity, PetscInt component,
                            EdgeAxis axis, MeshIndex dimensions);
    PetscMPIInt VertexOwner(MeshIndex vertex) const;
    MeshRange BlockRange(const FieldBlock& block, MeshRange vertices) const;

    bool initialized_ = false;
    DofSpace space_ = DofSpace::CellPressure;
    PetscMPIInt rank_ = 0, ranks_ = 0;
    MeshIndex vertices_{}, cells_{};
    MeshRange ownedVertices_{}, ghostVertices_{}, availableCells_{};
    PetscInt vertexCount_ = 0, iEdgeCount_ = 0, edgeCount_ = 0;
    PetscInt elementDofs_ = 0, globalDofs_ = 0, ownedBegin_ = 0, ownedEnd_ = 0;
    std::vector<FieldBlock> blocks_;
    std::vector<MeshRange> partitions_;
    std::vector<PetscInt> xCuts_, yCuts_, rankOffsets_, ghosts_;
    std::vector<PetscMPIInt> rankAtBlock_;
};

#endif
