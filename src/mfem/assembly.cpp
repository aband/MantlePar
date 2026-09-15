#include "assembly.h"

#include <cstdint>
#include <exception>
#include <limits>
#include <new>
#include <stdexcept>
#include <utility>
#include <vector>

namespace {

// Call only at points reached by ALL ranks. A local error is reported where it
// originates; this agreement also returns an error on all other ranks.
PetscErrorCode AgreeError(MPI_Comm comm, PetscErrorCode local, const char* stage)
{
    PetscFunctionBeginUser;
    const int own = static_cast<int>(local);
    int all = 0;
    PetscCallMPI(MPI_Allreduce(&own, &all, 1, MPI_INT, MPI_MAX, comm));
    PetscCheck(all == 0, comm, static_cast<PetscErrorCode>(all),
               "%s failed on at least one MPI rank; see the originating error", stage);
    PetscFunctionReturn(PETSC_SUCCESS);
}

template <class T>
PetscErrorCode Resize(std::vector<T>& values, std::size_t count)
{
    PetscFunctionBeginUser;
    try {
        values.resize(count);
    } catch (const std::bad_alloc&) {
        SETERRQ(PETSC_COMM_SELF, PETSC_ERR_MEM, "Cannot allocate assembly storage");
    } catch (const std::length_error&) {
        SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_SIZ, "Assembly array is too large");
    }
    PetscFunctionReturn(PETSC_SUCCESS);
}

struct MatrixCOO {
    std::vector<PetscInt> rows, columns;
    std::vector<PetscScalar> values;
};

struct VectorCOO {
    std::vector<PetscInt> rows;
    std::vector<PetscScalar> values;
};

struct MixedCOO {
    MatrixCOO A, B, C;
    VectorCOO f;
};

PetscErrorCode EntryCount(std::size_t cells, std::size_t perCell, std::size_t& count)
{
    PetscFunctionBeginUser;
    PetscCheck(cells <= std::numeric_limits<std::size_t>::max() / perCell,
               PETSC_COMM_SELF, PETSC_ERR_ARG_SIZ, "Assembly entry count overflows size_t");
    count = cells * perCell;
    PetscCheck(static_cast<std::uintmax_t>(count) <=
                   static_cast<std::uintmax_t>(std::numeric_limits<PetscCount>::max()),
               PETSC_COMM_SELF, PETSC_ERR_ARG_SIZ, "Assembly entry count overflows PetscCount");
    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode Allocate(MatrixCOO& data, std::size_t cells, std::size_t perCell)
{
    PetscFunctionBeginUser;
    std::size_t count = 0;
    PetscCall(EntryCount(cells, perCell, count));
    PetscCall(Resize(data.rows, count));
    PetscCall(Resize(data.columns, count));
    PetscCall(Resize(data.values, count));
    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode Allocate(VectorCOO& data, std::size_t cells, std::size_t perCell)
{
    PetscFunctionBeginUser;
    std::size_t count = 0;
    PetscCall(EntryCount(cells, perCell, count));
    PetscCall(Resize(data.rows, count));
    PetscCall(Resize(data.values, count));
    PetscFunctionReturn(PETSC_SUCCESS);
}

std::size_t OwnedCellCount(const MeshInfo& mesh)
{
    const auto size = mesh.OwnedCells().Size();
    return static_cast<std::size_t>(size.i) * static_cast<std::size_t>(size.j);
}

// MeshInfo and DofMap retain no communicator, so validate both local entity
// ownership and rank-contiguous layouts rather than using a coordinate Vec's
// layout. The corner-cell query also detects incompatible logical dimensions.
PetscErrorCode CheckMap(const MeshInfo& mesh, const DofMap& map, DofSpace space)
{
    PetscFunctionBeginUser;
    PetscCheck(mesh.IsInitialized() && map.IsInitialized(), PETSC_COMM_SELF,
               PETSC_ERR_ARG_WRONG, "Initialize MeshInfo and DofMap before assembly");
    PetscCheck(map.Space() == space, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG,
               "Wrong finite-element space for an assembly map");
    const auto maximum = std::numeric_limits<PetscInt>::max();
    const auto nv = mesh.VertexCount(), ne = mesh.EdgeCount();
    const auto vertices = mesh.OwnedVertices().Size();
    PetscInt global = mesh.CellCount();
    PetscInt owned = static_cast<PetscInt>(OwnedCellCount(mesh));
    PetscInt element = 1;
    if (space == DofSpace::BRVelocity) {
        PetscCheck(nv <= (maximum-ne)/2, PETSC_COMM_SELF, PETSC_ERR_ARG_SIZ,
                   "BR DOF count overflows PetscInt");
        global = 2*nv + ne;
        owned = 2*(vertices.i*vertices.j) + static_cast<PetscInt>(mesh.OwnedEdgeIds().size());
        element = BRMixed::ElementDofs;
    } else if (space == DofSpace::HDivVelocity) {
        PetscCheck(ne <= maximum/2, PETSC_COMM_SELF, PETSC_ERR_ARG_SIZ,
                   "HDiv DOF count overflows PetscInt");
        global = 2*ne;
        owned = 2*static_cast<PetscInt>(mesh.OwnedEdgeIds().size());
        element = HDivMixed::ElementDofs;
    }
    PetscCheck(map.GlobalDofs() == global && map.OwnedDofs() == owned &&
                   map.ElementDofs() == element,
               PETSC_COMM_SELF, PETSC_ERR_ARG_SIZ, "DofMap counts do not match MeshInfo");
    const auto cells = mesh.CellDimensions();
    std::vector<PetscInt> corner;
    PetscCall(map.GetCellNaturalDofs({cells.i-1, cells.j-1}, corner));
    for (PetscInt id = map.OwnershipBegin(); id < map.OwnershipEnd(); ++id) {
        DofInfo info;
        PetscCall(map.GetDofInfo(id, info));
        bool owns = false;
        if (info.entity == DofEntity::Vertex) {
            MeshIndex vertex;
            PetscCall(mesh.VertexIndex(info.entityId, vertex));
            owns = mesh.OwnedVertices().Contains(vertex);
        } else if (info.entity == DofEntity::Edge) {
            owns = mesh.OwnsEdge(info.entityId);
        } else {
            MeshIndex cell;
            PetscCall(mesh.CellIndex(info.entityId, cell));
            owns = mesh.OwnsCell(cell);
        }
        PetscCheck(owns, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG,
                   "DofMap entity ownership differs from MeshInfo; rebuild after repartitioning");
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
                         "DOF layout allocation"));
    const PetscInt own[3] = {map.OwnershipBegin(), map.OwnershipEnd(), map.GlobalDofs()};
    PetscCallMPI(MPI_Allgather(own, 3, MPIU_INT, all.data(), 3, MPIU_INT, comm));
    PetscInt end = 0;
    for (PetscMPIInt r = 0; r < ranks; ++r) {
        const auto offset = static_cast<std::size_t>(r)*3;
        PetscCheck(all[offset] == end && all[offset+1] >= end &&
                       all[offset+2] == all[2] && all[offset+1] <= all[2],
                   comm, PETSC_ERR_ARG_WRONG,
                   "DofMap layout is incompatible with this communicator/rank ordering");
        end = all[offset+1];
    }
    PetscCheck(end == all[2], comm, PETSC_ERR_ARG_SIZ,
               "Owned DOF ranges do not cover the global space");
    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode ReadPorosity(const CellPorosityFunction& provider, MeshIndex cell,
                             LocalPorositySamples& samples)
{
    PetscFunctionBeginUser;
    PetscErrorCode error = PETSC_SUCCESS;
    try {
        error = provider(cell, samples);
    } catch (const std::bad_alloc&) {
        SETERRQ(PETSC_COMM_SELF, PETSC_ERR_MEM, "Porosity callback allocation failed");
    } catch (const std::exception& exception) {
        SETERRQ(PETSC_COMM_SELF, PETSC_ERR_USER,
                "Porosity callback threw in cell (%" PetscInt_FMT ",%" PetscInt_FMT "): %s",
                cell.i, cell.j, exception.what());
    } catch (...) {
        SETERRQ(PETSC_COMM_SELF, PETSC_ERR_USER, "Porosity callback threw an unknown exception");
    }
    PetscCall(error);
    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode CheckMixedInputs(const MeshInfo& mesh, const DofMap& velocity,
                                const DofMap& pressure, DofSpace space,
                                const GaussRule1D& cellRule,
                                const CellPorosityFunction& porosity,
                                const MixedBlocks& result)
{
    PetscFunctionBeginUser;
    PetscCheck(result.IsEmpty(), PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG,
               "Assembly output is not empty; call DestroyMixedBlocks first");
    PetscCheck(static_cast<bool>(porosity), PETSC_COMM_SELF, PETSC_ERR_ARG_NULL,
               "Supply a cell porosity callback");
    PetscCall(CheckMap(mesh, velocity, space));
    PetscCall(CheckMap(mesh, pressure, DofSpace::CellPressure));
    PetscCall(ValidateGaussRule(cellRule));
    PetscFunctionReturn(PETSC_SUCCESS);
}

template <class Basis, class Compute>
PetscErrorCode BuildMixedCOO(const MeshInfo& mesh, const DofMap& velocity,
                            const DofMap& pressure, const CellPorosityFunction& porosity,
                            const Compute& compute, MixedCOO& data)
{
    PetscFunctionBeginUser;
    constexpr std::size_t n = static_cast<std::size_t>(Basis::ElementDofs);
    const auto cells = OwnedCellCount(mesh);
    PetscCall(Allocate(data.A, cells, n*n));
    PetscCall(Allocate(data.B, cells, n));
    PetscCall(Allocate(data.C, cells, 1));
    PetscCall(Allocate(data.f, cells, n));
    const auto range = mesh.OwnedCells();
    std::size_t k = 0;
    std::vector<PetscInt> u, p;
    for (PetscInt j = range.begin.j; j < range.end.j; ++j) {
        for (PetscInt i = range.begin.i; i < range.end.i; ++i, ++k) {
            const MeshIndex cell{i,j};
            Basis basis;
            PetscCall(basis.Initialize(mesh, cell));
            LocalPorositySamples samples;
            PetscCall(ReadPorosity(porosity, cell, samples));
            LocalMatrixBlock<n> local;
            PetscCall(compute(basis, samples, local));
            PetscCall(velocity.GetCellGlobalDofs(cell, u));
            PetscCall(pressure.GetCellGlobalDofs(cell, p));
            PetscCheck(u.size() == n && p.size() == 1, PETSC_COMM_SELF,
                       PETSC_ERR_PLIB, "Unexpected element DOF count");
            for (std::size_t row = 0; row < n; ++row) {
                for (std::size_t column = 0; column < n; ++column) {
                    const auto q = k*n*n + row*n + column;
                    data.A.rows[q] = u[row];
                    data.A.columns[q] = u[column];
                    data.A.values[q] = local.A[row*n+column];
                }
                const auto q = k*n + row;
                data.B.rows[q] = p[0];
                data.B.columns[q] = u[row];
                data.B.values[q] = local.B[row];
                data.f.rows[q] = u[row];
                data.f.values[q] = local.f[row];
            }
            data.C.rows[k] = p[0];
            data.C.columns[k] = p[0];
            data.C.values[k] = local.C;
        }
    }
    PetscFunctionReturn(PETSC_SUCCESS);
}

// COO permits duplicate and remote entries. INSERT_VALUES sums duplicates
// before replacing the matrix, so this is full additive element assembly.
// PETSc may overwrite the COO index arrays; each block has its own arrays.
PetscErrorCode CreateMatrix(MPI_Comm comm, const DofMap& rows, const DofMap& columns,
                            MatrixCOO& data, Mat& matrix)
{
    PetscFunctionBeginUser;
    PetscCall(AgreeError(comm, MatCreate(comm, &matrix), "MatCreate"));
    PetscCall(AgreeError(comm, MatSetSizes(matrix, rows.OwnedDofs(), columns.OwnedDofs(),
                                          rows.GlobalDofs(), columns.GlobalDofs()), "MatSetSizes"));
    PetscCall(AgreeError(comm, MatSetType(matrix, MATAIJ), "MatSetType"));
    PetscCall(AgreeError(comm, MatSetPreallocationCOO(matrix,
                         static_cast<PetscCount>(data.values.size()), data.rows.data(),
                         data.columns.data()), "Matrix COO preallocation"));
    PetscCall(AgreeError(comm, MatSetValuesCOO(matrix, data.values.data(), INSERT_VALUES),
                         "Matrix COO values"));
    // MatSetValuesCOO performs final assembly, including on empty cell owners.
    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode CreateBlocks(MPI_Comm comm, const DofMap& velocity,
                            const DofMap& pressure, MixedCOO& data, MixedBlocks& work)
{
    PetscFunctionBeginUser;
    PetscCall(CreateMatrix(comm, velocity, velocity, data.A, work.A));
    PetscCall(CreateMatrix(comm, pressure, velocity, data.B, work.B));
    PetscCall(CreateMatrix(comm, pressure, pressure, data.C, work.C));
    PetscCall(AgreeError(comm, VecCreateMPI(comm, velocity.OwnedDofs(), velocity.GlobalDofs(),
                                           &work.f), "Velocity Vec creation"));
    PetscCall(AgreeError(comm, VecSetPreallocationCOO(work.f,
                         static_cast<PetscCount>(data.f.values.size()), data.f.rows.data()),
                         "Velocity load COO preallocation"));
    PetscCall(AgreeError(comm, VecSetValuesCOO(work.f, data.f.values.data(), INSERT_VALUES),
                         "Velocity load assembly"));
    PetscCall(AgreeError(comm, VecCreateMPI(comm, pressure.OwnedDofs(), pressure.GlobalDofs(),
                                           &work.g), "Pressure Vec creation"));
    PetscCall(AgreeError(comm, VecSet(work.g, 0), "Pressure load initialization"));
    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode CommitBlocks(MPI_Comm comm, const DofMap& velocity,
                            const DofMap& pressure, MixedCOO& data, MixedBlocks& result)
{
    PetscFunctionBeginUser;
    MixedBlocks work;
    const PetscErrorCode error = CreateBlocks(comm, velocity, pressure, data, work);
    if (error) {
        const PetscErrorCode cleanup = DestroyMixedBlocks(work);
        (void)cleanup; // Preserve the originating error after attempting cleanup.
        PetscCall(error);
    }
    std::swap(result.A, work.A);
    std::swap(result.B, work.B);
    std::swap(result.C, work.C);
    std::swap(result.f, work.f);
    std::swap(result.g, work.g);
    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode CheckCouplingInputs(const MeshInfo& mesh, const DofMap& stokes,
                                   const DofMap& darcy, const GaussRule1D& rule,
                                   const CellPorosityFunction& porosity, Mat output)
{
    PetscFunctionBeginUser;
    PetscCheck(!output, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG,
               "Pressure coupling output must be nullptr");
    PetscCheck(static_cast<bool>(porosity), PETSC_COMM_SELF, PETSC_ERR_ARG_NULL,
               "Supply a cell porosity callback");
    PetscCall(CheckMap(mesh, stokes, DofSpace::CellPressure));
    PetscCall(CheckMap(mesh, darcy, DofSpace::CellPressure));
    PetscCall(ValidateGaussRule(rule));
    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode BuildCouplingCOO(const MeshInfo& mesh, const DofMap& stokes,
                               const DofMap& darcy, const GaussRule1D& rule,
                               const CellPorosityFunction& porosity,
                               const LocalMatrixParameters& parameters, MatrixCOO& data)
{
    PetscFunctionBeginUser;
    PetscCall(Allocate(data, OwnedCellCount(mesh), 1));
    const auto range = mesh.OwnedCells();
    std::size_t k = 0;
    std::vector<PetscInt> s, d;
    for (PetscInt j = range.begin.j; j < range.end.j; ++j) {
        for (PetscInt i = range.begin.i; i < range.end.i; ++i, ++k) {
            const MeshIndex cell{i,j};
            QuadBasis geometry;
            PetscCall(geometry.Initialize(mesh, cell));
            LocalPorositySamples samples;
            PetscCall(ReadPorosity(porosity, cell, samples));
            PetscCall(ComputeLocalCoupling(geometry, rule, samples, parameters, data.values[k]));
            PetscCall(stokes.GetCellGlobalDofs(cell, s));
            PetscCall(darcy.GetCellGlobalDofs(cell, d));
            PetscCheck(s.size() == 1 && d.size() == 1, PETSC_COMM_SELF,
                       PETSC_ERR_PLIB, "Unexpected pressure element DOF count");
            data.rows[k] = s[0];
            data.columns[k] = d[0];
        }
    }
    PetscFunctionReturn(PETSC_SUCCESS);
}

} // namespace

PetscErrorCode DestroyMixedBlocks(MixedBlocks& blocks)
{
    PetscFunctionBeginUser;
    PetscErrorCode first = PETSC_SUCCESS;
    const auto remember = [&first](PetscErrorCode error) {
        if (first == PETSC_SUCCESS) first = error;
    };
    remember(MatDestroy(&blocks.A));
    remember(MatDestroy(&blocks.B));
    remember(MatDestroy(&blocks.C));
    remember(VecDestroy(&blocks.f));
    remember(VecDestroy(&blocks.g));
    PetscCall(first);
    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode AssembleStokesBlocks(
    MPI_Comm comm, const MeshInfo& mesh,
    const DofMap& velocityMap, const DofMap& pressureMap,
    const GaussRule1D& cellRule, const CellPorosityFunction& porosity,
    const LocalForceFunction& force, MixedBlocks& result)
{
    PetscFunctionBeginUser;
    PetscCall(AgreeError(comm, CheckMixedInputs(mesh, velocityMap, pressureMap,
                         DofSpace::BRVelocity, cellRule, porosity, result), "Stokes inputs"));
    PetscCall(CheckLayout(comm, velocityMap));
    PetscCall(CheckLayout(comm, pressureMap));
    MixedCOO data;
    const auto compute = [&](const BRMixed& basis, const LocalPorositySamples& samples,
                              StokesLocalMatrix& local) {
        return ComputeLocalStokes(basis, cellRule, samples, force, local);
    };
    PetscCall(AgreeError(comm, BuildMixedCOO<BRMixed>(mesh, velocityMap, pressureMap,
                         porosity, compute, data), "Stokes element evaluation"));
    PetscCall(CommitBlocks(comm, velocityMap, pressureMap, data, result));
    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode AssembleDarcyBlocks(
    MPI_Comm comm, const MeshInfo& mesh,
    const DofMap& velocityMap, const DofMap& pressureMap,
    const GaussRule1D& cellRule, const GaussRule1D& edgeRule,
    const CellPorosityFunction& porosity, const LocalMatrixParameters& parameters,
    const LocalForceFunction& force, MixedBlocks& result)
{
    PetscFunctionBeginUser;
    PetscCall(AgreeError(comm, CheckMixedInputs(mesh, velocityMap, pressureMap,
                         DofSpace::HDivVelocity, cellRule, porosity, result), "Darcy inputs"));
    PetscCall(AgreeError(comm, ValidateGaussRule(edgeRule), "Darcy edge quadrature"));
    PetscCall(CheckLayout(comm, velocityMap));
    PetscCall(CheckLayout(comm, pressureMap));
    MixedCOO data;
    const auto compute = [&](const HDivMixed& basis, const LocalPorositySamples& samples,
                              DarcyLocalMatrix& local) {
        return ComputeLocalDarcy(basis, cellRule, edgeRule, samples, parameters, force, local);
    };
    PetscCall(AgreeError(comm, BuildMixedCOO<HDivMixed>(mesh, velocityMap, pressureMap,
                         porosity, compute, data), "Darcy element evaluation"));
    PetscCall(CommitBlocks(comm, velocityMap, pressureMap, data, result));
    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode AssemblePressureCoupling(
    MPI_Comm comm, const MeshInfo& mesh,
    const DofMap& stokesPressureMap, const DofMap& darcyPressureMap,
    const GaussRule1D& cellRule, const CellPorosityFunction& porosity,
    const LocalMatrixParameters& parameters, Mat& K_sd)
{
    PetscFunctionBeginUser;
    PetscCall(AgreeError(comm, CheckCouplingInputs(mesh, stokesPressureMap, darcyPressureMap,
                         cellRule, porosity, K_sd), "Pressure coupling inputs"));
    PetscCall(CheckLayout(comm, stokesPressureMap));
    PetscCall(CheckLayout(comm, darcyPressureMap));
    MatrixCOO data;
    PetscCall(AgreeError(comm, BuildCouplingCOO(mesh, stokesPressureMap, darcyPressureMap,
                         cellRule, porosity, parameters, data), "Pressure coupling evaluation"));
    Mat work = nullptr;
    const PetscErrorCode error = CreateMatrix(comm, stokesPressureMap, darcyPressureMap, data, work);
    if (error) {
        const PetscErrorCode cleanup = MatDestroy(&work);
        (void)cleanup;
        PetscCall(error);
    }
    K_sd = work;
    PetscFunctionReturn(PETSC_SUCCESS);
}
