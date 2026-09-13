#include "field_initialization.h"

#include <petscao.h>

#include <algorithm>
#include <exception>
#include <limits>
#include <new>
#include <utility>

namespace {

struct FieldLayout {
    MeshIndex dimensions{};
    MeshRange owned{};
    PetscInt globalSize = 0;
    PetscInt localSize = 0;
};

class TemporaryVector {
public:
    Vec value = nullptr;
    TemporaryVector() = default;
    TemporaryVector(const TemporaryVector&) = delete;
    TemporaryVector& operator=(const TemporaryVector&) = delete;
    ~TemporaryVector() { if (value) (void)VecDestroy(&value); }
};

// Use only around LOCAL operations. All ranks then synchronize the error before
// entering the next collective, even if just one rank had a bad callback/input.
template <typename Operation>
PetscErrorCode GuardLocal(Operation&& operation)
{
    PetscFunctionBeginUser;
    try {
        const PetscErrorCode error = std::forward<Operation>(operation)();
        PetscFunctionReturn(error);
    } catch (const std::bad_alloc&) {
        SETERRQ(PETSC_COMM_SELF, PETSC_ERR_MEM, "Field initialization ran out of memory");
    } catch (const std::exception& error) {
        SETERRQ(PETSC_COMM_SELF, PETSC_ERR_USER,
                "Field initialization raised an exception: %s", error.what());
    } catch (...) {
        SETERRQ(PETSC_COMM_SELF, PETSC_ERR_USER,
                "Field initialization raised an unknown exception");
    }
}

PetscErrorCode SynchronizeError(MPI_Comm comm, PetscErrorCode error, const char* operation)
{
    PetscFunctionBeginUser;
    const int local = static_cast<int>(error);
    int global = 0;
    PetscCallMPI(MPI_Allreduce(&local, &global, 1, MPI_INT, MPI_MAX, comm));
    PetscCheck(global == 0, comm, static_cast<PetscErrorCode>(global),
               "Field initialization failed on at least one rank: %s", operation);
    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode InspectField(DM dm, Vec field, FieldLayout& layout)
{
    PetscFunctionBeginUser;
    PetscBool isDMDA = PETSC_FALSE;
    PetscCall(PetscObjectTypeCompare(reinterpret_cast<PetscObject>(dm), DMDA, &isDMDA));
    PetscCheck(isDMDA, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "cellDM must be a DMDA");
    PetscCheck(field, PETSC_COMM_SELF, PETSC_ERR_ARG_NULL, "field cannot be null");
    const MPI_Comm comm = PetscObjectComm(reinterpret_cast<PetscObject>(dm));
    int comparison = MPI_UNEQUAL;
    PetscCallMPI(MPI_Comm_compare(comm, PetscObjectComm(reinterpret_cast<PetscObject>(field)),
                                  &comparison));
    PetscCheck(comparison == MPI_IDENT || comparison == MPI_CONGRUENT,
               PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG,
               "cellDM and field must use congruent communicators with the same rank order");

    PetscInt dimension = 0, dof = 0;
    PetscCall(DMDAGetInfo(dm, &dimension, &layout.dimensions.i, &layout.dimensions.j,
                          nullptr, nullptr, nullptr, nullptr, &dof, nullptr,
                          nullptr, nullptr, nullptr, nullptr));
    PetscCheck(dimension == 2 && dof == 1, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG,
               "Require a 2-D cell DMDA with one scalar degree of freedom per cell");
    const auto size = layout.dimensions;
    PetscCheck(size.i > 0 && size.j > 0 &&
               size.i <= std::numeric_limits<PetscInt>::max() / size.j,
               PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Invalid cell grid dimensions");
    PetscInt xs, ys, xm, ym;
    PetscCall(DMDAGetCorners(dm, &xs, &ys, nullptr, &xm, &ym, nullptr));
    PetscCheck(xs >= 0 && ys >= 0 && xs <= size.i && ys <= size.j &&
               xm >= 0 && ym >= 0 && xm <= size.i - xs && ym <= size.j - ys,
               PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Invalid cell ownership range");
    layout.owned = {{xs, ys}, {xs + xm, ys + ym}};
    layout.globalSize = size.i * size.j;
    layout.localSize = xm * ym; // Bounded by the validated global grid size.
    PetscInt globalSize, localSize;
    PetscCall(VecGetSize(field, &globalSize));
    PetscCall(VecGetLocalSize(field, &localSize));
    PetscCheck(globalSize == layout.globalSize && localSize == layout.localSize,
               PETSC_COMM_SELF, PETSC_ERR_ARG_SIZ,
               "field must be a global vector with the cell DM's global and local sizes");
    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode CheckVectorLayout(Vec expected, Vec field)
{
    PetscFunctionBeginUser;
    PetscInt expectedBegin, expectedEnd, actualBegin, actualEnd;
    PetscCall(VecGetOwnershipRange(expected, &expectedBegin, &expectedEnd));
    PetscCall(VecGetOwnershipRange(field, &actualBegin, &actualEnd));
    PetscCheck(expectedBegin == actualBegin && expectedEnd == actualEnd,
               PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG,
               "field's algebraic ownership must match a global vector created from cellDM");
    PetscFunctionReturn(PETSC_SUCCESS);
}

struct CellSamples {
    std::vector<PetscInt> indices;
    std::vector<PetscScalar> values;
    std::vector<PetscScalar> ones;
};

PetscErrorCode SampleOwnedCells(const MeshInfo& mesh, const FieldLayout& layout,
                               const CellInitialFunction& profile, PetscInt gaussPoints,
                               CellSamples& samples)
{
    PetscFunctionBeginUser;
    PetscCheck(mesh.IsInitialized(), PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG,
               "BuildMeshInfo must be called before initializing cell averages");
    PetscCheck(mesh.CellDimensions() == layout.dimensions, PETSC_COMM_SELF, PETSC_ERR_ARG_SIZ,
               "Cell DM dimensions must equal the mesh vertex dimensions minus one");
    PetscCheck(static_cast<bool>(profile), PETSC_COMM_SELF, PETSC_ERR_ARG_NULL,
               "The initial-value callback cannot be empty");
    GaussRule1D rule;
    PetscCall(CreateGaussRule(gaussPoints, rule));
    const MeshRange owned = mesh.OwnedCells();
    const auto size = layout.dimensions;
    PetscCheck(owned.begin.i >= 0 && owned.begin.j >= 0 &&
               owned.end.i >= owned.begin.i && owned.end.j >= owned.begin.j &&
               owned.end.i <= size.i && owned.end.j <= size.j,
               PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Invalid MeshInfo cell ownership");
    const auto extent = owned.Size();
    const auto count = static_cast<std::size_t>(extent.i) * static_cast<std::size_t>(extent.j);
    samples.indices.resize(count);
    samples.values.resize(count);
    samples.ones.assign(count, PetscScalar(1));
    std::size_t k = 0;
    for (PetscInt j = owned.begin.j; j < owned.end.j; ++j) {
        for (PetscInt i = owned.begin.i; i < owned.end.i; ++i, ++k) {
            QuadVertices corners;
            PetscReal area = 0, integral = 0;
            PetscCall(mesh.GetCellCorners({i, j}, corners));
            PetscCall(mesh.GetCellArea({i, j}, area));
            PetscCheck(!PetscIsInfOrNanReal(area) && area > 0, PETSC_COMM_SELF,
                       PETSC_ERR_FP, "Cell area must be finite and positive");
            // Do not unwind through IntegrateCell's PETSc stack frame if a
            // user callback throws. Finish that call normally, then report the
            // exception from this frame. The callback is skipped after failure.
            std::exception_ptr callbackException;
            const auto guardedProfile = [&](const Point& point) -> PetscReal {
                if (callbackException) return PetscReal(0);
                try {
                    return profile(point);
                } catch (...) {
                    callbackException = std::current_exception();
                    return PetscReal(0);
                }
            };
            const PetscErrorCode integrationError = IntegrateCell(corners, rule, guardedProfile, integral);
            if (callbackException) {
                try {
                    std::rethrow_exception(callbackException);
                } catch (const std::bad_alloc&) {
                    SETERRQ(PETSC_COMM_SELF, PETSC_ERR_MEM, "Initial-value callback ran out of memory");
                } catch (const std::exception& error) {
                    SETERRQ(PETSC_COMM_SELF, PETSC_ERR_USER,
                            "Initial-value callback failed: %s", error.what());
                } catch (...) {
                    SETERRQ(PETSC_COMM_SELF, PETSC_ERR_USER, "Initial-value callback raised an unknown exception");
                }
            }
            PetscCall(integrationError);
            const PetscReal average = integral / area;
            PetscCheck(!PetscIsInfOrNanReal(average), PETSC_COMM_SELF, PETSC_ERR_FP,
                       "Cell average is not finite");
            samples.indices[k] = j * size.i + i; // Natural, not PETSc algebraic order.
            samples.values[k] = PetscScalar(average);
        }
    }
    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode CheckMappedIndices(const CellSamples& samples, PetscInt globalSize)
{
    PetscFunctionBeginUser;
    for (const auto index : samples.indices)
        PetscCheck(index >= 0 && index < globalSize, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE,
                   "DMDA application ordering returned an invalid cell index");
    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode CheckCoverage(Vec coverage)
{
    PetscFunctionBeginUser;
    PetscInt count;
    PetscCall(VecGetLocalSize(coverage, &count));
    const PetscScalar* data = nullptr;
    PetscCall(VecGetArrayRead(coverage, &data));
    bool valid = true;
    for (PetscInt i = 0; i < count; ++i) valid = valid && data[i] == PetscScalar(1);
    PetscCall(VecRestoreArrayRead(coverage, &data));
    PetscCheck(valid, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG,
               "MeshInfo snapshots must collectively own every physical cell exactly once");
    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode CheckOwnedValues(const std::vector<PetscReal>& values, PetscInt count)
{
    PetscFunctionBeginUser;
    PetscCheck(values.size() == static_cast<std::size_t>(count), PETSC_COMM_SELF,
               PETSC_ERR_ARG_SIZ, "ownedValues must contain exactly xm*ym field-owned entries");
    for (const PetscReal value : values)
        PetscCheck(!PetscIsInfOrNanReal(value), PETSC_COMM_SELF, PETSC_ERR_FP,
                   "ownedValues contains a non-finite value");
    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode CopyOwnedValues(DM dm, Vec destination, const FieldLayout& layout,
                              const std::vector<PetscReal>& values)
{
    PetscFunctionBeginUser;
    PetscScalar** array = nullptr;
    PetscCall(DMDAVecGetArray(dm, destination, &array));
    std::size_t k = 0;
    for (PetscInt j = layout.owned.begin.j; j < layout.owned.end.j; ++j)
        for (PetscInt i = layout.owned.begin.i; i < layout.owned.end.i; ++i, ++k)
            array[j][i] = PetscScalar(values[k]);
    PetscCall(DMDAVecRestoreArray(dm, destination, &array));
    PetscFunctionReturn(PETSC_SUCCESS);
}

} // namespace

PetscErrorCode InitializeCellAverages(DM cellDM, Vec field, const MeshInfo& mesh,
                                     const CellInitialFunction& initialValue,
                                     PetscInt gaussPoints)
{
    PetscFunctionBeginUser;
    PetscCheck(cellDM, PETSC_COMM_SELF, PETSC_ERR_ARG_NULL, "cellDM cannot be null");
    const MPI_Comm comm = PetscObjectComm(reinterpret_cast<PetscObject>(cellDM));
    FieldLayout layout;
    PetscCall(SynchronizeError(comm, InspectField(cellDM, field, layout), "validate cell field"));
    PetscInt minimumPoints, maximumPoints;
    PetscCallMPI(MPI_Allreduce(&gaussPoints, &minimumPoints, 1, MPIU_INT, MPI_MIN, comm));
    PetscCallMPI(MPI_Allreduce(&gaussPoints, &maximumPoints, 1, MPIU_INT, MPI_MAX, comm));
    PetscCheck(minimumPoints == maximumPoints, comm, PETSC_ERR_ARG_WRONG,
               "gaussPoints must agree on every rank");

    CellSamples samples;
    PetscCall(SynchronizeError(comm, GuardLocal([&] {
        return SampleOwnedCells(mesh, layout, initialValue, gaussPoints, samples);
    }), "evaluate initial cell averages"));

    TemporaryVector work;
    PetscCall(DMCreateGlobalVector(cellDM, &work.value));
    PetscCall(SynchronizeError(comm, CheckVectorLayout(work.value, field), "validate vector layout"));
    AO ordering = nullptr;
    PetscCall(DMDAGetAO(cellDM, &ordering));
    // These calls are collective even on ranks owning zero mesh cells.
    const PetscInt count = static_cast<PetscInt>(samples.indices.size());
    PetscInt unusedIndex = 0;
    PetscScalar unusedValue = 0;
    PetscInt* indices = count ? samples.indices.data() : &unusedIndex;
    PetscCall(AOApplicationToPetsc(ordering, count, indices));
    PetscCall(SynchronizeError(comm, CheckMappedIndices(samples, layout.globalSize), "map cell indices"));

    // Check ownership before accepting values. ADD_VALUES detects both missing
    // and duplicate cells, including accidental use of a subset communicator.
    PetscCall(VecSet(work.value, PetscScalar(0)));
    PetscCall(SynchronizeError(comm, VecSetValues(work.value, count, indices,
        count ? samples.ones.data() : &unusedValue, ADD_VALUES), "insert ownership counts"));
    PetscCall(VecAssemblyBegin(work.value));
    PetscCall(VecAssemblyEnd(work.value));
    PetscCall(SynchronizeError(comm, CheckCoverage(work.value), "check mesh ownership coverage"));

    PetscCall(VecSet(work.value, PetscScalar(0)));
    PetscCall(SynchronizeError(comm, VecSetValues(work.value, count, indices,
        count ? samples.values.data() : &unusedValue, INSERT_VALUES), "insert cell averages"));
    PetscCall(VecAssemblyBegin(work.value));
    PetscCall(VecAssemblyEnd(work.value));
    PetscCall(VecCopy(work.value, field));
    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode AssignOwnedCellValues(DM cellDM, Vec field,
                                    const std::vector<PetscReal>& ownedValues)
{
    PetscFunctionBeginUser;
    PetscCheck(cellDM, PETSC_COMM_SELF, PETSC_ERR_ARG_NULL, "cellDM cannot be null");
    const MPI_Comm comm = PetscObjectComm(reinterpret_cast<PetscObject>(cellDM));
    FieldLayout layout;
    PetscCall(SynchronizeError(comm, InspectField(cellDM, field, layout), "validate cell field"));
    PetscCall(SynchronizeError(comm, CheckOwnedValues(ownedValues, layout.localSize), "validate owned values"));
    TemporaryVector work;
    PetscCall(DMCreateGlobalVector(cellDM, &work.value));
    PetscCall(SynchronizeError(comm, CheckVectorLayout(work.value, field), "validate vector layout"));
    PetscCall(SynchronizeError(comm, CopyOwnedValues(cellDM, work.value, layout, ownedValues), "copy owned values"));
    // Direct array writes to the global vector do not require VecAssembly.
    PetscCall(VecCopy(work.value, field));
    PetscFunctionReturn(PETSC_SUCCESS);
}
