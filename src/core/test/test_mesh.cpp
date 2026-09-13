#include "mesh.h"

#include <algorithm>
#include <array>
#include <limits>

namespace {

using Generator = PetscErrorCode (*)(DM, Vec, const MeshParam&);
enum class Shape { Uniform, Perturbed, Stretched };

PetscErrorCode RequireAll(bool condition, MPI_Comm comm, const char* message)
{
    PetscMPIInt local = condition ? 1 : 0, global = 0;
    PetscFunctionBeginUser;
    PetscCallMPI(MPI_Allreduce(&local, &global, 1, MPI_INT, MPI_MIN, comm));
    PetscCheck(global, comm, PETSC_ERR_PLIB, "%s", message);
    PetscFunctionReturn(0);
}

PetscErrorCode MakeDM(MPI_Comm comm, PetscInt M, PetscInt N,
                      PetscInt px, PetscInt py, DM* dm)
{
    PetscFunctionBeginUser;
    PetscCall(DMDACreate2d(comm, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE,
                          DMDA_STENCIL_BOX, M, N, px, py, 2, 1,
                          nullptr, nullptr, dm));
    PetscCall(DMSetUp(*dm));
    PetscFunctionReturn(0);
}

PetscErrorCode CheckCase(PetscInt M, PetscInt N, PetscInt px, PetscInt py,
                         const MeshParam& mp, Generator generate, Shape shape)
{
    DM dm = nullptr, serialDM = nullptr;
    Vec vertices = nullptr, reference = nullptr, snapshot = nullptr, local = nullptr;
    const PetscScalar*** a = nullptr;
    const PetscScalar*** b = nullptr;
    PetscInt xs, ys, xm, ym;
    const MPI_Comm comm = PETSC_COMM_WORLD;
    PetscFunctionBeginUser;
    PetscCall(MakeDM(comm, M, N, px, py, &dm));
    PetscCall(DMCreateGlobalVector(dm, &vertices));
    PetscCall(VecSet(vertices, 12345.0));
    PetscCall(generate(dm, vertices, mp));

    // An independent serial partition on each rank exposes any dependence
    // of the generated coordinates on MPI rank, ownership, or traversal.
    PetscCall(MakeDM(PETSC_COMM_SELF, M, N, 1, 1, &serialDM));
    PetscCall(DMCreateGlobalVector(serialDM, &reference));
    PetscCall(generate(serialDM, reference, mp));
    PetscCall(DMDAGetCorners(dm, &xs, &ys, nullptr, &xm, &ym, nullptr));
    PetscCall(DMDAVecGetArrayDOFRead(dm, vertices, &a));
    PetscCall(DMDAVecGetArrayDOFRead(serialDM, reference, &b));
    const PetscReal hx = mp.L / (M - 1), hy = mp.H / (N - 1);
    bool same = true, bounds = true;
    PetscMPIInt changed = 0, anyChanged = 0;
    for (PetscInt j = ys; j < ys + ym; ++j) {
        for (PetscInt i = xs; i < xs + xm; ++i) {
            const PetscReal x = PetscRealPart(a[j][i][0]);
            const PetscReal y = PetscRealPart(a[j][i][1]);
            same = same && a[j][i][0] == b[j][i][0]
                        && a[j][i][1] == b[j][i][1];
            if (i == 0) bounds = bounds && x == mp.xstart;
            if (i == M - 1) bounds = bounds && x == mp.xstart + mp.L;
            if (j == 0) bounds = bounds && y == mp.ystart;
            if (j == N - 1) bounds = bounds && y == mp.ystart + mp.H;
            if (shape == Shape::Stretched) continue;

            const PetscReal ux = i == M - 1 ? mp.xstart + mp.L : mp.xstart + i * hx;
            const PetscReal uy = j == N - 1 ? mp.ystart + mp.H : mp.ystart + j * hy;
            const bool boundary = i == 0 || j == 0 || i == M - 1 || j == N - 1;
            if (shape == Shape::Uniform || boundary) {
                bounds = bounds && x == ux && y == uy;
            } else {
                const PetscReal tol = 128.0 * PETSC_MACHINE_EPSILON
                                    * (1.0 + PetscAbsReal(ux) + PetscAbsReal(uy));
                bounds = bounds && PetscAbsReal(x - ux) <= mp.perturbation * hx + tol
                                && PetscAbsReal(y - uy) <= mp.perturbation * hy + tol;
                if (x != ux || y != uy) changed = 1;
            }
        }
    }
    PetscCall(DMDAVecRestoreArrayDOFRead(serialDM, reference, &b));
    PetscCall(DMDAVecRestoreArrayDOFRead(dm, vertices, &a));
    PetscCall(RequireAll(same, comm, "MPI coordinates differ from the serial mesh"));
    PetscCall(RequireAll(bounds, comm, "Wrong coordinates, boundary, or perturbation scale"));
    PetscCallMPI(MPI_Allreduce(&changed, &anyChanged, 1, MPI_INT, MPI_MAX, comm));
    if (shape == Shape::Perturbed && mp.perturbation > 0 && M > 2 && N > 2) {
        PetscCall(RequireAll(anyChanged != 0, comm, "Interior vertices were not perturbed"));
    }

    PetscCall(VecDuplicate(vertices, &snapshot));
    PetscCall(VecCopy(vertices, snapshot));
    PetscCall(generate(dm, vertices, mp));
    PetscBool equal = PETSC_FALSE;
    PetscCall(VecEqual(vertices, snapshot, &equal));
    PetscCall(RequireAll(equal == PETSC_TRUE, comm, "Generation is not repeatable"));
    if (shape == Shape::Perturbed && M > 2 && N > 2) {
        MeshParam different = mp;
        ++different.seed;
        PetscCall(generate(dm, vertices, different));
        PetscCall(VecEqual(vertices, snapshot, &equal));
        PetscCall(RequireAll(equal == PETSC_FALSE, comm, "Seed did not change the mesh"));
        PetscCall(VecCopy(snapshot, vertices));
    }

    // Sum signed polygon areas, independently of the validator's corner tests.
    PetscCall(DMGetLocalVector(dm, &local));
    PetscCall(DMGlobalToLocalBegin(dm, vertices, INSERT_VALUES, local));
    PetscCall(DMGlobalToLocalEnd(dm, vertices, INSERT_VALUES, local));
    PetscCall(DMDAVecGetArrayDOFRead(dm, local, &a));
    PetscReal localArea = 0.0, totalArea = 0.0;
    bool positiveAreas = true;
    for (PetscInt j = ys; j < std::min(ys + ym, N - 1); ++j) {
        for (PetscInt i = xs; i < std::min(xs + xm, M - 1); ++i) {
            const PetscReal x0 = PetscRealPart(a[j][i][0]);
            const PetscReal y0 = PetscRealPart(a[j][i][1]);
            const PetscReal x1 = PetscRealPart(a[j][i + 1][0]) - x0;
            const PetscReal y1 = PetscRealPart(a[j][i + 1][1]) - y0;
            const PetscReal x2 = PetscRealPart(a[j + 1][i + 1][0]) - x0;
            const PetscReal y2 = PetscRealPart(a[j + 1][i + 1][1]) - y0;
            const PetscReal x3 = PetscRealPart(a[j + 1][i][0]) - x0;
            const PetscReal y3 = PetscRealPart(a[j + 1][i][1]) - y0;
            const PetscReal area = 0.5 * (x1 * y2 - y1 * x2 + x2 * y3 - y2 * x3);
            positiveAreas = positiveAreas && area > 0.0;
            localArea += area;
        }
    }
    PetscCall(DMDAVecRestoreArrayDOFRead(dm, local, &a));
    PetscCall(DMRestoreLocalVector(dm, &local));
    PetscCallMPI(MPI_Allreduce(&localArea, &totalArea, 1, MPIU_REAL, MPI_SUM, comm));
    const PetscReal areaTolerance = 2048.0 * PETSC_MACHINE_EPSILON * mp.L * mp.H;
    PetscCall(RequireAll(positiveAreas && PetscAbsReal(totalArea - mp.L * mp.H)
                                        <= areaTolerance,
                         comm, "Cell areas are invalid or do not sum to the domain area"));
    PetscCall(VecDestroy(&snapshot));
    PetscCall(VecDestroy(&reference));
    PetscCall(VecDestroy(&vertices));
    PetscCall(DMDestroy(&serialDM));
    PetscCall(DMDestroy(&dm));
    PetscFunctionReturn(0);
}

PetscErrorCode CheckErrors(PetscInt px, PetscInt py)
{
    DM dm = nullptr;
    Vec vertices = nullptr;
    PetscScalar*** a = nullptr;
    PetscInt xs, ys, xm, ym;
    MeshParam mp;
    const MPI_Comm comm = PETSC_COMM_WORLD;
    PetscFunctionBeginUser;
    PetscCall(MakeDM(comm, 12, 8, px, py, &dm));
    PetscCall(DMCreateGlobalVector(dm, &vertices));
    PetscCall(PetscPushErrorHandler(PetscReturnErrorHandler, nullptr));
    mp.L = -1.0;
    const PetscErrorCode badLength = CreateFullMesh(dm, vertices, mp);
    mp.L = 1.0;
    mp.perturbation = 0.25;
    const PetscErrorCode badAmplitude = LogicRectMesh(dm, vertices, mp);
    mp.perturbation = std::numeric_limits<PetscReal>::quiet_NaN();
    const PetscErrorCode nanAmplitude = LogicRectMesh(dm, vertices, mp);
    PetscCall(PetscPopErrorHandler());
    PetscCall(RequireAll(badLength == PETSC_ERR_ARG_OUTOFRANGE &&
                         badAmplitude == PETSC_ERR_ARG_OUTOFRANGE &&
                         nanAmplitude == PETSC_ERR_ARG_OUTOFRANGE,
                         comm, "Invalid parameters were not rejected"));

    mp.perturbation = 0.15;
    PetscCall(CreateFullMesh(dm, vertices, mp));
    PetscCall(DMDAGetCorners(dm, &xs, &ys, nullptr, &xm, &ym, nullptr));
    PetscCall(DMDAVecGetArrayDOF(dm, vertices, &a));
    // Only one rank changes this vertex. Every rank must receive the error.
    if (xs <= 1 && 1 < xs + xm && ys <= 1 && 1 < ys + ym) {
        a[1][1][0] = mp.xstart + mp.L / 11.0;
        a[1][1][1] = mp.ystart;
    }
    PetscCall(DMDAVecRestoreArrayDOF(dm, vertices, &a));
    PetscCall(PetscPushErrorHandler(PetscReturnErrorHandler, nullptr));
    const PetscErrorCode collapsed = ValidateMesh(dm, vertices);
    PetscCall(PetscPopErrorHandler());
    PetscCall(RequireAll(collapsed == PETSC_ERR_ARG_WRONG, comm,
                         "A collapsed cell was not rejected on every rank"));
    PetscCall(VecDestroy(&vertices));
    PetscCall(DMDestroy(&dm));
    PetscFunctionReturn(0);
}

PetscErrorCode RunTests()
{
    PetscInt px = PETSC_DECIDE, py = PETSC_DECIDE;
    MeshParam mp;
    PetscFunctionBeginUser;
    PetscCall(PetscOptionsGetInt(nullptr, nullptr, "-mesh_px", &px, nullptr));
    PetscCall(PetscOptionsGetInt(nullptr, nullptr, "-mesh_py", &py, nullptr));
    // Unequal spacings and nonzero origins expose the old swapped-axis and
    // unnormalized sine-stretch bugs.
    mp.xstart = 3.5;
    mp.ystart = -2.25;
    mp.L = 17.0;
    mp.H = 0.8;
    PetscCall(CheckCase(12, 8, px, py, mp, CreateFullMesh, Shape::Uniform));
    PetscCall(CheckCase(12, 8, px, py, mp, LogicRectMesh, Shape::Perturbed));
    PetscCall(CheckCase(12, 8, px, py, mp, RefineMesh, Shape::Stretched));
    mp.perturbation = 0.0;
    PetscCall(CheckCase(12, 8, px, py, mp, LogicRectMesh, Shape::Uniform));
    mp.perturbation = 0.249;
    PetscCall(CheckCase(12, 8, px, py, mp, LogicRectMesh, Shape::Perturbed));
    // A single cell has no interior vertices, even in the perturbed mode.
    PetscCall(CheckCase(2, 2, px, py, mp, LogicRectMesh, Shape::Uniform));
    PetscCall(CheckErrors(px, py));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "mesh tests passed\n"));
    PetscFunctionReturn(0);
}

} // namespace

int main(int argc, char** argv)
{
    PetscErrorCode error = PetscInitialize(&argc, &argv, nullptr, nullptr);
    if (error) return static_cast<int>(error);
    error = RunTests();
    const PetscErrorCode finalError = PetscFinalize();
    return static_cast<int>(error ? error : finalError);
}
