#include "mesh.h"

#include <algorithm>
#include <array>

namespace {

enum class MeshKind { Uniform, Perturbed, SineStretched };

// Stateless integer mixing: no rank-dependent random stream or time seed.
std::uint64_t Mix64(std::uint64_t value)
{
    value += UINT64_C(0x9e3779b97f4a7c15);
    value = (value ^ (value >> 30)) * UINT64_C(0xbf58476d1ce4e5b9);
    value = (value ^ (value >> 27)) * UINT64_C(0x94d049bb133111eb);
    return value ^ (value >> 31);
}

PetscReal SignedNoise(std::uint64_t key)
{
    const PetscReal unit = static_cast<PetscReal>(Mix64(key) >> 11)
                         * static_cast<PetscReal>(0x1.0p-53);
    return 2.0 * unit - 1.0;
}

Point VertexAt(PetscInt i, PetscInt j, PetscInt M, PetscInt N,
               const MeshParam& mp, MeshKind kind)
{
    const PetscReal hx = mp.L / static_cast<PetscReal>(M - 1);
    const PetscReal hy = mp.H / static_cast<PetscReal>(N - 1);
    Point point{{mp.xstart + static_cast<PetscReal>(i) * hx,
                 mp.ystart + static_cast<PetscReal>(j) * hy}};

    if (kind == MeshKind::SineStretched) {
        const PetscReal s = static_cast<PetscReal>(i) / (M - 1);
        const PetscReal t = static_cast<PetscReal>(j) / (N - 1);
        point.p[0] = mp.xstart + mp.L * PetscSinReal(0.5 * PETSC_PI * s);
        point.p[1] = mp.ystart + mp.H * PetscSinReal(0.5 * PETSC_PI * t);
    } else if (kind == MeshKind::Perturbed &&
               i > 0 && i < M - 1 && j > 0 && j < N - 1) {
        const auto index = static_cast<std::uint64_t>(j)
                         * static_cast<std::uint64_t>(M)
                         + static_cast<std::uint64_t>(i);
        point.p[0] += mp.perturbation * hx * SignedNoise(mp.seed + 2 * index);
        point.p[1] += mp.perturbation * hy * SignedNoise(mp.seed + 2 * index + 1);
    }

    // Avoid accumulated rounding at the physical endpoints.
    if (i == 0) point.p[0] = mp.xstart;
    if (i == M - 1) point.p[0] = mp.xstart + mp.L;
    if (j == 0) point.p[1] = mp.ystart;
    if (j == N - 1) point.p[1] = mp.ystart + mp.H;
    return point;
}

bool ValidQuad(const std::array<Point, 4>& p)
{
    PetscReal xmin = p[0].p[0], xmax = xmin;
    PetscReal ymin = p[0].p[1], ymax = ymin;
    for (const auto& point : p) {
        if (PetscIsInfOrNanReal(point.p[0]) ||
            PetscIsInfOrNanReal(point.p[1])) return false;
        xmin = std::min(xmin, point.p[0]);
        xmax = std::max(xmax, point.p[0]);
        ymin = std::min(ymin, point.p[1]);
        ymax = std::max(ymax, point.p[1]);
    }

    const PetscReal sx = xmax - xmin, sy = ymax - ymin;
    if (PetscIsInfOrNanReal(sx) || PetscIsInfOrNanReal(sy) ||
        sx <= 0.0 || sy <= 0.0) return false;

    // Scale the axes separately so rectangular aspect ratio alone is not
    // mistaken for a folded cell. These are the four Q1 corner determinants,
    // divided by sx*sy (and without the 1/4 for reference coordinates [-1,1]).
    constexpr PetscReal tolerance = 64.0 * PETSC_MACHINE_EPSILON;
    for (std::size_t k = 0; k < p.size(); ++k) {
        const Point& next = p[(k + 1) % 4];
        const Point& prev = p[(k + 3) % 4];
        const PetscReal ux = (next.p[0] - p[k].p[0]) / sx;
        const PetscReal uy = (next.p[1] - p[k].p[1]) / sy;
        const PetscReal vx = (prev.p[0] - p[k].p[0]) / sx;
        const PetscReal vy = (prev.p[1] - p[k].p[1]) / sy;
        if (!(ux * vy - uy * vx > tolerance)) return false;
    }
    return true;
}

PetscErrorCode CheckLayout(DM dm, PetscInt& M, PetscInt& N)
{
    PetscInt dim, dof, width;
    DMBoundaryType bx, by;
    DMDAStencilType stencil;
    PetscFunctionBeginUser;
    PetscCall(DMDAGetInfo(dm, &dim, &M, &N, nullptr, nullptr, nullptr,
                          nullptr, &dof, &width, &bx, &by, nullptr, &stencil));
    const MPI_Comm comm = PetscObjectComm(reinterpret_cast<PetscObject>(dm));
    PetscCheck(dim == 2 && dof == 2, comm, PETSC_ERR_ARG_WRONG,
               "Mesh requires a 2-D vertex DMDA with two degrees of freedom");
    PetscCheck(M >= 2 && N >= 2, comm, PETSC_ERR_ARG_OUTOFRANGE,
               "Each direction needs at least two vertices");
    PetscCheck(bx == DM_BOUNDARY_NONE && by == DM_BOUNDARY_NONE,
               comm, PETSC_ERR_SUP,
               "Use DM_BOUNDARY_NONE for the vertex DM; boundary vertices "
               "must be owned, not stored as exterior ghosts");
    PetscCheck(stencil == DMDA_STENCIL_BOX && width >= 1,
               comm, PETSC_ERR_ARG_WRONG,
               "Mesh validation requires a box stencil of width at least one");
    PetscFunctionReturn(0);
}

PetscErrorCode GenerateMesh(DM dm, Vec vertices, const MeshParam& mp,
                            MeshKind kind)
{
    PetscInt M, N, xs, ys, xm, ym;
    PetscScalar*** a = nullptr;
    PetscFunctionBeginUser;
    PetscCall(CheckLayout(dm, M, N));
    const MPI_Comm comm = PetscObjectComm(reinterpret_cast<PetscObject>(dm));
    PetscCheck(!PetscIsInfOrNanReal(mp.xstart) &&
               !PetscIsInfOrNanReal(mp.ystart) &&
               !PetscIsInfOrNanReal(mp.L) && !PetscIsInfOrNanReal(mp.H) &&
               mp.L > 0.0 && mp.H > 0.0, comm, PETSC_ERR_ARG_OUTOFRANGE,
               "Mesh origin must be finite and lengths must be finite and positive");
    PetscCheck(!PetscIsInfOrNanReal(mp.xstart + mp.L) &&
               !PetscIsInfOrNanReal(mp.ystart + mp.H) &&
               mp.xstart + mp.L > mp.xstart &&
               mp.ystart + mp.H > mp.ystart,
               comm, PETSC_ERR_ARG_OUTOFRANGE,
               "Domain endpoints must be finite and distinguishable at this precision");
    if (kind == MeshKind::Perturbed) {
        PetscCheck(!PetscIsInfOrNanReal(mp.perturbation) &&
                   mp.perturbation >= 0.0 && mp.perturbation < 0.25,
                   comm, PETSC_ERR_ARG_OUTOFRANGE,
                   "Perturbation must satisfy 0 <= perturbation < 0.25");
    }

    PetscCall(DMDAGetCorners(dm, &xs, &ys, nullptr, &xm, &ym, nullptr));
    PetscCall(DMDAVecGetArrayDOF(dm, vertices, &a));
    for (PetscInt j = ys; j < ys + ym; ++j) {
        for (PetscInt i = xs; i < xs + xm; ++i) {
            const Point p = VertexAt(i, j, M, N, mp, kind);
            a[j][i][0] = p.p[0];
            a[j][i][1] = p.p[1];
        }
    }
    PetscCall(DMDAVecRestoreArrayDOF(dm, vertices, &a));
    PetscCall(ValidateMesh(dm, vertices));
    PetscFunctionReturn(0);
}

} // namespace

PetscErrorCode CreateFullMesh(DM dm, Vec vertices, const MeshParam& mp)
{
    PetscFunctionBeginUser;
    PetscCall(GenerateMesh(dm, vertices, mp, MeshKind::Uniform));
    PetscFunctionReturn(0);
}

PetscErrorCode LogicRectMesh(DM dm, Vec vertices, const MeshParam& mp)
{
    PetscFunctionBeginUser;
    PetscCall(GenerateMesh(dm, vertices, mp, MeshKind::Perturbed));
    PetscFunctionReturn(0);
}

PetscErrorCode RefineMesh(DM dm, Vec vertices, const MeshParam& mp)
{
    PetscFunctionBeginUser;
    PetscCall(GenerateMesh(dm, vertices, mp, MeshKind::SineStretched));
    PetscFunctionReturn(0);
}

PetscErrorCode ValidateMesh(DM dm, Vec vertices)
{
    PetscInt M, N, xs, ys, xm, ym;
    Vec local = nullptr;
    const PetscScalar*** a = nullptr;
    PetscMPIInt localBad = 0, globalBad = 0;
    PetscFunctionBeginUser;
    PetscCall(CheckLayout(dm, M, N));
    PetscCall(DMDAGetCorners(dm, &xs, &ys, nullptr, &xm, &ym, nullptr));
    PetscCall(DMGetLocalVector(dm, &local));
    PetscCall(DMGlobalToLocalBegin(dm, vertices, INSERT_VALUES, local));
    PetscCall(DMGlobalToLocalEnd(dm, vertices, INSERT_VALUES, local));
    PetscCall(DMDAVecGetArrayDOFRead(dm, local, &a));

    // Assign each cell to the rank owning its lower-left vertex.
    // The last vertex row/column does not start any physical cells.
    for (PetscInt j = ys; j < std::min(ys + ym, N - 1); ++j) {
        for (PetscInt i = xs; i < std::min(xs + xm, M - 1); ++i) {
            const PetscInt ci[4] = {i, i + 1, i + 1, i};
            const PetscInt cj[4] = {j, j, j + 1, j + 1};
            std::array<Point, 4> p{};
            for (std::size_t k = 0; k < p.size(); ++k) {
                for (int d = 0; d < 2; ++d) {
                    const PetscScalar value = a[cj[k]][ci[k]][d];
                    if (PetscIsInfOrNanScalar(value) ||
                        PetscImaginaryPart(value) != 0.0) localBad = 1;
                    p[k].p[d] = PetscRealPart(value);
                }
            }
            if (!ValidQuad(p)) localBad = 1;
        }
    }
    PetscCall(DMDAVecRestoreArrayDOFRead(dm, local, &a));
    PetscCall(DMRestoreLocalVector(dm, &local));

    const MPI_Comm comm = PetscObjectComm(reinterpret_cast<PetscObject>(dm));
    PetscCallMPI(MPI_Allreduce(&localBad, &globalBad, 1, MPI_INT, MPI_MAX, comm));
    PetscCheck(!globalBad, comm, PETSC_ERR_ARG_WRONG,
               "Mesh has non-real/non-finite coordinates or an inverted, "
               "degenerate, or nearly singular quadrilateral");
    PetscFunctionReturn(0);
}

