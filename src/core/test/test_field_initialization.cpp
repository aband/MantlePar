#include "field_initialization.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <new>
#include <stdexcept>
#include <vector>

namespace {

// No assert(): all checks remain active in Release builds. Reduce test failures
// before returning so another rank cannot continue into the next collective.
PetscErrorCode CheckAll(MPI_Comm comm, bool condition, const char* message)
{
    PetscFunctionBeginUser;
    int bad = condition ? 0 : 1, anyBad = 0;
    PetscCallMPI(MPI_Allreduce(&bad, &anyBad, 1, MPI_INT, MPI_MAX, comm));
    PetscCheck(!anyBad, comm, PETSC_ERR_PLIB, "%s", message);
    PetscFunctionReturn(PETSC_SUCCESS);
}

struct Vector {
    Vec value = nullptr;
    Vector() = default;
    Vector(const Vector&) = delete;
    Vector& operator=(const Vector&) = delete;
    ~Vector() { if (value) (void)VecDestroy(&value); }
};

struct Domain {
    DM value = nullptr;
    Domain() = default;
    Domain(const Domain&) = delete;
    Domain& operator=(const Domain&) = delete;
    ~Domain() { if (value) (void)DMDestroy(&value); }
};

constexpr PetscReal tolerance = 2048 * PETSC_MACHINE_EPSILON;

bool Near(PetscReal actual, PetscReal expected)
{
    return !PetscIsInfOrNanReal(actual) &&
           PetscAbsReal(actual - expected) <=
               tolerance * std::max(PetscReal(1), PetscAbsReal(expected));
}

PetscScalar Sentinel()
{
#if defined(PETSC_USE_COMPLEX)
    return PetscCMPLX(123.25, -7.5);
#else
    return PetscScalar(123.25);
#endif
}

PetscErrorCode MakeDM(MPI_Comm comm, PetscInt nx, PetscInt ny, PetscInt dof,
                      PetscInt px, PetscInt py, Domain& dm)
{
    PetscFunctionBeginUser;
    PetscCall(DMDACreate2d(comm, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE,
                          DMDA_STENCIL_BOX, nx, ny, px, py, dof, 1,
                          nullptr, nullptr, &dm.value));
    // Keep the requested grid fixed; unrelated -da_* runtime options must not
    // silently change the mesh or field decomposition used by this test.
    PetscCall(DMSetUp(dm.value));
    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode Generate(DM dm, Vec vertices, const MeshParam& p, PetscInt kind)
{
    PetscFunctionBeginUser;
    if (kind == 0) PetscCall(CreateFullMesh(dm, vertices, p));
    else if (kind == 1) PetscCall(LogicRectMesh(dm, vertices, p));
    else PetscCall(RefineMesh(dm, vertices, p));
    PetscFunctionReturn(PETSC_SUCCESS);
}

// Independent reference: integrate monomials over the two physical triangles
// (0,1,2) and (0,2,3). Do not use IntegrateCell, GetCellArea, MapCellPoint or the
// production Gauss rule to compute expected averages.
struct Moments {
    PetscReal area = 0, x = 0, y = 0, xx = 0, xy = 0, yy = 0, xxxx = 0;
    PetscReal referenceMidpointX = 0;
};

Moments PolygonMoments(const QuadVertices& q)
{
    std::array<long double, 7> integral{};
    for (const auto& ids : {std::array<int, 3>{{0, 1, 2}},
                            std::array<int, 3>{{0, 2, 3}}}) {
        std::array<long double, 3> x{}, y{};
        for (int k = 0; k < 3; ++k) {
            x[k] = q[ids[k]].p[0];
            y[k] = q[ids[k]].p[1];
        }
        const long double area = ((x[1] - x[0]) * (y[2] - y[0]) -
                                  (x[2] - x[0]) * (y[1] - y[0])) / 2;
        const long double sx = x[0] + x[1] + x[2];
        const long double sy = y[0] + y[1] + y[2];
        const long double sxx = x[0]*x[0] + x[1]*x[1] + x[2]*x[2];
        const long double syy = y[0]*y[0] + y[1]*y[1] + y[2]*y[2];
        const long double sxy = x[0]*y[0] + x[1]*y[1] + x[2]*y[2];
        long double x4 = 0;
        // Triangle mean of x^p is 2/((p+1)(p+2)) times the complete
        // homogeneous polynomial of degree p in its three vertex x values.
        for (int a = 0; a <= 4; ++a)
            for (int b = 0; b <= 4 - a; ++b)
                x4 += std::pow(x[0], a) * std::pow(x[1], b) *
                      std::pow(x[2], 4 - a - b) / 15;
        const std::array<long double, 7> mean{{
            1, sx / 3, sy / 3, (sx*sx + sxx) / 12,
            (sx*sy + sxy) / 12, (sy*sy + syy) / 12, x4}};
        for (std::size_t k = 0; k < integral.size(); ++k)
            integral[k] += area * mean[k];
    }
    Moments m;
    m.area = static_cast<PetscReal>(integral[0]);
    m.x = static_cast<PetscReal>(integral[1] / integral[0]);
    m.y = static_cast<PetscReal>(integral[2] / integral[0]);
    m.xx = static_cast<PetscReal>(integral[3] / integral[0]);
    m.xy = static_cast<PetscReal>(integral[4] / integral[0]);
    m.yy = static_cast<PetscReal>(integral[5] / integral[0]);
    m.xxxx = static_cast<PetscReal>(integral[6] / integral[0]);
    for (const auto& p : q) m.referenceMidpointX += p.p[0] / 4;
    return m;
}

struct Fixture {
    // DMs outlive vectors, and both outlive no borrowed array views.
    Domain vertexDM, fieldDM;
    Vector vertices, field;
    MeshInfo mesh, reference;
    MeshParam parameters;
    PetscInt nx = 0, ny = 0;
    std::vector<Moments> moments;

    const Moments& At(PetscInt i, PetscInt j) const
    { return moments[static_cast<std::size_t>(j * nx + i)]; }
};

PetscErrorCode BuildFixture(MPI_Comm comm, PetscInt nx, PetscInt ny,
                            PetscInt meshPx, PetscInt meshPy,
                            PetscInt fieldPx, PetscInt fieldPy,
                            PetscInt kind, Fixture& f)
{
    PetscFunctionBeginUser;
    f.nx = nx;
    f.ny = ny;
    f.parameters.xstart = -0.75;
    f.parameters.ystart = 0.2;
    f.parameters.L = 2.5;
    f.parameters.H = 1.3;
    f.parameters.perturbation = 0.249;
    f.parameters.seed = 991;
    PetscCall(MakeDM(comm, nx + 1, ny + 1, 2, meshPx, meshPy, f.vertexDM));
    PetscCall(DMCreateGlobalVector(f.vertexDM.value, &f.vertices.value));
    PetscCall(Generate(f.vertexDM.value, f.vertices.value, f.parameters, kind));
    PetscCall(BuildMeshInfo(f.vertexDM.value, f.vertices.value, f.mesh));
    PetscCall(MakeDM(comm, nx, ny, 1, fieldPx, fieldPy, f.fieldDM));
    PetscCall(DMCreateGlobalVector(f.fieldDM.value, &f.field.value));

    // A small, complete SERIAL reference on each rank gives expected geometry
    // even where the distributed field owner has no local mesh geometry.
    // This replication belongs to the test only, not the production helper.
    Domain serialDM;
    Vector serialVertices;
    PetscCall(MakeDM(PETSC_COMM_SELF, nx + 1, ny + 1, 2, 1, 1, serialDM));
    PetscCall(DMCreateGlobalVector(serialDM.value, &serialVertices.value));
    PetscCall(Generate(serialDM.value, serialVertices.value, f.parameters, kind));
    PetscCall(BuildMeshInfo(serialDM.value, serialVertices.value, f.reference));
    f.moments.resize(static_cast<std::size_t>(nx * ny));
    for (PetscInt j = 0; j < ny; ++j) {
        for (PetscInt i = 0; i < nx; ++i) {
            QuadVertices q;
            PetscCall(f.reference.GetCellCorners({i, j}, q));
            f.moments[static_cast<std::size_t>(j * nx + i)] = PolygonMoments(q);
        }
    }
    bool valid = true;
    PetscReal area = 0;
    for (const auto& m : f.moments) {
        valid = valid && m.area > 0 && !PetscIsInfOrNanReal(m.area);
        area += m.area;
    }
    PetscCall(CheckAll(comm, valid && Near(area, f.parameters.L * f.parameters.H),
                       "The independent polygon reference has invalid total area"));
    PetscFunctionReturn(PETSC_SUCCESS);
}

template <typename Expected>
PetscErrorCode CheckGrid(DM dm, Vec field, Expected expected,
                         const char* message, bool ghosted = false)
{
    PetscFunctionBeginUser;
    PetscInt xs, ys, xm, ym;
    if (ghosted)
        PetscCall(DMDAGetGhostCorners(dm, &xs, &ys, nullptr, &xm, &ym, nullptr));
    else
        PetscCall(DMDAGetCorners(dm, &xs, &ys, nullptr, &xm, &ym, nullptr));
    const PetscScalar** a = nullptr;
    PetscCall(DMDAVecGetArrayRead(dm, field, &a));
    bool valid = true;
    for (PetscInt j = ys; j < ys + ym; ++j)
        for (PetscInt i = xs; i < xs + xm; ++i)
            valid = valid && Near(PetscRealPart(a[j][i]), expected(i, j)) &&
                    PetscImaginaryPart(a[j][i]) == PetscReal(0);
    PetscCall(DMDAVecRestoreArrayRead(dm, field, &a));
    PetscCall(CheckAll(PetscObjectComm(reinterpret_cast<PetscObject>(dm)), valid, message));
    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode CheckConstantVector(MPI_Comm comm, Vec field, PetscScalar expected)
{
    PetscFunctionBeginUser;
    PetscInt n;
    const PetscScalar* a = nullptr;
    PetscCall(VecGetLocalSize(field, &n));
    PetscCall(VecGetArrayRead(field, &a));
    bool valid = true;
    for (PetscInt k = 0; k < n; ++k) valid = valid && a[k] == expected;
    PetscCall(VecRestoreArrayRead(field, &a));
    PetscCall(CheckAll(comm, valid, "A separate local vector changed without a ghost exchange"));
    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscReal Quadratic(const Point& p)
{
    const PetscReal x = p.p[0], y = p.p[1];
    return 1.2 + 2*x - 0.7*y + 0.5*x*x + 1.3*x*y - 0.9*y*y;
}

PetscReal QuadraticMean(const Moments& m)
{ return 1.2 + 2*m.x - 0.7*m.y + 0.5*m.xx + 1.3*m.xy - 0.9*m.yy; }

PetscErrorCode TestAverages(MPI_Comm comm, Fixture& f)
{
    PetscFunctionBeginUser;
    const auto dm = f.fieldDM.value;
    const auto field = f.field.value;
    for (const PetscInt order : {1, 2, 3, 5}) {
        // Sentinel also tests clearing the imaginary component in complex PETSc.
        PetscCall(VecSet(field, Sentinel()));
        PetscCall(InitializeCellAverages(dm, field, f.mesh,
            [](const Point&) { return PetscReal(2.5); }, order));
        PetscCall(CheckGrid(dm, field, [](PetscInt, PetscInt) { return PetscReal(2.5); },
                             "Constant profile was not preserved"));
    }

    PetscReal offset = 0.4, slope = 1.7;
    const CellInitialFunction affine = [&](const Point& p) {
        return offset + slope*p.p[0] - 0.3*p.p[1];
    };
    for (int pass = 0; pass < 2; ++pass) {
        PetscCall(InitializeCellAverages(dm, field, f.mesh, affine)); // Default order.
        PetscCall(CheckGrid(dm, field, [&](PetscInt i, PetscInt j) {
            const auto& m = f.At(i, j);
            return offset + slope*m.x - 0.3*m.y;
        }, "Affine mean or runtime parameter update is incorrect"));
        offset = -2.1;
        slope = -0.6;
    }

    for (const PetscInt order : {2, 3}) {
        PetscCall(InitializeCellAverages(dm, field, f.mesh, Quadratic, order));
        PetscCall(CheckGrid(dm, field, [&](PetscInt i, PetscInt j) {
            return QuadraticMean(f.At(i, j));
        }, "Quadratic cell average disagrees with exact polygon moments"));
    }

    // Global conservation, summed over FIELD ownership, independently of mesh
    // ownership. All three generators keep the same outer rectangle.
    PetscInt xs, ys, xm, ym;
    PetscCall(DMDAGetCorners(dm, &xs, &ys, nullptr, &xm, &ym, nullptr));
    const PetscScalar** a = nullptr;
    PetscCall(DMDAVecGetArrayRead(dm, field, &a));
    PetscReal localIntegral = 0, integral = 0;
    for (PetscInt j = ys; j < ys + ym; ++j)
        for (PetscInt i = xs; i < xs + xm; ++i)
            localIntegral += PetscRealPart(a[j][i]) * f.At(i, j).area;
    PetscCall(DMDAVecRestoreArrayRead(dm, field, &a));
    PetscCallMPI(MPI_Allreduce(&localIntegral, &integral, 1, MPIU_REAL, MPI_SUM, comm));
    const auto& p = f.parameters;
    const PetscReal x0 = p.xstart, x1 = x0 + p.L;
    const PetscReal y0 = p.ystart, y1 = y0 + p.H;
    Moments domain;
    domain.x = (x0 + x1) / 2;
    domain.y = (y0 + y1) / 2;
    domain.xx = (x0*x0 + x0*x1 + x1*x1) / 3;
    domain.yy = (y0*y0 + y0*y1 + y1*y1) / 3;
    domain.xy = domain.x * domain.y;
    PetscCall(CheckAll(comm, Near(integral, p.L*p.H*QuadraticMean(domain)),
                       "Cell averages do not conserve the domain integral"));

    const CellInitialFunction quartic = [](const Point& p) {
        const PetscReal x2 = p.p[0] * p.p[0];
        return x2 * x2;
    };
    PetscCall(InitializeCellAverages(dm, field, f.mesh, quartic, 1));
    PetscCall(CheckGrid(dm, field, [&](PetscInt i, PetscInt j) {
        const PetscReal x = f.At(i, j).referenceMidpointX;
        return x*x*x*x;
    }, "One-point quadrature did not evaluate the mapped reference midpoint"));
    bool orderMatters = false;
    for (const auto& m : f.moments) {
        const PetscReal x = m.referenceMidpointX;
        orderMatters = orderMatters || !Near(x*x*x*x, m.xxxx);
    }
    PetscCall(CheckAll(comm, orderMatters, "Quartic fixture cannot distinguish quadrature orders"));
    for (const PetscInt order : {3, 5}) {
        PetscCall(InitializeCellAverages(dm, field, f.mesh, quartic, order));
        PetscCall(CheckGrid(dm, field, [&](PetscInt i, PetscInt j) { return f.At(i, j).xxxx; },
                             "Higher-order quadrature did not recover the exact quartic average"));
    }
    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscReal IndexedValue(PetscInt i, PetscInt j)
{ return 37*PetscReal(j) - 11*PetscReal(i) + 0.25; }

PetscErrorCode OwnedValues(DM dm, std::vector<PetscReal>& values)
{
    PetscFunctionBeginUser;
    PetscInt xs, ys, xm, ym;
    PetscCall(DMDAGetCorners(dm, &xs, &ys, nullptr, &xm, &ym, nullptr));
    values.clear();
    for (PetscInt j = ys; j < ys + ym; ++j)
        for (PetscInt i = xs; i < xs + xm; ++i)
            values.push_back(IndexedValue(i, j));
    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode TestAssignmentAndGhosts(MPI_Comm comm, Fixture& f)
{
    PetscFunctionBeginUser;
    Vector local;
    PetscCall(DMCreateLocalVector(f.fieldDM.value, &local.value));
    PetscCall(VecSet(local.value, Sentinel()));
    PetscCall(InitializeCellAverages(f.fieldDM.value, f.field.value, f.mesh, Quadratic));
    PetscCall(CheckConstantVector(comm, local.value, Sentinel()));
    PetscCall(DMGlobalToLocalBegin(f.fieldDM.value, f.field.value, INSERT_VALUES, local.value));
    PetscCall(DMGlobalToLocalEnd(f.fieldDM.value, f.field.value, INSERT_VALUES, local.value));
    PetscCall(CheckGrid(f.fieldDM.value, local.value, [&](PetscInt i, PetscInt j) {
        return QuadraticMean(f.At(i, j));
    }, "Cell averages did not reach the correct ghost cells after exchange", true));

    PetscCall(VecSet(local.value, Sentinel()));
    PetscCall(VecSet(f.field.value, Sentinel()));
    std::vector<PetscReal> values;
    PetscCall(OwnedValues(f.fieldDM.value, values));
    PetscCall(AssignOwnedCellValues(f.fieldDM.value, f.field.value, values));
    PetscCall(CheckGrid(f.fieldDM.value, f.field.value, IndexedValue,
                         "Owned values were not assigned in field-owned, i-fastest order"));
    PetscCall(CheckConstantVector(comm, local.value, Sentinel()));
    PetscCall(DMGlobalToLocalBegin(f.fieldDM.value, f.field.value, INSERT_VALUES, local.value));
    PetscCall(DMGlobalToLocalEnd(f.fieldDM.value, f.field.value, INSERT_VALUES, local.value));
    PetscCall(CheckGrid(f.fieldDM.value, local.value, IndexedValue,
                         "Assigned values did not reach the correct ghost cells after exchange", true));
    PetscFunctionReturn(PETSC_SUCCESS);
}

template <typename Operation>
PetscErrorCode ExpectFailure(MPI_Comm comm, Vec field, PetscErrorCode expected,
                             const char* message, Operation operation)
{
    PetscFunctionBeginUser;
    Vector before;
    PetscCall(VecSet(field, Sentinel()));
    PetscCall(VecDuplicate(field, &before.value));
    PetscCall(VecCopy(field, before.value));
    PetscCall(PetscPushErrorHandler(PetscReturnErrorHandler, nullptr));
    const PetscErrorCode actual = operation();
    PetscCall(PetscPopErrorHandler());
    PetscCall(CheckAll(comm, actual == expected, message));
    PetscBool unchanged = PETSC_FALSE;
    PetscCall(VecEqual(field, before.value, &unchanged));
    PetscCall(CheckAll(comm, unchanged == PETSC_TRUE,
                       "A rejected operation changed the destination field"));
    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode TestFailures(MPI_Comm comm, Fixture& f)
{
    PetscFunctionBeginUser;
    PetscMPIInt rank, ranks;
    PetscCallMPI(MPI_Comm_rank(comm, &rank));
    PetscCallMPI(MPI_Comm_size(comm, &ranks));
    const auto dm = f.fieldDM.value;
    const auto field = f.field.value;
    std::vector<PetscReal> valid;
    PetscCall(OwnedValues(dm, valid));
    PetscCall(CheckAll(comm, !valid.empty(), "Failure fixture must give every field rank cells"));

    for (int extra = 0; extra < 2; ++extra) {
        auto bad = valid;
        if (rank == ranks - 1) {
            if (extra) bad.push_back(0);
            else bad.pop_back();
        }
        PetscCall(ExpectFailure(comm, field, PETSC_ERR_ARG_SIZ,
            "One rank's wrong owned-array length was not rejected collectively", [&] {
                return AssignOwnedCellValues(dm, field, bad);
            }));
    }
    const std::array<PetscReal, 3> nonfinite{{
        std::numeric_limits<PetscReal>::quiet_NaN(),
        std::numeric_limits<PetscReal>::infinity(),
        -std::numeric_limits<PetscReal>::infinity()}};
    for (const auto value : nonfinite) {
        auto bad = valid;
        if (rank == ranks - 1) bad.front() = value;
        PetscCall(ExpectFailure(comm, field, PETSC_ERR_FP,
            "One rank's nonfinite owned value was not rejected collectively", [&] {
                return AssignOwnedCellValues(dm, field, bad);
            }));
        PetscCall(ExpectFailure(comm, field, PETSC_ERR_FP,
            "One rank's nonfinite callback was not rejected collectively", [&] {
                return InitializeCellAverages(dm, field, f.mesh,
                    [&](const Point&) { return rank == 0 ? value : PetscReal(1); });
            }));
    }
    CellInitialFunction emptyOnOneRank = Quadratic;
    if (rank == ranks - 1) emptyOnOneRank = {};
    PetscCall(ExpectFailure(comm, field, PETSC_ERR_ARG_NULL,
        "An empty callback was not rejected collectively", [&] {
            return InitializeCellAverages(dm, field, f.mesh, emptyOnOneRank);
        }));
    for (int exception = 0; exception < 3; ++exception) {
        PetscCall(ExpectFailure(comm, field, exception == 2 ? PETSC_ERR_MEM : PETSC_ERR_USER,
            "A callback exception was not converted to the expected collective error", [&] {
                return InitializeCellAverages(dm, field, f.mesh, [&](const Point&) -> PetscReal {
                    if (rank == 0) {
                        if (exception == 0) throw std::runtime_error("intentional test failure");
                        if (exception == 1) throw 42;
                        throw std::bad_alloc();
                    }
                    return 1;
                });
            }));
    }
    for (const PetscInt order : {0, -2}) {
        PetscCall(ExpectFailure(comm, field, PETSC_ERR_ARG_OUTOFRANGE,
            "A nonpositive Gauss-point count was not rejected", [&] {
                return InitializeCellAverages(dm, field, f.mesh, Quadratic, order);
            }));
    }
    if (ranks > 1) {
        PetscCall(ExpectFailure(comm, field, PETSC_ERR_ARG_WRONG,
            "Different Gauss-point counts across ranks were not rejected", [&] {
                return InitializeCellAverages(dm, field, f.mesh, Quadratic, rank == 0 ? 2 : 3);
            }));
        // Every serial snapshot is valid on its own but owns the ENTIRE mesh.
        // Supplying all of them to a parallel field duplicates every cell.
        PetscCall(ExpectFailure(comm, field, PETSC_ERR_ARG_WRONG,
            "Duplicate mesh ownership was not rejected", [&] {
                return InitializeCellAverages(dm, field, f.reference, Quadratic);
            }));
    }
    MeshInfo uninitialized;
    PetscCall(ExpectFailure(comm, field, PETSC_ERR_ARG_WRONG,
        "One rank's uninitialized mesh snapshot was not rejected", [&] {
            return InitializeCellAverages(dm, field,
                rank == ranks - 1 ? uninitialized : f.mesh, Quadratic);
        }));
    PetscCall(ExpectFailure(comm, field, PETSC_ERR_ARG_NULL,
        "A null field on one rank was not rejected by initialization", [&] {
            return InitializeCellAverages(dm, rank == 0 ? nullptr : field, f.mesh, Quadratic);
        }));
    PetscCall(ExpectFailure(comm, field, PETSC_ERR_ARG_NULL,
        "A null field on one rank was not rejected by assignment", [&] {
            return AssignOwnedCellValues(dm, rank == 0 ? nullptr : field, valid);
        }));

    PetscInt px, py;
    PetscCall(DMDAGetInfo(dm, nullptr, nullptr, nullptr, nullptr, &px, &py,
                          nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr));
    Domain wrongSize, wrongDof;
    Vector wrongSizeField, wrongDofField;
    PetscCall(MakeDM(comm, f.nx + 1, f.ny, 1, px, py, wrongSize));
    PetscCall(DMCreateGlobalVector(wrongSize.value, &wrongSizeField.value));
    PetscCall(ExpectFailure(comm, wrongSizeField.value, PETSC_ERR_ARG_SIZ,
        "The mesh vertex/cell dimension mismatch was not rejected", [&] {
            return InitializeCellAverages(wrongSize.value, wrongSizeField.value, f.mesh, Quadratic);
        }));
    PetscCall(ExpectFailure(comm, wrongSizeField.value, PETSC_ERR_ARG_SIZ,
        "An incompatible vector was not rejected by initialization", [&] {
            return InitializeCellAverages(dm, wrongSizeField.value, f.mesh, Quadratic);
        }));
    PetscCall(ExpectFailure(comm, wrongSizeField.value, PETSC_ERR_ARG_SIZ,
        "An incompatible vector was not rejected by assignment", [&] {
            return AssignOwnedCellValues(dm, wrongSizeField.value, valid);
        }));
    PetscCall(MakeDM(comm, f.nx, f.ny, 2, px, py, wrongDof));
    PetscCall(DMCreateGlobalVector(wrongDof.value, &wrongDofField.value));
    PetscCall(ExpectFailure(comm, wrongDofField.value, PETSC_ERR_ARG_WRONG,
        "A nonscalar DM was not rejected by initialization", [&] {
            return InitializeCellAverages(wrongDof.value, wrongDofField.value, f.mesh, Quadratic);
        }));
    PetscCall(ExpectFailure(comm, wrongDofField.value, PETSC_ERR_ARG_WRONG,
        "A nonscalar DM was not rejected by assignment", [&] {
            return AssignOwnedCellValues(wrongDof.value, wrongDofField.value, valid);
        }));
    if (ranks > 1) {
        Domain selfDM;
        Vector selfField;
        PetscCall(MakeDM(PETSC_COMM_SELF, f.nx, f.ny, 1, 1, 1, selfDM));
        PetscCall(DMCreateGlobalVector(selfDM.value, &selfField.value));
        PetscCall(ExpectFailure(comm, selfField.value, PETSC_ERR_ARG_WRONG,
            "An incompatible vector communicator was not rejected", [&] {
                return InitializeCellAverages(dm, selfField.value, f.mesh, Quadratic);
            }));
    }
    // Confirm successful use still works after all intentional error paths.
    PetscCall(InitializeCellAverages(dm, field, f.mesh, Quadratic));
    PetscCall(CheckGrid(dm, field, [&](PetscInt i, PetscInt j) {
        return QuadraticMean(f.At(i, j));
    }, "Initialization failed to recover after rejected input"));
    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode TestEmptyMeshOwners(MPI_Comm comm, PetscInt kind, PetscMPIInt ranks)
{
    PetscFunctionBeginUser;
    // 1 rank: 2x2 vertices / 1 cell. 2 ranks: 3x2 vertices / 2 cells.
    // 4 ranks: 3x3 vertices / 4 cells. Only rank 0 owns lower-left vertices;
    // other ranks own boundary vertices but must still receive field values.
    const PetscInt px = ranks == 1 ? 1 : 2;
    const PetscInt py = ranks == 4 ? 2 : 1;
    Fixture f;
    PetscCall(BuildFixture(comm, px, py, px, py, px, py, kind, f));
    const auto extent = f.mesh.OwnedCells().Size();
    int empty = (extent.i == 0 || extent.j == 0) ? 1 : 0, emptyRanks = 0;
    PetscCallMPI(MPI_Allreduce(&empty, &emptyRanks, 1, MPI_INT, MPI_SUM, comm));
    PetscCall(CheckAll(comm, emptyRanks == ranks - 1,
                       "Tiny fixture did not produce the expected empty mesh owners"));
    bool called = false;
    PetscCall(InitializeCellAverages(f.fieldDM.value, f.field.value, f.mesh,
        [&](const Point& p) { called = true; return Quadratic(p); }));
    PetscCall(CheckAll(comm, called == (empty == 0),
                       "The callback was called on an empty mesh owner or skipped on a nonempty one"));
    PetscCall(CheckGrid(f.fieldDM.value, f.field.value, [&](PetscInt i, PetscInt j) {
        return QuadraticMean(f.At(i, j));
    }, "Empty mesh owners did not receive their field-owned cell averages"));
    PetscCall(TestAssignmentAndGhosts(comm, f));
    if (ranks > 1) {
        PetscMPIInt rank;
        PetscCallMPI(MPI_Comm_rank(comm, &rank));
        CellInitialFunction profile = Quadratic;
        if (rank == ranks - 1) profile = {};
        PetscCall(ExpectFailure(comm, f.field.value, PETSC_ERR_ARG_NULL,
            "An empty callback on an empty mesh owner was not rejected", [&] {
                return InitializeCellAverages(f.fieldDM.value, f.field.value, f.mesh, profile);
            }));
    }
    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode RunTests()
{
    PetscFunctionBeginUser;
    const MPI_Comm comm = PETSC_COMM_WORLD;
    PetscMPIInt ranks;
    PetscCallMPI(MPI_Comm_size(comm, &ranks));
    PetscInt expectedRanks = 1, kind = 0, meshPx = 1, meshPy = 1, fieldPx = 1, fieldPy = 1;
    PetscCall(PetscOptionsGetInt(nullptr, nullptr, "-expected_ranks", &expectedRanks, nullptr));
    PetscCall(PetscOptionsGetInt(nullptr, nullptr, "-field_mesh_type", &kind, nullptr));
    PetscCall(PetscOptionsGetInt(nullptr, nullptr, "-mesh_px", &meshPx, nullptr));
    PetscCall(PetscOptionsGetInt(nullptr, nullptr, "-mesh_py", &meshPy, nullptr));
    PetscCall(PetscOptionsGetInt(nullptr, nullptr, "-field_px", &fieldPx, nullptr));
    PetscCall(PetscOptionsGetInt(nullptr, nullptr, "-field_py", &fieldPy, nullptr));
    PetscCall(CheckAll(comm, ranks == expectedRanks,
        "MPI size differs from -expected_ranks; check the PETSc/MPI build and launcher pairing"));
    PetscCall(CheckAll(comm, ranks == 1 || ranks == 2 || ranks == 4,
                       "These tests support 1, 2, or 4 ranks"));
    PetscCall(CheckAll(comm, kind >= 0 && kind <= 2 && meshPx >= 1 && meshPx <= ranks &&
        meshPy >= 1 && meshPy <= ranks && fieldPx >= 1 && fieldPx <= ranks &&
        fieldPy >= 1 && fieldPy <= ranks && meshPx*meshPy == ranks && fieldPx*fieldPy == ranks,
        "Invalid mesh type or process-grid dimensions"));

    Fixture f;
    // 9x7 cells: uneven divisions in both directions, with enough cells for
    // every registered process grid (including 4x1 field ownership).
    PetscCall(BuildFixture(comm, 9, 7, meshPx, meshPy, fieldPx, fieldPy, kind, f));
    PetscCall(TestAverages(comm, f));
    PetscCall(TestAssignmentAndGhosts(comm, f));
    PetscCall(TestFailures(comm, f));
    PetscCall(TestEmptyMeshOwners(comm, kind, ranks));
    PetscCall(PetscPrintf(comm,
        "PASS field initialization: mesh type %" PetscInt_FMT
        ", mesh grid %" PetscInt_FMT "x%" PetscInt_FMT
        ", field grid %" PetscInt_FMT "x%" PetscInt_FMT ", %d rank(s)\n",
        kind, meshPx, meshPy, fieldPx, fieldPy, static_cast<int>(ranks)));
    PetscFunctionReturn(PETSC_SUCCESS);
}

} // namespace

int main(int argc, char** argv)
{
    PetscCall(PetscInitialize(&argc, &argv, nullptr,
        "Tests for cell-field initialization on rectangular and quadrilateral meshes.\n"));
    PetscCallAbort(PETSC_COMM_WORLD, RunTests());
    PetscCall(PetscFinalize());
    return 0;
}
