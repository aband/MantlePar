#include "assembly.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <type_traits>
#include <vector>

namespace {

// No assert(): all checks also run in Release builds. The dense reference uses
// natural numbering and a serial geometry snapshot, independently of COO and
// rank ownership. Analytic identities additionally check signs and scaling.
constexpr PetscReal tolerance = 131072 * PETSC_MACHINE_EPSILON;
enum class Field { Stokes, Darcy, Coupling };

PetscErrorCode Require(bool condition, const char* message)
{
    PetscFunctionBeginUser;
    PetscCheck(condition, PETSC_COMM_SELF, PETSC_ERR_PLIB, "%s", message);
    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode Near(PetscScalar actual, PetscScalar expected, const char* message)
{
    PetscFunctionBeginUser;
    const PetscReal limit = tolerance * std::max(PetscReal(1), PetscAbsScalar(expected));
    PetscCheck(!PetscIsInfOrNanScalar(actual) && !PetscIsInfOrNanScalar(expected) &&
                   PetscAbsScalar(actual-expected) <= limit,
               PETSC_COMM_SELF, PETSC_ERR_PLIB,
               "%s: got %.17g%+.17gi, expected %.17g%+.17gi, tolerance %.3g",
               message, static_cast<double>(PetscRealPart(actual)),
               static_cast<double>(PetscImaginaryPart(actual)),
               static_cast<double>(PetscRealPart(expected)),
               static_cast<double>(PetscImaginaryPart(expected)), static_cast<double>(limit));
    PetscFunctionReturn(PETSC_SUCCESS);
}

template<class Function>
PetscErrorCode ExpectCollectiveError(Function&& function, PetscErrorCode expected,
                                    const char* message)
{
    PetscFunctionBeginUser;
    PetscCall(PetscPushErrorHandler(PetscReturnErrorHandler, nullptr));
    // Forwarding lambdas contain no PetscCall/PetscCheck. Every other callback
    // that uses PETSc error macros has its own BeginUser/Return frame.
    const PetscErrorCode error = function();
    PetscCall(PetscPopErrorHandler());
    const int wrong = error == expected ? 0 : 1;
    int anyWrong = 0;
    PetscCallMPI(MPI_Allreduce(&wrong, &anyWrong, 1, MPI_INT, MPI_MAX, PETSC_COMM_WORLD));
    PetscCheck(!anyWrong, PETSC_COMM_WORLD, PETSC_ERR_PLIB,
               "%s: at least one rank did not return error %d (this rank returned %d)",
               message, static_cast<int>(expected), static_cast<int>(error));
    PetscFunctionReturn(PETSC_SUCCESS);
}

MeshParam Domain()
{
    MeshParam p;
    p.xstart = -0.35; p.ystart = 0.2; p.L = 2.7; p.H = 1.1;
    p.seed = 141; p.perturbation = 0.20;
    return p;
}

PetscErrorCode MakeMesh(MPI_Comm comm, PetscInt nx, PetscInt ny,
                        PetscInt px, PetscInt py, PetscInt kind, DM& dm, MeshInfo& mesh)
{
    PetscFunctionBeginUser;
    Vec coordinates = nullptr;
    PetscCall(DMDACreate2d(comm, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE, DMDA_STENCIL_BOX,
                          nx, ny, px, py, 2, 1, nullptr, nullptr, &dm));
    PetscCall(DMSetUp(dm));
    PetscCall(DMCreateGlobalVector(dm, &coordinates));
    if (kind == 0) PetscCall(CreateFullMesh(dm, coordinates, Domain()));
    else if (kind == 1) PetscCall(LogicRectMesh(dm, coordinates, Domain()));
    else PetscCall(RefineMesh(dm, coordinates, Domain()));
    PetscCall(BuildMeshInfo(dm, coordinates, mesh));
    PetscCall(VecDestroy(&coordinates));
    PetscFunctionReturn(PETSC_SUCCESS);
}

// Shoelace area/centroid: independent of bilinear quadrature and cached areas.
PetscReal PolygonArea(const QuadVertices& q, Point* centroid = nullptr)
{
    PetscReal twiceArea = 0, cx = 0, cy = 0;
    for (std::size_t e = 0; e < 4; ++e) {
        const auto& a = q[e]; const auto& b = q[(e+1)%4];
        const PetscReal cross = a.p[0]*b.p[1] - b.p[0]*a.p[1];
        twiceArea += cross;
        cx += (a.p[0]+b.p[0])*cross;
        cy += (a.p[1]+b.p[1])*cross;
    }
    if (centroid) *centroid = {{cx/(3*twiceArea), cy/(3*twiceArea)}};
    return twiceArea/2;
}

PetscReal PorosityProfile(const Point& p)
{ return PetscReal(0.18) + PetscReal(0.012)*p.p[0] + PetscReal(0.009)*p.p[1]; }
Point VariableForce(const Point& p)
{ return {{PetscReal(0.7)-PetscReal(0.2)*p.p[0]+PetscReal(0.4)*p.p[1],
           PetscReal(-0.3)+PetscReal(0.5)*p.p[0]+PetscReal(0.1)*p.p[1]}}; }
Point ConstantForce(const Point&) { return {{1,-2}}; }

struct Samples {
    const MeshInfo& mesh;
    const GaussRule1D& cellRule;
    const GaussRule1D& edgeRule;
    Field field;
    PetscReal uniformPhi = -1; // Negative selects a variable physical profile.
    std::vector<int>* visits = nullptr;

    PetscErrorCode operator()(MeshIndex index, LocalPorositySamples& result) const
    {
        PetscFunctionBeginUser;
        PetscInt id = 0;
        PetscCall(mesh.CellId(index, id));
        if (visits) {
            PetscCall(Require(mesh.OwnsCell(index), "Assembly evaluated a ghost/unowned cell"));
            PetscCall(Require(result.cell.empty() && result.edge[0].empty() &&
                              result.edge[1].empty() && result.edge[2].empty() &&
                              result.edge[3].empty() && PetscIsInfOrNanReal(result.average),
                              "Porosity callback did not receive fresh samples"));
            ++(*visits)[static_cast<std::size_t>(id)];
        }
        QuadVertices q; Point centroid;
        PetscCall(mesh.GetCellCorners(index, q));
        PetscCall(Require(PolygonArea(q, &centroid) > 0, "Invalid reference cell area"));
        const PetscReal jump = PetscReal(0.003)*static_cast<PetscReal>(id%7);
        const std::size_t n = cellRule.points.size();
        result.cell.resize(n*n);
        for (std::size_t j = 0; j < n; ++j)
            for (std::size_t i = 0; i < n; ++i) {
                const Point x = MapCellPoint({{cellRule.points[i],cellRule.points[j]}}, q);
                result.cell[j*n+i] = uniformPhi >= 0 ? uniformPhi : PorosityProfile(x)+jump;
            }
        // Stokes deliberately leaves average unset and all edges empty.
        if (field != Field::Stokes)
            result.average = uniformPhi >= 0 ? uniformPhi : PorosityProfile(centroid)+jump;
        if (field == Field::Darcy) {
            for (std::size_t e = 0; e < 4; ++e) {
                result.edge[e].resize(edgeRule.points.size());
                for (std::size_t g = 0; g < edgeRule.points.size(); ++g) {
                    const Point x = MapEdgePoint(edgeRule.points[g], {{q[e],q[(e+1)%4]}});
                    // Continuous edge trace, with different cell averages.
                    result.edge[e][g] = uniformPhi >= 0 ? uniformPhi : PorosityProfile(x);
                }
            }
        }
        PetscFunctionReturn(PETSC_SUCCESS);
    }
};

PetscErrorCode CheckCoverage(const std::vector<int>& visits)
{
    PetscFunctionBeginUser;
    std::vector<int> total(visits.size());
    PetscCallMPI(MPI_Allreduce(visits.data(), total.data(), static_cast<int>(visits.size()),
                               MPI_INT, MPI_SUM, PETSC_COMM_WORLD));
    for (int count : total)
        PetscCall(Require(count == 1, "An element was omitted or assembled more than once"));
    PetscFunctionReturn(PETSC_SUCCESS);
}

template<class Expected>
PetscErrorCode CheckMatrix(Mat matrix, const DofMap& rows, const DofMap& columns,
                           const Expected& expected, const char* name)
{
    PetscFunctionBeginUser;
    PetscCall(Require(matrix != nullptr, "Missing assembled matrix"));
    PetscInt m=0,n=0,M=0,N=0,begin=0,end=0,cbegin=0,cend=0;
    PetscBool assembled = PETSC_FALSE;
    PetscCall(MatGetSize(matrix, &M, &N));
    PetscCall(MatGetLocalSize(matrix, &m, &n));
    PetscCall(MatGetOwnershipRange(matrix, &begin, &end));
    PetscCall(MatGetOwnershipRangeColumn(matrix, &cbegin, &cend));
    PetscCall(MatAssembled(matrix, &assembled));
    PetscCall(Require(assembled && M == rows.GlobalDofs() && N == columns.GlobalDofs() &&
                      m == rows.OwnedDofs() && n == columns.OwnedDofs() &&
                      begin == rows.OwnershipBegin() && end == rows.OwnershipEnd() &&
                      cbegin == columns.OwnershipBegin() && cend == columns.OwnershipEnd(),
                      "Matrix dimensions, orientation, ownership, or assembled state is wrong"));
    std::vector<PetscInt> ids(static_cast<std::size_t>(N)), natural(ids.size());
    std::iota(ids.begin(), ids.end(), PetscInt(0));
    for (PetscInt col = 0; col < N; ++col)
        PetscCall(columns.GlobalToNatural(col, natural[static_cast<std::size_t>(col)]));
    std::vector<PetscScalar> values(ids.size());
    // MatGetValues permits only OWNED rows. All global columns are valid.
    for (PetscInt row = begin; row < end; ++row) {
        PetscInt r = 0;
        PetscCall(rows.GlobalToNatural(row, r));
        PetscCall(MatGetValues(matrix, 1, &row, N, ids.data(), values.data()));
        for (std::size_t col = 0; col < values.size(); ++col)
            PetscCall(Near(values[col], expected(r, natural[col]), name));
    }
    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode ReadVector(Vec vector, const DofMap& map, std::vector<PetscScalar>& values)
{
    PetscFunctionBeginUser;
    PetscInt global=0,local=0,begin=0,end=0;
    PetscCall(VecGetSize(vector, &global));
    PetscCall(VecGetLocalSize(vector, &local));
    PetscCall(VecGetOwnershipRange(vector, &begin, &end));
    PetscCall(Require(global == map.GlobalDofs() && local == map.OwnedDofs() &&
                      begin == map.OwnershipBegin() && end == map.OwnershipEnd(),
                      "Vector does not use the field's full DOF layout"));
    values.resize(static_cast<std::size_t>(local));
    const PetscScalar* array = nullptr;
    PetscCall(VecGetArrayRead(vector, &array));
    if (local) std::copy(array, array+local, values.begin());
    PetscCall(VecRestoreArrayRead(vector, &array));
    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode CheckVector(Vec vector, const DofMap& map,
                           const std::vector<PetscScalar>& expected, const char* name)
{
    PetscFunctionBeginUser;
    std::vector<PetscScalar> values;
    PetscCall(ReadVector(vector, map, values));
    for (PetscInt local = 0; local < map.OwnedDofs(); ++local) {
        PetscInt natural = 0;
        PetscCall(map.GlobalToNatural(map.OwnershipBegin()+local, natural));
        PetscCall(Near(values[static_cast<std::size_t>(local)],
                        expected[static_cast<std::size_t>(natural)], name));
    }
    PetscFunctionReturn(PETSC_SUCCESS);
}

struct Reference {
    std::vector<PetscScalar> A, B, C, f; // C stores its diagonal in natural order.
};

template<class Basis>
PetscErrorCode MakeReference(const MeshInfo& serial, const DofMap& velocity,
                             const Samples& samples, const LocalMatrixParameters& parameters,
                             const LocalForceFunction& force, Reference& reference)
{
    PetscFunctionBeginUser;
    constexpr std::size_t n = static_cast<std::size_t>(Basis::ElementDofs);
    const auto N = static_cast<std::size_t>(velocity.GlobalDofs());
    const auto P = static_cast<std::size_t>(serial.CellCount());
    reference.A.assign(N*N, 0); reference.B.assign(P*N, 0);
    reference.C.assign(P, 0); reference.f.assign(N, 0);
    std::vector<PetscInt> ids;
    for (PetscInt p = 0; p < serial.CellCount(); ++p) {
        MeshIndex cell;
        PetscCall(serial.CellIndex(p, cell));
        Basis basis;
        PetscCall(basis.Initialize(serial, cell));
        LocalPorositySamples phi;
        PetscCall(samples(cell, phi));
        LocalMatrixBlock<n> local;
        if constexpr (std::is_same_v<Basis, BRMixed>)
            PetscCall(ComputeLocalStokes(basis, samples.cellRule, phi, force, local));
        else
            PetscCall(ComputeLocalDarcy(basis, samples.cellRule, samples.edgeRule,
                                        phi, parameters, force, local));
        PetscCall(velocity.GetCellNaturalDofs(cell, ids));
        for (std::size_t a = 0; a < n; ++a) {
            const auto row = static_cast<std::size_t>(ids[a]);
            reference.f[row] += local.f[a];
            reference.B[static_cast<std::size_t>(p)*N+row] += local.B[a];
            for (std::size_t b = 0; b < n; ++b)
                reference.A[row*N+static_cast<std::size_t>(ids[b])] += local.A[a*n+b];
        }
        reference.C[static_cast<std::size_t>(p)] += local.C;
    }
    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode CheckReference(const MixedBlocks& blocks, const DofMap& velocity,
                              const DofMap& pressure, const Reference& reference)
{
    PetscFunctionBeginUser;
    const auto N = static_cast<std::size_t>(velocity.GlobalDofs());
    PetscCall(CheckMatrix(blocks.A, velocity, velocity, [&](PetscInt r, PetscInt c) {
        return reference.A[static_cast<std::size_t>(r)*N+static_cast<std::size_t>(c)];
    }, "Velocity matrix versus serial natural-numbering reference"));
    PetscCall(CheckMatrix(blocks.B, pressure, velocity, [&](PetscInt r, PetscInt c) {
        return reference.B[static_cast<std::size_t>(r)*N+static_cast<std::size_t>(c)];
    }, "Pressure-by-velocity coupling versus serial reference"));
    PetscCall(CheckMatrix(blocks.C, pressure, pressure, [&](PetscInt r, PetscInt c) {
        return r == c ? reference.C[static_cast<std::size_t>(r)] : PetscScalar(0);
    }, "Compaction block versus serial reference"));
    PetscCall(CheckVector(blocks.f, velocity, reference.f, "Assembled force vector"));
    PetscCall(CheckVector(blocks.g, pressure,
                         std::vector<PetscScalar>(static_cast<std::size_t>(pressure.GlobalDofs())),
                         "Pressure load must start at zero"));
    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode Assemble(Field field, const MeshInfo& mesh, const DofMap& velocity,
                        const DofMap& pressure, const Samples& samples,
                        const LocalMatrixParameters& parameters,
                        const LocalForceFunction& force, MixedBlocks& blocks)
{
    PetscFunctionBeginUser;
    if (field == Field::Stokes)
        PetscCall(AssembleStokesBlocks(PETSC_COMM_WORLD, mesh, velocity, pressure,
                                       samples.cellRule, samples, force, blocks));
    else
        PetscCall(AssembleDarcyBlocks(PETSC_COMM_WORLD, mesh, velocity, pressure,
                                      samples.cellRule, samples.edgeRule, samples, parameters,
                                      force, blocks));
    PetscFunctionReturn(PETSC_SUCCESS);
}

// Coefficients for u=(x,y), or the rigid translation u=(1,0). For HDiv the
// radial field has zero linear edge mode, and midpoint normal constant mode.
PetscErrorCode SetAffineVelocity(Vec vector, const DofMap& map,
                                 const MeshInfo& serial, bool translation)
{
    PetscFunctionBeginUser;
    std::vector<PetscScalar> values(static_cast<std::size_t>(map.OwnedDofs()), 0);
    for (PetscInt id = map.OwnershipBegin(); id < map.OwnershipEnd(); ++id) {
        DofInfo info; PetscCall(map.GetDofInfo(id, info));
        PetscReal coefficient = 0;
        if (info.entity == DofEntity::Vertex) {
            MeshIndex index; Point point;
            PetscCall(serial.VertexIndex(info.entityId, index));
            PetscCall(serial.GetVertex(index, point));
            coefficient = translation ? (info.component == 0 ? 1 : 0) : point.p[info.component];
        } else if (map.Space() == DofSpace::HDivVelocity && info.component == 1) {
            EdgeVertices edge; EdgeTopology topology;
            PetscCall(serial.GetEdgeVertices(info.entityId, edge));
            PetscCall(serial.GetEdgeTopology(info.entityId, topology));
            const PetscReal dx = edge[1].p[0]-edge[0].p[0];
            const PetscReal dy = edge[1].p[1]-edge[0].p[1];
            const PetscReal length = std::hypot(dx,dy);
            const PetscReal sign = topology.axis == EdgeAxis::AlongI ? -1 : 1;
            const Point normal{{sign*dy/length,-sign*dx/length}};
            const Point value = translation ? Point{{1,0}} :
                Point{{(edge[0].p[0]+edge[1].p[0])/2,(edge[0].p[1]+edge[1].p[1])/2}};
            coefficient = normal.p[0]*value.p[0]+normal.p[1]*value.p[1];
        }
        values[static_cast<std::size_t>(id-map.OwnershipBegin())] = coefficient;
    }
    PetscScalar* array = nullptr;
    PetscCall(VecGetArray(vector, &array));
    if (!values.empty()) std::copy(values.begin(), values.end(), array);
    PetscCall(VecRestoreArray(vector, &array));
    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode CheckAnalytic(Field field, const MixedBlocks& blocks,
                            const DofMap& velocity, const DofMap& pressure,
                            const MeshInfo& serial, PetscReal phi, PetscReal theta)
{
    PetscFunctionBeginUser;
    Vec u=nullptr, Au=nullptr, Bu=nullptr;
    PetscCall(MatCreateVecs(blocks.A, &u, &Au));
    PetscCall(MatCreateVecs(blocks.B, nullptr, &Bu));
    PetscCall(SetAffineVelocity(u, velocity, serial, false));
    PetscCall(MatMult(blocks.A, u, Au));
    PetscScalar energy = 0, load = 0;
    PetscCall(VecDot(u, Au, &energy));
    PetscCall(VecDot(u, blocks.f, &load));
    const auto domain = Domain();
    const PetscReal area = domain.L*domain.H;
    const PetscReal xx = domain.xstart*domain.xstart + domain.xstart*domain.L + domain.L*domain.L/3;
    const PetscReal yy = domain.ystart*domain.ystart + domain.ystart*domain.H + domain.H*domain.H/3;
    const bool stokes = field == Field::Stokes;
    PetscCall(Near(energy, stokes ? PetscReal(4)*(1-phi)*area/3 : area*(xx+yy),
                    "Analytic affine velocity energy; Darcy mass must be unweighted"));
    const PetscReal forceWork = area*(domain.xstart+domain.L/2-2*(domain.ystart+domain.H/2));
    PetscCall(Near(load, (stokes ? 1-phi : PetscReal(1))*forceWork,
                    "Analytic affine velocity/body-force pairing"));
    PetscCall(MatMult(blocks.B, u, Bu));
    std::vector<PetscScalar> expected(static_cast<std::size_t>(pressure.GlobalDofs()));
    std::vector<PetscScalar> compaction(expected.size());
    const PetscReal weight = stokes ? PetscReal(1) :
        (phi == 0 ? PetscReal(0) : std::pow(phi,1+theta)/PetscSqrtReal(phi));
    for (PetscInt p = 0; p < serial.CellCount(); ++p) {
        MeshIndex cell; QuadVertices q;
        PetscCall(serial.CellIndex(p, cell)); PetscCall(serial.GetCellCorners(cell, q));
        const PetscReal cellArea = PolygonArea(q);
        expected[static_cast<std::size_t>(p)] = 2*weight*cellArea;
        compaction[static_cast<std::size_t>(p)] = cellArea*(stokes ? phi : PetscReal(1))/(1-phi);
    }
    PetscCall(CheckVector(Bu, pressure, expected, "Analytic cell divergence and raw B sign"));
    PetscCall(CheckMatrix(blocks.C, pressure, pressure, [&](PetscInt r, PetscInt c) {
        return r == c ? compaction[static_cast<std::size_t>(r)] : PetscScalar(0);
    }, "Analytic compaction diagonal and raw C sign"));
    if (stokes) {
        PetscCall(SetAffineVelocity(u, velocity, serial, true));
        PetscCall(MatMult(blocks.A, u, Au));
        PetscReal norm = 0;
        PetscCall(VecNorm(Au, NORM_INFINITY, &norm));
        PetscCall(Near(norm, 0, "Unconstrained Stokes A must annihilate rigid translation"));
    } else if (phi == 0) {
        PetscCall(CheckMatrix(blocks.B, pressure, velocity,
                              [](PetscInt, PetscInt) { return PetscScalar(0); },
                              "Dry Darcy B must be zero"));
    }
    // Boundary DOFs must still be present, with their raw domain contributions.
    for (PetscInt id = velocity.OwnershipBegin(); id < velocity.OwnershipEnd(); ++id) {
        DofInfo info; PetscCall(velocity.GetDofInfo(id, info));
        if (info.IsBoundary()) {
            PetscScalar diagonal = 0;
            PetscCall(MatGetValues(blocks.A, 1, &id, 1, &id, &diagonal));
            PetscCall(Require(!PetscIsInfOrNanScalar(diagonal) && PetscRealPart(diagonal) > 0,
                              "A boundary velocity DOF lost its diagonal contribution"));
        }
    }
    PetscCall(VecDestroy(&u)); PetscCall(VecDestroy(&Au)); PetscCall(VecDestroy(&Bu));
    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode CheckSubsystem(Field field, DM dm, const MeshInfo& mesh, const MeshInfo& serial)
{
    PetscFunctionBeginUser;
    // No maps or blocks from the OTHER velocity subsystem are constructed here.
    DofMap velocity, pressure;
    PetscCall(velocity.Initialize(dm, mesh, field == Field::Stokes ? DofSpace::BRVelocity : DofSpace::HDivVelocity));
    PetscCall(pressure.Initialize(dm, mesh, DofSpace::CellPressure));
    GaussRule1D cellRule, edgeRule;
    PetscCall(CreateGaussRule(4, cellRule)); PetscCall(CreateGaussRule(3, edgeRule));
    LocalMatrixParameters parameters; parameters.theta = PetscReal(0.5);
    std::vector<int> visits(static_cast<std::size_t>(mesh.CellCount()), 0);
    Samples samples{mesh,cellRule,edgeRule,field,-1,&visits};
    Samples referenceSamples{serial,cellRule,edgeRule,field,-1,nullptr};
    MixedBlocks blocks;
    PetscCall(Assemble(field, mesh, velocity, pressure, samples, parameters, VariableForce, blocks));
    PetscCall(CheckCoverage(visits));
    Reference reference;
    if (field == Field::Stokes)
        PetscCall(MakeReference<BRMixed>(serial, velocity, referenceSamples, parameters, VariableForce, reference));
    else
        PetscCall(MakeReference<HDivMixed>(serial, velocity, referenceSamples, parameters, VariableForce, reference));
    PetscCall(CheckReference(blocks, velocity, pressure, reference));
    PetscCall(DestroyMixedBlocks(blocks));
    PetscCall(Require(blocks.IsEmpty(), "DestroyMixedBlocks did not clear all handles"));

    // Reuse the same output with changed quadrature, theta, porosity and force.
    PetscCall(CreateGaussRule(5, cellRule)); PetscCall(CreateGaussRule(4, edgeRule));
    parameters.theta = PetscReal(0.7);
    for (PetscReal phi : {PetscReal(0.25), PetscReal(0)}) {
        Samples uniform{mesh,cellRule,edgeRule,field,phi,nullptr};
        PetscCall(Assemble(field, mesh, velocity, pressure, uniform, parameters, ConstantForce, blocks));
        PetscCall(CheckAnalytic(field, blocks, velocity, pressure, serial, phi, parameters.theta));
        PetscCall(DestroyMixedBlocks(blocks));
    }
    PetscCall(DestroyMixedBlocks(blocks)); // Empty cleanup must be harmless.
    if (mesh.CellCount() == 1) {
        const int empty = velocity.OwnedDofs() == 0 ? 1 : 0;
        int emptyRanks = 0; PetscMPIInt ranks = 0;
        PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD, &ranks));
        PetscCallMPI(MPI_Allreduce(&empty, &emptyRanks, 1, MPI_INT, MPI_SUM, PETSC_COMM_WORLD));
        if (field == Field::Darcy && ranks == 4)
            PetscCall(Require(emptyRanks == 1, "Minimal 2x2 layout did not exercise an empty HDiv owner"));
    }
    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode CheckCoupling(DM dm, const MeshInfo& mesh, const MeshInfo& serial)
{
    PetscFunctionBeginUser;
    // Pressure-only assembly must work without either velocity DofMap.
    DofMap s, d;
    PetscCall(s.Initialize(dm, mesh, DofSpace::CellPressure));
    PetscCall(d.Initialize(dm, mesh, DofSpace::CellPressure));
    GaussRule1D cellRule, edgeRule;
    PetscCall(CreateGaussRule(4, cellRule)); PetscCall(CreateGaussRule(2, edgeRule));
    LocalMatrixParameters parameters;
    for (PetscReal phi : {PetscReal(-1), PetscReal(0.25), PetscReal(0)}) {
        std::vector<int> visits(static_cast<std::size_t>(mesh.CellCount()), 0);
        Samples source{mesh,cellRule,edgeRule,Field::Coupling,phi,&visits};
        Samples referenceSource{serial,cellRule,edgeRule,Field::Coupling,phi,nullptr};
        Mat matrix = nullptr;
        // Exercise both a distinct pressure map and the documented shared map.
        const DofMap& columns = phi == 0 ? s : d;
        PetscCall(AssemblePressureCoupling(PETSC_COMM_WORLD, mesh, s, columns,
                                           cellRule, source, parameters, matrix));
        PetscCall(CheckCoverage(visits));
        std::vector<PetscScalar> expected(static_cast<std::size_t>(serial.CellCount()));
        for (PetscInt p = 0; p < serial.CellCount(); ++p) {
            MeshIndex index; QuadVertices q;
            PetscCall(serial.CellIndex(p, index)); PetscCall(serial.GetCellCorners(index, q));
            if (phi >= 0) {
                expected[static_cast<std::size_t>(p)] = -PolygonArea(q)*PetscSqrtReal(phi)/(1-phi);
            } else {
                QuadBasis basis; LocalPorositySamples samples;
                PetscCall(basis.Initialize(q)); PetscCall(referenceSource(index, samples));
                PetscCall(ComputeLocalCoupling(basis, cellRule, samples, parameters,
                                               expected[static_cast<std::size_t>(p)]));
            }
        }
        PetscCall(CheckMatrix(matrix, s, columns, [&](PetscInt r, PetscInt c) {
            return r == c ? expected[static_cast<std::size_t>(r)] : PetscScalar(0);
        }, "Independent pressure coupling, negative sign and dry limit"));
        PetscCall(MatDestroy(&matrix));
        PetscCall(Require(!matrix, "Coupling cleanup did not clear its handle"));
    }
    // Runtime cutoff, rather than a compiled default, must select the branch.
    parameters.couplingAverageCutoff = PetscReal(0.5);
    Samples source{mesh,cellRule,edgeRule,Field::Coupling,PetscReal(0.25),nullptr};
    Mat matrix = nullptr;
    PetscCall(AssemblePressureCoupling(PETSC_COMM_WORLD, mesh, s, d,
                                       cellRule, source, parameters, matrix));
    PetscCall(CheckMatrix(matrix, s, d, [](PetscInt, PetscInt) { return PetscScalar(0); },
                          "Updated coupling cutoff was ignored"));
    PetscCall(MatDestroy(&matrix));
    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode CheckErrors(DM dm, const MeshInfo& mesh, const MeshInfo& serial)
{
    PetscFunctionBeginUser;
    DofMap u,w,p,uninitialized;
    PetscCall(u.Initialize(dm, mesh, DofSpace::BRVelocity));
    PetscCall(w.Initialize(dm, mesh, DofSpace::HDivVelocity));
    PetscCall(p.Initialize(dm, mesh, DofSpace::CellPressure));
    PetscMPIInt rank=0,ranks=0;
    PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD, &rank));
    PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD, &ranks));
    const bool badRank = rank == ranks-1;
    GaussRule1D cell,edge;
    PetscCall(CreateGaussRule(3, cell)); PetscCall(CreateGaussRule(2, edge));
    LocalMatrixParameters parameters;
    // Superset of samples is valid for all three assembly functions.
    Samples source{mesh,cell,edge,Field::Darcy,PetscReal(0.25),nullptr};
    CellPorosityFunction provider = source;
    MixedBlocks blocks; Mat coupling = nullptr;
    const auto stokes = [&]() { return AssembleStokesBlocks(PETSC_COMM_WORLD,mesh,u,p,cell,provider,{},blocks); };
    const auto darcy = [&]() { return AssembleDarcyBlocks(PETSC_COMM_WORLD,mesh,w,p,cell,edge,provider,parameters,{},blocks); };
    const auto couple = [&]() { return AssemblePressureCoupling(PETSC_COMM_WORLD,mesh,p,p,cell,provider,parameters,coupling); };

    PetscCall(ExpectCollectiveError([&] {
        return AssembleStokesBlocks(PETSC_COMM_WORLD,mesh,badRank ? uninitialized : u,p,cell,provider,{},blocks);
    }, PETSC_ERR_ARG_WRONG, "Uninitialized map on one rank"));
    PetscCall(ExpectCollectiveError([&] {
        return AssembleStokesBlocks(PETSC_COMM_WORLD,mesh,badRank ? w : u,p,cell,provider,{},blocks);
    }, PETSC_ERR_ARG_WRONG, "Wrong velocity space on one rank"));
    PetscCall(ExpectCollectiveError([&] {
        return AssembleDarcyBlocks(PETSC_COMM_WORLD,mesh,w,badRank ? u : p,cell,edge,provider,parameters,{},blocks);
    }, PETSC_ERR_ARG_WRONG, "Wrong pressure space on one rank"));
    const MeshInfo emptyMesh;
    PetscCall(ExpectCollectiveError([&] {
        return AssemblePressureCoupling(PETSC_COMM_WORLD,badRank ? emptyMesh : mesh,p,p,cell,provider,parameters,coupling);
    }, PETSC_ERR_ARG_WRONG, "Uninitialized mesh on one rank"));

    if (badRank) provider = {};
    PetscCall(ExpectCollectiveError(stokes, PETSC_ERR_ARG_NULL, "Missing Stokes provider"));
    PetscCall(ExpectCollectiveError(darcy, PETSC_ERR_ARG_NULL, "Missing Darcy provider"));
    PetscCall(ExpectCollectiveError(couple, PETSC_ERR_ARG_NULL, "Missing coupling provider"));
    provider = source;
    const auto goodCell = cell, goodEdge = edge;
    if (badRank) cell.weights[0] = -1;
    PetscCall(ExpectCollectiveError(stokes, PETSC_ERR_ARG_OUTOFRANGE, "Invalid cell rule on one rank"));
    cell = goodCell;
    if (badRank) edge.weights.clear();
    PetscCall(ExpectCollectiveError(darcy, PETSC_ERR_ARG_WRONG, "Invalid edge rule on one rank"));
    edge = goodEdge;

    provider = [&](MeshIndex index, LocalPorositySamples& samples) -> PetscErrorCode {
        if (index == MeshIndex{0,0}) return static_cast<PetscErrorCode>(PETSC_ERR_USER);
        return source(index, samples);
    };
    PetscCall(ExpectCollectiveError(stokes, PETSC_ERR_USER, "Rank-local Stokes callback error"));
    PetscCall(ExpectCollectiveError(darcy, PETSC_ERR_USER, "Rank-local Darcy callback error"));
    PetscCall(ExpectCollectiveError(couple, PETSC_ERR_USER, "Rank-local coupling callback error"));
    provider = [&](MeshIndex index, LocalPorositySamples& samples) -> PetscErrorCode {
        if (index == MeshIndex{0,0}) throw std::runtime_error("intentional callback exception");
        return source(index, samples);
    };
    PetscCall(ExpectCollectiveError(darcy, PETSC_ERR_USER, "Porosity callback exception"));
    provider = [&](MeshIndex index, LocalPorositySamples& samples) -> PetscErrorCode {
        PetscFunctionBeginUser;
        PetscCall(source(index, samples));
        if (index == MeshIndex{0,0}) samples.cell[0] = 1;
        PetscFunctionReturn(PETSC_SUCCESS);
    };
    PetscCall(ExpectCollectiveError(stokes, PETSC_ERR_ARG_OUTOFRANGE, "Invalid cell sample on one rank"));
    provider = [&](MeshIndex index, LocalPorositySamples& samples) -> PetscErrorCode {
        PetscFunctionBeginUser;
        PetscCall(source(index, samples));
        if (index == MeshIndex{0,0}) samples.edge[2].clear();
        PetscFunctionReturn(PETSC_SUCCESS);
    };
    PetscCall(ExpectCollectiveError(darcy, PETSC_ERR_ARG_SIZ, "Missing Darcy edge samples on one rank"));
    provider = [&](MeshIndex index, LocalPorositySamples& samples) -> PetscErrorCode {
        PetscFunctionBeginUser;
        PetscCall(source(index, samples));
        if (index == MeshIndex{0,0}) samples.average = std::numeric_limits<PetscReal>::quiet_NaN();
        PetscFunctionReturn(PETSC_SUCCESS);
    };
    PetscCall(ExpectCollectiveError(couple, PETSC_ERR_ARG_OUTOFRANGE, "Missing coupling average on one rank"));
    provider = source;
    PetscInt first = 0; PetscMPIInt firstOwner = 0;
    PetscCall(p.NaturalToGlobal(0, first)); PetscCall(p.GetOwnerRank(first, firstOwner));
    const LocalForceFunction badForce = [&](const Point&) -> Point {
        if (rank == firstOwner) throw std::runtime_error("intentional force exception");
        return {{0,0}};
    };
    PetscCall(ExpectCollectiveError([&] {
        return AssembleStokesBlocks(PETSC_COMM_WORLD,mesh,u,p,cell,provider,badForce,blocks);
    }, PETSC_ERR_USER, "Force callback exception on one cell owner"));
    PetscCall(Require(blocks.IsEmpty() && !coupling, "A failed assembly published partial output"));

    // Successful recovery, nonempty-output rejection and unchanged contents.
    PetscCall(stokes());
    const std::array<Mat,3> oldMatrices{{blocks.A,blocks.B,blocks.C}};
    const std::array<Vec,2> oldVectors{{blocks.f,blocks.g}};
    PetscCall(ExpectCollectiveError(stokes, PETSC_ERR_ARG_WRONG, "Nonempty subsystem output"));
    PetscCall(Require(oldMatrices == std::array<Mat,3>{{blocks.A,blocks.B,blocks.C}} &&
                      oldVectors == std::array<Vec,2>{{blocks.f,blocks.g}},
                      "Rejected reuse replaced a live output handle"));
    Samples referenceSamples{serial,cell,edge,Field::Darcy,PetscReal(0.25),nullptr};
    Reference reference;
    PetscCall(MakeReference<BRMixed>(serial,u,referenceSamples,parameters,{},reference));
    PetscCall(CheckReference(blocks,u,p,reference));
    PetscCall(DestroyMixedBlocks(blocks));
    PetscCall(darcy());
    PetscCall(MakeReference<HDivMixed>(serial,w,referenceSamples,parameters,{},reference));
    PetscCall(CheckReference(blocks,w,p,reference));
    PetscCall(DestroyMixedBlocks(blocks));
    PetscCall(couple());
    const Mat oldCoupling = coupling;
    PetscCall(ExpectCollectiveError(couple, PETSC_ERR_ARG_WRONG, "Nonempty coupling output"));
    PetscCall(Require(coupling == oldCoupling, "Rejected coupling reuse replaced its handle"));
    PetscCall(CheckMatrix(coupling,p,p,[&](PetscInt r, PetscInt c) {
        // At phi=average=1/4, K = -sqrt(phi)*C_D = -C_D/2.
        return r == c ? PetscScalar(-0.5)*reference.C[static_cast<std::size_t>(r)] : PetscScalar(0);
    }, "Coupling contents after rejected reuse"));
    PetscCall(MatDestroy(&coupling));
    PetscCall(DestroyMixedBlocks(blocks));
    PetscCall(Require(blocks.IsEmpty() && !coupling, "Final cleanup left a live handle"));
    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode Run()
{
    PetscFunctionBeginUser;
    PetscInt kind=0,suite=0,expectedRanks=1,px=1,py=1,nx=9,ny=7;
    PetscCall(PetscOptionsGetInt(nullptr,nullptr,"-assembly_mesh_type",&kind,nullptr));
    PetscCall(PetscOptionsGetInt(nullptr,nullptr,"-assembly_case",&suite,nullptr));
    PetscCall(PetscOptionsGetInt(nullptr,nullptr,"-expected_ranks",&expectedRanks,nullptr));
    PetscCall(PetscOptionsGetInt(nullptr,nullptr,"-mesh_px",&px,nullptr));
    PetscCall(PetscOptionsGetInt(nullptr,nullptr,"-mesh_py",&py,nullptr));
    PetscCall(PetscOptionsGetInt(nullptr,nullptr,"-mesh_nx",&nx,nullptr));
    PetscCall(PetscOptionsGetInt(nullptr,nullptr,"-mesh_ny",&ny,nullptr));
    PetscMPIInt ranks = 0; PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD,&ranks));
    PetscCheck(expectedRanks == ranks,PETSC_COMM_WORLD,PETSC_ERR_ARG_WRONG,
               "Expected %d ranks but PETSc sees %d; use the MPICH launcher matching PETSc",
               static_cast<int>(expectedRanks),static_cast<int>(ranks));
    PetscCall(Require(kind >= 0 && kind <= 2 && (suite == 0 || suite == 1),
                      "Use assembly_mesh_type 0/1/2 and assembly_case 0 (numerics) or 1 (errors)"));
    PetscCall(Require((px == 1 || px == 2) && (py == 1 || py == 2) && px*py == ranks,
                      "Use a matching 1x1, 2x1, 1x2 or 2x2 process grid"));
    // Small meshes are intentional: each rank holds a dense numerical oracle.
    PetscCall(Require(nx >= 2 && ny >= 2 && nx >= px && ny >= py && nx <= 16 && ny <= 16,
                      "Dense-reference tests require 2..16 vertices in each direction"));
    DM dm=nullptr,serialDM=nullptr; MeshInfo mesh,serial;
    PetscCall(MakeMesh(PETSC_COMM_WORLD,nx,ny,px,py,kind,dm,mesh));
    PetscCall(MakeMesh(PETSC_COMM_SELF,nx,ny,1,1,kind,serialDM,serial));
    PetscCall(DMDestroy(&serialDM));
    if (suite == 0) {
        PetscCall(CheckSubsystem(Field::Stokes,dm,mesh,serial));
        PetscCall(CheckSubsystem(Field::Darcy,dm,mesh,serial));
        PetscCall(CheckCoupling(dm,mesh,serial));
        if (nx == 2 && ny == 2) {
            const auto size = mesh.OwnedCells().Size();
            const int empty = size.i*size.j == 0 ? 1 : 0;
            int total = 0;
            PetscCallMPI(MPI_Allreduce(&empty,&total,1,MPI_INT,MPI_SUM,PETSC_COMM_WORLD));
            PetscCall(Require(total == ranks-1, "Minimal mesh did not exercise empty cell owners"));
        }
    } else {
        PetscCall(CheckErrors(dm,mesh,serial));
    }
    PetscCall(DMDestroy(&dm));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD,"Assembly case %d, mesh type %d passed on %d rank(s)\n",
                          static_cast<int>(suite),static_cast<int>(kind),static_cast<int>(ranks)));
    PetscFunctionReturn(PETSC_SUCCESS);
}

} // namespace

int main(int argc, char** argv)
{
    const PetscErrorCode error = PetscInitialize(&argc,&argv,nullptr,
        "Assembly tests: -assembly_mesh_type 0/1/2; -assembly_case 0 numerics, 1 expected errors.\n");
    if (error) return static_cast<int>(error);
    // Unexpected local failure aborts WORLD, rather than stranding peers at a
    // subsequent collective. Expected errors are handled explicitly above.
    PetscCallAbort(PETSC_COMM_WORLD,Run());
    return static_cast<int>(PetscFinalize());
}
