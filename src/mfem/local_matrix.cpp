#include "local_matrix.h"

#include <cmath>
#include <exception>
#include <limits>

namespace {

PetscErrorCode CheckCellPorosity(const GaussRule1D& rule,
                                const LocalPorositySamples& porosity)
{
    PetscFunctionBeginUser;
    PetscCall(ValidateGaussRule(rule));
    const std::size_t n = rule.points.size();
    PetscCheck(n <= std::numeric_limits<std::size_t>::max() / n,
               PETSC_COMM_SELF, PETSC_ERR_ARG_SIZ,
               "Cell quadrature sample count overflows size_t");
    PetscCheck(porosity.cell.size() == n * n,
               PETSC_COMM_SELF, PETSC_ERR_ARG_SIZ,
               "Cell porosity has %zu samples; expected %zu for this cell rule",
               porosity.cell.size(), n * n);
    for (std::size_t g = 0; g < porosity.cell.size(); ++g) {
        const PetscReal phi = porosity.cell[g];
        PetscCheck(!PetscIsInfOrNanReal(phi) && phi >= 0 && phi < 1,
                   PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE,
                   "Cell porosity sample %zu must be finite and in [0,1); "
                   "phi=1 makes the compaction terms singular", g);
    }
    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode CheckEdgePorosity(const GaussRule1D& rule,
                                const LocalPorositySamples& porosity)
{
    PetscFunctionBeginUser;
    PetscCall(ValidateGaussRule(rule));
    for (std::size_t e = 0; e < porosity.edge.size(); ++e) {
        PetscCheck(porosity.edge[e].size() == rule.points.size(),
                   PETSC_COMM_SELF, PETSC_ERR_ARG_SIZ,
                   "Edge %zu has %zu porosity samples; expected %zu",
                   e, porosity.edge[e].size(), rule.points.size());
        for (std::size_t g = 0; g < porosity.edge[e].size(); ++g) {
            const PetscReal phi = porosity.edge[e][g];
            PetscCheck(!PetscIsInfOrNanReal(phi) && phi >= 0 && phi <= 1,
                       PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE,
                       "Porosity at edge %zu, sample %zu must be finite and in [0,1]",
                       e, g);
        }
    }
    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode CheckAverage(PetscReal average)
{
    PetscFunctionBeginUser;
    PetscCheck(!PetscIsInfOrNanReal(average) && average >= 0 && average < 1,
               PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE,
               "Supply a finite cell-average porosity in [0,1)");
    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode CheckCutoff(PetscReal cutoff, const char* name)
{
    PetscFunctionBeginUser;
    PetscCheck(!PetscIsInfOrNanReal(cutoff) && cutoff >= 0 && cutoff < 1,
               PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE,
               "%s must be finite and in [0,1)", name);
    PetscFunctionReturn(PETSC_SUCCESS);
}

// The geometry comes from an initialized, validated basis. Do not take abs(J):
// a nonpositive physical Jacobian must be reported rather than hidden.
PetscErrorCode CellQuadraturePoint(const QuadVertices& corners,
                                   const GaussRule1D& rule,
                                   std::size_t i, std::size_t j,
                                   Point& point, PetscReal& weight)
{
    PetscFunctionBeginUser;
    const Point reference{{rule.points[i], rule.points[j]}};
    const Point mapped = MapCellPoint(reference, corners);
    const PetscReal jacobian = CellJacobian(reference, corners);
    PetscCheck(!PetscIsInfOrNanReal(mapped.p[0]) &&
                   !PetscIsInfOrNanReal(mapped.p[1]) &&
                   !PetscIsInfOrNanReal(jacobian) && jacobian > 0,
               PETSC_COMM_SELF, PETSC_ERR_FP,
               "Cell quadrature (%zu,%zu) has an invalid point or Jacobian", i, j);
    const PetscReal physicalWeight = rule.weights[i] * rule.weights[j] * jacobian;
    PetscCheck(!PetscIsInfOrNanReal(physicalWeight) && physicalWeight > 0,
               PETSC_COMM_SELF, PETSC_ERR_FP,
               "Cell quadrature (%zu,%zu) has a non-finite or underflowed weight", i, j);
    point = mapped;
    weight = physicalWeight;
    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode EvaluateForce(const LocalForceFunction& force, const Point& point,
                             const char* name, Point& result)
{
    PetscFunctionBeginUser;
    Point value{};
    if (force) {
        // Only user code can throw here. No partially computed matrix is exposed.
        try {
            value = force(point);
        } catch (const std::exception& error) {
            SETERRQ(PETSC_COMM_SELF, PETSC_ERR_USER,
                    "%s force callback threw: %s", name, error.what());
        } catch (...) {
            SETERRQ(PETSC_COMM_SELF, PETSC_ERR_USER,
                    "%s force callback threw an unknown exception", name);
        }
    }
    PetscCheck(!PetscIsInfOrNanReal(value.p[0]) &&
                   !PetscIsInfOrNanReal(value.p[1]),
               PETSC_COMM_SELF, PETSC_ERR_FP,
               "%s force callback returned a non-finite component", name);
    result = value;
    PetscFunctionReturn(PETSC_SUCCESS);
}

// Both velocity bilinear forms are symmetric and real. Accumulate the upper
// triangle once, then copy it to the lower triangle for full row-major storage.
template <std::size_t N>
void CompleteSymmetry(LocalMatrixBlock<N>& block) noexcept
{
    for (std::size_t row = 1; row < N; ++row)
        for (std::size_t column = 0; column < row; ++column)
            block.A[row * N + column] = block.A[column * N + row];
}

template <std::size_t N>
PetscErrorCode CheckBlock(const LocalMatrixBlock<N>& block, const char* name)
{
    PetscFunctionBeginUser;
    for (std::size_t q = 0; q < block.A.size(); ++q)
        PetscCheck(!PetscIsInfOrNanScalar(block.A[q]),
                   PETSC_COMM_SELF, PETSC_ERR_FP,
                   "%s A[%zu] is not finite", name, q);
    for (std::size_t q = 0; q < N; ++q) {
        PetscCheck(!PetscIsInfOrNanScalar(block.B[q]),
                   PETSC_COMM_SELF, PETSC_ERR_FP,
                   "%s B[%zu] is not finite", name, q);
        PetscCheck(!PetscIsInfOrNanScalar(block.f[q]),
                   PETSC_COMM_SELF, PETSC_ERR_FP,
                   "%s f[%zu] is not finite", name, q);
    }
    PetscCheck(!PetscIsInfOrNanScalar(block.C),
               PETSC_COMM_SELF, PETSC_ERR_FP, "%s C is not finite", name);
    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode EdgePorosityPower(PetscReal phi, PetscReal exponent,
                                 std::size_t edge, std::size_t sample,
                                 PetscReal& result)
{
    PetscFunctionBeginUser;
    PetscReal value = 0;
    if (phi == 0) {
        PetscCheck(exponent >= 0, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE,
                   "Zero porosity at edge %zu, sample %zu requires 1+theta >= 0",
                   edge, sample);
        // Preserve the legacy pow(0,0)==1 convention explicitly.
        value = exponent == 0 ? PetscReal(1) : PetscReal(0);
    } else {
        value = std::pow(phi, exponent);
    }
    PetscCheck(!PetscIsInfOrNanReal(value), PETSC_COMM_SELF, PETSC_ERR_FP,
               "Porosity power is not finite at edge %zu, sample %zu", edge, sample);
    result = value;
    PetscFunctionReturn(PETSC_SUCCESS);
}

} // namespace

PetscErrorCode ComputeLocalStokes(
    const BRMixed& basis, const GaussRule1D& cellRule,
    const LocalPorositySamples& porosity, const LocalForceFunction& force,
    StokesLocalMatrix& result)
{
    PetscFunctionBeginUser;
    QuadVertices corners{};
    PetscCall(basis.GetCorners(corners));
    PetscCall(CheckCellPorosity(cellRule, porosity));

    StokesLocalMatrix work{};
    constexpr std::size_t ndof = BRMixed::ElementDofs;
    const PetscReal pressure = BRMixed::Pressure();
    const std::size_t n = cellRule.points.size();
    for (std::size_t j = 0; j < n; ++j) {
        for (std::size_t i = 0; i < n; ++i) {
            Point point{}, load{};
            PetscReal weight = 0;
            PetscCall(CellQuadraturePoint(corners, cellRule, i, j, point, weight));
            BRMixed::Values values{};
            PetscCall(basis.EvaluateAll(point, values));
            PetscCall(EvaluateForce(force, point, "Stokes", load));
            const PetscReal phi = porosity.cell[j * n + i];
            const PetscReal solid = 1 - phi;
            std::array<PetscReal, ndof> divergence{}, shear{};
            for (std::size_t a = 0; a < ndof; ++a) {
                const auto& gradient = values[a].gradient;
                divergence[a] = gradient[0] + gradient[3];
                shear[a] = PetscReal(0.5) * (gradient[1] + gradient[2]);
            }
            for (std::size_t row = 0; row < ndof; ++row) {
                const auto& gradientRow = values[row].gradient;
                for (std::size_t column = row; column < ndof; ++column) {
                    const auto& gradientColumn = values[column].gradient;
                    const PetscReal strainProduct =
                        gradientRow[0] * gradientColumn[0] +
                        2 * shear[row] * shear[column] +
                        gradientRow[3] * gradientColumn[3];
                    work.A[row * ndof + column] += 2 * solid * weight *
                        (strainProduct - divergence[row] * divergence[column] / 3);
                }
                work.B[row] += weight * divergence[row] * pressure;
                work.f[row] += weight * solid *
                    (load.p[0] * values[row].value.p[0] +
                     load.p[1] * values[row].value.p[1]);
            }
            work.C += weight * (phi / solid) * pressure * pressure;
        }
    }
    CompleteSymmetry(work);
    PetscCall(CheckBlock(work, "Stokes"));
    result = work;
    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode ComputeLocalDarcy(
    const HDivMixed& basis, const GaussRule1D& cellRule,
    const GaussRule1D& edgeRule, const LocalPorositySamples& porosity,
    const LocalMatrixParameters& parameters, const LocalForceFunction& force,
    DarcyLocalMatrix& result)
{
    PetscFunctionBeginUser;
    QuadVertices corners{};
    PetscCall(basis.GetCorners(corners));
    PetscCall(CheckCellPorosity(cellRule, porosity));
    PetscCall(CheckEdgePorosity(edgeRule, porosity));
    PetscCall(CheckAverage(porosity.average));
    PetscCall(CheckCutoff(parameters.darcyCompactionAverageCutoff,
                          "darcyCompactionAverageCutoff"));
    const PetscReal exponent = 1 + parameters.theta;
    PetscCheck(!PetscIsInfOrNanReal(parameters.theta) &&
                   !PetscIsInfOrNanReal(exponent),
               PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE,
               "theta and 1+theta must be finite");

    DarcyLocalMatrix work{};
    constexpr std::size_t ndof = HDivMixed::ElementDofs;
    const PetscReal pressure = HDivMixed::Pressure();
    const bool scaleCompaction =
        porosity.average > parameters.darcyCompactionAverageCutoff;
    const std::size_t n = cellRule.points.size();
    for (std::size_t j = 0; j < n; ++j) {
        for (std::size_t i = 0; i < n; ++i) {
            Point point{}, load{};
            PetscReal weight = 0;
            PetscCall(CellQuadraturePoint(corners, cellRule, i, j, point, weight));
            HDivMixed::Values values{};
            PetscCall(basis.EvaluateAll(point, values));
            PetscCall(EvaluateForce(force, point, "Darcy", load));
            const PetscReal phi = porosity.cell[j * n + i];
            const PetscReal solid = 1 - phi;
            for (std::size_t row = 0; row < ndof; ++row) {
                const Point& u = values[row].value;
                for (std::size_t column = row; column < ndof; ++column) {
                    const Point& v = values[column].value;
                    // Preserve the rescaled Darcy mass matrix without an
                    // additional porosity, viscosity, or permeability factor.
                    work.A[row * ndof + column] += weight *
                        (u.p[0] * v.p[0] + u.p[1] * v.p[1]);
                }
                work.f[row] += weight * (load.p[0] * u.p[0] + load.p[1] * u.p[1]);
            }
            const PetscReal ratio = scaleCompaction ? phi / porosity.average : PetscReal(1);
            work.C += weight * (ratio / solid) * pressure * pressure;
        }
    }

    // Exact-zero substitution is distinct from BOTH volume cutoffs.
    const PetscReal phiHat = porosity.average == 0 ? PetscReal(1) : porosity.average;
    const PetscReal inverseRootAverage = 1 / PetscSqrtReal(phiHat);
    PetscCheck(!PetscIsInfOrNanReal(inverseRootAverage),
               PETSC_COMM_SELF, PETSC_ERR_FP,
               "Darcy inverse square-root average is not finite");
    for (std::size_t e = 0; e < corners.size(); ++e) {
        const EdgeVertices edge{{corners[e], corners[(e + 1) % 4]}};
        Point outward{};
        PetscReal length = 0;
        // Right normal of this CCW-directed physical edge is outward. The
        // shared normal returned by HDivMixed::GetEdgeNormal is not suitable.
        PetscCall(GetEdgeGeometry(edge, length, outward));
        for (std::size_t g = 0; g < edgeRule.points.size(); ++g) {
            const Point point = MapEdgePoint(edgeRule.points[g], edge);
            const PetscReal weight = PetscReal(0.5) * length * edgeRule.weights[g];
            PetscCheck(!PetscIsInfOrNanReal(point.p[0]) &&
                           !PetscIsInfOrNanReal(point.p[1]) &&
                           !PetscIsInfOrNanReal(weight) && weight > 0,
                       PETSC_COMM_SELF, PETSC_ERR_FP,
                       "Invalid physical quadrature at edge %zu, sample %zu", e, g);
            PetscReal power = 0;
            PetscCall(EdgePorosityPower(porosity.edge[e][g], exponent, e, g, power));
            const PetscReal boundaryCoefficient = inverseRootAverage * power;
            PetscCheck(!PetscIsInfOrNanReal(boundaryCoefficient),
                       PETSC_COMM_SELF, PETSC_ERR_FP,
                       "Darcy boundary coefficient is not finite at edge %zu, sample %zu", e, g);
            HDivMixed::Values values{};
            PetscCall(basis.EvaluateAll(point, values));
            for (std::size_t row = 0; row < ndof; ++row) {
                const Point& u = values[row].value;
                work.B[row] += weight * boundaryCoefficient * pressure *
                    (u.p[0] * outward.p[0] + u.p[1] * outward.p[1]);
            }
        }
    }
    CompleteSymmetry(work);
    PetscCall(CheckBlock(work, "Darcy"));
    result = work;
    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode ComputeLocalCoupling(
    const QuadBasis& geometry, const GaussRule1D& cellRule,
    const LocalPorositySamples& porosity,
    const LocalMatrixParameters& parameters, PetscScalar& result)
{
    PetscFunctionBeginUser;
    QuadVertices corners{};
    PetscCall(geometry.GetCorners(corners));
    PetscCall(CheckCellPorosity(cellRule, porosity));
    PetscCall(CheckAverage(porosity.average));
    PetscCall(CheckCutoff(parameters.couplingAverageCutoff, "couplingAverageCutoff"));

    PetscScalar work{};
    const bool coupled = porosity.average > parameters.couplingAverageCutoff;
    // Avoid taking a square root or dividing by zero in the inactive branch.
    const PetscReal rootAverage = coupled ? PetscSqrtReal(porosity.average) : PetscReal(1);
    const PetscReal pressureProduct = BRMixed::Pressure() * HDivMixed::Pressure();
    const std::size_t n = cellRule.points.size();
    for (std::size_t j = 0; j < n; ++j) {
        for (std::size_t i = 0; i < n; ++i) {
            Point point{};
            PetscReal weight = 0;
            PetscCall(CellQuadraturePoint(corners, cellRule, i, j, point, weight));
            const PetscReal phi = porosity.cell[j * n + i];
            const PetscReal scale = coupled ? phi / rootAverage : PetscReal(0);
            work -= weight * (scale / (1 - phi)) * pressureProduct;
        }
    }
    PetscCheck(!PetscIsInfOrNanScalar(work), PETSC_COMM_SELF, PETSC_ERR_FP,
               "Local pressure coupling is not finite");
    result = work;
    PetscFunctionReturn(PETSC_SUCCESS);
}
