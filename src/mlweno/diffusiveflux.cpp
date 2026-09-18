#include "diffusiveflux.h"

#include <algorithm>
#include <cmath>
#include <utility>

namespace {

PetscErrorCode RealValue(long double value, PetscReal& result)
{
    PetscFunctionBeginUser;
    PetscCheck(std::isfinite(value) &&
               std::abs(value) <= std::numeric_limits<PetscReal>::max(),
               PETSC_COMM_SELF, PETSC_ERR_FP,
               "Diffusion geometry, flux or derivative is not representable");
    result = static_cast<PetscReal>(value);
    PetscFunctionReturn(PETSC_SUCCESS);
}

bool SamePoint(const Point& a, const Point& b)
{ return a.p[0] == b.p[0] && a.p[1] == b.p[1]; }

PetscErrorCode FindFace(const QuadVertices& cell, const EdgeVertices& edge,
                        bool reverse, std::size_t& face)
{
    PetscFunctionBeginUser;
    PetscCall(ValidateQuad(cell));
    for (std::size_t f = 0; f < 4; ++f) {
        if (SamePoint(cell[f], edge[reverse ? 1 : 0]) &&
            SamePoint(cell[(f + 1) % 4], edge[reverse ? 0 : 1])) {
            face = f;
            PetscFunctionReturn(PETSC_SUCCESS);
        }
    }
    PetscCheck(false, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG,
               "Edge must match the cell vertices with the required orientation");
    PetscFunctionReturn(PETSC_SUCCESS);
}

// Intersection of an inward unit ray from this face with the other three
// half-planes of a convex CCW quad. All subtractions precede products; this
// avoids introducing absolute coordinate offsets into determinant products.
PetscErrorCode RayDistance(const QuadVertices& cell, std::size_t face,
                           const Point& origin, const Point& direction,
                           PetscReal& distance)
{
    PetscFunctionBeginUser;
    long double exit = std::numeric_limits<long double>::infinity();
    for (std::size_t f = 0; f < 4; ++f) {
        if (f == face) continue;
        const Point& a = cell[f];
        const Point& b = cell[(f + 1) % 4];
        const long double tx = static_cast<long double>(b.p[0]) - a.p[0];
        const long double ty = static_cast<long double>(b.p[1]) - a.p[1];
        const long double rx = static_cast<long double>(origin.p[0]) - a.p[0];
        const long double ry = static_cast<long double>(origin.p[1]) - a.p[1];
        const long double inside = tx * ry - ty * rx;
        PetscCheck(std::isfinite(inside) && inside > 0, PETSC_COMM_SELF,
                   PETSC_ERR_ARG_WRONG,
                   "Edge quadrature point must lie strictly between the other cell faces");
        const long double rate = tx * direction.p[1] - ty * direction.p[0];
        PetscCheck(std::isfinite(rate), PETSC_COMM_SELF, PETSC_ERR_FP,
                   "Normal-ray intersection overflowed");
        if (rate < 0) exit = std::min(exit, inside / (-rate));
    }
    PetscReal work;
    PetscCall(RealValue(exit, work));
    PetscCheck(work > 0, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE,
               "Normal sampling requires positive available distance");
    distance = work;
    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode ShiftPoint(const Point& point, const Point& normal,
                          long double distance, Point& result)
{
    PetscFunctionBeginUser;
    Point work;
    for (int d = 0; d < 2; ++d)
        PetscCall(RealValue(static_cast<long double>(point.p[d]) +
                            distance * normal.p[d], work.p[d]));
    result = work;
    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode Quantity(PetscReal u, const DiffusiveFluxPoint& point,
                        const DiffusiveFluxLaw& law, bool derivatives,
                        DiffusiveQuantity& result)
{
    PetscFunctionBeginUser;
    PetscCheck(std::isfinite(u), PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE,
               "Diffusion sample state must be finite");
    PetscCheck(static_cast<bool>(law.evaluate), PETSC_COMM_SELF,
               PETSC_ERR_ARG_WRONG, "Missing diffused-quantity callback");
    DiffusiveQuantity work;
    PetscCall(law.evaluate(u, point, work));
    PetscCheck(std::isfinite(work.value) &&
               (!derivatives || std::isfinite(work.derivative)),
               PETSC_COMM_SELF, PETSC_ERR_FP,
               "Diffused quantity must return a finite value and requested derivative");
    result = work;
    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode IntegrateImpl(
    const DiffusiveSampling& sampling, const std::vector<PetscReal>& samples,
    const std::vector<PetscReal>& diffusivity,
    const std::vector<DiffusiveBoundaryValue>* boundary, PetscReal time,
    const DiffusiveFluxLaw& law, bool derivatives, DiffusiveEdgeFluxResult& result)
{
    PetscFunctionBeginUser;
    PetscCheck(sampling.IsInitialized(), PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG,
               "Create diffusion sampling before evaluating fluxes");
    PetscCheck(sampling.IsBoundary() == (boundary != nullptr), PETSC_COMM_SELF,
               PETSC_ERR_ARG_WRONG, "Use the matching interior or boundary flux function");
    PetscCheck(std::isfinite(time), PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE,
               "Diffusion time must be finite");
    const std::size_t nq = sampling.QuadraturePoints().size();
    const std::size_t count = sampling.SamplesPerPoint();
    PetscCheck(samples.size() == sampling.SamplePoints().size() &&
               diffusivity.size() == nq && (!boundary || boundary->size() == nq),
               PETSC_COMM_SELF, PETSC_ERR_ARG_SIZ,
               "Incorrect sample, diffusivity or boundary-data array size");
    DiffusiveEdgeFluxResult work;
    if (derivatives) {
        work.derivativeSamples.assign(samples.size(), 0);
        work.derivativeDiffusivity.assign(nq, 0);
        if (boundary) work.derivativeBoundaryData.assign(nq, 0);
    }
    std::vector<DiffusiveQuantity> values(count);
    // Use differences from the final value to preserve constant fields exactly.
    // The final coefficient is consequently -sum(other coefficients), also in
    // the Jacobian. This removes roundoff in the endpoint harmonic-sum identity.
    long double lastCoefficient = 0;
    for (std::size_t k = 0; k + 1 < count; ++k)
        lastCoefficient -= sampling.Coefficients()[k];
    long double integral = 0;
    for (std::size_t q = 0; q < nq; ++q) {
        const PetscReal weight = sampling.QuadratureWeights()[q];
        const auto type = boundary ? (*boundary)[q].type
                                  : DiffusiveBoundaryType::PrescribedState;
        if (boundary) {
            PetscCheck(type == DiffusiveBoundaryType::PrescribedState ||
                       type == DiffusiveBoundaryType::PrescribedQuantity ||
                       type == DiffusiveBoundaryType::PrescribedFlux ||
                       type == DiffusiveBoundaryType::ZeroFlux,
                       PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG,
                       "Unknown diffusion boundary condition");
            if (type == DiffusiveBoundaryType::ZeroFlux) continue;
            PetscCheck(std::isfinite((*boundary)[q].value), PETSC_COMM_SELF,
                       PETSC_ERR_ARG_OUTOFRANGE, "Boundary data must be finite");
            if (type == DiffusiveBoundaryType::PrescribedFlux) {
                integral += static_cast<long double>(weight) * (*boundary)[q].value;
                if (derivatives) work.derivativeBoundaryData[q] = weight;
                continue;
            }
        }
        const PetscReal kappa = diffusivity[q];
        PetscCheck(std::isfinite(kappa) && kappa >= 0, PETSC_COMM_SELF,
                   PETSC_ERR_ARG_OUTOFRANGE, "Diffusivity must be finite and nonnegative");
        // Even kappa=0 needs D_n(psi) for its derivative with respect to kappa.
        for (std::size_t k = 0; k < count; ++k) {
            const std::size_t a = q * count + k;
            const DiffusiveFluxPoint point{sampling.SamplePoints()[a], sampling.Normal(),
                                            time, sampling.SampleSides()[a]};
            if (boundary && k + 1 == count) {
                if (type == DiffusiveBoundaryType::PrescribedQuantity)
                    values[k] = {(*boundary)[q].value, 1};
                else PetscCall(Quantity((*boundary)[q].value, point, law,
                                         derivatives, values[k]));
            } else PetscCall(Quantity(samples[a], point, law, derivatives, values[k]));
        }
        long double gradient = 0;
        for (std::size_t k = 0; k + 1 < count; ++k)
            gradient += static_cast<long double>(sampling.Coefficients()[k]) *
                (static_cast<long double>(values[k].value) - values.back().value);
        gradient /= sampling.Spacing()[q];
        PetscCheck(std::isfinite(gradient), PETSC_COMM_SELF, PETSC_ERR_FP,
                   "Sampled normal derivative overflowed");
        integral -= static_cast<long double>(weight) * kappa * gradient;
        if (derivatives) {
            PetscCall(RealValue(-static_cast<long double>(weight) * gradient,
                                work.derivativeDiffusivity[q]));
            const long double scale = -static_cast<long double>(weight) * kappa /
                                       sampling.Spacing()[q];
            for (std::size_t k = 0; k < count; ++k) {
                const long double coefficient = k + 1 == count ? lastCoefficient
                                                              : sampling.Coefficients()[k];
                PetscReal partial;
                PetscCall(RealValue(scale * coefficient * values[k].derivative, partial));
                if (boundary && k + 1 == count) work.derivativeBoundaryData[q] = partial;
                else work.derivativeSamples[q * count + k] = partial;
            }
        }
    }
    PetscCall(RealValue(integral, work.flux));
    result = std::move(work);
    PetscFunctionReturn(PETSC_SUCCESS);
}

} // namespace

PetscErrorCode CreateLagrangeDerivativeWeights(
    PetscInt numberOfPoints, LagrangeDerivativeLocation location,
    std::vector<PetscReal>& weights)
{
    PetscFunctionBeginUser;
    PetscCheck(numberOfPoints >= 2, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE,
               "A first-derivative rule requires at least two points");
    PetscCheck(location == LagrangeDerivativeLocation::Middle ||
               location == LagrangeDerivativeLocation::EndLo ||
               location == LagrangeDerivativeLocation::EndHi,
               PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Unknown Lagrange derivative location");
    PetscCheck(location != LagrangeDerivativeLocation::Middle || numberOfPoints % 2 == 0,
               PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE,
               "Legacy midpoint derivative requires an even number of points");
    const PetscInt n = numberOfPoints - 1;
    std::vector<PetscReal> work(static_cast<std::size_t>(numberOfPoints));
    if (location == LagrangeDerivativeLocation::Middle) {
        // Legacy constantFactor and binomial formula; enforce exact paired
        // antisymmetry when casting the coefficients to PetscReal.
        long double factor = std::exp2(1.0L - n);
        for (PetscInt k = 3; k <= n; k += 2)
            factor *= static_cast<long double>(k) / (k - 1);
        if ((numberOfPoints / 2) % 2) factor = -factor;
        PetscCheck(factor != 0 && std::isfinite(factor), PETSC_COMM_SELF,
                   PETSC_ERR_FP, "Lagrange order exceeds representable coefficient range");
        long double choose = 1;
        for (PetscInt i = 0; i < numberOfPoints / 2; ++i) {
            const long double denominator = static_cast<long double>(n) - 2.0L * i;
            PetscCall(RealValue(factor * choose / (denominator * denominator), work[i]));
            work[n - i] = -work[i];
            choose *= static_cast<long double>(n - i) / (i + 1);
            factor = -factor;
        }
    } else {
        long double harmonic = 0;
        for (PetscInt j = 1; j <= n; ++j) harmonic += 1.0L / j;
        PetscCall(RealValue(harmonic, work[n]));
        long double choose = 1, sign = n % 2 ? -1 : 1;
        for (PetscInt i = 0; i < n; ++i) {
            PetscCall(RealValue(sign * choose / (n - i), work[i]));
            choose *= static_cast<long double>(n - i) / (i + 1);
            sign = -sign;
        }
        if (location == LagrangeDerivativeLocation::EndLo) {
            std::reverse(work.begin(), work.end());
            for (auto& value : work) value = -value;
        }
    }
    weights = std::move(work);
    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode DiffusiveSampling::Build(
    const EdgeVertices& edge, const QuadVertices& left,
    const QuadVertices* right, const GaussRule1D& rule,
    const DiffusiveSamplingOptions& options)
{
    PetscFunctionBeginUser;
    PetscCheck(options.numberOfSamples >= 2 && options.numberOfSamples % 2 == 0 &&
               options.numberOfSamples < std::numeric_limits<PetscInt>::max(),
               PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE,
               "Diffusion requires an even sample count >= 2 with room for a boundary node");
    PetscCheck(std::isfinite(options.extentFraction) && options.extentFraction > 0 &&
               options.extentFraction <= 1, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE,
               "Normal sampling extentFraction must lie in (0,1]");
    PetscCall(ValidateGaussRule(rule));
    PetscReal length;
    PetscCall(GetEdgeGeometry(edge, length, normal_));
    std::size_t leftFace, rightFace = 0;
    PetscCall(FindFace(left, edge, false, leftFace));
    if (right) PetscCall(FindFace(*right, edge, true, rightFace));
    boundary_ = right == nullptr;
    const PetscInt n = options.numberOfSamples;
    const PetscInt count = boundary_ ? n + 1 : n;
    const std::size_t nq = rule.points.size(), m = static_cast<std::size_t>(count);
    PetscCheck(nq <= std::numeric_limits<std::size_t>::max() / m,
               PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Diffusion sample array is too large");
    PetscCall(CreateLagrangeDerivativeWeights(count, boundary_
        ? LagrangeDerivativeLocation::EndHi : LagrangeDerivativeLocation::Middle,
        coefficients_));
    points_.resize(nq); spacing_.resize(nq); weights_.resize(nq);
    samples_.resize(nq * m); sides_.resize(nq * m);
    const Point inward{{-normal_.p[0], -normal_.p[1]}};
    for (std::size_t q = 0; q < nq; ++q) {
        PetscCheck(rule.points[q] > -1 && rule.points[q] < 1, PETSC_COMM_SELF,
                   PETSC_ERR_ARG_OUTOFRANGE, "Normal sampling needs open-edge quadrature points");
        points_[q] = MapEdgePoint(rule.points[q], edge);
        PetscReal available, other;
        PetscCall(RayDistance(left, leftFace, points_[q], inward, available));
        if (right) {
            PetscCall(RayDistance(*right, rightFace, points_[q], normal_, other));
            available = std::min(available, other);
        }
        const long double spacing = static_cast<long double>(options.extentFraction) *
            available * (boundary_ ? 1.0L / n : 2.0L / (n - 1));
        PetscCall(RealValue(spacing, spacing_[q]));
        PetscCheck(spacing_[q] > 0, PETSC_COMM_SELF, PETSC_ERR_FP,
                   "Normal sample spacing underflowed");
        PetscCall(RealValue(static_cast<long double>(length) * rule.weights[q] / 2,
                            weights_[q]));
        PetscCheck(weights_[q] > 0, PETSC_COMM_SELF, PETSC_ERR_FP,
                   "Physical edge quadrature weight underflowed");
        for (std::size_t k = 0; k < m; ++k) {
            const long double offset = boundary_ ? static_cast<long double>(k) - n
                                                 : static_cast<long double>(k) - (n - 1) / 2.0L;
            const std::size_t a = q * m + k;
            PetscCall(ShiftPoint(points_[q], normal_, offset * spacing_[q], samples_[a]));
            sides_[a] = boundary_ ? (k + 1 == m ? DiffusiveSampleSide::Boundary
                                                : DiffusiveSampleSide::Left)
                                  : (k < m / 2 ? DiffusiveSampleSide::Left
                                               : DiffusiveSampleSide::Right);
            long double signedDistance = 0;
            for (int d = 0; d < 2; ++d)
                signedDistance += (static_cast<long double>(samples_[a].p[d]) -
                                    points_[q].p[d]) * normal_.p[d];
            PetscCheck((offset < 0 && signedDistance < 0) ||
                       (offset > 0 && signedDistance > 0) ||
                       (offset == 0 && signedDistance == 0),
                       PETSC_COMM_SELF, PETSC_ERR_FP,
                       "Coordinates cannot resolve the requested normal sample offset");
            if (k > 0) {
                long double step = 0;
                for (int d = 0; d < 2; ++d)
                    step += (static_cast<long double>(samples_[a].p[d]) -
                              samples_[a - 1].p[d]) * normal_.p[d];
                PetscCheck(step > 0, PETSC_COMM_SELF, PETSC_ERR_FP,
                           "Coordinates cannot resolve distinct normal samples");
            }
        }
    }
    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode CreateInteriorDiffusiveSampling(
    const EdgeVertices& edge, const QuadVertices& leftCell,
    const QuadVertices& rightCell, const GaussRule1D& rule,
    const DiffusiveSamplingOptions& options, DiffusiveSampling& sampling)
{
    PetscFunctionBeginUser;
    DiffusiveSampling work;
    PetscCall(work.Build(edge, leftCell, &rightCell, rule, options));
    sampling = std::move(work);
    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode CreateBoundaryDiffusiveSampling(
    const EdgeVertices& outwardEdge, const QuadVertices& interiorCell,
    const GaussRule1D& rule, const DiffusiveSamplingOptions& options,
    DiffusiveSampling& sampling)
{
    PetscFunctionBeginUser;
    DiffusiveSampling work;
    PetscCall(work.Build(outwardEdge, interiorCell, nullptr, rule, options));
    sampling = std::move(work);
    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode IdentityDiffusiveQuantity(
    PetscReal u, const DiffusiveFluxPoint&, DiffusiveQuantity& result)
{
    PetscFunctionBeginUser;
    PetscCheck(std::isfinite(u), PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE,
               "Diffusion state must be finite");
    result = {u, 1};
    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode IntegrateDiffusiveFlux(
    const DiffusiveSampling& sampling, const std::vector<PetscReal>& samples,
    const std::vector<PetscReal>& diffusivity, PetscReal time,
    const DiffusiveFluxLaw& law, PetscReal& flux)
{
    PetscFunctionBeginUser;
    DiffusiveEdgeFluxResult work;
    PetscCall(IntegrateImpl(sampling, samples, diffusivity, nullptr, time, law, false, work));
    flux = work.flux;
    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode IntegrateDiffusiveFluxWithDerivatives(
    const DiffusiveSampling& sampling, const std::vector<PetscReal>& samples,
    const std::vector<PetscReal>& diffusivity, PetscReal time,
    const DiffusiveFluxLaw& law, DiffusiveEdgeFluxResult& result)
{
    PetscFunctionBeginUser;
    PetscCall(IntegrateImpl(sampling, samples, diffusivity, nullptr, time, law, true, result));
    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode IntegrateDiffusiveBoundaryFlux(
    const DiffusiveSampling& sampling, const std::vector<PetscReal>& samples,
    const std::vector<PetscReal>& diffusivity,
    const std::vector<DiffusiveBoundaryValue>& data, PetscReal time,
    const DiffusiveFluxLaw& law, PetscReal& flux)
{
    PetscFunctionBeginUser;
    DiffusiveEdgeFluxResult work;
    PetscCall(IntegrateImpl(sampling, samples, diffusivity, &data, time, law, false, work));
    flux = work.flux;
    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode IntegrateDiffusiveBoundaryFluxWithDerivatives(
    const DiffusiveSampling& sampling, const std::vector<PetscReal>& samples,
    const std::vector<PetscReal>& diffusivity,
    const std::vector<DiffusiveBoundaryValue>& data, PetscReal time,
    const DiffusiveFluxLaw& law, DiffusiveEdgeFluxResult& result)
{
    PetscFunctionBeginUser;
    PetscCall(IntegrateImpl(sampling, samples, diffusivity, &data, time, law, true, result));
    PetscFunctionReturn(PETSC_SUCCESS);
}
