#include "advectiveflux.h"

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
               "Advective flux or derivative is not representable");
    result = static_cast<PetscReal>(value);
    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode CheckPoint(const AdvectiveFluxPoint& point)
{
    PetscFunctionBeginUser;
    for (int d = 0; d < 2; ++d)
        PetscCheck(std::isfinite(point.position.p[d]) &&
                   std::isfinite(point.velocity.p[d]) &&
                   std::isfinite(point.normal.p[d]),
                   PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE,
                   "Flux position, velocity and normal must be finite");
    PetscCheck(std::isfinite(point.time), PETSC_COMM_SELF,
               PETSC_ERR_ARG_OUTOFRANGE, "Flux time must be finite");
    const PetscReal norm = std::hypot(point.normal.p[0], point.normal.p[1]);
    PetscCheck(std::isfinite(norm) &&
               std::abs(norm - 1) <= 128 * PETSC_MACHINE_EPSILON,
               PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG,
               "Advective flux requires a unit normal");
    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode NormalVelocity(const AdvectiveFluxPoint& point, PetscReal& value)
{
    PetscFunctionBeginUser;
    PetscCall(RealValue(static_cast<long double>(point.velocity.p[0]) * point.normal.p[0]
                       + static_cast<long double>(point.velocity.p[1]) * point.normal.p[1],
                       value));
    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode CheckOptions(const LaxFriedrichsOptions& options)
{
    PetscFunctionBeginUser;
    PetscCheck(options.mode == LaxFriedrichsMode::Local ||
               options.mode == LaxFriedrichsMode::Global,
               PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Unknown LF mode");
    PetscCheck(options.linearization == LaxFriedrichsLinearization::Full ||
               options.linearization == LaxFriedrichsLinearization::FrozenSpeed,
               PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Unknown LF linearization");
    if (options.mode == LaxFriedrichsMode::Global)
        PetscCheck(std::isfinite(options.globalSpeed) && options.globalSpeed >= 0,
                   PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE,
                   "Global LF speed must be finite and nonnegative");
    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode PhysicalFlux(PetscReal u, const AdvectiveFluxPoint& point,
                           const AdvectiveFluxLaw& law, AdvectivePhysicalFlux& flux)
{
    PetscFunctionBeginUser;
    PetscCheck(std::isfinite(u), PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE,
               "Advective trace must be finite");
    PetscCheck(static_cast<bool>(law.evaluate), PETSC_COMM_SELF,
               PETSC_ERR_ARG_WRONG, "Missing physical flux callback");
    AdvectivePhysicalFlux work;
    PetscCall(law.evaluate(u, point, work));
    PetscCheck(std::isfinite(work.value) && std::isfinite(work.derivative),
               PETSC_COMM_SELF, PETSC_ERR_FP,
               "Physical flux must return finite value and first derivative");
    flux = work;
    PetscFunctionReturn(PETSC_SUCCESS);
}

bool Bounds(PetscReal bound, PetscReal required)
{
    // Relative roundoff allowance; no absolute tolerance hiding small speeds.
    return bound >= required || required - bound <=
        64 * PETSC_MACHINE_EPSILON * std::max(bound, required);
}

PetscReal Sign(PetscReal value)
{ return value > 0 ? 1 : (value < 0 ? -1 : 0); }

PetscErrorCode LocalSpeed(PetscReal left, PetscReal right,
                         const AdvectiveFluxPoint& point, const AdvectiveFluxLaw& law,
                         const AdvectivePhysicalFlux& fLeft,
                         const AdvectivePhysicalFlux& fRight,
                         bool derivatives, AdvectiveWaveSpeed& speed)
{
    PetscFunctionBeginUser;
    const PetscReal a = std::abs(fLeft.derivative), b = std::abs(fRight.derivative);
    AdvectiveWaveSpeed work;
    if (law.waveSpeed) {
        PetscCall(law.waveSpeed(left, right, point, work));
    } else {
        work.value = std::max(a, b);
        work.derivativeLeft = work.derivativeRight = 0;
        if (derivatives) {
            PetscCheck(std::isfinite(fLeft.secondDerivative) &&
                       std::isfinite(fRight.secondDerivative), PETSC_COMM_SELF,
                       PETSC_ERR_FP, "Full local LF derivatives require F_n'' or a speed callback");
            const PetscReal l = Sign(fLeft.derivative) * fLeft.secondDerivative;
            const PetscReal r = Sign(fRight.derivative) * fRight.secondDerivative;
            if (a > b) work.derivativeLeft = l;
            else if (b > a) work.derivativeRight = r;
            else { work.derivativeLeft = l / 2; work.derivativeRight = r / 2; }
        }
    }
    PetscCheck(std::isfinite(work.value) && work.value >= 0 &&
               Bounds(work.value, std::max(a, b)), PETSC_COMM_SELF,
               PETSC_ERR_ARG_OUTOFRANGE,
               "LF speed must bound both endpoint characteristic speeds");
    if (derivatives)
        PetscCheck(std::isfinite(work.derivativeLeft) &&
                   std::isfinite(work.derivativeRight), PETSC_COMM_SELF,
                   PETSC_ERR_FP, "Local LF speed derivatives must be finite");
    else work.derivativeLeft = work.derivativeRight = 0;
    speed = work;
    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode FluxImpl(PetscReal left, PetscReal right, const AdvectiveFluxPoint& point,
                       const AdvectiveFluxLaw& law, const LaxFriedrichsOptions& options,
                       bool derivatives, AdvectiveFluxResult& result,
                       PetscReal& localMaximum)
{
    PetscFunctionBeginUser;
    PetscCall(CheckPoint(point));
    PetscCall(CheckOptions(options));
    AdvectivePhysicalFlux fLeft, fRight;
    PetscCall(PhysicalFlux(left, point, law, fLeft));
    PetscCall(PhysicalFlux(right, point, law, fRight));
    const bool localFull = derivatives && options.mode == LaxFriedrichsMode::Local &&
        options.linearization == LaxFriedrichsLinearization::Full;
    AdvectiveWaveSpeed speed;
    PetscCall(LocalSpeed(left, right, point, law, fLeft, fRight, localFull, speed));
    const PetscReal alpha = options.mode == LaxFriedrichsMode::Local
                          ? speed.value : options.globalSpeed;
    PetscCheck(Bounds(alpha, speed.value), PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE,
               "Supplied global LF speed is smaller than the current local bound");
    const long double jump = static_cast<long double>(right) - left;
    AdvectiveFluxResult work;
    work.waveSpeed = alpha;
    PetscCall(RealValue((static_cast<long double>(fLeft.value) + fRight.value
                        - static_cast<long double>(alpha) * jump) / 2, work.flux));
    if (derivatives) {
        PetscCall(RealValue((static_cast<long double>(fLeft.derivative) + alpha
                            - jump * speed.derivativeLeft) / 2, work.derivativeLeft));
        PetscCall(RealValue((static_cast<long double>(fRight.derivative) - alpha
                            - jump * speed.derivativeRight) / 2, work.derivativeRight));
        if (options.mode == LaxFriedrichsMode::Global)
            PetscCall(RealValue(-jump / 2, work.derivativeGlobalSpeed));
    }
    result = work;
    localMaximum = speed.value;
    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode PrepareEdge(const EdgeVertices& edge, const GaussRule1D& rule,
                          const std::vector<Point>& velocity, PetscReal time,
                          PetscReal& length, Point& normal)
{
    PetscFunctionBeginUser;
    PetscCall(ValidateGaussRule(rule));
    PetscCall(GetEdgeGeometry(edge, length, normal));
    PetscCheck(velocity.size() == rule.points.size(), PETSC_COMM_SELF,
               PETSC_ERR_ARG_SIZ, "Need one velocity per edge quadrature point");
    PetscCheck(std::isfinite(time), PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE,
               "Flux time must be finite");
    for (const auto& v : velocity)
        PetscCheck(std::isfinite(v.p[0]) && std::isfinite(v.p[1]), PETSC_COMM_SELF,
                   PETSC_ERR_ARG_OUTOFRANGE, "Edge velocity must be finite");
    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode CheckTraces(std::size_t count, const std::vector<PetscReal>& left,
                          const std::vector<PetscReal>& right)
{
    PetscFunctionBeginUser;
    PetscCheck(left.size() == count && right.size() == count, PETSC_COMM_SELF,
               PETSC_ERR_ARG_SIZ, "Need two traces per edge quadrature point");
    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode EdgeImpl(const EdgeVertices& edge, const GaussRule1D& rule,
                       const std::vector<Point>& velocity,
                       const std::vector<PetscReal>& left,
                       const std::vector<PetscReal>& right, PetscReal time,
                       const AdvectiveFluxLaw& law, const LaxFriedrichsOptions& options,
                       bool derivatives, AdvectiveEdgeFluxResult& result)
{
    PetscFunctionBeginUser;
    PetscReal length; Point normal;
    PetscCall(PrepareEdge(edge, rule, velocity, time, length, normal));
    PetscCall(CheckTraces(rule.points.size(), left, right));
    AdvectiveEdgeFluxResult work;
    if (derivatives) {
        work.derivativeLeft.resize(left.size());
        work.derivativeRight.resize(right.size());
    }
    long double sum = 0, globalDerivative = 0;
    for (std::size_t q = 0; q < rule.points.size(); ++q) {
        const AdvectiveFluxPoint point{MapEdgePoint(rule.points[q], edge),
                                       velocity[q], normal, time};
        AdvectiveFluxResult flux; PetscReal maximum;
        PetscCall(FluxImpl(left[q], right[q], point, law, options, derivatives, flux, maximum));
        const long double weight = static_cast<long double>(rule.weights[q]) * length / 2;
        sum += weight * flux.flux;
        work.maximumLocalSpeed = std::max(work.maximumLocalSpeed, maximum);
        if (derivatives) {
            PetscCall(RealValue(weight * flux.derivativeLeft, work.derivativeLeft[q]));
            PetscCall(RealValue(weight * flux.derivativeRight, work.derivativeRight[q]));
            globalDerivative += weight * flux.derivativeGlobalSpeed;
        }
    }
    PetscCall(RealValue(sum, work.flux));
    PetscCall(RealValue(globalDerivative, work.derivativeGlobalSpeed));
    result = std::move(work);
    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode BoundaryImpl(const EdgeVertices& edge, const GaussRule1D& rule,
                           const std::vector<Point>& velocity,
                           const std::vector<PetscReal>& interior,
                           const std::vector<AdvectiveBoundaryValue>& data,
                           PetscReal time, const AdvectiveFluxLaw& law,
                           bool derivatives, AdvectiveBoundaryFluxResult& result)
{
    PetscFunctionBeginUser;
    PetscReal length; Point normal;
    PetscCall(PrepareEdge(edge, rule, velocity, time, length, normal));
    const auto count = rule.points.size();
    PetscCheck(interior.size() == count && data.size() == count, PETSC_COMM_SELF,
               PETSC_ERR_ARG_SIZ, "Need an interior trace and boundary policy per quadrature point");
    AdvectiveBoundaryFluxResult work;
    if (derivatives) work.derivativeInterior.assign(count, 0);
    long double sum = 0;
    for (std::size_t q = 0; q < count; ++q) {
        const auto type = data[q].type;
        PetscCheck(type == AdvectiveBoundaryType::PrescribedState ||
                   type == AdvectiveBoundaryType::ExtrapolatedState ||
                   type == AdvectiveBoundaryType::ZeroFlux, PETSC_COMM_SELF,
                   PETSC_ERR_ARG_WRONG, "Unknown advection boundary policy");
        PetscCheck(std::isfinite(interior[q]), PETSC_COMM_SELF,
                   PETSC_ERR_ARG_OUTOFRANGE, "Interior boundary trace must be finite");
        const AdvectiveFluxPoint point{MapEdgePoint(rule.points[q], edge),
                                       velocity[q], normal, time};
        PetscCall(CheckPoint(point));
        if (type == AdvectiveBoundaryType::ZeroFlux) continue;
        const bool outflow = type == AdvectiveBoundaryType::ExtrapolatedState;
        AdvectivePhysicalFlux flux;
        PetscCall(PhysicalFlux(outflow ? interior[q] : data[q].value, point, law, flux));
        const long double weight = static_cast<long double>(rule.weights[q]) * length / 2;
        sum += weight * flux.value;
        if (derivatives && outflow)
            PetscCall(RealValue(weight * flux.derivative, work.derivativeInterior[q]));
    }
    PetscCall(RealValue(sum, work.flux));
    result = std::move(work);
    PetscFunctionReturn(PETSC_SUCCESS);
}

} // namespace

PetscErrorCode LinearAdvectiveFlux(PetscReal u, const AdvectiveFluxPoint& point,
                                  AdvectivePhysicalFlux& result)
{
    PetscFunctionBeginUser;
    PetscCall(CheckPoint(point));
    PetscCheck(std::isfinite(u), PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE,
               "Linear-advection state must be finite");
    AdvectivePhysicalFlux work;
    PetscCall(NormalVelocity(point, work.derivative));
    PetscCall(RealValue(static_cast<long double>(work.derivative) * u, work.value));
    work.secondDerivative = 0;
    result = work;
    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode BurgersAdvectiveFlux(PetscReal u, const AdvectiveFluxPoint& point,
                                   AdvectivePhysicalFlux& result)
{
    PetscFunctionBeginUser;
    PetscCall(CheckPoint(point));
    PetscCheck(std::isfinite(u), PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE,
               "Burgers state must be finite");
    AdvectivePhysicalFlux work;
    PetscCall(NormalVelocity(point, work.secondDerivative));
    PetscCall(RealValue(static_cast<long double>(work.secondDerivative) * u,
                       work.derivative));
    PetscCall(RealValue(static_cast<long double>(work.secondDerivative) * u * u / 2,
                       work.value));
    result = work;
    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode EvaluateAdvectiveFlux(
    PetscReal left, PetscReal right, const AdvectiveFluxPoint& point,
    const AdvectiveFluxLaw& law, const LaxFriedrichsOptions& options, PetscReal& flux)
{
    PetscFunctionBeginUser;
    AdvectiveFluxResult work; PetscReal maximum;
    PetscCall(FluxImpl(left, right, point, law, options, false, work, maximum));
    flux = work.flux;
    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode EvaluateAdvectiveFluxWithDerivatives(
    PetscReal left, PetscReal right, const AdvectiveFluxPoint& point,
    const AdvectiveFluxLaw& law, const LaxFriedrichsOptions& options,
    AdvectiveFluxResult& result)
{
    PetscFunctionBeginUser;
    AdvectiveFluxResult work; PetscReal maximum;
    PetscCall(FluxImpl(left, right, point, law, options, true, work, maximum));
    result = work;
    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode IntegrateAdvectiveFlux(
    const EdgeVertices& edge, const GaussRule1D& rule, const std::vector<Point>& velocity,
    const std::vector<PetscReal>& left, const std::vector<PetscReal>& right,
    PetscReal time, const AdvectiveFluxLaw& law, const LaxFriedrichsOptions& options,
    PetscReal& flux)
{
    PetscFunctionBeginUser;
    AdvectiveEdgeFluxResult work;
    PetscCall(EdgeImpl(edge, rule, velocity, left, right, time, law, options, false, work));
    flux = work.flux;
    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode IntegrateAdvectiveFluxWithDerivatives(
    const EdgeVertices& edge, const GaussRule1D& rule, const std::vector<Point>& velocity,
    const std::vector<PetscReal>& left, const std::vector<PetscReal>& right,
    PetscReal time, const AdvectiveFluxLaw& law, const LaxFriedrichsOptions& options,
    AdvectiveEdgeFluxResult& result)
{
    PetscFunctionBeginUser;
    AdvectiveEdgeFluxResult work;
    PetscCall(EdgeImpl(edge, rule, velocity, left, right, time, law, options, true, work));
    result = std::move(work);
    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode EstimateAdvectiveEdgeWaveSpeed(
    const EdgeVertices& edge, const GaussRule1D& rule, const std::vector<Point>& velocity,
    const std::vector<PetscReal>& left, const std::vector<PetscReal>& right,
    PetscReal time, const AdvectiveFluxLaw& law, PetscReal& maximum)
{
    PetscFunctionBeginUser;
    PetscReal length; Point normal;
    PetscCall(PrepareEdge(edge, rule, velocity, time, length, normal));
    PetscCall(CheckTraces(rule.points.size(), left, right));
    PetscReal work = 0;
    for (std::size_t q = 0; q < rule.points.size(); ++q) {
        const AdvectiveFluxPoint point{MapEdgePoint(rule.points[q], edge),
                                       velocity[q], normal, time};
        PetscCall(CheckPoint(point));
        AdvectivePhysicalFlux fLeft, fRight;
        PetscCall(PhysicalFlux(left[q], point, law, fLeft));
        PetscCall(PhysicalFlux(right[q], point, law, fRight));
        AdvectiveWaveSpeed speed;
        PetscCall(LocalSpeed(left[q], right[q], point, law, fLeft, fRight, false, speed));
        work = std::max(work, speed.value);
    }
    maximum = work;
    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode ReduceGlobalLaxFriedrichsSpeed(
    MPI_Comm comm, PetscReal localMaximum, PetscReal& globalMaximum)
{
    PetscFunctionBeginUser;
    // Agree on invalid input before any rank returns; empty owners pass zero.
    const int invalid = !std::isfinite(localMaximum) || localMaximum < 0;
    int anyInvalid = 0;
    PetscCallMPI(MPI_Allreduce(&invalid, &anyInvalid, 1, MPI_INT, MPI_MAX, comm));
    PetscCheck(!anyInvalid, comm, PETSC_ERR_ARG_OUTOFRANGE,
               "Every rank must supply a finite nonnegative local LF bound");
    PetscReal work = 0;
    PetscCallMPI(MPI_Allreduce(&localMaximum, &work, 1, MPIU_REAL, MPI_MAX, comm));
    globalMaximum = work;
    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode BuildLinearAdvectionBoundaryData(
    const EdgeVertices& outwardEdge, const GaussRule1D& rule,
    const std::vector<Point>& velocity, PetscReal time,
    const AdvectiveBoundaryFunction& inflow, std::vector<AdvectiveBoundaryValue>& data)
{
    PetscFunctionBeginUser;
    PetscReal length; Point normal;
    PetscCall(PrepareEdge(outwardEdge, rule, velocity, time, length, normal));
    std::vector<AdvectiveBoundaryValue> work(rule.points.size());
    for (std::size_t q = 0; q < rule.points.size(); ++q) {
        const AdvectiveFluxPoint point{MapEdgePoint(rule.points[q], outwardEdge),
                                       velocity[q], normal, time};
        PetscCall(CheckPoint(point));
        PetscReal vn; PetscCall(NormalVelocity(point, vn));
        if (vn < 0) {
            PetscCheck(static_cast<bool>(inflow), PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG,
                       "An inflow point requires prescribed boundary data");
            work[q].type = AdvectiveBoundaryType::PrescribedState;
            work[q].value = std::numeric_limits<PetscReal>::quiet_NaN();
            PetscCall(inflow(point, work[q].value));
            PetscCheck(std::isfinite(work[q].value), PETSC_COMM_SELF, PETSC_ERR_FP,
                       "Inflow boundary callback must return a finite state");
        } else if (vn > 0) {
            work[q].type = AdvectiveBoundaryType::ExtrapolatedState;
        } else {
            work[q].type = AdvectiveBoundaryType::ZeroFlux;
        }
    }
    data = std::move(work);
    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode IntegrateAdvectiveBoundaryFlux(
    const EdgeVertices& outwardEdge, const GaussRule1D& rule,
    const std::vector<Point>& velocity, const std::vector<PetscReal>& interior,
    const std::vector<AdvectiveBoundaryValue>& data, PetscReal time,
    const AdvectiveFluxLaw& law, PetscReal& flux)
{
    PetscFunctionBeginUser;
    AdvectiveBoundaryFluxResult work;
    PetscCall(BoundaryImpl(outwardEdge, rule, velocity, interior, data, time, law, false, work));
    flux = work.flux;
    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode IntegrateAdvectiveBoundaryFluxWithDerivatives(
    const EdgeVertices& outwardEdge, const GaussRule1D& rule,
    const std::vector<Point>& velocity, const std::vector<PetscReal>& interior,
    const std::vector<AdvectiveBoundaryValue>& data, PetscReal time,
    const AdvectiveFluxLaw& law, AdvectiveBoundaryFluxResult& result)
{
    PetscFunctionBeginUser;
    AdvectiveBoundaryFluxResult work;
    PetscCall(BoundaryImpl(outwardEdge, rule, velocity, interior, data, time, law, true, work));
    result = std::move(work);
    PetscFunctionReturn(PETSC_SUCCESS);
}
