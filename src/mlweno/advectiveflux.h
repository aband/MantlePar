#ifndef MANTLE_TRANSPORT_ADVECTIVEFLUX_H
#define MANTLE_TRANSPORT_ADVECTIVEFLUX_H

#include "integral.h"
#include <petscsys.h>

#include <functional>
#include <limits>
#include <vector>

// Scalar face-flux kernels for MantlePar. Link against mantle_core (C++17).
// Except ReduceGlobalLaxFriedrichsSpeed, all functions are noncollective.
// Call after PetscInitialize. Checked outputs are replaced only on success.
// Geometry, supplied velocity, time and physical-law parameters are held fixed
// in the scalar derivatives. A monolithic flow/transport Jacobian must add its
// velocity/coefficient derivatives separately.

enum class LaxFriedrichsMode { Local, Global };
enum class LaxFriedrichsLinearization { Full, FrozenSpeed };

struct LaxFriedrichsOptions {
    LaxFriedrichsMode mode = LaxFriedrichsMode::Local;
    LaxFriedrichsLinearization linearization = LaxFriedrichsLinearization::Full;
    PetscReal globalSpeed = 0; // Used only in Global mode; finite and nonnegative.
};

struct AdvectiveFluxPoint {
    Point position{};
    Point velocity{}; // Physical transport velocity supplied at THIS point.
    Point normal{};   // Unit normal directed from the left state to the right.
    PetscReal time = 0;
};

struct AdvectivePhysicalFlux {
    PetscReal value = std::numeric_limits<PetscReal>::quiet_NaN();
    PetscReal derivative = std::numeric_limits<PetscReal>::quiet_NaN();
    PetscReal secondDerivative = std::numeric_limits<PetscReal>::quiet_NaN();
};

// F_n(u) and derivatives with respect to u, already including velocity.normal.
// These built-in laws give F_n=(velocity.normal)*u and (velocity.normal)*u^2/2.
PetscErrorCode LinearAdvectiveFlux(PetscReal u, const AdvectiveFluxPoint& point,
                                  AdvectivePhysicalFlux& result);
PetscErrorCode BurgersAdvectiveFlux(PetscReal u, const AdvectiveFluxPoint& point,
                                   AdvectivePhysicalFlux& result);

using AdvectiveFluxFunction = std::function<PetscErrorCode(
    PetscReal, const AdvectiveFluxPoint&, AdvectivePhysicalFlux&)>;

struct AdvectiveWaveSpeed {
    PetscReal value = std::numeric_limits<PetscReal>::quiet_NaN();
    PetscReal derivativeLeft = std::numeric_limits<PetscReal>::quiet_NaN();
    PetscReal derivativeRight = std::numeric_limits<PetscReal>::quiet_NaN();
};

using AdvectiveWaveSpeedFunction = std::function<PetscErrorCode(
    PetscReal, PetscReal, const AdvectiveFluxPoint&, AdvectiveWaveSpeed&)>;

struct AdvectiveFluxLaw {
    AdvectiveFluxFunction evaluate = LinearAdvectiveFlux;
    // Optional bound on |dF_n/du| over the interval between the two states.
    // Empty selects the legacy endpoint estimate max(|F'_L|,|F'_R|), which is
    // sufficient for linear advection and Burgers. Supply a stronger bound
    // for laws whose characteristic speed has an interior maximum.
    // The bound must be symmetric under (L,R,n)->(R,L,-n).
    AdvectiveWaveSpeedFunction waveSpeed;
};

// evaluate must set value and derivative, even for value-only calls. Its
// secondDerivative is needed only for Full LOCAL endpoint-speed derivatives.
// A custom waveSpeed instead supplies its two partial derivatives in that case.
// FrozenSpeed is an approximate linearization; it does not freeze WENO weights.
// Callbacks return PETSc errors and must not throw C++ exceptions.
// At an endpoint-speed tie, Full uses equal shares of both active branches;
// at abs(0), it uses derivative zero. These are stated generalized derivatives,
// not a claim of classical differentiability at switching points.

struct AdvectiveFluxResult {
    PetscReal flux = 0;
    PetscReal waveSpeed = 0;
    PetscReal derivativeLeft = 0;
    PetscReal derivativeRight = 0;
    PetscReal derivativeGlobalSpeed = 0; // Global mode only: -(uR-uL)/2.
};

// Fhat = (F_n(uL)+F_n(uR)-alpha*(uR-uL))/2.
// Global mode checks globalSpeed against the current local bound. Its trace
// derivatives hold this SUPPLIED speed fixed. If alpha_global is recomputed
// from U inside a residual, the exact Jacobian ALSO requires
// derivativeGlobalSpeed * d(alpha_global)/dU, possibly with nonlocal columns.
// The reduction helper does not compute that derivative. Alternatively supply
// a state-independent bound or deliberately lag/freeze the global speed.
PetscErrorCode EvaluateAdvectiveFlux(
    PetscReal left, PetscReal right, const AdvectiveFluxPoint& point,
    const AdvectiveFluxLaw& law, const LaxFriedrichsOptions& options,
    PetscReal& flux);
PetscErrorCode EvaluateAdvectiveFluxWithDerivatives(
    PetscReal left, PetscReal right, const AdvectiveFluxPoint& point,
    const AdvectiveFluxLaw& law, const LaxFriedrichsOptions& options,
    AdvectiveFluxResult& result);

struct AdvectiveEdgeFluxResult {
    PetscReal flux = 0; // Physical line integral, NOT divided by cell area.
    PetscReal maximumLocalSpeed = 0;
    PetscReal derivativeGlobalSpeed = 0;
    // Partials with respect to the supplied trace at each quadrature point.
    // These ALREADY include rule.weights[q]*edgeLength/2.
    std::vector<PetscReal> derivativeLeft, derivativeRight;
};

// edge follows core's convention: normal=(dy,-dx)/length. The left state is
// behind this normal, the right state ahead of it. For a CCW cell edge, left
// is the interior cell and the normal is outward. Reversing an interior face
// requires swapping L/R and reordering ALL sampled arrays by physical point.
// For Gauss-Legendre rules, this is reversal of the quadrature array order.
//
// Every array has rule.points.size() entries in MapEdgePoint(rule.points[q],
// edge) order. Velocity comes from the flow solver; it is not reconstructed
// here. Both scalar traces must be evaluated at those SAME physical points.
// No assumptions about stencil size, level or MPI ownership are made.
//
// For reconstruction columns j, chain sum_q(dLeft[q]*dR_L(q)/dU_j
// + dRight[q]*dR_R(q)/dU_j). Merge overlapping supports and apply opposite
// signs to neighboring cell residuals. Map logical cells to PETSc columns
// outside this module. Use consistent reconstruction states for values/Jacobians.
PetscErrorCode IntegrateAdvectiveFlux(
    const EdgeVertices& edge, const GaussRule1D& rule,
    const std::vector<Point>& velocity,
    const std::vector<PetscReal>& left, const std::vector<PetscReal>& right,
    PetscReal time, const AdvectiveFluxLaw& law,
    const LaxFriedrichsOptions& options, PetscReal& flux);
PetscErrorCode IntegrateAdvectiveFluxWithDerivatives(
    const EdgeVertices& edge, const GaussRule1D& rule,
    const std::vector<Point>& velocity,
    const std::vector<PetscReal>& left, const std::vector<PetscReal>& right,
    PetscReal time, const AdvectiveFluxLaw& law,
    const LaxFriedrichsOptions& options, AdvectiveEdgeFluxResult& result);

// First pass for a global speed: estimate each relevant edge (including any
// boundary trace pairs used by your operator), take the maximum on each rank,
// then reduce. Empty ranks contribute zero. The caller supplies complete edge
// coverage and calls the collective on every rank in the same order.
PetscErrorCode EstimateAdvectiveEdgeWaveSpeed(
    const EdgeVertices& edge, const GaussRule1D& rule,
    const std::vector<Point>& velocity,
    const std::vector<PetscReal>& left, const std::vector<PetscReal>& right,
    PetscReal time, const AdvectiveFluxLaw& law, PetscReal& maximum);
PetscErrorCode ReduceGlobalLaxFriedrichsSpeed(
    MPI_Comm comm, PetscReal localMaximum, PetscReal& globalMaximum);

enum class AdvectiveBoundaryType { PrescribedState, ExtrapolatedState, ZeroFlux };

struct AdvectiveBoundaryValue {
    AdvectiveBoundaryType type = AdvectiveBoundaryType::ExtrapolatedState;
    PetscReal value = 0; // Used only for PrescribedState; independent of U.
};
using AdvectiveBoundaryFunction = std::function<PetscErrorCode(
    const AdvectiveFluxPoint&, PetscReal&)>;

struct AdvectiveBoundaryFluxResult {
    PetscReal flux = 0;
    std::vector<PetscReal> derivativeInterior; // Includes quadrature/length factor.
};

// Convenience policy for LINEAR advection only, using the outward normal:
// velocity.normal < 0: prescribed inflow; > 0: extrapolated outflow; == 0:
// zero flux. Decide separately at every quadrature point. The callback is
// required/called only at inflow points and receives position and time.
// Nonlinear laws require caller-selected characteristic boundary data instead.
PetscErrorCode BuildLinearAdvectionBoundaryData(
    const EdgeVertices& outwardEdge, const GaussRule1D& rule,
    const std::vector<Point>& velocity, PetscReal time,
    const AdvectiveBoundaryFunction& inflow,
    std::vector<AdvectiveBoundaryValue>& data);

// Preserve the legacy PHYSICAL boundary flux: F_n(g) for prescribed inflow,
// F_n(uInterior) for free outflow, and exactly zero for ZeroFlux. Equivalently
// both LF traces are the selected state, so stabilization cancels in BOTH LF
// modes. This is not a two-state exterior-ghost numerical boundary flux.
// For such a closure, call the interior kernel with your chosen exterior trace
// and include its derivative. Here prescribed data are independent of U.
PetscErrorCode IntegrateAdvectiveBoundaryFlux(
    const EdgeVertices& outwardEdge, const GaussRule1D& rule,
    const std::vector<Point>& velocity, const std::vector<PetscReal>& interior,
    const std::vector<AdvectiveBoundaryValue>& data, PetscReal time,
    const AdvectiveFluxLaw& law, PetscReal& flux);
PetscErrorCode IntegrateAdvectiveBoundaryFluxWithDerivatives(
    const EdgeVertices& outwardEdge, const GaussRule1D& rule,
    const std::vector<Point>& velocity, const std::vector<PetscReal>& interior,
    const std::vector<AdvectiveBoundaryValue>& data, PetscReal time,
    const AdvectiveFluxLaw& law, AdvectiveBoundaryFluxResult& result);

#endif
