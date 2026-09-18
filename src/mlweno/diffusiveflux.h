#ifndef MANTLE_MLWENO_DIFFUSIVEFLUX_H
#define MANTLE_MLWENO_DIFFUSIVEFLUX_H

#include "integral.h"
#include <petscsys.h>

#include <functional>
#include <limits>
#include <vector>

// Scalar isotropic diffusion by normal-line sampling, for MantlePar (C++17).
// Add diffusiveflux.cpp to mantle_mlweno; its mantle_core dependency suffices.
// All routines are local/noncollective; call after PetscInitialize. Checked
// outputs are replaced only on success. No reconstruction, DM or Vec is owned.
// Geometry is held fixed in all derivatives. Refresh sampling after mesh changes.

enum class LagrangeDerivativeLocation { Middle, EndLo, EndHi };

// Adapted from Todd's LagrangeBasisDeriv in the legacy
// MantleSolver_cpp/src/transport_array/clean/lagrange_tmp.cpp:
// middle(n,i), endLo(n,i), endHi(n,i), where n = numberOfPoints-1.
// Returns dimensionless first-derivative weights on uniformly spaced nodes
// 0,...,n, evaluated at n/2, 0 or n. Middle requires an EVEN point count.
// Divide by physical spacing once when differentiating. Uses the legacy
// binomial formulas, with per-call storage instead of shared mutable caches.
PetscErrorCode CreateLagrangeDerivativeWeights(
    PetscInt numberOfPoints, LagrangeDerivativeLocation location,
    std::vector<PetscReal>& weights);

struct DiffusiveSamplingOptions {
    // Any even count >= 2: total interior-edge samples, half from each cell.
    // A Dirichlet edge uses this many INTERIOR samples plus the boundary point:
    // one extra node preserves the formal derivative order of the centered rule.
    PetscInt numberOfSamples = 4;
    // Farthest sample / available normal-ray distance; require 0 < value <= 1.
    // 0.9 keeps samples away from the other cell faces. Independent of Gauss order.
    PetscReal extentFraction = 0.9;
};

enum class DiffusiveSampleSide { Left, Right, Boundary };

// Built once and reused. Left is BEHIND the oriented normal, right is ahead;
// at a boundary Left means interior, Boundary is the prescribed face value.
// Flat sample arrays use q*SamplesPerPoint()+k. Coefficients() is dimensionless;
// Spacing()[q] is physical; QuadratureWeights()[q] already includes edgeLength/2.
// A boundary row is ordered from farthest interior sample to the face itself.
// Its final slot is a placeholder in the supplied solution samples, because
// boundary data supply that value. Ignore it when evaluating reconstructions.
class DiffusiveSampling {
public:
    bool IsInitialized() const noexcept { return !points_.empty() && !samples_.empty(); }
    bool IsBoundary() const noexcept { return boundary_; }
    std::size_t SamplesPerPoint() const noexcept { return coefficients_.size(); }
    const Point& Normal() const noexcept { return normal_; }
    const std::vector<Point>& QuadraturePoints() const noexcept { return points_; }
    const std::vector<Point>& SamplePoints() const noexcept { return samples_; }
    const std::vector<DiffusiveSampleSide>& SampleSides() const noexcept { return sides_; }
    const std::vector<PetscReal>& Spacing() const noexcept { return spacing_; }
    const std::vector<PetscReal>& QuadratureWeights() const noexcept { return weights_; }
    const std::vector<PetscReal>& Coefficients() const noexcept { return coefficients_; }

private:
    friend PetscErrorCode CreateInteriorDiffusiveSampling(
        const EdgeVertices&, const QuadVertices&, const QuadVertices&,
        const GaussRule1D&, const DiffusiveSamplingOptions&, DiffusiveSampling&);
    friend PetscErrorCode CreateBoundaryDiffusiveSampling(
        const EdgeVertices&, const QuadVertices&, const GaussRule1D&,
        const DiffusiveSamplingOptions&, DiffusiveSampling&);
    PetscErrorCode Build(const EdgeVertices&, const QuadVertices&,
                         const QuadVertices*, const GaussRule1D&,
                         const DiffusiveSamplingOptions&);
    bool boundary_ = false;
    Point normal_{};
    std::vector<Point> points_, samples_;
    std::vector<DiffusiveSampleSide> sides_;
    std::vector<PetscReal> spacing_, weights_, coefficients_;
};

// Core convention: normal=(dy,-dx)/length. Edge endpoints must be the actual
// shared vertices, in CCW boundary order for leftCell, reversed for rightCell.
// Both cells must be valid convex CCW quads. Gauss points must be strictly
// inside (-1,1), since a corner can have zero available normal-ray distance.
//
// At each Gauss point, ray/quad intersections give available distances dL,dR.
// Set ds=2*extentFraction*min(dL,dR)/(N-1); samples are at
// xq+(k-(N-1)/2)*ds*normal. Thus side assignment is geometric, including on
// perturbed, stretched and pseudo-1D meshes. No cell-area length surrogate.
PetscErrorCode CreateInteriorDiffusiveSampling(
    const EdgeVertices& edge, const QuadVertices& leftCell,
    const QuadVertices& rightCell, const GaussRule1D& rule,
    const DiffusiveSamplingOptions& options, DiffusiveSampling& sampling);

// outwardEdge must follow interiorCell's CCW boundary. With N interior samples,
// ds=extentFraction*dInterior/N and offsets are -N*ds,...,-ds,0.
// The endpoint derivative reuses legacy endHi(N,k). No repeated exterior values.
PetscErrorCode CreateBoundaryDiffusiveSampling(
    const EdgeVertices& outwardEdge, const QuadVertices& interiorCell,
    const GaussRule1D& rule, const DiffusiveSamplingOptions& options,
    DiffusiveSampling& sampling);

struct DiffusiveFluxPoint {
    Point position{}; // Actual sample position (boundary position for boundary data).
    Point normal{};
    PetscReal time = 0;
    DiffusiveSampleSide side = DiffusiveSampleSide::Left;
};

struct DiffusiveQuantity {
    PetscReal value = std::numeric_limits<PetscReal>::quiet_NaN();
    PetscReal derivative = std::numeric_limits<PetscReal>::quiet_NaN();
};

PetscErrorCode IdentityDiffusiveQuantity(
    PetscReal u, const DiffusiveFluxPoint& point, DiffusiveQuantity& result);

using DiffusiveQuantityFunction = std::function<PetscErrorCode(
    PetscReal, const DiffusiveFluxPoint&, DiffusiveQuantity&)>;

struct DiffusiveFluxLaw {
    // psi(u,x,t), the quantity DIFFERENTIATED, matching legacy diffunc(u).
    // Identity gives -kappa*grad(u); kappa=1 gives -grad(psi(u)). For
    // -D(u)*grad(u), one option is psi'(u)=D(u), NOT psi(u)=D(u)*u.
    // Callbacks return PETSc errors, must not throw, and must set value.
    // derivative=dpsi/du is required only by WithDerivatives calls. Other
    // dependencies captured by a callback are fixed in this scalar derivative.
    DiffusiveQuantityFunction evaluate = IdentityDiffusiveQuantity;
};

struct DiffusiveEdgeFluxResult {
    PetscReal flux = 0; // Integral of -kappa*D_n(psi); not divided by cell area.
    // Partials already include quadrature, length, spacing and psi'(u).
    // Flat order matches SamplePoints(). Boundary placeholder entries are zero.
    std::vector<PetscReal> derivativeSamples;
    // One partial per Gauss point, holding sample values fixed.
    std::vector<PetscReal> derivativeDiffusivity;
    // Empty for an interior edge; one partial per point for boundary data.
    std::vector<PetscReal> derivativeBoundaryData;
};

// samples: one reconstructed u per SamplePoints() entry, using SampleSides().
// diffusivity: one finite nonnegative SCALAR kappa per QuadraturePoints() entry.
// The caller supplies its physical interface coefficient; this module performs
// no averaging of discontinuous material coefficients and no tensor diffusion.
//
// For implicit assembly, sum_a derivativeSamples[a]*dR_a/dU_j over BOTH
// reconstructions; merge their overlapping JacobianCells(). Include nonlinear
// weight derivatives by using Reconstruction::EvaluateWithJacobian in Full mode.
// If kappa depends on U, also add sum_q derivativeDiffusivity[q]*dkappa_q/dU_j.
// Coupled constitutive variables need their additional chain-rule terms.
// Apply the same edge flux with opposite signs to the adjacent cell balances.
// u_t+div(F_adv+F_diff)=s uses -outwardFlux/area in its spatial RHS.
// Reverse an interior edge by swapping cells and resampling/reordering arrays.
PetscErrorCode IntegrateDiffusiveFlux(
    const DiffusiveSampling& sampling, const std::vector<PetscReal>& samples,
    const std::vector<PetscReal>& diffusivity, PetscReal time,
    const DiffusiveFluxLaw& law, PetscReal& flux);
PetscErrorCode IntegrateDiffusiveFluxWithDerivatives(
    const DiffusiveSampling& sampling, const std::vector<PetscReal>& samples,
    const std::vector<PetscReal>& diffusivity, PetscReal time,
    const DiffusiveFluxLaw& law, DiffusiveEdgeFluxResult& result);

enum class DiffusiveBoundaryType {
    PrescribedState,    // value=g for u; apply psi(g) at the face.
    PrescribedQuantity, // value=g for psi itself, e.g. prescribed temperature.
    PrescribedFlux,     // value=g for PHYSICAL OUTWARD -kappa*D_n(psi), per length.
    ZeroFlux
};

struct DiffusiveBoundaryValue {
    DiffusiveBoundaryType type = DiffusiveBoundaryType::ZeroFlux;
    PetscReal value = 0;
};

// data has one entry per Gauss point, allowing different conditions along an
// edge. The outward normal was fixed when boundary sampling was constructed.
// Supply full-size samples and diffusivity arrays. At prescribed/zero-flux
// points both arrays' entries are ignored; no constitutive callback is needed.
// At Dirichlet points the final sample slot is ignored and filled from data.
// Data normally depend on position/time only; if state-dependent, also chain
// derivativeBoundaryData[q]*d(data[q].value)/dU_j. For a prescribed flux this
// partial is exactly the physical quadrature weight. ZeroFlux has zero partials.
PetscErrorCode IntegrateDiffusiveBoundaryFlux(
    const DiffusiveSampling& sampling, const std::vector<PetscReal>& samples,
    const std::vector<PetscReal>& diffusivity,
    const std::vector<DiffusiveBoundaryValue>& data, PetscReal time,
    const DiffusiveFluxLaw& law, PetscReal& flux);
PetscErrorCode IntegrateDiffusiveBoundaryFluxWithDerivatives(
    const DiffusiveSampling& sampling, const std::vector<PetscReal>& samples,
    const std::vector<PetscReal>& diffusivity,
    const std::vector<DiffusiveBoundaryValue>& data, PetscReal time,
    const DiffusiveFluxLaw& law, DiffusiveEdgeFluxResult& result);

#endif
