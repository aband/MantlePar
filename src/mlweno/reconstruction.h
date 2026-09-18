#ifndef MANTLE_MLWENO_RECONSTRUCTION_H
#define MANTLE_MLWENO_RECONSTRUCTION_H

#include "tensorstencilpoly.h"

#include <cstddef>
#include <vector>

enum class ReconstructionFamilyKind { Large, Small };
enum class ReconstructionSmoothness { ReferenceSquare, TargetCell };
enum class ReconstructionJacobianMode { Full, FrozenWeights };

// One family may contain any number of candidates with the same size/order.
// Polynomial space is Q_(size.i-1,size.j-1). As in the legacy implementation,
// order controls smoothness derivatives AND the weight parameter r=order+1;
// it does not set the polynomial dimensions.
struct ReconstructionFamily {
    MeshIndex size{1, 1};
    PetscInt order = 0;
    // Stencil origins relative to the target cell. Every prescribed stencil
    // must contain the target: 1-size.d <= offset.d <= 0 in each direction.
    std::vector<MeshIndex> offsets;
    // Empty means weight 1 for each offset. Otherwise one weight per offset.
    std::vector<PetscReal> linearWeights;
    ReconstructionSmoothness smoothness = ReconstructionSmoothness::ReferenceSquare;
};

struct ReconstructionOptions {
    ReconstructionFamily large, small;
    bool useConstant = false;
    PetscReal constantLinearWeight = 1e-5;
    PetscReal epsilon = 1e-2;
    PetscInt s = 1;
};

struct ReconstructionCandidateInfo {
    ReconstructionFamilyKind family = ReconstructionFamilyKind::Large;
    std::size_t configuredIndex = 0; // Index in the family's original offsets.
    MeshIndex start{}, size{}, targetOffset{};
    PetscInt order = 0;
    ReconstructionSmoothness smoothnessRegion = ReconstructionSmoothness::ReferenceSquare;
    PetscReal linearWeight = 0;
    // Meaningful after HasWeights() becomes true.
    PetscReal smoothness = 0;
    PetscReal nonlinearWeight = 0;
};

// ML-WENO reconstruction for ONE target cell, using two configurable families
// and an optional constant candidate p0 = the target cell average.
//
// Workflow: Initialize geometry once; Update for each new solution/stage; then
// Evaluate at any number of points. Geometry and gathered solution data are
// owned. No MeshInfo, DM, Vec, or caller-array pointers are retained.
//
// All methods are local/noncollective; call after PetscInitialize. The caller
// performs MPI halo exchanges. Candidates outside the physical domain are
// discarded (legacy policy); missing local geometry/data are errors. Geometry
// must be rebuilt after coordinate or partition changes. No stencil shifting
// or automatic constant fallback is performed.
//
// Failed Initialize/Update calls preserve the previous valid state. Checked
// output arguments are assigned only on success. Copies are independent;
// moved-from objects are uninitialized. EvaluateDerivative returns a spatial
// derivative. The separate Jacobian methods differentiate with respect to cell
// averages, including nonlinear weights in Full mode. Geometry, candidate
// selection, h, epsilon, area and linear weights are held fixed.
class Reconstruction {
public:
    Reconstruction() = default;
    Reconstruction(const Reconstruction&) = default;
    Reconstruction& operator=(const Reconstruction&) = default;
    Reconstruction(Reconstruction&& other) noexcept;
    Reconstruction& operator=(Reconstruction&& other) noexcept;

    // h is the TensorStencilPoly scale; sqrt(L*H/(Nx*Ny)) is the legacy choice.
    // Disabled (zero linear weight) candidates are omitted. The retained order
    // is large candidates followed by small candidates, in input order.
    PetscErrorCode Initialize(const MeshInfo& mesh, MeshIndex target,
                              const ReconstructionOptions& options, PetscReal h);

    bool IsInitialized() const noexcept { return initialized_; }
    bool HasWeights() const noexcept { return hasWeights_; }
    bool HasJacobian() const noexcept { return hasJacobian_; }
    MeshIndex Target() const noexcept { return target_; }
    // Smallest logical rectangle covering the target and active candidates.
    const MeshRange& RequiredCells() const noexcept { return required_; }
    // Sorted unique support: increasing global j, then i. Includes the target;
    // omits holes in RequiredCells(). Available after Initialize, independent
    // of the supplied solution patch. These are logical cell coordinates,
    // NOT PETSc Vec algebraic column numbers. All Jacobian columns and input
    // directions below use this ordering. Reinitialize after repartitioning.
    const std::vector<MeshIndex>& JacobianCells() const noexcept { return jacobianCells_; }
    const ReconstructionOptions& Options() const noexcept { return options_; }
    std::size_t CandidateCount() const noexcept { return candidates_.size(); }
    PetscErrorCode GetCandidate(std::size_t index,
                               ReconstructionCandidateInfo& info) const;
    PetscErrorCode ConstantWeight(PetscReal& weight) const;
    // Legacy sum_j omega_j*r_j, including r0=1; a diagnostic, not a rate proof.
    PetscErrorCode EffectiveOrder(PetscReal& order) const;

    // range is a half-open rectangle of GLOBAL physical cell indices. Values
    // are row-major: (j-range.begin.j)*range.Size().i + i-range.begin.i.
    // range must cover RequiredCells(); a larger owned/ghost patch is allowed.
    // Only the referenced values need be finite. The array is copied by gather,
    // so it may be changed/released after this call.
    //
    // area is the EXPLICIT scale A in epsilon*A, not inferred from geometry.
    // The caller can preserve its legacy nominal area or supply the target's
    // physical area. Both epsilon and A must be finite and strictly positive.
    //
    // alpha_j = d_j/(sigma_j + epsilon*A)^(s*r_j + eta(r_j)),
    // r_j=order_j+1, eta(1)=1, eta(2)=3, eta(r>=3)=4.
    // The constant candidate uses sigma0=0 and r0=1. Weights are normalized
    // together using logarithmic rescaling. TensorStencilPoly returns sigma>=0.
    PetscErrorCode Update(const std::vector<PetscReal>& localAverages,
                          const MeshRange& range, PetscReal area);

    // Update preserves the value-only cost and invalidates any old Jacobian.
    // This variant also prepares weight sensitivities, once per solution.
    // Failure preserves the previous solution, weights AND Jacobian cache.
    PetscErrorCode UpdateWithJacobian(const std::vector<PetscReal>& localAverages,
                                      const MeshRange& range, PetscReal area);
    // Alternatively prepare sensitivities after a successful ordinary Update.
    // Repeated calls reuse the cache until the next Update/Initialize.
    PetscErrorCode PrepareJacobian();

    // d(omega[row])/d(u[column]). Rows follow GetCandidate(), plus a final
    // constant row (always present, identically zero if disabled). The active
    // constant weight also varies through the normalization denominator.
    // Requires HasJacobian(). Outputs are replaced only on success.
    PetscErrorCode WeightJacobian(std::vector<std::vector<PetscReal>>& jacobian) const;

    PetscErrorCode Evaluate(const Point& point, PetscReal& value) const;
    PetscErrorCode Evaluate(const std::vector<Point>& points,
                            std::vector<PetscReal>& values) const;
    PetscErrorCode EvaluateDerivative(const Point& point, PetscInt dx,
                                      PetscInt dy, PetscReal& value) const;

    // Full includes d(omega)/du and requires PrepareJacobian or
    // UpdateWithJacobian. FrozenWeights holds current nonlinear weights fixed
    // and needs only Update (a Picard/approximate-Newton derivative).
    PetscErrorCode EvaluateWithJacobian(const Point& point, PetscReal& value,
                                        std::vector<PetscReal>& jacobian,
                                        ReconstructionJacobianMode mode =
                                            ReconstructionJacobianMode::Full) const;
    PetscErrorCode EvaluateWithJacobian(const std::vector<Point>& points,
                                        std::vector<PetscReal>& values,
                                        std::vector<std::vector<PetscReal>>& jacobian,
                                        ReconstructionJacobianMode mode =
                                            ReconstructionJacobianMode::Full) const;
    // Jacobian of the PHYSICAL spatial derivative d_x^dx d_y^dy R.
    PetscErrorCode EvaluateDerivativeWithJacobian(const Point& point,
                                                  PetscInt dx, PetscInt dy,
                                                  PetscReal& value,
                                                  std::vector<PetscReal>& jacobian,
                                                  ReconstructionJacobianMode mode =
                                                      ReconstructionJacobianMode::Full) const;
    // Apply the local Jacobian to a direction ordered as JacobianCells().
    // Uses directional polynomial evaluation without constructing a point
    // Jacobian row. Both modes use the solution from the latest Update.
    PetscErrorCode ApplyJacobian(const Point& point,
                                 const std::vector<PetscReal>& direction,
                                 PetscReal& action,
                                 ReconstructionJacobianMode mode =
                                     ReconstructionJacobianMode::Full) const;
    PetscErrorCode ApplyDerivativeJacobian(const Point& point, PetscInt dx, PetscInt dy,
                                           const std::vector<PetscReal>& direction,
                                           PetscReal& action,
                                           ReconstructionJacobianMode mode =
                                               ReconstructionJacobianMode::Full) const;

private:
    struct Candidate {
        TensorStencilPoly polynomial;
        ReconstructionCandidateInfo info;
        // Averages of u - targetAverage_; the target entry is exactly zero.
        std::vector<PetscReal> centeredAverages;
        // Map polynomial's local row-major cells to JacobianCells().
        std::vector<std::size_t> columns;
    };
    void Swap(Reconstruction& other) noexcept;
    PetscErrorCode NormalizeWeights(const std::vector<PetscReal>& sigma,
                                    PetscReal area,
                                    std::vector<PetscReal>& weights,
                                    std::vector<long double>* logWeights = nullptr) const;
    PetscErrorCode UpdateImpl(const std::vector<PetscReal>& localAverages,
                              const MeshRange& range, PetscReal area, bool withJacobian);
    PetscErrorCode BuildWeightJacobian(const std::vector<std::vector<PetscReal>>& gradients,
                                       const std::vector<PetscReal>& sigma,
                                       const std::vector<long double>& logWeights,
                                       PetscReal area,
                                       std::vector<std::vector<long double>>& jacobian) const;
    PetscErrorCode CheckJacobianMode(ReconstructionJacobianMode mode) const;

    bool initialized_ = false, hasWeights_ = false, hasJacobian_ = false;
    MeshIndex dimensions_{}, target_{};
    MeshRange required_{};
    ReconstructionOptions options_;
    std::vector<Candidate> candidates_;
    std::vector<MeshIndex> jacobianCells_;
    std::size_t targetColumn_ = 0;
    std::vector<std::vector<long double>> weightJacobian_;
    PetscReal targetAverage_ = 0, constantWeight_ = 0, area_ = 0;
};

#endif
