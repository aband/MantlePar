#include "reconstruction.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <utility>

namespace {

PetscErrorCode CheckFamily(const ReconstructionFamily& family)
{
    PetscFunctionBeginUser;
    const PetscInt limit = std::numeric_limits<PetscInt>::max();
    PetscCheck(family.size.i > 0 && family.size.j > 0 &&
               family.size.i <= limit / family.size.j,
               PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE,
               "Stencil dimensions must be positive with a representable product");
    PetscCheck(family.order >= 0 && family.order < limit,
               PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE,
               "Smoothness order must be nonnegative with order+1 representable");
    PetscCheck(family.smoothness == ReconstructionSmoothness::ReferenceSquare ||
               family.smoothness == ReconstructionSmoothness::TargetCell,
               PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE,
               "Unknown reconstruction smoothness region");
    PetscCheck(family.linearWeights.empty() ||
               family.linearWeights.size() == family.offsets.size(),
               PETSC_COMM_SELF, PETSC_ERR_ARG_SIZ,
               "Provide one linear weight per prescribed offset, or none");
    for (std::size_t k = 0; k < family.offsets.size(); ++k) {
        const MeshIndex offset = family.offsets[k];
        PetscCheck(offset.i <= 0 && offset.i >= 1 - family.size.i &&
                   offset.j <= 0 && offset.j >= 1 - family.size.j,
                   PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE,
                   "Every prescribed stencil must contain the target cell");
        const PetscReal weight = family.linearWeights.empty() ? 1 : family.linearWeights[k];
        PetscCheck(!PetscIsInfOrNanReal(weight) && weight >= 0,
                   PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE,
                   "Linear weights must be finite and nonnegative");
    }
    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscInt Eta(PetscInt r)
{ return r == 1 ? 1 : (r == 2 ? 3 : 4); }

// log(sigma + epsilon*A), without overflowing/underflowing epsilon*A.
long double LogDenominator(PetscReal sigma, long double logRegularization)
{
    if (sigma == 0) return logRegularization;
    const long double logSigma = std::log(static_cast<long double>(sigma));
    const long double high = std::max(logSigma, logRegularization);
    const long double low = std::min(logSigma, logRegularization);
    return high + std::log1p(std::exp(low - high));
}

std::size_t PatchIndex(const MeshRange& range, MeshIndex cell)
{
    return static_cast<std::size_t>(cell.j - range.begin.j) *
           static_cast<std::size_t>(range.Size().i) +
           static_cast<std::size_t>(cell.i - range.begin.i);
}

bool CellLess(MeshIndex a, MeshIndex b)
{ return a.j < b.j || (a.j == b.j && a.i < b.i); }

PetscErrorCode ToReal(long double input, PetscReal& output)
{
    PetscFunctionBeginUser;
    PetscCheck(std::isfinite(input) &&
               std::abs(input) <= std::numeric_limits<PetscReal>::max(),
               PETSC_COMM_SELF, PETSC_ERR_FP, "Jacobian result is not representable");
    output = static_cast<PetscReal>(input);
    PetscFunctionReturn(PETSC_SUCCESS);
}

} // namespace

void Reconstruction::Swap(Reconstruction& other) noexcept
{
    using std::swap;
    swap(initialized_, other.initialized_);
    swap(hasWeights_, other.hasWeights_);
    swap(hasJacobian_, other.hasJacobian_);
    swap(dimensions_, other.dimensions_);
    swap(target_, other.target_);
    swap(required_, other.required_);
    swap(options_, other.options_);
    candidates_.swap(other.candidates_);
    jacobianCells_.swap(other.jacobianCells_);
    swap(targetColumn_, other.targetColumn_);
    weightJacobian_.swap(other.weightJacobian_);
    swap(targetAverage_, other.targetAverage_);
    swap(constantWeight_, other.constantWeight_);
    swap(area_, other.area_);
}

Reconstruction::Reconstruction(Reconstruction&& other) noexcept
{ Swap(other); }

Reconstruction& Reconstruction::operator=(Reconstruction&& other) noexcept
{
    if (this != &other) {
        Reconstruction work(std::move(other));
        Swap(work);
    }
    return *this;
}

PetscErrorCode Reconstruction::Initialize(const MeshInfo& mesh, MeshIndex target,
                                         const ReconstructionOptions& options,
                                         PetscReal h)
{
    PetscFunctionBeginUser;
    PetscCheck(mesh.IsInitialized(), PETSC_COMM_SELF, PETSC_ERR_ARG_WRONGSTATE,
               "BuildMeshInfo must be called before reconstruction setup");
    const MeshIndex dimensions = mesh.CellDimensions();
    PetscCheck(target.i >= 0 && target.i < dimensions.i &&
               target.j >= 0 && target.j < dimensions.j,
               PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE,
               "Reconstruction target must be a physical cell");
    PetscCheck(mesh.HasCell(target), PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE,
               "Target geometry is unavailable locally; increase the geometry halo");
    PetscCheck(!PetscIsInfOrNanReal(h) && h > 0 &&
               !PetscIsInfOrNanReal(options.epsilon) && options.epsilon > 0 &&
               options.s >= 0,
               PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE,
               "h and epsilon must be finite and positive; s must be nonnegative");
    PetscCheck(!PetscIsInfOrNanReal(options.constantLinearWeight) &&
               options.constantLinearWeight >= 0,
               PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE,
               "Constant linear weight must be finite and nonnegative");
    PetscCall(CheckFamily(options.large));
    PetscCall(CheckFamily(options.small));

    Reconstruction work;
    work.dimensions_ = dimensions;
    work.target_ = target;
    work.required_ = {target, {target.i + 1, target.j + 1}};
    work.options_ = options;
    const auto addFamily = [&](const ReconstructionFamily& family,
                               ReconstructionFamilyKind kind) -> PetscErrorCode {
        PetscFunctionBeginUser;
        for (std::size_t k = 0; k < family.offsets.size(); ++k) {
            const PetscReal weight = family.linearWeights.empty() ? 1 : family.linearWeights[k];
            if (weight == 0) continue;
            const MeshIndex offset = family.offsets[k];
            // CheckFamily bounds offsets by 1-size <= offset <= 0, so these
            // additions cannot overflow PetscInt for a valid target.
            const MeshIndex start{target.i + offset.i, target.j + offset.j};
            if (start.i < 0 || start.j < 0 ||
                family.size.i > dimensions.i || family.size.j > dimensions.j ||
                start.i > dimensions.i - family.size.i ||
                start.j > dimensions.j - family.size.j) continue;

            Candidate candidate;
            candidate.info.family = kind;
            candidate.info.configuredIndex = k;
            candidate.info.start = start;
            candidate.info.size = family.size;
            candidate.info.targetOffset = {-offset.i, -offset.j};
            candidate.info.order = family.order;
            candidate.info.smoothnessRegion = family.smoothness;
            candidate.info.linearWeight = weight;
            // TensorStencilPoly checks every cell's LOCAL geometry availability.
            // A missing MPI neighbor is an error, never a boundary filter.
            PetscCall(candidate.polynomial.Initialize(mesh, start, family.size,
                                                       family.order, h));
            if (family.smoothness == ReconstructionSmoothness::TargetCell)
                PetscCall(candidate.polynomial.SetTargetSmoothness(candidate.info.targetOffset));
            work.required_.begin.i = std::min(work.required_.begin.i, start.i);
            work.required_.begin.j = std::min(work.required_.begin.j, start.j);
            work.required_.end.i = std::max(work.required_.end.i, start.i + family.size.i);
            work.required_.end.j = std::max(work.required_.end.j, start.j + family.size.j);
            work.candidates_.push_back(std::move(candidate));
        }
        PetscFunctionReturn(PETSC_SUCCESS);
    };
    PetscCall(addFamily(options.large, ReconstructionFamilyKind::Large));
    PetscCall(addFamily(options.small, ReconstructionFamilyKind::Small));
    PetscCheck(!work.candidates_.empty() ||
               (options.useConstant && options.constantLinearWeight > 0),
               PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG,
               "No positive-weight candidate remains at this target cell");

    work.jacobianCells_.push_back(target);
    for (const auto& candidate : work.candidates_) {
        const auto& info = candidate.info;
        for (PetscInt j = 0; j < info.size.j; ++j)
            for (PetscInt i = 0; i < info.size.i; ++i)
                work.jacobianCells_.push_back({info.start.i + i, info.start.j + j});
    }
    auto& cells = work.jacobianCells_;
    std::sort(cells.begin(), cells.end(), CellLess);
    cells.erase(std::unique(cells.begin(), cells.end()), cells.end());
    const auto column = [&](MeshIndex cell) -> std::size_t {
        return static_cast<std::size_t>(
            std::lower_bound(cells.begin(), cells.end(), cell, CellLess) - cells.begin());
    };
    work.targetColumn_ = column(target);
    for (auto& candidate : work.candidates_) {
        const auto& info = candidate.info;
        candidate.columns.reserve(static_cast<std::size_t>(candidate.polynomial.CellCount()));
        for (PetscInt j = 0; j < info.size.j; ++j)
            for (PetscInt i = 0; i < info.size.i; ++i)
                candidate.columns.push_back(column({info.start.i + i, info.start.j + j}));
    }
    work.initialized_ = true;
    Swap(work);
    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode Reconstruction::NormalizeWeights(const std::vector<PetscReal>& sigma,
                                               PetscReal area,
                                               std::vector<PetscReal>& weights,
                                               std::vector<long double>* logWeights) const
{
    PetscFunctionBeginUser;
    // Final entry is always reserved for the optional constant candidate.
    std::vector<long double> logAlpha(candidates_.size() + 1,
                                     -std::numeric_limits<long double>::infinity());
    const long double logRegularization = std::log(static_cast<long double>(options_.epsilon)) +
                                               std::log(static_cast<long double>(area));
    for (std::size_t k = 0; k < candidates_.size(); ++k) {
        const auto& info = candidates_[k].info;
        const PetscInt r = info.order + 1;
        const long double exponent = static_cast<long double>(options_.s) * r + Eta(r);
        logAlpha[k] = std::log(static_cast<long double>(info.linearWeight)) -
                       exponent * LogDenominator(sigma[k], logRegularization);
        PetscCheck(std::isfinite(logAlpha[k]), PETSC_COMM_SELF, PETSC_ERR_FP,
                   "Polynomial log weight is not finite");
    }
    if (options_.useConstant && options_.constantLinearWeight > 0) {
        const long double exponent = static_cast<long double>(options_.s) + Eta(1);
        logAlpha.back() = std::log(static_cast<long double>(options_.constantLinearWeight)) -
                          exponent * logRegularization;
        PetscCheck(std::isfinite(logAlpha.back()), PETSC_COMM_SELF, PETSC_ERR_FP,
                   "Constant log weight is not finite");
    }
    const long double maximum = *std::max_element(logAlpha.begin(), logAlpha.end());
    PetscCheck(std::isfinite(maximum), PETSC_COMM_SELF, PETSC_ERR_FP,
               "No finite active log weight");
    std::vector<long double> logarithms;
    if (logWeights) {
        logarithms.resize(logAlpha.size());
        for (std::size_t k = 0; k < logarithms.size(); ++k)
            logarithms[k] = logAlpha[k] - maximum;
    }
    long double sum = 0;
    for (long double& value : logAlpha) {
        value = std::exp(value - maximum);
        sum += value;
    }
    PetscCheck(std::isfinite(sum) && sum > 0, PETSC_COMM_SELF, PETSC_ERR_FP,
               "Invalid ML-WENO weight normalization");
    std::vector<PetscReal> work(logAlpha.size());
    for (std::size_t k = 0; k < work.size(); ++k)
        work[k] = static_cast<PetscReal>(logAlpha[k] / sum);
    if (logWeights) {
        const long double logSum = std::log(sum);
        for (long double& logarithm : logarithms) logarithm -= logSum;
        logWeights->swap(logarithms);
    }
    weights.swap(work);
    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode Reconstruction::Update(const std::vector<PetscReal>& localAverages,
                                     const MeshRange& range, PetscReal area)
{ return UpdateImpl(localAverages, range, area, false); }

PetscErrorCode Reconstruction::UpdateWithJacobian(
    const std::vector<PetscReal>& localAverages, const MeshRange& range, PetscReal area)
{ return UpdateImpl(localAverages, range, area, true); }

PetscErrorCode Reconstruction::UpdateImpl(const std::vector<PetscReal>& localAverages,
                                         const MeshRange& range, PetscReal area,
                                         bool withJacobian)
{
    PetscFunctionBeginUser;
    PetscCheck(initialized_, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONGSTATE,
               "Initialize reconstruction before updating its solution");
    PetscCheck(!PetscIsInfOrNanReal(area) && area > 0,
               PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE,
               "Weight area scale must be finite and positive");
    PetscCheck(range.begin.i >= 0 && range.begin.j >= 0 &&
               range.end.i > range.begin.i && range.end.j > range.begin.j &&
               range.end.i <= dimensions_.i && range.end.j <= dimensions_.j,
               PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE,
               "Cell-average range must be a nonempty rectangle of physical cells");
    const auto nx = static_cast<std::size_t>(range.Size().i);
    const auto ny = static_cast<std::size_t>(range.Size().j);
    PetscCheck(nx <= localAverages.max_size() / ny && localAverages.size() == nx * ny,
               PETSC_COMM_SELF, PETSC_ERR_ARG_SIZ,
               "Cell-average array size must equal the supplied range area in cells");
    PetscCheck(range.begin.i <= required_.begin.i && range.begin.j <= required_.begin.j &&
               range.end.i >= required_.end.i && range.end.j >= required_.end.j,
               PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE,
               "Cell averages are unavailable locally; update/increase the solution halo");
    const PetscReal targetAverage = localAverages[PatchIndex(range, target_)];
    PetscCheck(!PetscIsInfOrNanReal(targetAverage), PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE,
               "Target cell average must be finite");

    std::vector<std::vector<PetscReal>> centered(candidates_.size());
    std::vector<std::vector<PetscReal>> gradients(withJacobian ? candidates_.size() : 0);
    const bool needGradients = withJacobian &&
        (candidates_.size() + (options_.useConstant && options_.constantLinearWeight > 0) > 1);
    std::vector<PetscReal> sigma(candidates_.size()), weights;
    std::vector<long double> logWeights;
    std::vector<std::vector<long double>> weightJacobian;
    for (std::size_t k = 0; k < candidates_.size(); ++k) {
        const auto& candidate = candidates_[k];
        const auto& info = candidate.info;
        auto& values = centered[k];
        values.resize(static_cast<std::size_t>(candidate.polynomial.CellCount()));
        for (PetscInt j = 0; j < info.size.j; ++j) {
            for (PetscInt i = 0; i < info.size.i; ++i) {
                const MeshIndex cell{info.start.i + i, info.start.j + j};
                const PetscReal average = localAverages[PatchIndex(range, cell)];
                PetscCheck(!PetscIsInfOrNanReal(average), PETSC_COMM_SELF,
                           PETSC_ERR_ARG_OUTOFRANGE, "Referenced cell averages must be finite");
                const PetscReal difference = average - targetAverage;
                PetscCheck(!PetscIsInfOrNanReal(difference), PETSC_COMM_SELF,
                           PETSC_ERR_FP, "Centered cell average is not representable");
                values[static_cast<std::size_t>(j) * info.size.i + i] = difference;
            }
        }
        if (needGradients && info.smoothnessRegion == ReconstructionSmoothness::TargetCell) {
            PetscCall(candidate.polynomial.SmoothnessWithGradient(
                values, info.targetOffset, sigma[k], gradients[k]));
        } else if (needGradients) {
            PetscCall(candidate.polynomial.SmoothnessWithGradient(values, sigma[k], gradients[k]));
        } else if (info.smoothnessRegion == ReconstructionSmoothness::TargetCell) {
            PetscCall(candidate.polynomial.Smoothness(values, info.targetOffset, sigma[k]));
        } else {
            PetscCall(candidate.polynomial.Smoothness(values, sigma[k]));
        }
    }
    PetscCall(NormalizeWeights(sigma, area, weights, withJacobian ? &logWeights : nullptr));
    if (withJacobian)
        PetscCall(BuildWeightJacobian(gradients, sigma, logWeights, area, weightJacobian));

    // Commit solution, indicators, and weights together after all checks pass.
    for (std::size_t k = 0; k < candidates_.size(); ++k) {
        candidates_[k].centeredAverages.swap(centered[k]);
        candidates_[k].info.smoothness = sigma[k];
        candidates_[k].info.nonlinearWeight = weights[k];
    }
    targetAverage_ = targetAverage;
    constantWeight_ = weights.back();
    area_ = area;
    weightJacobian_.swap(weightJacobian);
    hasJacobian_ = withJacobian;
    hasWeights_ = true;
    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode Reconstruction::BuildWeightJacobian(
    const std::vector<std::vector<PetscReal>>& gradients,
    const std::vector<PetscReal>& sigma, const std::vector<long double>& logWeights,
    PetscReal area, std::vector<std::vector<long double>>& jacobian) const
{
    PetscFunctionBeginUser;
    const std::size_t rows = candidates_.size() + 1;
    const std::size_t columns = jacobianCells_.size();
    std::vector<std::vector<long double>> work(rows, std::vector<long double>(columns, 0));
    const long double logRegularization = std::log(static_cast<long double>(options_.epsilon))
                                       + std::log(static_cast<long double>(area));

    // g_k = d(log alpha_k)/du = -p_k/(sigma_k+epsilon*A) * d(sigma_k)/du.
    // d(omega_k) = sum_(l!=k) omega_k*omega_l*(g_k-g_l).
    // Accumulate each g_k contribution antisymmetrically into rows k and l.
    // Pair products are formed in log space BEFORE multiplying by a gradient,
    // avoiding epsilon*A underflow, unnormalized-weight overflow, and the
    // cancellation in omega_k*(1-omega_k) for a nearly dominant candidate.
    for (std::size_t k = 0; k < candidates_.size(); ++k) {
        const auto& candidate = candidates_[k];
        const PetscInt r = candidate.info.order + 1;
        const long double exponent = static_cast<long double>(options_.s) * r + Eta(r);
        const long double logFactor = std::log(exponent) - LogDenominator(sigma[k], logRegularization);
        for (std::size_t other = 0; other < rows; ++other) {
            if (other == k || !std::isfinite(logWeights[k]) ||
                !std::isfinite(logWeights[other])) continue;
            const long double logPair = logWeights[k] + logWeights[other] + logFactor;
            for (std::size_t a = 0; a < candidate.columns.size(); ++a) {
                const std::size_t column = candidate.columns[a];
                if (column == targetColumn_ || gradients[k][a] == 0) continue;
                const long double gradient = gradients[k][a];
                const long double entry = -std::copysign(
                    std::exp(logPair + std::log(std::abs(gradient))), gradient);
                PetscCheck(std::isfinite(entry), PETSC_COMM_SELF, PETSC_ERR_FP,
                           "Nonlinear weight sensitivity overflowed");
                work[k][column] += entry;
                work[other][column] -= entry;
            }
        }
    }
    for (auto& row : work) {
        // Update gathers z_a=u_a-u_target. Its chain rule makes the target
        // derivative minus the sum of every other column (including overlap).
        long double target = 0;
        for (std::size_t column = 0; column < columns; ++column) {
            if (column != targetColumn_) target -= row[column];
            PetscCheck(std::isfinite(row[column]), PETSC_COMM_SELF, PETSC_ERR_FP,
                       "Nonlinear weight sensitivity accumulation overflowed");
        }
        PetscCheck(std::isfinite(target), PETSC_COMM_SELF, PETSC_ERR_FP,
                   "Target weight sensitivity overflowed");
        row[targetColumn_] = target;
    }
    jacobian.swap(work);
    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode Reconstruction::PrepareJacobian()
{
    PetscFunctionBeginUser;
    PetscCheck(hasWeights_, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONGSTATE,
               "Update reconstruction before preparing its Jacobian");
    if (hasJacobian_) PetscFunctionReturn(PETSC_SUCCESS);
    if (candidates_.size() + (options_.useConstant && options_.constantLinearWeight > 0) == 1) {
        weightJacobian_.assign(candidates_.size() + 1,
                               std::vector<long double>(jacobianCells_.size(), 0));
        hasJacobian_ = true;
        PetscFunctionReturn(PETSC_SUCCESS);
    }
    std::vector<PetscReal> sigma(candidates_.size()), weights;
    std::vector<std::vector<PetscReal>> gradients(candidates_.size());
    for (std::size_t k = 0; k < candidates_.size(); ++k) {
        const auto& candidate = candidates_[k];
        if (candidate.info.smoothnessRegion == ReconstructionSmoothness::TargetCell)
            PetscCall(candidate.polynomial.SmoothnessWithGradient(candidate.centeredAverages,
                candidate.info.targetOffset, sigma[k], gradients[k]));
        else
            PetscCall(candidate.polynomial.SmoothnessWithGradient(candidate.centeredAverages,
                sigma[k], gradients[k]));
    }
    std::vector<long double> logWeights;
    PetscCall(NormalizeWeights(sigma, area_, weights, &logWeights));
    std::vector<std::vector<long double>> jacobian;
    PetscCall(BuildWeightJacobian(gradients, sigma, logWeights, area_, jacobian));
    weightJacobian_.swap(jacobian);
    hasJacobian_ = true;
    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode Reconstruction::CheckJacobianMode(ReconstructionJacobianMode mode) const
{
    PetscFunctionBeginUser;
    PetscCheck(hasWeights_, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONGSTATE,
               "Update reconstruction before evaluating its Jacobian");
    PetscCheck(mode == ReconstructionJacobianMode::Full ||
               mode == ReconstructionJacobianMode::FrozenWeights,
               PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Unknown reconstruction Jacobian mode");
    PetscCheck(mode != ReconstructionJacobianMode::Full || hasJacobian_,
               PETSC_COMM_SELF, PETSC_ERR_ARG_WRONGSTATE,
               "Call UpdateWithJacobian or PrepareJacobian before requesting the full Jacobian");
    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode Reconstruction::WeightJacobian(
    std::vector<std::vector<PetscReal>>& jacobian) const
{
    PetscFunctionBeginUser;
    PetscCall(CheckJacobianMode(ReconstructionJacobianMode::Full));
    std::vector<std::vector<PetscReal>> work(weightJacobian_.size(),
                                           std::vector<PetscReal>(jacobianCells_.size()));
    for (std::size_t row = 0; row < work.size(); ++row)
        for (std::size_t column = 0; column < work[row].size(); ++column)
            PetscCall(ToReal(weightJacobian_[row][column], work[row][column]));
    jacobian.swap(work);
    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode Reconstruction::Evaluate(const Point& point, PetscReal& value) const
{ return EvaluateDerivative(point, 0, 0, value); }

PetscErrorCode Reconstruction::EvaluateDerivative(const Point& point, PetscInt dx,
                                                  PetscInt dy, PetscReal& value) const
{
    PetscFunctionBeginUser;
    PetscCheck(hasWeights_, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONGSTATE,
               "Update reconstruction before evaluation");
    PetscCheck(dx >= 0 && dy >= 0 && !PetscIsInfOrNanReal(point.p[0]) &&
               !PetscIsInfOrNanReal(point.p[1]),
               PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE,
               "Derivative orders must be nonnegative and the point finite");
    // Equivalent to sum_j omega_j*p_j + omega0*targetAverage. Every centered
    // candidate has zero target-cell mean. This form preserves constants exactly.
    long double result = (dx == 0 && dy == 0) ? targetAverage_ : 0;
    for (const auto& candidate : candidates_) {
        if (candidate.info.nonlinearWeight == 0) continue;
        PetscReal residual = 0;
        PetscCall(candidate.polynomial.EvaluateDerivative(candidate.centeredAverages,
                                                          point, dx, dy, residual));
        result += static_cast<long double>(candidate.info.nonlinearWeight) * residual;
    }
    PetscCheck(std::isfinite(result) &&
               std::abs(result) <= std::numeric_limits<PetscReal>::max(),
               PETSC_COMM_SELF, PETSC_ERR_FP, "Reconstructed value is not representable");
    value = static_cast<PetscReal>(result);
    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode Reconstruction::EvaluateWithJacobian(
    const Point& point, PetscReal& value, std::vector<PetscReal>& jacobian,
    ReconstructionJacobianMode mode) const
{ return EvaluateDerivativeWithJacobian(point, 0, 0, value, jacobian, mode); }

PetscErrorCode Reconstruction::EvaluateDerivativeWithJacobian(
    const Point& point, PetscInt dx, PetscInt dy, PetscReal& value,
    std::vector<PetscReal>& jacobian, ReconstructionJacobianMode mode) const
{
    PetscFunctionBeginUser;
    PetscCall(CheckJacobianMode(mode));
    PetscReal result = 0;
    PetscCall(EvaluateDerivative(point, dx, dy, result));
    const long double constant = (dx == 0 && dy == 0) ? 1 : 0;
    std::vector<long double> derivative(jacobianCells_.size(), 0);
    for (std::size_t k = 0; k < candidates_.size(); ++k) {
        const auto& candidate = candidates_[k];
        const long double weight = candidate.info.nonlinearWeight;
        if (weight != 0) {
            // TensorStencilPoly evaluates z_0 + sum_(a>0)(z_a-z_0)*phi_a.
            // Its effective first basis is therefore 1-sum_(a>0)phi_a,
            // or -sum_(a>0)D(phi_a) for a nonzero spatial derivative.
            long double first = constant;
            for (std::size_t a = 1; a < candidate.columns.size(); ++a) {
                PetscReal basis = 0;
                PetscCall(candidate.polynomial.EvaluateBasisDerivative(
                    static_cast<PetscInt>(a), point, dx, dy, basis));
                first -= basis;
                if (candidate.columns[a] != targetColumn_)
                    derivative[candidate.columns[a]] += weight * basis;
            }
            if (candidate.columns[0] != targetColumn_)
                derivative[candidate.columns[0]] += weight * first;
        }
        if (mode == ReconstructionJacobianMode::Full) {
            PetscReal residual = 0;
            PetscCall(candidate.polynomial.EvaluateDerivative(candidate.centeredAverages,
                                                               point, dx, dy, residual));
            for (std::size_t column = 0; column < derivative.size(); ++column)
                if (column != targetColumn_)
                    derivative[column] += static_cast<long double>(residual) *
                                          weightJacobian_[k][column];
        }
    }
    // R = u_target + sum omega_k*P_k(u-u_target). Its target sensitivity
    // follows from the chain rule, including both levels of centering.
    // This also enforces sum(dR/du)=1, or zero for a spatial derivative.
    long double target = constant;
    std::vector<PetscReal> work(derivative.size());
    for (std::size_t column = 0; column < derivative.size(); ++column) {
        if (column == targetColumn_) continue;
        PetscCall(ToReal(derivative[column], work[column]));
        target -= derivative[column];
    }
    PetscCall(ToReal(target, work[targetColumn_]));
    jacobian.swap(work);
    value = result;
    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode Reconstruction::EvaluateWithJacobian(
    const std::vector<Point>& points, std::vector<PetscReal>& values,
    std::vector<std::vector<PetscReal>>& jacobian, ReconstructionJacobianMode mode) const
{
    PetscFunctionBeginUser;
    PetscCall(CheckJacobianMode(mode));
    std::vector<PetscReal> workValues(points.size());
    std::vector<std::vector<PetscReal>> workJacobian(points.size());
    for (std::size_t point = 0; point < points.size(); ++point)
        PetscCall(EvaluateWithJacobian(points[point], workValues[point], workJacobian[point], mode));
    values.swap(workValues);
    jacobian.swap(workJacobian);
    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode Reconstruction::ApplyJacobian(
    const Point& point, const std::vector<PetscReal>& direction,
    PetscReal& action, ReconstructionJacobianMode mode) const
{ return ApplyDerivativeJacobian(point, 0, 0, direction, action, mode); }

PetscErrorCode Reconstruction::ApplyDerivativeJacobian(
    const Point& point, PetscInt dx, PetscInt dy, const std::vector<PetscReal>& direction,
    PetscReal& action, ReconstructionJacobianMode mode) const
{
    PetscFunctionBeginUser;
    PetscCall(CheckJacobianMode(mode));
    PetscCheck(dx >= 0 && dy >= 0 && !PetscIsInfOrNanReal(point.p[0]) &&
               !PetscIsInfOrNanReal(point.p[1]),
               PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE,
               "Derivative orders must be nonnegative and the point finite");
    PetscCheck(direction.size() == jacobianCells_.size(), PETSC_COMM_SELF, PETSC_ERR_ARG_SIZ,
               "Jacobian direction must have one entry per JacobianCells() cell");
    for (PetscReal entry : direction)
        PetscCheck(!PetscIsInfOrNanReal(entry), PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE,
                   "Jacobian direction must be finite");
    const PetscReal targetDirection = direction[targetColumn_];
    long double result = (dx == 0 && dy == 0) ? targetDirection : 0;
    for (std::size_t k = 0; k < candidates_.size(); ++k) {
        const auto& candidate = candidates_[k];
        const long double weight = candidate.info.nonlinearWeight;
        if (weight != 0) {
            std::vector<PetscReal> localDirection(candidate.columns.size());
            for (std::size_t a = 0; a < candidate.columns.size(); ++a)
                PetscCall(ToReal(static_cast<long double>(direction[candidate.columns[a]]) -
                                 targetDirection, localDirection[a]));
            PetscReal polynomialAction = 0;
            PetscCall(candidate.polynomial.EvaluateDerivative(localDirection, point,
                                                               dx, dy, polynomialAction));
            result += weight * polynomialAction;
        }
        if (mode == ReconstructionJacobianMode::Full) {
            long double weightAction = 0;
            for (std::size_t column = 0; column < direction.size(); ++column)
                if (column != targetColumn_)
                    weightAction += weightJacobian_[k][column] *
                        (static_cast<long double>(direction[column]) - targetDirection);
            PetscReal residual = 0;
            PetscCall(candidate.polynomial.EvaluateDerivative(candidate.centeredAverages,
                                                               point, dx, dy, residual));
            result += weightAction * residual;
        }
    }
    PetscReal work = 0;
    PetscCall(ToReal(result, work));
    action = work;
    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode Reconstruction::Evaluate(const std::vector<Point>& points,
                                        std::vector<PetscReal>& values) const
{
    PetscFunctionBeginUser;
    PetscCheck(hasWeights_, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONGSTATE,
               "Update reconstruction before evaluation");
    std::vector<PetscReal> work(points.size());
    for (std::size_t k = 0; k < points.size(); ++k) PetscCall(Evaluate(points[k], work[k]));
    values.swap(work);
    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode Reconstruction::GetCandidate(std::size_t index,
                                           ReconstructionCandidateInfo& info) const
{
    PetscFunctionBeginUser;
    PetscCheck(initialized_, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONGSTATE,
               "Initialize reconstruction before querying candidates");
    PetscCheck(index < candidates_.size(), PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE,
               "Reconstruction candidate index is out of range");
    info = candidates_[index].info;
    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode Reconstruction::ConstantWeight(PetscReal& weight) const
{
    PetscFunctionBeginUser;
    PetscCheck(hasWeights_, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONGSTATE,
               "Update reconstruction before querying its constant weight");
    weight = constantWeight_;
    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode Reconstruction::EffectiveOrder(PetscReal& order) const
{
    PetscFunctionBeginUser;
    PetscCheck(hasWeights_, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONGSTATE,
               "Update reconstruction before querying its effective order");
    long double result = constantWeight_; // r0=1.
    for (const auto& candidate : candidates_)
        result += static_cast<long double>(candidate.info.nonlinearWeight) *
                  (candidate.info.order + 1);
    order = static_cast<PetscReal>(result);
    PetscFunctionReturn(PETSC_SUCCESS);
}
