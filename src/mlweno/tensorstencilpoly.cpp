#include "tensorstencilpoly.h"

#include <lapacke.h>
#include <algorithm>
#include <cmath>
#include <limits>
#include <type_traits>
#include <utility>

static_assert(std::is_same<PetscReal, double>::value,
              "TensorStencilPoly currently requires double-precision PetscReal");

namespace {

PetscErrorCode CheckSetup(MeshIndex start, MeshIndex size, PetscInt order,
                         PetscReal h, PetscInt& count)
{
    PetscFunctionBeginUser;
    const PetscInt limit = std::numeric_limits<PetscInt>::max();
    PetscCheck(size.i > 0 && size.j > 0 && size.i <= limit / size.j,
               PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE,
               "Stencil dimensions must be positive and their product representable");
    PetscCheck(start.i >= 0 && start.j >= 0 &&
               start.i <= limit - (size.i - 1) &&
               start.j <= limit - (size.j - 1),
               PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE,
               "Stencil origin or last cell index is out of range");
    PetscCheck(order >= 0 && !PetscIsInfOrNanReal(h) && h > 0,
               PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE,
               "Smoothness order must be nonnegative and h finite and positive");
    const PetscInt n = size.i * size.j;
    PetscCheck(static_cast<unsigned long long>(n) <=
               static_cast<unsigned long long>(std::numeric_limits<lapack_int>::max()),
               PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE,
               "Stencil size exceeds the LAPACK integer range");
    const auto length = static_cast<std::size_t>(n);
    PetscCheck(length <= std::vector<PetscReal>().max_size() / length,
               PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE,
               "Stencil coefficient matrix is too large");
    count = n;
    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscReal Power(PetscReal x, PetscInt degree)
{
    PetscReal result = 1;
    for (PetscInt k = 0; k < degree; ++k) result *= x;
    return result;
}

PetscReal Falling(PetscInt degree, PetscInt derivative)
{
    PetscReal result = 1;
    for (PetscInt k = 0; k < derivative; ++k)
        result *= static_cast<PetscReal>(degree - k);
    return result;
}

// Horner evaluation of derivatives in NORMALIZED coordinates.
// The polynomial dimensions determine indexing, including 1-by-n stencils.
PetscReal NormalizedDerivative(const PetscReal* coefficients, MeshIndex size,
                               PetscReal x, PetscReal y, PetscInt dx, PetscInt dy)
{
    if (dx >= size.i || dy >= size.j) return 0;
    PetscReal result = 0;
    for (PetscInt j = size.j; j-- > dy;) {
        PetscReal row = 0;
        for (PetscInt i = size.i; i-- > dx;)
            row = row * x + coefficients[static_cast<std::size_t>(j) * size.i + i]
                              * Falling(i, dx);
        result = result * y + row * Falling(j, dy);
    }
    return result;
}

PetscErrorCode Normalize(const QuadVertices& corners, const Point& center,
                         PetscReal h, QuadVertices& normalized)
{
    PetscFunctionBeginUser;
    QuadVertices work{};
    for (std::size_t k = 0; k < 4; ++k)
        for (int d = 0; d < 2; ++d)
            work[k].p[d] = (corners[k].p[d] - center.p[d]) / h;
    PetscCall(ValidateQuad(work));
    normalized = work;
    PetscFunctionReturn(PETSC_SUCCESS);
}

} // namespace

void TensorStencilPoly::Swap(TensorStencilPoly& other) noexcept
{
    using std::swap;
    swap(start_, other.start_);
    swap(size_, other.size_);
    swap(count_, other.count_);
    swap(order_, other.order_);
    swap(center_, other.center_);
    swap(h_, other.h_);
    corners_.swap(other.corners_);
    coefficients_.swap(other.coefficients_);
    sigma_.swap(other.sigma_);
    targetSigma_.swap(other.targetSigma_);
}

TensorStencilPoly::TensorStencilPoly(TensorStencilPoly&& other) noexcept
{ Swap(other); }

TensorStencilPoly& TensorStencilPoly::operator=(TensorStencilPoly&& other) noexcept
{
    if (this != &other) {
        TensorStencilPoly work(std::move(other));
        Swap(work);
    }
    return *this;
}

PetscErrorCode TensorStencilPoly::Initialize(const MeshInfo& mesh, MeshIndex start,
                                            MeshIndex size, PetscInt order,
                                            PetscReal h)
{
    PetscFunctionBeginUser;
    PetscInt count;
    PetscCall(CheckSetup(start, size, order, h, count));
    PetscCheck(mesh.IsInitialized(), PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG,
               "BuildMeshInfo must be called before stencil setup");
    const MeshIndex dimensions = mesh.CellDimensions();
    PetscCheck(size.i <= dimensions.i && size.j <= dimensions.j &&
               start.i <= dimensions.i - size.i && start.j <= dimensions.j - size.j,
               PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE,
               "Stencil extends outside the physical mesh");
    std::vector<QuadVertices> corners(static_cast<std::size_t>(count));
    for (PetscInt j = 0; j < size.j; ++j) {
        for (PetscInt i = 0; i < size.i; ++i) {
            const MeshIndex cell{start.i + i, start.j + j};
            PetscCheck(mesh.HasCell(cell), PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE,
                       "Stencil geometry is unavailable locally; increase the geometry halo");
            PetscCall(mesh.GetCellCorners(cell,
                corners[static_cast<std::size_t>(j) * size.i + i]));
        }
    }
    PetscCall(Initialize(corners, start, size, order, h));
    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode TensorStencilPoly::Initialize(
    const std::vector<QuadVertices>& corners, MeshIndex start, MeshIndex size,
    PetscInt order, PetscReal h)
{
    PetscFunctionBeginUser;
    TensorStencilPoly work;
    PetscCall(CheckSetup(start, size, order, h, work.count_));
    const auto n = static_cast<std::size_t>(work.count_);
    PetscCheck(corners.size() == n, PETSC_COMM_SELF, PETSC_ERR_ARG_SIZ,
               "Expected exactly size.i*size.j cell quadrilaterals");
    for (const auto& cell : corners) PetscCall(ValidateQuad(cell));
    work.start_ = start;
    work.size_ = size;
    work.order_ = order;
    work.h_ = h;
    work.corners_ = corners;

    // Preserve the legacy average of the stencil's four outer vertices.
    // Form differences first to avoid summing large translated coordinates.
    const Point anchor = corners.front()[0];
    const Point outer[4] = {anchor, corners[static_cast<std::size_t>(size.i - 1)][1],
                            corners.back()[2],
                            corners[static_cast<std::size_t>(size.j - 1) * size.i][3]};
    for (int d = 0; d < 2; ++d) {
        PetscReal offset = 0;
        for (const Point& point : outer)
            offset += (point.p[d] - anchor.p[d]) / 4;
        work.center_.p[d] = anchor.p[d] + offset;
        PetscCheck(!PetscIsInfOrNanReal(work.center_.p[d]),
                   PETSC_COMM_SELF, PETSC_ERR_FP, "Stencil center is not finite");
    }

    // Sufficient for all tensor moments on bilinear quadrilaterals.
    const PetscInt gaussPoints = std::max(DefaultGaussPoints, std::max(size.i, size.j));
    GaussRule1D rule;
    PetscCall(CreateGaussRule(gaussPoints, rule));
    std::vector<double> moments(n * n, 0), inverse(n * n, 0);
    for (std::size_t cell = 0; cell < n; ++cell) {
        QuadVertices normalized{};
        PetscCall(Normalize(corners[cell], work.center_, h, normalized));
        PetscReal area = 0;
        PetscCall(IntegrateCell(normalized, rule,
            [](const Point&) { return PetscReal(1); }, area));
        PetscCheck(area > 0, PETSC_COMM_SELF, PETSC_ERR_FP,
                   "Normalized cell area must be positive");
        for (PetscInt j = 0; j < size.j; ++j) {
            for (PetscInt i = 0; i < size.i; ++i) {
                PetscReal moment = 0;
                PetscCall(IntegrateCell(normalized, rule,
                    [i,j](const Point& p) { return Power(p.p[0], i) * Power(p.p[1], j); },
                    moment));
                const auto column = static_cast<std::size_t>(j) * size.i + i;
                moments[cell * n + column] = moment / area;
                PetscCheck(!std::isinf(moments[cell * n + column]) &&
                           !std::isnan(moments[cell * n + column]),
                           PETSC_COMM_SELF, PETSC_ERR_FP, "Cell moment is not finite");
            }
        }
        inverse[cell * n + cell] = 1;
    }

    // M*C = I, M[cell,monomial] = the cell AVERAGE of that monomial.
    // This is the legacy integral system after dividing each row by its area.
    const auto dimension = static_cast<lapack_int>(work.count_);
    std::vector<lapack_int> pivots(n);
    const lapack_int info = LAPACKE_dgesv(LAPACK_ROW_MAJOR, dimension, dimension,
        moments.data(), dimension, pivots.data(), inverse.data(), dimension);
    PetscCheck(info == 0, PETSC_COMM_SELF, PETSC_ERR_LIB,
               "Stencil moment solve failed (LAPACK info=%lld); check geometry and stencil size",
               static_cast<long long>(info));
    work.coefficients_.resize(n * n);
    for (std::size_t cell = 0; cell < n; ++cell) {
        for (std::size_t monomial = 0; monomial < n; ++monomial) {
            const PetscReal value = inverse[monomial * n + cell];
            PetscCheck(!PetscIsInfOrNanReal(value), PETSC_COMM_SELF, PETSC_ERR_FP,
                       "Stencil solve produced a non-finite coefficient");
            work.coefficients_[cell * n + monomial] = value;
        }
    }

    const QuadVertices reference{{Point{{-0.5,-0.5}}, Point{{0.5,-0.5}},
                                  Point{{0.5,0.5}}, Point{{-0.5,0.5}}}};
    PetscCall(work.BuildSmoothness(reference, gaussPoints, work.sigma_));
    Swap(work); // Publish only the complete new state; discard old target caches.
    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode TensorStencilPoly::CheckAverages(
    const std::vector<PetscReal>& averages) const
{
    PetscFunctionBeginUser;
    PetscCheck(IsInitialized(), PETSC_COMM_SELF, PETSC_ERR_ARG_WRONGSTATE,
               "Initialize the stencil before evaluation");
    PetscCheck(averages.size() == static_cast<std::size_t>(count_),
               PETSC_COMM_SELF, PETSC_ERR_ARG_SIZ, "Wrong number of stencil cell averages");
    for (PetscReal value : averages)
        PetscCheck(!PetscIsInfOrNanReal(value), PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE,
                   "Cell averages must be finite");
    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode TensorStencilPoly::PolynomialValue(
    const PetscReal* coefficients, const Point& point, PetscInt dx, PetscInt dy,
    PetscReal& value) const
{
    PetscFunctionBeginUser;
    PetscCheck(dx >= 0 && dy >= 0 &&
               !PetscIsInfOrNanReal(point.p[0]) && !PetscIsInfOrNanReal(point.p[1]),
               PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE,
               "Derivative orders must be nonnegative and the point finite");
    const PetscReal x = (point.p[0] - center_.p[0]) / h_;
    const PetscReal y = (point.p[1] - center_.p[1]) / h_;
    PetscCheck(!PetscIsInfOrNanReal(x) && !PetscIsInfOrNanReal(y),
               PETSC_COMM_SELF, PETSC_ERR_FP, "Normalized evaluation point is not finite");
    PetscReal result = NormalizedDerivative(coefficients, size_, x, y, dx, dy);
    // Derivatives exceeding the polynomial degree are identically zero.
    if (dx < size_.i && dy < size_.j) {
        for (PetscInt k = 0; k < dx; ++k) result /= h_;
        for (PetscInt k = 0; k < dy; ++k) result /= h_;
    }
    PetscCheck(!PetscIsInfOrNanReal(result), PETSC_COMM_SELF, PETSC_ERR_FP,
               "Polynomial value or derivative is not finite");
    value = result;
    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode TensorStencilPoly::EvaluateBasis(
    PetscInt cell, const Point& point, PetscReal& value) const
{ return EvaluateBasisDerivative(cell, point, 0, 0, value); }

PetscErrorCode TensorStencilPoly::EvaluateBasisDerivative(
    PetscInt cell, const Point& point, PetscInt dx, PetscInt dy, PetscReal& value) const
{
    PetscFunctionBeginUser;
    PetscCheck(IsInitialized(), PETSC_COMM_SELF, PETSC_ERR_ARG_WRONGSTATE,
               "Initialize the stencil before evaluation");
    PetscCheck(cell >= 0 && cell < count_, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE,
               "Basis index lies outside the stencil");
    PetscCall(PolynomialValue(coefficients_.data() +
        static_cast<std::size_t>(cell) * count_, point, dx, dy, value));
    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode TensorStencilPoly::Evaluate(
    const std::vector<PetscReal>& averages, const Point& point, PetscReal& value) const
{ return EvaluateDerivative(averages, point, 0, 0, value); }

PetscErrorCode TensorStencilPoly::EvaluateDerivative(
    const std::vector<PetscReal>& averages, const Point& point,
    PetscInt dx, PetscInt dy, PetscReal& value) const
{
    PetscFunctionBeginUser;
    PetscCall(CheckAverages(averages));
    const auto n = static_cast<std::size_t>(count_);
    std::vector<PetscReal> polynomial(n, 0);
    // The basis is a partition of unity. Remove a constant before summation
    // to preserve constant fields exactly and reduce cancellation.
    polynomial[0] = averages[0];
    for (std::size_t cell = 0; cell < n; ++cell) {
        const PetscReal difference = averages[cell] - averages[0];
        PetscCheck(!PetscIsInfOrNanReal(difference), PETSC_COMM_SELF, PETSC_ERR_FP,
                   "Cell-average difference is not representable");
        for (std::size_t k = 0; k < n; ++k)
            polynomial[k] += difference * coefficients_[cell * n + k];
    }
    for (PetscReal coefficient : polynomial)
        PetscCheck(!PetscIsInfOrNanReal(coefficient), PETSC_COMM_SELF, PETSC_ERR_FP,
                   "Reconstructed coefficient is not finite");
    PetscCall(PolynomialValue(polynomial.data(), point, dx, dy, value));
    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode TensorStencilPoly::BuildSmoothness(
    const QuadVertices& normalizedCorners, PetscInt gaussPoints,
    std::vector<PetscReal>& matrix) const
{
    PetscFunctionBeginUser;
    GaussRule1D rule;
    PetscCall(CreateGaussRule(gaussPoints, rule));
    PetscReal area = 0;
    PetscCall(IntegrateCell(normalizedCorners, rule,
        [](const Point&) { return PetscReal(1); }, area));
    PetscCheck(area > 0, PETSC_COMM_SELF, PETSC_ERR_FP,
               "Smoothness integration area must be positive");
    const auto n = static_cast<std::size_t>(count_);
    std::vector<PetscReal> work(n * n, 0), derivative(n);
    for (std::size_t qy = 0; qy < rule.points.size(); ++qy) {
        for (std::size_t qx = 0; qx < rule.points.size(); ++qx) {
            const Point reference{{rule.points[qx], rule.points[qy]}};
            const Point p = MapCellPoint(reference, normalizedCorners);
            const PetscReal weight = rule.weights[qx] * rule.weights[qy]
                * (CellJacobian(reference, normalizedCorners) / area);
            PetscCheck(!PetscIsInfOrNanReal(weight) && weight > 0,
                       PETSC_COMM_SELF, PETSC_ERR_FP, "Invalid smoothness quadrature weight");
            for (PetscInt dy = 0; dy < size_.j; ++dy) {
                for (PetscInt dx = 0; dx < size_.i; ++dx) {
                    if (dx + dy == 0 || dx + dy > order_) continue;
                    for (std::size_t cell = 0; cell < n; ++cell) {
                        derivative[cell] = NormalizedDerivative(
                            coefficients_.data() + cell * n, size_,
                            p.p[0], p.p[1], dx, dy);
                        PetscCheck(!PetscIsInfOrNanReal(derivative[cell]),
                                   PETSC_COMM_SELF, PETSC_ERR_FP,
                                   "Smoothness basis derivative is not finite");
                    }
                    for (std::size_t row = 0; row < n; ++row)
                        for (std::size_t column = 0; column <= row; ++column)
                            work[row * n + column] += weight
                                * derivative[row] * derivative[column];
                }
            }
        }
    }
    for (std::size_t row = 0; row < n; ++row) {
        for (std::size_t column = 0; column <= row; ++column) {
            PetscCheck(!PetscIsInfOrNanReal(work[row * n + column]),
                       PETSC_COMM_SELF, PETSC_ERR_FP, "Smoothness matrix is not finite");
            work[column * n + row] = work[row * n + column];
        }
    }
    matrix = std::move(work);
    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode TensorStencilPoly::TargetIndex(MeshIndex localCell, PetscInt& index) const
{
    PetscFunctionBeginUser;
    PetscCheck(IsInitialized(), PETSC_COMM_SELF, PETSC_ERR_ARG_WRONGSTATE,
               "Initialize the stencil before selecting a target cell");
    PetscCheck(localCell.i >= 0 && localCell.i < size_.i &&
               localCell.j >= 0 && localCell.j < size_.j,
               PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE,
               "Target must be a local cell offset within the stencil");
    index = localCell.j * size_.i + localCell.i;
    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode TensorStencilPoly::SetTargetSmoothness(MeshIndex localCell)
{
    PetscFunctionBeginUser;
    PetscInt index;
    PetscCall(TargetIndex(localCell, index));
    QuadVertices normalized{};
    PetscCall(Normalize(corners_[static_cast<std::size_t>(index)],
                        center_, h_, normalized));
    // On a bilinear physical quad, squared first derivatives have reference
    // degree at most 2*(px+py)-1 including det(J). Q2 therefore needs 4 points.
    const PetscInt degree = (size_.i - 1) + (size_.j - 1);
    std::vector<PetscReal> matrix;
    PetscCall(BuildSmoothness(normalized, std::max(DefaultGaussPoints, degree), matrix));
    targetSigma_.insert_or_assign(index, std::move(matrix));
    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode TensorStencilPoly::ApplySmoothness(
    const std::vector<PetscReal>& matrix, const std::vector<PetscReal>& averages,
    PetscReal& value, std::vector<PetscReal>* gradient) const
{
    PetscFunctionBeginUser;
    PetscCall(CheckAverages(averages));
    const auto n = static_cast<std::size_t>(count_);
    // Derivatives annihilate constants. Centering avoids a large cancellation
    // for nearly constant fields, and makes a constant's indicator exactly zero.
    std::vector<long double> differences(n);
    for (std::size_t i = 0; i < n; ++i)
        differences[i] = static_cast<long double>(averages[i]) - averages[0];
    std::vector<long double> product(gradient ? n : 0, 0);
    long double sum = 0, magnitude = 0;
    for (std::size_t row = 0; row < n; ++row) {
        for (std::size_t column = 0; column < n; ++column) {
            const long double term = differences[row] * matrix[row * n + column]
                                     * differences[column];
            sum += term;
            magnitude += std::abs(term);
            if (gradient)
                product[row] += static_cast<long double>(matrix[row * n + column])
                                * differences[column];
        }
    }
    PetscCheck(std::isfinite(sum) && std::isfinite(magnitude),
               PETSC_COMM_SELF, PETSC_ERR_FP, "Smoothness evaluation overflowed");
    const long double tolerance = 64 * std::numeric_limits<PetscReal>::epsilon() * magnitude;
    PetscCheck(sum >= -tolerance, PETSC_COMM_SELF, PETSC_ERR_FP,
               "Smoothness quadratic form is negative beyond roundoff");
    const PetscReal result = static_cast<PetscReal>(std::max(0.0L, sum));
    PetscCheck(!PetscIsInfOrNanReal(result), PETSC_COMM_SELF, PETSC_ERR_FP,
               "Smoothness indicator is not representable");

    if (gradient) {
        std::vector<PetscReal> work(n, 0);
        if (sum > 0) {
            // v = C*u, C = I - 1*e_0^T, so d(v^T*S*v)/du = 2*C^T*S*v.
            // S is symmetric. For k>0 this is 2*(S*v)[k]; component zero
            // is minus the sum of those components. Do not omit this chain
            // rule: numerical row sums of S need not be exactly zero.
            long double first = 0;
            for (std::size_t k = 1; k < n; ++k) {
                const long double component = 2 * product[k];
                work[k] = static_cast<PetscReal>(component);
                PetscCheck(std::isfinite(component) && !PetscIsInfOrNanReal(work[k]),
                           PETSC_COMM_SELF, PETSC_ERR_FP,
                           "Smoothness gradient is not representable");
                first -= component;
            }
            work[0] = static_cast<PetscReal>(first);
            PetscCheck(std::isfinite(first) && !PetscIsInfOrNanReal(work[0]),
                       PETSC_COMM_SELF, PETSC_ERR_FP,
                       "Smoothness gradient is not representable");
        }
        *gradient = std::move(work);
    }
    value = result;
    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode TensorStencilPoly::Smoothness(
    const std::vector<PetscReal>& averages, PetscReal& value) const
{
    PetscFunctionBeginUser;
    PetscCall(ApplySmoothness(sigma_, averages, value));
    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode TensorStencilPoly::SmoothnessWithGradient(
    const std::vector<PetscReal>& averages, PetscReal& value,
    std::vector<PetscReal>& gradient) const
{
    PetscFunctionBeginUser;
    PetscCall(ApplySmoothness(sigma_, averages, value, &gradient));
    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode TensorStencilPoly::Smoothness(
    const std::vector<PetscReal>& averages, MeshIndex localCell, PetscReal& value) const
{
    PetscFunctionBeginUser;
    PetscInt index;
    PetscCall(TargetIndex(localCell, index));
    const auto entry = targetSigma_.find(index);
    PetscCheck(entry != targetSigma_.end(), PETSC_COMM_SELF, PETSC_ERR_ARG_WRONGSTATE,
               "Call SetTargetSmoothness for this target cell before querying it");
    PetscCall(ApplySmoothness(entry->second, averages, value));
    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode TensorStencilPoly::SmoothnessWithGradient(
    const std::vector<PetscReal>& averages, MeshIndex localCell,
    PetscReal& value, std::vector<PetscReal>& gradient) const
{
    PetscFunctionBeginUser;
    PetscInt index;
    PetscCall(TargetIndex(localCell, index));
    const auto entry = targetSigma_.find(index);
    PetscCheck(entry != targetSigma_.end(), PETSC_COMM_SELF, PETSC_ERR_ARG_WRONGSTATE,
               "Call SetTargetSmoothness for this target cell before querying it");
    PetscCall(ApplySmoothness(entry->second, averages, value, &gradient));
    PetscFunctionReturn(PETSC_SUCCESS);
}
