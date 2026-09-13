#ifndef MANTLE_CORE_INTEGRAL_H
#define MANTLE_CORE_INTEGRAL_H

#include "mesh.h"

#include <array>
#include <utility>
#include <vector>

using EdgeVertices = std::array<Point, 2>;
using QuadVertices = std::array<Point, 4>;

inline constexpr PetscInt DefaultGaussPoints = 3;

// Reference Gauss-Legendre rule on [-1,1]. n points have degree 2*n-1
// exactness in one dimension; cell integration uses its n-by-n tensor product.
// Create once and reuse. Treat points and weights as read-only after creation.
struct GaussRule1D {
    std::vector<PetscReal> points;
    std::vector<PetscReal> weights;
};

// All routines here are local (not collective). Use after PetscInitialize.
// The caller supplies owned-cell vertices, including any required MPI ghosts,
// and performs any global reductions outside these routines.
PetscErrorCode CreateGaussRule(PetscInt npoints, GaussRule1D& rule);
PetscErrorCode ValidateGaussRule(const GaussRule1D& rule);

// Require finite, convex, counterclockwise corners in mesh.h's order:
// (i,j), (i+1,j), (i+1,j+1), (i,j+1). Uses the same scaled corner-Jacobian
// criterion as ValidateMesh, including its treatment of rectangular aspect ratio.
PetscErrorCode ValidateQuad(const QuadVertices& corners);

// Require two distinct finite endpoints. The normal is the unit right normal
// (dy,-dx)/length. It points outward when the edge follows a cell's CCW boundary.
PetscErrorCode GetEdgeGeometry(const EdgeVertices& edge, PetscReal& length,
                               Point& normal);

// Low-level mapping helpers: callers must supply validated geometry and
// reference coordinates in [-1,1]. The integration functions check geometry.
Point MapEdgePoint(PetscReal reference, const EdgeVertices& edge);
Point MapCellPoint(const Point& reference, const QuadVertices& corners);
PetscReal CellJacobian(const Point& reference, const QuadVertices& corners);

// Physical cell integral: integral_K f(x,y) dx dy. The callback accepts
// const Point& and returns a finite real scalar. Capturing lambdas can carry
// parameters formerly passed through vector<int> or vector<double>.
// The result is assigned only on success. All validation remains active in Release.
template <typename Function>
PetscErrorCode IntegrateCell(const QuadVertices& corners,
                             const GaussRule1D& rule, Function&& function,
                             PetscReal& integral)
{
    PetscFunctionBeginUser;
    PetscCall(ValidateQuad(corners));
    PetscCall(ValidateGaussRule(rule));
    PetscReal sum = 0.0;
    for (std::size_t j = 0; j < rule.points.size(); ++j) {
        for (std::size_t i = 0; i < rule.points.size(); ++i) {
            const Point reference{{rule.points[i], rule.points[j]}};
            const Point point = MapCellPoint(reference, corners);
            const PetscReal jacobian = CellJacobian(reference, corners);
            PetscCheck(!PetscIsInfOrNanReal(point.p[0]) &&
                       !PetscIsInfOrNanReal(point.p[1]) &&
                       !PetscIsInfOrNanReal(jacobian) && jacobian > 0.0,
                       PETSC_COMM_SELF, PETSC_ERR_FP,
                       "Cell mapping produced an invalid point or Jacobian");
            const PetscReal value = function(point);
            PetscCheck(!PetscIsInfOrNanReal(value), PETSC_COMM_SELF, PETSC_ERR_FP,
                       "Cell integrand returned a non-finite value");
            sum += rule.weights[i] * rule.weights[j] * jacobian * value;
        }
    }
    PetscCheck(!PetscIsInfOrNanReal(sum), PETSC_COMM_SELF, PETSC_ERR_FP,
               "Cell integral is not finite");
    integral = sum;
    PetscFunctionReturn(PETSC_SUCCESS);
}

// Normalized-domain integral, preserving the old NumIntegralFace convention:
// K' = (K-center)/h; return integral_K' f(z) dz, with finite h > 0.
// This equals integral_K f((x-center)/h) dx / h^2. In particular, f=1
// returns area(K)/h^2. For physical integration use IntegrateCell instead.
template <typename Function>
PetscErrorCode IntegrateCellNormalized(const QuadVertices& corners,
                                       const GaussRule1D& rule,
                                       const Point& center, PetscReal h,
                                       Function&& function, PetscReal& integral)
{
    PetscFunctionBeginUser;
    PetscCall(ValidateQuad(corners));
    PetscCheck(!PetscIsInfOrNanReal(h) && h > 0.0,
               PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE,
               "Normalization scale h must be finite and positive");
    PetscCheck(!PetscIsInfOrNanReal(center.p[0]) &&
               !PetscIsInfOrNanReal(center.p[1]),
               PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE,
               "Normalization center must be finite");
    QuadVertices normalized{};
    for (std::size_t k = 0; k < corners.size(); ++k) {
        for (int d = 0; d < 2; ++d) {
            normalized[k].p[d] = (corners[k].p[d] - center.p[d]) / h;
        }
    }
    PetscCall(IntegrateCell(normalized, rule, std::forward<Function>(function), integral));
    PetscFunctionReturn(PETSC_SUCCESS);
}

// Physical scalar line integral: integral_edge f(x,y) ds.
// Reversing the endpoints leaves this integral unchanged (up to roundoff).
template <typename Function>
PetscErrorCode IntegrateEdge(const EdgeVertices& edge,
                             const GaussRule1D& rule, Function&& function,
                             PetscReal& integral)
{
    PetscFunctionBeginUser;
    PetscCall(ValidateGaussRule(rule));
    PetscReal length = 0.0;
    Point normal{};
    PetscCall(GetEdgeGeometry(edge, length, normal));
    PetscReal sum = 0.0;
    for (std::size_t i = 0; i < rule.points.size(); ++i) {
        const Point point = MapEdgePoint(rule.points[i], edge);
        PetscCheck(!PetscIsInfOrNanReal(point.p[0]) &&
                   !PetscIsInfOrNanReal(point.p[1]),
                   PETSC_COMM_SELF, PETSC_ERR_FP,
                   "Edge mapping produced a non-finite point");
        const PetscReal value = function(point);
        PetscCheck(!PetscIsInfOrNanReal(value), PETSC_COMM_SELF, PETSC_ERR_FP,
                   "Edge integrand returned a non-finite value");
        sum += rule.weights[i] * value;
    }
    sum *= 0.5 * length;
    PetscCheck(!PetscIsInfOrNanReal(sum), PETSC_COMM_SELF, PETSC_ERR_FP,
               "Edge integral is not finite");
    integral = sum;
    PetscFunctionReturn(PETSC_SUCCESS);
}

// Normal flux: integral_edge F(x,y).n ds. The callback returns Point{{Fx,Fy}}.
// Reversing the endpoints reverses the normal and the sign of the integral.
template <typename Function>
PetscErrorCode IntegrateNormalFlux(const EdgeVertices& edge,
                                   const GaussRule1D& rule, Function&& function,
                                   PetscReal& integral)
{
    PetscFunctionBeginUser;
    PetscReal length = 0.0;
    Point normal{};
    PetscCall(GetEdgeGeometry(edge, length, normal));
    const auto normalComponent = [&](const Point& point) -> PetscReal {
        const Point value = function(point);
        return value.p[0] * normal.p[0] + value.p[1] * normal.p[1];
    };
    PetscCall(IntegrateEdge(edge, rule, normalComponent, integral));
    PetscFunctionReturn(PETSC_SUCCESS);
}

#endif

