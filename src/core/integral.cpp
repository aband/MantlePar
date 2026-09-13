#include "integral.h"

#include <petscdt.h>
#include <algorithm>

PetscErrorCode CreateGaussRule(PetscInt npoints, GaussRule1D& rule)
{
    PetscFunctionBeginUser;
    PetscCheck(npoints >= 1, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE,
               "Gauss quadrature requires at least one point per direction");
    GaussRule1D generated;
    generated.points.resize(static_cast<std::size_t>(npoints));
    generated.weights.resize(static_cast<std::size_t>(npoints));
    PetscCall(PetscDTGaussQuadrature(npoints, -1.0, 1.0,
                                     generated.points.data(), generated.weights.data()));
    PetscCall(ValidateGaussRule(generated));
    rule = std::move(generated);
    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode ValidateGaussRule(const GaussRule1D& rule)
{
    PetscFunctionBeginUser;
    PetscCheck(!rule.points.empty() && rule.points.size() == rule.weights.size(),
               PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG,
               "Quadrature points and weights must have the same nonzero size");
    PetscReal sum = 0.0;
    for (std::size_t i = 0; i < rule.points.size(); ++i) {
        const PetscReal point = rule.points[i], weight = rule.weights[i];
        PetscCheck(!PetscIsInfOrNanReal(point) && point >= -1.0 && point <= 1.0 &&
                   !PetscIsInfOrNanReal(weight) && weight > 0.0,
                   PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE,
                   "Quadrature requires finite points in [-1,1] and positive finite weights");
        sum += weight;
    }
    const PetscReal tolerance = 64.0 * PETSC_MACHINE_EPSILON
                             * static_cast<PetscReal>(rule.points.size());
    PetscCheck(!PetscIsInfOrNanReal(sum) && PetscAbsReal(sum - 2.0) <= tolerance,
               PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG,
               "Reference quadrature weights must sum to 2");
    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode ValidateQuad(const QuadVertices& corners)
{
    PetscFunctionBeginUser;
    PetscReal xmin = corners[0].p[0], xmax = xmin;
    PetscReal ymin = corners[0].p[1], ymax = ymin;
    for (const Point& point : corners) {
        PetscCheck(!PetscIsInfOrNanReal(point.p[0]) &&
                   !PetscIsInfOrNanReal(point.p[1]),
                   PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE,
                   "Quadrilateral coordinates must be finite");
        xmin = std::min(xmin, point.p[0]);
        xmax = std::max(xmax, point.p[0]);
        ymin = std::min(ymin, point.p[1]);
        ymax = std::max(ymax, point.p[1]);
    }
    const PetscReal sx = xmax - xmin, sy = ymax - ymin;
    PetscCheck(!PetscIsInfOrNanReal(sx) && !PetscIsInfOrNanReal(sy) &&
               sx > 0.0 && sy > 0.0,
               PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE,
               "Quadrilateral extents must be finite and positive");

    // det(J) is affine on the reference square: check its four corners.
    // Separate axis scaling matches mesh.cpp and permits thin rectangles.
    const PetscReal tolerance = 64.0 * PETSC_MACHINE_EPSILON;
    for (std::size_t k = 0; k < corners.size(); ++k) {
        const Point& point = corners[k];
        const Point& next = corners[(k + 1) % 4];
        const Point& prev = corners[(k + 3) % 4];
        const PetscReal ux = (next.p[0] - point.p[0]) / sx;
        const PetscReal uy = (next.p[1] - point.p[1]) / sy;
        const PetscReal vx = (prev.p[0] - point.p[0]) / sx;
        const PetscReal vy = (prev.p[1] - point.p[1]) / sy;
        PetscCheck(ux * vy - uy * vx > tolerance,
                   PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG,
                   "Quadrilateral must be counterclockwise, convex and nonsingular");
    }
    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode GetEdgeGeometry(const EdgeVertices& edge, PetscReal& length,
                               Point& normal)
{
    PetscFunctionBeginUser;
    for (const Point& point : edge) {
        PetscCheck(!PetscIsInfOrNanReal(point.p[0]) &&
                   !PetscIsInfOrNanReal(point.p[1]),
                   PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE,
                   "Edge coordinates must be finite");
    }
    const PetscReal dx = edge[1].p[0] - edge[0].p[0];
    const PetscReal dy = edge[1].p[1] - edge[0].p[1];
    PetscCheck(!PetscIsInfOrNanReal(dx) && !PetscIsInfOrNanReal(dy),
               PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE,
               "Edge displacement must be finite");
    const PetscReal scale = std::max(PetscAbsReal(dx), PetscAbsReal(dy));
    PetscCheck(scale > 0.0, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE,
               "Edge endpoints must be distinct");
    const PetscReal ux = dx / scale, uy = dy / scale;
    const PetscReal norm = PetscSqrtReal(ux * ux + uy * uy);
    const PetscReal edgeLength = scale * norm;
    PetscCheck(!PetscIsInfOrNanReal(edgeLength) && edgeLength > 0.0,
               PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE,
               "Edge length must be finite and positive");
    length = edgeLength;
    normal = Point{{uy / norm, -ux / norm}};
    PetscFunctionReturn(PETSC_SUCCESS);
}

Point MapEdgePoint(PetscReal reference, const EdgeVertices& edge)
{
    const PetscReal a = 0.5 * (1.0 - reference);
    const PetscReal b = 0.5 * (1.0 + reference);
    return Point{{a * edge[0].p[0] + b * edge[1].p[0],
                  a * edge[0].p[1] + b * edge[1].p[1]}};
}

Point MapCellPoint(const Point& reference, const QuadVertices& corners)
{
    const EdgeVertices lower{{corners[0], corners[1]}};
    const EdgeVertices upper{{corners[3], corners[2]}};
    const EdgeVertices vertical{{MapEdgePoint(reference.p[0], lower),
                                 MapEdgePoint(reference.p[0], upper)}};
    return MapEdgePoint(reference.p[1], vertical);
}

PetscReal CellJacobian(const Point& reference, const QuadVertices& corners)
{
    const PetscReal s = 0.5 * (1.0 + reference.p[0]);
    const PetscReal t = 0.5 * (1.0 + reference.p[1]);
    Point dxi{}, deta{};
    for (int d = 0; d < 2; ++d) {
        // Use differences so a large translation does not enter the derivatives.
        dxi.p[d] = 0.5 * ((1.0 - t) * (corners[1].p[d] - corners[0].p[d])
                         + t * (corners[2].p[d] - corners[3].p[d]));
        deta.p[d] = 0.5 * ((1.0 - s) * (corners[3].p[d] - corners[0].p[d])
                          + s * (corners[2].p[d] - corners[1].p[d]));
    }
    return dxi.p[0] * deta.p[1] - dxi.p[1] * deta.p[0];
}
