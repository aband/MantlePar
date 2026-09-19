#ifndef MANTLE_CORE_QUAD_VALIDATION_H
#define MANTLE_CORE_QUAD_VALIDATION_H

#include "mesh.h"

#include <algorithm>
#include <array>

namespace mantle::detail {

enum class QuadValidity { Valid, NonfiniteCoordinates, InvalidExtents, InvalidJacobian };

// Internal, local check; callers handle local or collective error reporting.
inline QuadValidity CheckQuadGeometry(const std::array<Point, 4>& corners)
{
    PetscReal xmin = corners[0].p[0], xmax = xmin;
    PetscReal ymin = corners[0].p[1], ymax = ymin;
    for (const Point& point : corners) {
        if (PetscIsInfOrNanReal(point.p[0]) || PetscIsInfOrNanReal(point.p[1]))
            return QuadValidity::NonfiniteCoordinates;
        xmin = std::min(xmin, point.p[0]);
        xmax = std::max(xmax, point.p[0]);
        ymin = std::min(ymin, point.p[1]);
        ymax = std::max(ymax, point.p[1]);
    }

    const PetscReal sx = xmax - xmin, sy = ymax - ymin;
    if (PetscIsInfOrNanReal(sx) || PetscIsInfOrNanReal(sy) || sx <= 0.0 || sy <= 0.0)
        return QuadValidity::InvalidExtents;

    // The Q1 Jacobian determinant is affine, so check its four corners.
    // Scale each axis separately to allow thin rectangles. These determinants
    // omit the positive sx*sy/4 factor for reference coordinates in [-1,1].
    constexpr PetscReal tolerance = 64.0 * PETSC_MACHINE_EPSILON;
    for (std::size_t k = 0; k < corners.size(); ++k) {
        const Point& point = corners[k];
        const Point& next = corners[(k + 1) % 4];
        const Point& prev = corners[(k + 3) % 4];
        const PetscReal ux = (next.p[0] - point.p[0]) / sx;
        const PetscReal uy = (next.p[1] - point.p[1]) / sy;
        const PetscReal vx = (prev.p[0] - point.p[0]) / sx;
        const PetscReal vy = (prev.p[1] - point.p[1]) / sy;
        if (!(ux * vy - uy * vx > tolerance)) return QuadValidity::InvalidJacobian;
    }
    return QuadValidity::Valid;
}

} // namespace mantle::detail

#endif
