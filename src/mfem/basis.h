#ifndef MANTLE_MFEM_BASIS_H
#define MANTLE_MFEM_BASIS_H

#include "mesh_info.h"

#include <array>
#include <cstddef>

// A scalar function and its PHYSICAL gradient: (d/dx, d/dy).
// Distances and projections have physical length units; the other helpers are
// dimensionless. Their gradients carry the corresponding inverse-length factor.
struct ScalarBasisValue {
    PetscReal value = 0;
    Point gradient{};
};

struct BasisEdgeGeometry {
    EdgeVertices vertices{};
    PetscReal length = 0;
    Point midpoint{};
    Point tangent{}; // From vertices[0] to vertices[1].
    Point normal{};  // Right normal (tangent.y, -tangent.x).
};

// Shared physical-quadrilateral geometry and scalar helpers for BR and H(div).
// All methods are LOCAL, with no MPI communication; use after PetscInitialize.
// Owns its geometry and retains no MeshInfo, DM, Vec, or external array pointers.
// Reinitialize after mesh coordinates change. Const evaluation does not mutate
// cached state. Copying this small object gives an independent geometry snapshot.
//
// Corners: v0=(i,j), v1=(i+1,j), v2=(i+1,j+1), v3=(i,j+1), CCW.
// Sides follow CellSide in core/mesh_info.h: Bottom=0, Right=1, Top=2, Left=3.
// Edge e is v[e] -> v[(e+1)%4]; its normal is outward.
// Legacy MFEM used Left,Bottom,Right,Top: old_edge=(new_edge+1)%4.
// No global edge signs or global DOF numbering are applied in this class.
//
// Initialization checks finite, convex, nondegenerate CCW geometry with the
// existing ValidateQuad(). Invalid reinitialization preserves the previous state.
// All checked output arguments are unchanged on failure, including in Release.
// Queries accept finite physical points, including exterior points wherever the
// requested expression is numerically nonsingular and the value/gradient
// calculations remain representable.
class QuadBasis {
public:
    QuadBasis() = default;

    PetscErrorCode Initialize(const QuadVertices& corners);
    // Requires all four cell corners to be available in this local snapshot.
    PetscErrorCode Initialize(const MeshInfo& mesh, MeshIndex cell);
    bool IsInitialized() const noexcept { return initialized_; }

    PetscErrorCode GetCorners(QuadVertices& corners) const;
    PetscErrorCode GetEdge(CellSide side, BasisEdgeGeometry& edge) const;
    // Diagonal 0: v0 -> v2. Diagonal 1: v1 -> v3.
    // A diagonal's right normal fixes its signed-distance convention.
    PetscErrorCode GetDiagonal(PetscInt diagonal, BasisEdgeGeometry& edge) const;

    // lambda_e(x)=-(x-v[e]).n_e, positive inside the cell; grad=-n_e.
    PetscErrorCode EdgeDistance(CellSide side, const Point& point,
                                 ScalarBasisValue& result) const;
    // The same signed-distance formula on a directed diagonal.
    PetscErrorCode DiagonalDistance(PetscInt diagonal, const Point& point,
                                     ScalarBasisValue& result) const;
    // s_e(x)=(x-v[e]).t_e; grad=t_e. This is physical arclength, not [-1,1].
    PetscErrorCode EdgeProjection(CellSide side, const Point& point,
                                   ScalarBasisValue& result) const;

    // Require first and second to be opposite sides.
    // (lambda_first-lambda_second)/|v[first+1]-v[second+1]|, indices modulo 4.
    // The END-corner diagonal preserves the old two-edge lambda convention
    // after converting from the legacy edge numbering.
    PetscErrorCode ScaledDistanceDifference(CellSide first, CellSide second,
                                             const Point& point,
                                             ScalarBasisValue& result) const;

    // R_e=lambda_opposite/(lambda_e+lambda_opposite).
    // One on edge e and zero on its opposite; grad is evaluated analytically.
    PetscErrorCode EdgeBlend(CellSide side, const Point& point,
                              ScalarBasisValue& result) const;
    // (lambda_first-lambda_second)/(lambda_first+lambda_second).
    // Opposite sides only: -1 on first, +1 on second. Completes the old R/dR pair.
    PetscErrorCode OppositeEdgeContrast(CellSide first, CellSide second,
                                         const Point& point,
                                         ScalarBasisValue& result) const;

    // Sum_i (-1)^i [lambda_(i+1)(x)/lambda_(i+1)(vi)]
    //                 [lambda_(i+2)(x)/lambda_(i+2)(vi)].
    // This is a quadratic POLYNOMIAL, with values (+1,-1,+1,-1) at the corners.
    // It replaces the misleadingly named legacy rational()/dRational().
    PetscErrorCode AlternatingVertexPolynomial(const Point& point,
                                                ScalarBasisValue& result) const;

    // b_e=lambda_(e+1)*lambda_(e+3)*R_e, normalized by b_e at the edge midpoint.
    // One at its own midpoint, zero on all other edges and at every corner.
    // Shared by BR edge bubbles and the DS2 scalar potentials used by H(div).
    PetscErrorCode EdgeBubble(CellSide side, const Point& point,
                               ScalarBasisValue& result) const;

private:
    PetscErrorCode CheckSide(CellSide side, std::size_t& index) const;
    PetscErrorCode CheckOppositeSides(CellSide first, CellSide second,
                                      std::size_t& index) const;

    bool initialized_ = false;
    QuadVertices corners_{};
    std::array<BasisEdgeGeometry, 4> edges_{};
    std::array<BasisEdgeGeometry, 2> diagonals_{};
    // Store positive distances separately: multiplying two tiny/large lengths
    // would unnecessarily underflow/overflow on otherwise valid cells.
    std::array<std::array<PetscReal, 2>, 4> vertexDistances_{};
    std::array<std::array<PetscReal, 2>, 4> bubbleDistances_{};
    std::array<PetscReal, 4> bubbleBlends_{};
};

#endif

