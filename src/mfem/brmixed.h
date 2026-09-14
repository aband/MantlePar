#ifndef MANTLE_MFEM_BRMIXED_H
#define MANTLE_MFEM_BRMIXED_H

#include "basis.h"

#include <array>

// A real vector basis function (u,v) and its PHYSICAL gradient, in row order:
// [du/dx, du/dy, dv/dx, dv/dy]. Its divergence is gradient[0]+gradient[3].
struct BRBasisValue {
    Point value{};
    std::array<PetscReal, 4> gradient{};
};

// Enriched DS1 H1-conforming velocity basis on a convex physical quadrilateral.
//
// Local DOFs:
//   0..3  : phi_i * (1,0), vertices v0,v1,v2,v3 in QuadBasis order.
//   4..7  : phi_i * (0,1), in the same vertex order.
//   8..11 : b_e * nBR_e, sides Bottom,Right,Top,Left (CellSide order).
// New DOFs 8,9,10,11 correspond to legacy DOFs 9,10,11,8.
//
// b_e is normalized to one at its own edge midpoint. For a represented
// velocity u, its edge coefficient is
//   nBR_e . [u(midpoint) - (u(v_e)+u(v_(e+1)))/2].
//
// Preserve the legacy shared-edge normal convention:
//   nBR_e = {-1,+1,+1,-1}[e] * outward_normal_e.
// For logical AlongI edges nBR is the LEFT normal of the canonical tangent;
// for AlongJ edges it is the RIGHT normal. On an axis-aligned rectangle these
// point up and right, respectively. Adjacent cells use the same nBR on their
// shared edge. Values and gradients already include these signs; later global
// assembly should use them directly. Outward normals for flux integrals still
// come from QuadBasis/MeshInfo.
//
// All operations are LOCAL; call after PetscInitialize. Initialize from a
// MeshInfo snapshot whose four cell corners are available on this rank, from
// four corners, or from an existing QuadBasis. Owns all geometry and constants;
// no DM, Vec, MeshInfo or QuadBasis pointers are retained. Reinitialize after a
// geometry change. Copies are independent; const evaluation changes no state.
// Failed initialization preserves the previous basis, and checked outputs are
// assigned only on success. No global numbering or boundary handling here.
class BRMixed {
public:
    static constexpr PetscInt ElementDofs = 12;
    static constexpr PetscInt PressureDofs = 1;
    using Values = std::array<BRBasisValue, ElementDofs>;

    BRMixed() = default;
    static constexpr const char* Name() noexcept { return "BR"; }
    // The constant P0 pressure basis on this cell; independent of geometry.
    static constexpr PetscReal Pressure() noexcept { return PetscReal(1); }

    PetscErrorCode Initialize(const QuadVertices& corners);
    PetscErrorCode Initialize(const MeshInfo& mesh, MeshIndex cell);
    PetscErrorCode Initialize(const QuadBasis& geometry);
    bool IsInitialized() const noexcept { return initialized_; }

    PetscErrorCode GetCorners(QuadVertices& corners) const;
    // Returns the consistently oriented nBR, including the sign above.
    PetscErrorCode GetEdgeNormal(CellSide side, Point& normal) const;

    // Scalar DS1 vertex function and normalized scalar edge bubble. The bubble
    // returned here has no normal/sign factor; vector Evaluate adds nBR.
    PetscErrorCode EvaluateVertex(PetscInt vertex, const Point& point,
                                   ScalarBasisValue& result) const;
    PetscErrorCode EvaluateBubble(CellSide side, const Point& point,
                                   ScalarBasisValue& result) const;
    // S = Q - sum_e Q(midpoint_e)*b_e; Q is AlternatingVertexPolynomial.
    // This is the corrected supplement used in the legacy vertex construction.
    PetscErrorCode EvaluateSupplement(const Point& point,
                                       ScalarBasisValue& result) const;

    // A single vector basis function, or all 12, with gradients in the same
    // result. Both interfaces share the scalar formulas and normal factors.
    PetscErrorCode Evaluate(const Point& point, PetscInt localDof,
                             BRBasisValue& result) const;
    PetscErrorCode EvaluateAll(const Point& point, Values& result) const;

private:
    PetscErrorCode CheckInitialized() const;
    PetscErrorCode EvaluateScalars(const Point& point,
                                    std::array<ScalarBasisValue, 4>& bubbles,
                                    ScalarBasisValue& supplement) const;
    PetscErrorCode MakeVertex(PetscInt vertex, const ScalarBasisValue& diagonal,
                               const ScalarBasisValue& supplement,
                               ScalarBasisValue& result) const;

    bool initialized_ = false;
    QuadBasis geometry_{};
    std::array<Point, 4> edgeNormals_{};
    std::array<PetscReal, 4> midpointPolynomial_{};
    // For vertex i, d_i(x) is distance to diagonal (i+1)%2. Store
    // D_i = d_i(v_i)-d_i(v_(i+2)) as scale_i*difference_i, with
    // 1 <= |difference_i| <= 2. Evaluate by dividing by these factors
    // separately, without forming D_i explicitly.
    std::array<PetscReal, 4> diagonalScale_{};
    std::array<PetscReal, 4> diagonalDifference_{};
    // weight_i = d_i(v_(i+2))/(2*D_i), a bounded dimensionless number.
    std::array<PetscReal, 4> oppositeWeight_{};
};

#endif
