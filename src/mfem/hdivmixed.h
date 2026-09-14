#ifndef MANTLE_MFEM_HDIVMIXED_H
#define MANTLE_MFEM_HDIVMIXED_H

#include "basis.h"

#include <array>
#include <cstddef>

// A physical vector basis function and its analytic PHYSICAL divergence.
struct HDivBasisValue {
    Point value{};
    PetscReal divergence = 0;
};

// Eight-DOF, direct-serendipity-based H(div) element on a convex quadrilateral.
// Preserves the legacy BDM-style space and normalization in physical coordinates.
//
// Sides: Bottom=0, Right=1, Top=2, Left=3 (CellSide / QuadBasis convention).
// Local DOFs 0..3 are LINEAR-normal-trace modes; 4..7 are CONSTANT-trace modes.
// New DOFs 0,1,2,3,4,5,6,7 correspond to legacy DOFs 1,2,3,0,5,6,7,4.
// Each mode has zero normal trace on the other three edges.
// Pressure is one cellwise constant P0 function.
//
// Let s run from 0 to 1 along the cell's CCW edge e, with outward normal n_e.
// Linear mode:   u_e . n_e = 1-2*s,       div(u_e) = 0.
// Constant mode: u_(e+4) . n_e = sigma_e, div(u_(e+4)) = sigma_e*|e|/|K|.
// Here sigma = {-1,+1,+1,-1}, and the shared normal is nShared=sigma_e*n_e.
// Thus the constant mode has unit trace with respect to nShared. Linear modes
// already account for edge-parameter reversal; DO NOT multiply them by sigma.
// On a canonical edge parameter increasing in i or j, their nShared traces
// are 2*s-1 on AlongI edges and 1-2*s on AlongJ edges, as in the legacy code.
// Values and divergences already contain these conventions: do not apply a
// second global orientation sign. Use QuadBasis/MeshInfo for outward normals.
// H(div) continuity concerns the normal component; tangential jumps are allowed.
//
// All operations are LOCAL; use after PetscInitialize. Owns a geometry snapshot
// and retains no MeshInfo, DM, Vec or external geometry pointers. Reinitialize
// after moving mesh vertices. Copies are independent; const evaluation changes
// no state. Invalid initialization preserves the previous state. Checked outputs
// are assigned only on success, including in Release builds. Finite exterior
// points are accepted where the required scalar expressions are nonsingular.
class HDivMixed {
public:
    static constexpr PetscInt ElementDofs = 8;
    static constexpr PetscInt PressureDofs = 1;
    using Values = std::array<HDivBasisValue, ElementDofs>;
    using EdgeValues = std::array<HDivBasisValue, 2>; // Linear, then constant.

    HDivMixed() = default;
    static constexpr const char* Name() noexcept { return "BDM"; }
    static constexpr PetscReal Pressure() noexcept { return PetscReal(1); }

    PetscErrorCode Initialize(const QuadVertices& corners);
    PetscErrorCode Initialize(const MeshInfo& mesh, MeshIndex cell);
    PetscErrorCode Initialize(const QuadBasis& geometry);
    bool IsInitialized() const noexcept { return initialized_; }

    PetscErrorCode GetCorners(QuadVertices& corners) const;
    // Returns nShared, which agrees on both sides of a logical interior edge.
    PetscErrorCode GetEdgeNormal(CellSide side, Point& normal) const;

    PetscErrorCode Evaluate(const Point& point, PetscInt localDof,
                             HDivBasisValue& result) const;
    PetscErrorCode EvaluateAll(const Point& point, Values& result) const;
    // Evaluates the two modes associated with side, at any admissible point.
    PetscErrorCode EvaluateEdge(const Point& point, CellSide side,
                                 EdgeValues& result) const;

private:
    struct EdgeData {
        PetscReal length = 0;
        Point normal{}, oppositeVertex{};
        std::array<PetscReal, 2> potentialDistances{};
        std::array<PetscReal, 2> potentialWeights{};
        PetscReal inverseDenominator = 0;
        PetscReal curlWeight = 0;
    };

    PetscErrorCode CheckInitialized() const;
    PetscErrorCode CheckSide(CellSide side, std::size_t& edge) const;
    PetscErrorCode MakeLinear(std::size_t edge, const ScalarBasisValue& bubble,
                               HDivBasisValue& result) const;
    PetscErrorCode MakeConstant(std::size_t edge, const Point& point,
                                 const std::array<ScalarBasisValue, 2>& distances,
                                 const std::array<ScalarBasisValue, 2>& bubbles,
                                 HDivBasisValue& result) const;

    bool initialized_ = false;
    QuadBasis geometry_{};
    std::array<EdgeData, 4> edges_{};
};

#endif
