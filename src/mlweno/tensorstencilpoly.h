#ifndef MANTLE_MLWENO_TENSORSTENCILPOLY_H
#define MANTLE_MLWENO_TENSORSTENCILPOLY_H

#include "mesh_info.h"

#include <map>
#include <vector>

// One cell-average polynomial on a logical rectangular stencil.
// Polynomial space: Q_(size.i-1,size.j-1) in PHYSICAL x,y, scaled by h.
// order is the maximum TOTAL derivative order in the smoothness indicator.
//
// All operations are local; use after PetscInitialize. This class owns its
// geometry and coefficient arrays. It retains no MeshInfo, DM, Vec or solution
// pointers. Copies are independent; moved-from objects are uninitialized.
// Rebuild after changing the geometry. Failed setup preserves the old object,
// and checked output arguments are assigned only on success.
//
// Cell averages and basis indices use k = j*size.i+i, with i varying fastest.
// Callers gather these averages from owned/ghost cells before evaluation.
// Coordinates must be finite convex CCW quads in core/mesh.h's corner order.
// LAPACKE_dgesv requires the usual double-precision PetscReal build.
class TensorStencilPoly {
public:
    TensorStencilPoly() = default;
    TensorStencilPoly(const TensorStencilPoly&) = default;
    TensorStencilPoly& operator=(const TensorStencilPoly&) = default;
    TensorStencilPoly(TensorStencilPoly&& other) noexcept;
    TensorStencilPoly& operator=(TensorStencilPoly&& other) noexcept;

    // h is explicit: sqrt(L*H/(Nx*Ny)) reproduces the legacy scale.
    // Every stencil cell must be physical AND available in mesh's local cache.
    PetscErrorCode Initialize(const MeshInfo& mesh, MeshIndex start,
                              MeshIndex size, PetscInt order, PetscReal h);
    // Geometry-only entry point. corners[k] has the same ordering as averages.
    // start records the global logical origin; no global mesh is needed.
    PetscErrorCode Initialize(const std::vector<QuadVertices>& corners,
                              MeshIndex start, MeshIndex size,
                              PetscInt order, PetscReal h);

    bool IsInitialized() const noexcept { return !coefficients_.empty(); }
    MeshIndex Start() const noexcept { return start_; }
    MeshIndex Size() const noexcept { return size_; }
    PetscInt CellCount() const noexcept { return count_; }
    PetscInt Order() const noexcept { return order_; }
    Point Center() const noexcept { return center_; }
    PetscReal Scale() const noexcept { return h_; }

    // Basis phi_k satisfies average_(cell j)(phi_k) = delta_jk.
    PetscErrorCode EvaluateBasis(PetscInt cell, const Point& point,
                                 PetscReal& value) const;
    PetscErrorCode EvaluateBasisDerivative(PetscInt cell, const Point& point,
                                           PetscInt dx, PetscInt dy,
                                           PetscReal& value) const;
    PetscErrorCode Evaluate(const std::vector<PetscReal>& averages,
                            const Point& point, PetscReal& value) const;
    // Returns a PHYSICAL derivative d_x^dx d_y^dy p.
    PetscErrorCode EvaluateDerivative(const std::vector<PetscReal>& averages,
                                      const Point& point, PetscInt dx, PetscInt dy,
                                      PetscReal& value) const;

    // Default indicator: sum_(1<=dx+dy<=order) h^(2*(dx+dy))
    //   * integral_R (d_x^dx d_y^dy p)^2 / area(R),
    // where R is the square of side h centered at Center().
    PetscErrorCode Smoothness(const std::vector<PetscReal>& averages,
                              PetscReal& value) const;

    // Return the same indicator and gradient[k] = d(value)/d(averages[k]).
    // This is a solution gradient for implicit Jacobians, not a spatial
    // derivative. Geometry, h and order are held fixed. The gradient uses
    // the same cell ordering as averages and includes the constant-centering
    // chain rule. Constant data gives exactly zero value and gradient.
    // If a nonpositive quadratic form is clipped to zero at roundoff level,
    // its returned gradient is zero. Outputs are unchanged on error.
    PetscErrorCode SmoothnessWithGradient(const std::vector<PetscReal>& averages,
                                          PetscReal& value,
                                          std::vector<PetscReal>& gradient) const;

    // Cache the same indicator integrated over one physical stencil cell.
    // localCell is an OFFSET within the stencil, not a global cell index.
    // Repeated calls replace that entry. Arbitrary subsets are supported.
    PetscErrorCode SetTargetSmoothness(MeshIndex localCell);
    // An uncached target returns an error; it never inserts an empty entry.
    PetscErrorCode Smoothness(const std::vector<PetscReal>& averages,
                              MeshIndex localCell, PetscReal& value) const;
    PetscErrorCode SmoothnessWithGradient(const std::vector<PetscReal>& averages,
                                          MeshIndex localCell, PetscReal& value,
                                          std::vector<PetscReal>& gradient) const;

private:
    void Swap(TensorStencilPoly& other) noexcept;
    PetscErrorCode CheckAverages(const std::vector<PetscReal>& averages) const;
    PetscErrorCode TargetIndex(MeshIndex localCell, PetscInt& index) const;
    PetscErrorCode PolynomialValue(const PetscReal* coefficients,
                                    const Point& point, PetscInt dx, PetscInt dy,
                                    PetscReal& value) const;
    PetscErrorCode BuildSmoothness(const QuadVertices& normalizedCorners,
                                   PetscInt gaussPoints,
                                   std::vector<PetscReal>& matrix) const;
    PetscErrorCode ApplySmoothness(const std::vector<PetscReal>& matrix,
                                   const std::vector<PetscReal>& averages,
                                   PetscReal& value,
                                   std::vector<PetscReal>* gradient = nullptr) const;

    MeshIndex start_{}, size_{};
    PetscInt count_ = 0, order_ = 0;
    Point center_{};
    PetscReal h_ = 0;
    std::vector<QuadVertices> corners_;
    // Basis-major storage: coefficients_[cell*count_ + j*size_.i+i].
    std::vector<PetscReal> coefficients_, sigma_;
    std::map<PetscInt, std::vector<PetscReal>> targetSigma_;
};

#endif
