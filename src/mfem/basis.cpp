#include "basis.h"

#include <algorithm>

namespace {

PetscErrorCode CheckPoint(const Point& point)
{
    PetscFunctionBeginUser;
    PetscCheck(!PetscIsInfOrNanReal(point.p[0]) && !PetscIsInfOrNanReal(point.p[1]),
               PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE,
               "Basis evaluation requires finite physical coordinates");
    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode StoreResult(const ScalarBasisValue& value, ScalarBasisValue& result)
{
    PetscFunctionBeginUser;
    PetscCheck(!PetscIsInfOrNanReal(value.value) &&
               !PetscIsInfOrNanReal(value.gradient.p[0]) &&
               !PetscIsInfOrNanReal(value.gradient.p[1]),
               PETSC_COMM_SELF, PETSC_ERR_FP,
               "Basis value or physical gradient is not finite");
    result = value;
    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode MakeSegment(const Point& a, const Point& b, BasisEdgeGeometry& edge)
{
    PetscFunctionBeginUser;
    BasisEdgeGeometry work;
    work.vertices = {{a, b}};
    PetscCall(::GetEdgeGeometry(work.vertices, work.length, work.normal));
    work.tangent = Point{{-work.normal.p[1], work.normal.p[0]}};
    for (int d = 0; d < 2; ++d)
        work.midpoint.p[d] = a.p[d] + PetscReal(0.5) * (b.p[d] - a.p[d]);
    PetscCall(CheckPoint(work.midpoint));
    edge = work;
    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode LinearEvaluation(const BasisEdgeGeometry& edge, const Point& point,
                                const Point& gradient, ScalarBasisValue& result)
{
    PetscFunctionBeginUser;
    PetscCall(CheckPoint(point));
    const PetscReal dx = point.p[0] - edge.vertices[0].p[0];
    const PetscReal dy = point.p[1] - edge.vertices[0].p[1];
    PetscCheck(!PetscIsInfOrNanReal(dx) && !PetscIsInfOrNanReal(dy),
               PETSC_COMM_SELF, PETSC_ERR_FP,
               "Basis evaluation displacement is not representable");
    const ScalarBasisValue work{dx * gradient.p[0] + dy * gradient.p[1], gradient};
    PetscCall(StoreResult(work, result));
    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode Distance(const BasisEdgeGeometry& edge, const Point& point,
                        ScalarBasisValue& result)
{
    PetscFunctionBeginUser;
    const Point gradient{{-edge.normal.p[0], -edge.normal.p[1]}};
    PetscCall(LinearEvaluation(edge, point, gradient, result));
    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode PositiveDistance(const BasisEdgeGeometry& edge, const Point& point,
                                PetscReal& distance)
{
    PetscFunctionBeginUser;
    ScalarBasisValue work;
    PetscCall(Distance(edge, point, work));
    PetscCheck(work.value > 0, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG,
               "Quadrilateral has a nonpositive basis normalization distance");
    distance = work.value;
    PetscFunctionReturn(PETSC_SUCCESS);
}

ScalarBasisValue Divide(const ScalarBasisValue& value, PetscReal divisor)
{
    return {value.value / divisor,
            Point{{value.gradient.p[0] / divisor, value.gradient.p[1] / divisor}}};
}

ScalarBasisValue Product(const ScalarBasisValue& a, const ScalarBasisValue& b)
{
    return {a.value * b.value,
            Point{{a.gradient.p[0] * b.value + a.value * b.gradient.p[0],
                   a.gradient.p[1] * b.value + a.value * b.gradient.p[1]}}};
}

} // namespace

PetscErrorCode QuadBasis::Initialize(const QuadVertices& corners)
{
    PetscFunctionBeginUser;
    PetscCall(ValidateQuad(corners));
    QuadBasis work;
    work.corners_ = corners;
    for (std::size_t e = 0; e < 4; ++e)
        PetscCall(MakeSegment(corners[e], corners[(e + 1) % 4], work.edges_[e]));
    for (std::size_t d = 0; d < 2; ++d)
        PetscCall(MakeSegment(corners[d], corners[d + 2], work.diagonals_[d]));

    // Only this temporary object is marked ready while its scalar constants are
    // filled. No partially initialized state is published to the caller.
    work.initialized_ = true;
    for (std::size_t i = 0; i < 4; ++i) {
        PetscCall(PositiveDistance(work.edges_[(i + 1) % 4], corners[i],
                                    work.vertexDistances_[i][0]));
        PetscCall(PositiveDistance(work.edges_[(i + 2) % 4], corners[i],
                                    work.vertexDistances_[i][1]));
        const Point midpoint = work.edges_[i].midpoint;
        PetscCall(PositiveDistance(work.edges_[(i + 1) % 4], midpoint,
                                    work.bubbleDistances_[i][0]));
        PetscCall(PositiveDistance(work.edges_[(i + 3) % 4], midpoint,
                                    work.bubbleDistances_[i][1]));
        ScalarBasisValue blend;
        PetscCall(work.EdgeBlend(static_cast<CellSide>(i), midpoint, blend));
        PetscCheck(blend.value > 0, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG,
                   "Quadrilateral has an invalid midpoint blending value");
        work.bubbleBlends_[i] = blend.value;
    }
    *this = work;
    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode QuadBasis::Initialize(const MeshInfo& mesh, MeshIndex cell)
{
    PetscFunctionBeginUser;
    PetscCheck(mesh.IsInitialized(), PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG,
               "BuildMeshInfo must be called before initializing a cell basis");
    QuadVertices corners;
    PetscCall(mesh.GetCellCorners(cell, corners));
    PetscCall(Initialize(corners));
    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode QuadBasis::CheckSide(CellSide side, std::size_t& index) const
{
    PetscFunctionBeginUser;
    PetscCheck(initialized_, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG,
               "Initialize the quadrilateral basis before querying it");
    const int e = static_cast<int>(side);
    PetscCheck(e >= 0 && e < 4, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE,
               "CellSide must be Bottom, Right, Top, or Left");
    index = static_cast<std::size_t>(e);
    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode QuadBasis::CheckOppositeSides(CellSide first, CellSide second,
                                           std::size_t& index) const
{
    PetscFunctionBeginUser;
    std::size_t opposite;
    PetscCall(CheckSide(first, index));
    PetscCall(CheckSide(second, opposite));
    PetscCheck(opposite == (index + 2) % 4, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG,
               "The two sides must be opposite sides of the cell");
    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode QuadBasis::GetCorners(QuadVertices& corners) const
{
    PetscFunctionBeginUser;
    PetscCheck(initialized_, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG,
               "Initialize the quadrilateral basis before querying it");
    corners = corners_;
    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode QuadBasis::GetEdge(CellSide side, BasisEdgeGeometry& edge) const
{
    PetscFunctionBeginUser;
    std::size_t e;
    PetscCall(CheckSide(side, e));
    edge = edges_[e];
    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode QuadBasis::GetDiagonal(PetscInt diagonal, BasisEdgeGeometry& edge) const
{
    PetscFunctionBeginUser;
    PetscCheck(initialized_, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG,
               "Initialize the quadrilateral basis before querying it");
    PetscCheck(diagonal == 0 || diagonal == 1, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE,
               "Diagonal index must be 0 (v0 to v2) or 1 (v1 to v3)");
    edge = diagonals_[static_cast<std::size_t>(diagonal)];
    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode QuadBasis::EdgeDistance(CellSide side, const Point& point,
                                      ScalarBasisValue& result) const
{
    PetscFunctionBeginUser;
    std::size_t e;
    PetscCall(CheckSide(side, e));
    PetscCall(Distance(edges_[e], point, result));
    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode QuadBasis::DiagonalDistance(PetscInt diagonal, const Point& point,
                                          ScalarBasisValue& result) const
{
    PetscFunctionBeginUser;
    BasisEdgeGeometry edge;
    PetscCall(GetDiagonal(diagonal, edge));
    PetscCall(Distance(edge, point, result));
    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode QuadBasis::EdgeProjection(CellSide side, const Point& point,
                                        ScalarBasisValue& result) const
{
    PetscFunctionBeginUser;
    std::size_t e;
    PetscCall(CheckSide(side, e));
    PetscCall(LinearEvaluation(edges_[e], point, edges_[e].tangent, result));
    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode QuadBasis::ScaledDistanceDifference(CellSide first, CellSide second,
                                                  const Point& point,
                                                  ScalarBasisValue& result) const
{
    PetscFunctionBeginUser;
    std::size_t e;
    PetscCall(CheckOppositeSides(first, second, e));
    ScalarBasisValue a, b;
    PetscCall(Distance(edges_[e], point, a));
    PetscCall(Distance(edges_[(e + 2) % 4], point, b));
    const PetscReal length = diagonals_[(e + 1) % 2].length;
    a = Divide(a, length);
    b = Divide(b, length);
    const ScalarBasisValue work{a.value - b.value,
        Point{{a.gradient.p[0] - b.gradient.p[0], a.gradient.p[1] - b.gradient.p[1]}}};
    PetscCall(StoreResult(work, result));
    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode QuadBasis::EdgeBlend(CellSide side, const Point& point,
                                   ScalarBasisValue& result) const
{
    PetscFunctionBeginUser;
    std::size_t e;
    PetscCall(CheckSide(side, e));
    const auto& current = edges_[e];
    const auto& opposite = edges_[(e + 2) % 4];
    ScalarBasisValue a, b;
    PetscCall(Distance(current, point, a));
    PetscCall(Distance(opposite, point, b));

    // Scaling avoids squaring tiny/huge physical distances in the quotient
    // rule. A relative cancellation check also detects singular exterior points.
    const PetscReal scale = std::max(PetscAbsReal(a.value), PetscAbsReal(b.value));
    PetscCheck(scale > 0, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE,
               "Opposite-edge distance sum is zero");
    const PetscReal x = a.value / scale, y = b.value / scale;
    const PetscReal sum = x + y;
    PetscCheck(PetscAbsReal(sum) > 64 * PETSC_MACHINE_EPSILON *
                                      (PetscAbsReal(x) + PetscAbsReal(y)),
               PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE,
               "Opposite-edge distance sum is zero or numerically singular");
    const PetscReal firstWeight = x / sum, oppositeWeight = y / sum;
    ScalarBasisValue work;
    work.value = oppositeWeight;
    for (int d = 0; d < 2; ++d)
        work.gradient.p[d] = (oppositeWeight * current.normal.p[d] -
                               firstWeight * opposite.normal.p[d]) / sum / scale;
    PetscCall(StoreResult(work, result));
    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode QuadBasis::OppositeEdgeContrast(CellSide first, CellSide second,
                                              const Point& point,
                                              ScalarBasisValue& result) const
{
    PetscFunctionBeginUser;
    std::size_t e;
    PetscCall(CheckOppositeSides(first, second, e));
    ScalarBasisValue blend;
    PetscCall(EdgeBlend(first, point, blend));
    const ScalarBasisValue work{1 - 2 * blend.value,
        Point{{-2 * blend.gradient.p[0], -2 * blend.gradient.p[1]}}};
    PetscCall(StoreResult(work, result));
    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode QuadBasis::AlternatingVertexPolynomial(const Point& point,
                                                     ScalarBasisValue& result) const
{
    PetscFunctionBeginUser;
    PetscCheck(initialized_, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG,
               "Initialize the quadrilateral basis before querying it");
    ScalarBasisValue sum;
    for (std::size_t i = 0; i < 4; ++i) {
        ScalarBasisValue a, b;
        PetscCall(Distance(edges_[(i + 1) % 4], point, a));
        PetscCall(Distance(edges_[(i + 2) % 4], point, b));
        const auto term = Product(Divide(a, vertexDistances_[i][0]),
                                  Divide(b, vertexDistances_[i][1]));
        const PetscReal sign = (i % 2 == 0) ? 1 : -1;
        sum.value += sign * term.value;
        for (int d = 0; d < 2; ++d) sum.gradient.p[d] += sign * term.gradient.p[d];
    }
    PetscCall(StoreResult(sum, result));
    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode QuadBasis::EdgeBubble(CellSide side, const Point& point,
                                    ScalarBasisValue& result) const
{
    PetscFunctionBeginUser;
    std::size_t e;
    PetscCall(CheckSide(side, e));
    ScalarBasisValue a, b, blend;
    PetscCall(Distance(edges_[(e + 1) % 4], point, a));
    PetscCall(Distance(edges_[(e + 3) % 4], point, b));
    PetscCall(EdgeBlend(side, point, blend));
    const auto product = Product(Divide(a, bubbleDistances_[e][0]),
                                 Divide(b, bubbleDistances_[e][1]));
    const auto work = Product(product, Divide(blend, bubbleBlends_[e]));
    PetscCall(StoreResult(work, result));
    PetscFunctionReturn(PETSC_SUCCESS);
}
