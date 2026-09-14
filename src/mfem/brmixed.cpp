#include "brmixed.h"

#include <algorithm>

namespace {

// CellSide order: Bottom,Right,Top,Left. The intended legacy signs were
// {-1,-1,+1,+1} in Left,Bottom,Right,Top order. All evaluators use this rule.
constexpr std::array<PetscReal, 4> normalSigns{{-1, 1, 1, -1}};

PetscErrorCode StoreScalar(const ScalarBasisValue& value, ScalarBasisValue& result)
{
    PetscFunctionBeginUser;
    PetscCheck(!PetscIsInfOrNanReal(value.value) &&
                 !PetscIsInfOrNanReal(value.gradient.p[0]) &&
                 !PetscIsInfOrNanReal(value.gradient.p[1]),
               PETSC_COMM_SELF, PETSC_ERR_FP,
               "BR scalar value or physical gradient is not finite");
    result = value;
    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode MakeVector(const ScalarBasisValue& scalar, const Point& direction,
                          BRBasisValue& result)
{
    PetscFunctionBeginUser;
    BRBasisValue work;
    for (int component = 0; component < 2; ++component) {
        work.value.p[component] = scalar.value * direction.p[component];
        PetscCheck(!PetscIsInfOrNanReal(work.value.p[component]),
                   PETSC_COMM_SELF, PETSC_ERR_FP, "BR vector value is not finite");
        for (int d = 0; d < 2; ++d) {
            work.gradient[2*component+d] = scalar.gradient.p[d] * direction.p[component];
            PetscCheck(!PetscIsInfOrNanReal(work.gradient[2*component+d]),
                       PETSC_COMM_SELF, PETSC_ERR_FP, "BR vector gradient is not finite");
        }
    }
    result = work;
    PetscFunctionReturn(PETSC_SUCCESS);
}

} // namespace

PetscErrorCode BRMixed::Initialize(const QuadVertices& corners)
{
    PetscFunctionBeginUser;
    QuadBasis geometry;
    PetscCall(geometry.Initialize(corners));
    PetscCall(Initialize(geometry));
    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode BRMixed::Initialize(const MeshInfo& mesh, MeshIndex cell)
{
    PetscFunctionBeginUser;
    QuadBasis geometry;
    PetscCall(geometry.Initialize(mesh, cell));
    PetscCall(Initialize(geometry));
    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode BRMixed::Initialize(const QuadBasis& geometry)
{
    PetscFunctionBeginUser;
    PetscCheck(geometry.IsInitialized(), PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG,
               "Initialize QuadBasis before using it to initialize BRMixed");
    BRMixed work;
    work.geometry_ = geometry;
    QuadVertices corners;
    PetscCall(geometry.GetCorners(corners));

    for (int e = 0; e < 4; ++e) {
        BasisEdgeGeometry edge;
        PetscCall(geometry.GetEdge(static_cast<CellSide>(e), edge));
        for (int d = 0; d < 2; ++d)
            work.edgeNormals_[e].p[d] = normalSigns[e] * edge.normal.p[d];
        ScalarBasisValue polynomial;
        PetscCall(geometry.AlternatingVertexPolynomial(edge.midpoint, polynomial));
        work.midpointPolynomial_[e] = polynomial.value;
    }

    for (PetscInt i = 0; i < 4; ++i) {
        const PetscInt diagonal = (i+1)%2;
        ScalarBasisValue own, opposite;
        PetscCall(geometry.DiagonalDistance(diagonal, corners[i], own));
        PetscCall(geometry.DiagonalDistance(diagonal, corners[(i+2)%4], opposite));
        PetscCheck((own.value > 0 && opposite.value < 0) ||
                     (own.value < 0 && opposite.value > 0),
                   PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG,
                   "Opposite vertices must lie strictly on opposite sides of the other diagonal");
        const PetscReal scale = std::max(PetscAbsReal(own.value), PetscAbsReal(opposite.value));
        const PetscReal a = own.value / scale, b = opposite.value / scale;
        work.diagonalScale_[i] = scale;
        work.diagonalDifference_[i] = a-b;
        work.oppositeWeight_[i] = PetscReal(0.5) * b / (a-b);
    }

    work.initialized_ = true;
    *this = work;
    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode BRMixed::CheckInitialized() const
{
    PetscFunctionBeginUser;
    PetscCheck(initialized_, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG,
               "Initialize BRMixed before evaluating or querying its geometry");
    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode BRMixed::GetCorners(QuadVertices& corners) const
{
    PetscFunctionBeginUser;
    PetscCall(CheckInitialized());
    PetscCall(geometry_.GetCorners(corners));
    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode BRMixed::GetEdgeNormal(CellSide side, Point& normal) const
{
    PetscFunctionBeginUser;
    PetscCall(CheckInitialized());
    const int e = static_cast<int>(side);
    PetscCheck(e >= 0 && e < 4, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE,
               "CellSide must be Bottom, Right, Top, or Left");
    normal = edgeNormals_[e];
    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode BRMixed::EvaluateScalars(const Point& point,
                                       std::array<ScalarBasisValue, 4>& bubbles,
                                       ScalarBasisValue& supplement) const
{
    PetscFunctionBeginUser;
    std::array<ScalarBasisValue, 4> work;
    ScalarBasisValue sum;
    PetscCall(geometry_.AlternatingVertexPolynomial(point, sum));
    for (int e = 0; e < 4; ++e) {
        PetscCall(geometry_.EdgeBubble(static_cast<CellSide>(e), point, work[e]));
        sum.value -= midpointPolynomial_[e] * work[e].value;
        for (int d = 0; d < 2; ++d)
            sum.gradient.p[d] -= midpointPolynomial_[e] * work[e].gradient.p[d];
    }
    PetscCall(StoreScalar(sum, supplement));
    bubbles = work;
    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode BRMixed::MakeVertex(PetscInt vertex, const ScalarBasisValue& diagonal,
                                  const ScalarBasisValue& supplement,
                                  ScalarBasisValue& result) const
{
    PetscFunctionBeginUser;
    // Same DS1 construction as legacy phiv/dPhiv:
    // phi_i = d_i(x)/D_i - weight_i*(1 + (-1)^i*S(x)).
    const PetscReal sign = vertex%2 == 0 ? PetscReal(1) : PetscReal(-1);
    const PetscReal weight = oppositeWeight_[vertex];
    const PetscReal scale = diagonalScale_[vertex];
    const PetscReal difference = diagonalDifference_[vertex];
    ScalarBasisValue work;
    // Divide by |difference| >= 1 first to avoid an unnecessary intermediate
    // overflow at very small cell scales. Never form scale*difference.
    work.value = diagonal.value / difference / scale - weight*(1 + sign*supplement.value);
    for (int d = 0; d < 2; ++d)
        work.gradient.p[d] = diagonal.gradient.p[d] / difference / scale -
                              weight*sign*supplement.gradient.p[d];
    PetscCall(StoreScalar(work, result));
    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode BRMixed::EvaluateVertex(PetscInt vertex, const Point& point,
                                      ScalarBasisValue& result) const
{
    PetscFunctionBeginUser;
    PetscCall(CheckInitialized());
    PetscCheck(vertex >= 0 && vertex < 4, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE,
               "BR vertex index must be in [0,4)");
    std::array<ScalarBasisValue, 4> bubbles;
    ScalarBasisValue supplement, diagonal;
    PetscCall(EvaluateScalars(point, bubbles, supplement));
    PetscCall(geometry_.DiagonalDistance((vertex+1)%2, point, diagonal));
    PetscCall(MakeVertex(vertex, diagonal, supplement, result));
    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode BRMixed::EvaluateBubble(CellSide side, const Point& point,
                                      ScalarBasisValue& result) const
{
    PetscFunctionBeginUser;
    PetscCall(CheckInitialized());
    PetscCall(geometry_.EdgeBubble(side, point, result));
    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode BRMixed::EvaluateSupplement(const Point& point,
                                          ScalarBasisValue& result) const
{
    PetscFunctionBeginUser;
    PetscCall(CheckInitialized());
    std::array<ScalarBasisValue, 4> bubbles;
    PetscCall(EvaluateScalars(point, bubbles, result));
    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode BRMixed::Evaluate(const Point& point, PetscInt localDof,
                                BRBasisValue& result) const
{
    PetscFunctionBeginUser;
    PetscCall(CheckInitialized());
    PetscCheck(localDof >= 0 && localDof < ElementDofs,
               PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE,
               "BR local DOF index must be in [0,12)");
    ScalarBasisValue scalar;
    Point direction{};
    if (localDof < 8) {
        PetscCall(EvaluateVertex(localDof%4, point, scalar));
        direction.p[localDof/4] = 1;
    } else {
        const PetscInt e = localDof-8;
        PetscCall(EvaluateBubble(static_cast<CellSide>(e), point, scalar));
        direction = edgeNormals_[e];
    }
    PetscCall(MakeVector(scalar, direction, result));
    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode BRMixed::EvaluateAll(const Point& point, Values& result) const
{
    PetscFunctionBeginUser;
    PetscCall(CheckInitialized());
    std::array<ScalarBasisValue, 4> bubbles;
    ScalarBasisValue supplement;
    PetscCall(EvaluateScalars(point, bubbles, supplement));
    std::array<ScalarBasisValue, 2> diagonals;
    for (PetscInt d = 0; d < 2; ++d)
        PetscCall(geometry_.DiagonalDistance(d, point, diagonals[d]));

    Values work;
    for (PetscInt i = 0; i < 4; ++i) {
        ScalarBasisValue vertex;
        PetscCall(MakeVertex(i, diagonals[(i+1)%2], supplement, vertex));
        PetscCall(MakeVector(vertex, Point{{1,0}}, work[i]));
        PetscCall(MakeVector(vertex, Point{{0,1}}, work[i+4]));
        PetscCall(MakeVector(bubbles[i], edgeNormals_[i], work[i+8]));
    }
    result = work;
    PetscFunctionReturn(PETSC_SUCCESS);
}
