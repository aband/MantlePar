#include "hdivmixed.h"

#include <algorithm>

namespace {

constexpr std::array<PetscReal, 4> normalSigns{{-1, 1, 1, -1}};

CellSide Side(std::size_t edge)
{ return static_cast<CellSide>(edge % 4); }

// curl(f) = (df/dy, -df/dx). For lambda_e, this is the CCW edge tangent.
Point Curl(const Point& gradient)
{ return Point{{gradient.p[1], -gradient.p[0]}}; }

PetscErrorCode PositiveDistance(const QuadBasis& geometry, CellSide side,
                                const Point& point, PetscReal& result)
{
    PetscFunctionBeginUser;
    ScalarBasisValue distance;
    PetscCall(geometry.EdgeDistance(side, point, distance));
    PetscCheck(distance.value > 0, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG,
               "HDivMixed requires positive normalization distances");
    result = distance.value;
    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode StoreResult(const HDivBasisValue& value, HDivBasisValue& result)
{
    PetscFunctionBeginUser;
    PetscCheck(!PetscIsInfOrNanReal(value.value.p[0]) &&
                 !PetscIsInfOrNanReal(value.value.p[1]) &&
                 !PetscIsInfOrNanReal(value.divergence),
               PETSC_COMM_SELF, PETSC_ERR_FP,
               "HDivMixed vector value or divergence is not finite");
    result = value;
    PetscFunctionReturn(PETSC_SUCCESS);
}

} // namespace

PetscErrorCode HDivMixed::Initialize(const QuadVertices& corners)
{
    PetscFunctionBeginUser;
    QuadBasis geometry;
    PetscCall(geometry.Initialize(corners));
    PetscCall(Initialize(geometry));
    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode HDivMixed::Initialize(const MeshInfo& mesh, MeshIndex cell)
{
    PetscFunctionBeginUser;
    QuadBasis geometry;
    PetscCall(geometry.Initialize(mesh, cell));
    PetscCall(Initialize(geometry));
    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode HDivMixed::Initialize(const QuadBasis& geometry)
{
    PetscFunctionBeginUser;
    PetscCheck(geometry.IsInitialized(), PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG,
               "Initialize QuadBasis before using it to initialize HDivMixed");
    HDivMixed work;
    work.geometry_ = geometry;
    QuadVertices corners;
    std::array<BasisEdgeGeometry, 4> edges;
    PetscCall(geometry.GetCorners(corners));
    for (std::size_t e = 0; e < 4; ++e)
        PetscCall(geometry.GetEdge(Side(e), edges[e]));

    for (std::size_t e = 0; e < 4; ++e) {
        auto& data = work.edges_[e];
        const auto& edge = edges[e];
        const auto& next = edges[(e+1)%4];
        const auto& vertex = corners[(e+1)%4];
        data.length = edge.length;
        data.oppositeVertex = corners[(e+3)%4];
        for (int d = 0; d < 2; ++d)
            data.normal.p[d] = normalSigns[e] * edge.normal.p[d];

        // P_e = lambda_(e+3)*lambda_(e+2), normalized to one at v_(e+1).
        // Divide each factor separately, avoiding products of physical lengths.
        PetscCall(PositiveDistance(geometry, Side(e+3), vertex, data.potentialDistances[0]));
        PetscCall(PositiveDistance(geometry, Side(e+2), vertex, data.potentialDistances[1]));
        for (std::size_t k = 0; k < 2; ++k) {
            const auto& midpoint = edges[(e+k)%4].midpoint;
            ScalarBasisValue a, b;
            PetscCall(geometry.EdgeDistance(Side(e+3), midpoint, a));
            PetscCall(geometry.EdgeDistance(Side(e+2), midpoint, b));
            data.potentialWeights[k] = PetscReal(0.5) -
                (a.value / data.potentialDistances[0]) * (b.value / data.potentialDistances[1]);
            PetscCheck(!PetscIsInfOrNanReal(data.potentialWeights[k]),
                       PETSC_COMM_SELF, PETSC_ERR_FP,
                       "HDivMixed scalar-potential normalization is not finite");
        }

        // Legacy phic = ((x-vOpp) + b*Lnext*curl(S_e)) / D,
        // D = a + b*Lnext/L = 2*area/L. Both a and b are positive distances.
        // Scale the positive sum to avoid forming length-squared products.
        PetscReal a, b;
        PetscCall(PositiveDistance(geometry, Side(e), data.oppositeVertex, a));
        PetscCall(PositiveDistance(geometry, Side(e+1), data.oppositeVertex, b));
        const PetscReal distanceScale = std::max(a, b);
        const PetscReal lengthScale = std::max(edge.length, next.length);
        const PetscReal currentLength = edge.length / lengthScale;
        const PetscReal nextContribution = (b / distanceScale) * (next.length / lengthScale);
        const PetscReal sum = (a / distanceScale) * currentLength + nextContribution;
        PetscCheck(sum > 0 && !PetscIsInfOrNanReal(sum),
                   PETSC_COMM_SELF, PETSC_ERR_FP,
                   "HDivMixed constant-mode denominator is not representable");
        data.inverseDenominator = (currentLength / sum) / distanceScale;
        data.curlWeight = (nextContribution / sum) * edge.length;
        PetscCheck(data.inverseDenominator > 0 &&
                     !PetscIsInfOrNanReal(2 * data.inverseDenominator) &&
                     !PetscIsInfOrNanReal(data.curlWeight),
                   PETSC_COMM_SELF, PETSC_ERR_FP,
                   "HDivMixed normalization or divergence is not representable");
    }

    work.initialized_ = true;
    *this = work;
    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode HDivMixed::CheckInitialized() const
{
    PetscFunctionBeginUser;
    PetscCheck(initialized_, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG,
               "Initialize HDivMixed before evaluating or querying its geometry");
    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode HDivMixed::CheckSide(CellSide side, std::size_t& edge) const
{
    PetscFunctionBeginUser;
    PetscCall(CheckInitialized());
    const int e = static_cast<int>(side);
    PetscCheck(e >= 0 && e < 4, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE,
               "CellSide must be Bottom, Right, Top, or Left");
    edge = static_cast<std::size_t>(e);
    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode HDivMixed::GetCorners(QuadVertices& corners) const
{
    PetscFunctionBeginUser;
    PetscCall(CheckInitialized());
    PetscCall(geometry_.GetCorners(corners));
    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode HDivMixed::GetEdgeNormal(CellSide side, Point& normal) const
{
    PetscFunctionBeginUser;
    std::size_t e;
    PetscCall(CheckSide(side, e));
    normal = edges_[e].normal;
    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode HDivMixed::MakeLinear(std::size_t e, const ScalarBasisValue& bubble,
                                    HDivBasisValue& result) const
{
    PetscFunctionBeginUser;
    HDivBasisValue work;
    const auto curl = Curl(bubble.gradient);
    for (int d = 0; d < 2; ++d)
        work.value.p[d] = (PetscReal(0.25) * edges_[e].length) * curl.p[d];
    // div(curl(b_e)) = 0 analytically. No additional orientation sign.
    work.divergence = 0;
    PetscCall(StoreResult(work, result));
    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode HDivMixed::MakeConstant(std::size_t e, const Point& point,
                                      const std::array<ScalarBasisValue, 2>& distances,
                                      const std::array<ScalarBasisValue, 2>& bubbles,
                                      HDivBasisValue& result) const
{
    PetscFunctionBeginUser;
    const auto& data = edges_[e];
    const PetscReal a = distances[0].value / data.potentialDistances[0];
    const PetscReal b = distances[1].value / data.potentialDistances[1];
    Point gradient{};
    // S_e = P_e + (1/2-P_e(mid_e))*b_e
    //             + (1/2-P_e(mid_next))*b_next.
    // This is the legacy normalized DS2 vertex potential plus its two half
    // edge bubbles, evaluated through the shared normalized scalar helpers.
    for (int d = 0; d < 2; ++d) {
        gradient.p[d] = (distances[0].gradient.p[d] / data.potentialDistances[0]) * b +
            a * (distances[1].gradient.p[d] / data.potentialDistances[1]) +
            data.potentialWeights[0] * bubbles[0].gradient.p[d] +
            data.potentialWeights[1] * bubbles[1].gradient.p[d];
    }
    const auto curl = Curl(gradient);
    HDivBasisValue work;
    for (int d = 0; d < 2; ++d) {
        const PetscReal displacement = point.p[d] - data.oppositeVertex.p[d];
        PetscCheck(!PetscIsInfOrNanReal(displacement), PETSC_COMM_SELF, PETSC_ERR_FP,
                   "HDivMixed evaluation displacement is not representable");
        work.value.p[d] = normalSigns[e] *
            (displacement * data.inverseDenominator + data.curlWeight * curl.p[d]);
    }
    // The curl term has zero divergence; div(x-vOpp)=2. Apply exactly the
    // same orientation sign as the vector value returned above.
    work.divergence = normalSigns[e] * (2 * data.inverseDenominator);
    PetscCall(StoreResult(work, result));
    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode HDivMixed::EvaluateEdge(const Point& point, CellSide side,
                                      EdgeValues& result) const
{
    PetscFunctionBeginUser;
    std::size_t e;
    PetscCall(CheckSide(side, e));
    std::array<ScalarBasisValue, 2> distances, bubbles;
    PetscCall(geometry_.EdgeDistance(Side(e+3), point, distances[0]));
    PetscCall(geometry_.EdgeDistance(Side(e+2), point, distances[1]));
    PetscCall(geometry_.EdgeBubble(Side(e), point, bubbles[0]));
    PetscCall(geometry_.EdgeBubble(Side(e+1), point, bubbles[1]));
    EdgeValues work;
    PetscCall(MakeLinear(e, bubbles[0], work[0]));
    PetscCall(MakeConstant(e, point, distances, bubbles, work[1]));
    result = work;
    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode HDivMixed::Evaluate(const Point& point, PetscInt localDof,
                                  HDivBasisValue& result) const
{
    PetscFunctionBeginUser;
    PetscCall(CheckInitialized());
    PetscCheck(localDof >= 0 && localDof < ElementDofs,
               PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE,
               "HDivMixed local DOF index must be in [0,8)");
    EdgeValues pair;
    PetscCall(EvaluateEdge(point, static_cast<CellSide>(localDof % 4), pair));
    result = pair[static_cast<std::size_t>(localDof / 4)];
    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode HDivMixed::EvaluateAll(const Point& point, Values& result) const
{
    PetscFunctionBeginUser;
    PetscCall(CheckInitialized());
    std::array<ScalarBasisValue, 4> distances, bubbles;
    for (std::size_t e = 0; e < 4; ++e) {
        PetscCall(geometry_.EdgeDistance(Side(e), point, distances[e]));
        PetscCall(geometry_.EdgeBubble(Side(e), point, bubbles[e]));
    }
    Values work;
    for (std::size_t e = 0; e < 4; ++e) {
        PetscCall(MakeLinear(e, bubbles[e], work[e]));
        const std::array<ScalarBasisValue, 2> vertexDistances{{distances[(e+3)%4], distances[(e+2)%4]}};
        const std::array<ScalarBasisValue, 2> edgeBubbles{{bubbles[e], bubbles[(e+1)%4]}};
        PetscCall(MakeConstant(e, point, vertexDistances, edgeBubbles, work[e+4]));
    }
    result = work;
    PetscFunctionReturn(PETSC_SUCCESS);
}
