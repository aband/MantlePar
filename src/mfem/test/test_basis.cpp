#include "basis.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <vector>

namespace {

// All checks remain active in Release builds. No external test framework.
PetscErrorCode Require(bool condition, const char* message)
{
    PetscFunctionBeginUser;
    PetscCheck(condition, PETSC_COMM_SELF, PETSC_ERR_PLIB, "%s", message);
    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode Near(PetscReal actual, PetscReal expected, PetscReal scale,
                     const char* message,
                     PetscReal tolerance = 4096 * PETSC_MACHINE_EPSILON)
{
    PetscFunctionBeginUser;
    const PetscReal limit = tolerance * std::max(PetscAbsReal(expected), scale);
    PetscCheck(!PetscIsInfOrNanReal(actual) && !PetscIsInfOrNanReal(expected) &&
                 PetscAbsReal(actual - expected) <= limit,
               PETSC_COMM_SELF, PETSC_ERR_PLIB,
               "%s: got %.17g, expected %.17g, tolerance %.3g", message,
               static_cast<double>(actual), static_cast<double>(expected),
               static_cast<double>(limit));
    PetscFunctionReturn(PETSC_SUCCESS);
}

template<class Call>
PetscErrorCode ExpectError(Call call, PetscErrorCode expected = PETSC_SUCCESS)
{
    PetscFunctionBeginUser;
    PetscCall(PetscPushErrorHandler(PetscReturnErrorHandler, nullptr));
    const PetscErrorCode error = call();
    PetscCall(PetscPopErrorHandler());
    PetscCall(Require(error != PETSC_SUCCESS, "Invalid input was accepted"));
    if (expected != PETSC_SUCCESS)
        PetscCall(Require(error == expected, "Unexpected error code"));
    PetscFunctionReturn(PETSC_SUCCESS);
}

CellSide Side(int e) { return static_cast<CellSide>(e); }
PetscReal Length(const Point& a, const Point& b)
{
    return std::hypot(b.p[0] - a.p[0], b.p[1] - a.p[1]);
}
Point Mix(const Point& a, const Point& b, PetscReal t)
{
    return {{a.p[0] + t * (b.p[0] - a.p[0]),
             a.p[1] + t * (b.p[1] - a.p[1])}};
}
Point Interior(const QuadVertices& q, PetscReal s, PetscReal t)
{
    return Mix(Mix(q[0], q[1], s), Mix(q[3], q[2], s), t);
}
Point Extents(const QuadVertices& q)
{
    Point lo = q[0], hi = q[0];
    for (const auto& p : q) for (int d = 0; d < 2; ++d) {
        lo.p[d] = std::min(lo.p[d], p.p[d]);
        hi.p[d] = std::max(hi.p[d], p.p[d]);
    }
    return {{hi.p[0] - lo.p[0], hi.p[1] - lo.p[1]}};
}

// Independent determinant oracle, used only on moderate-size fixtures.
// Expected values do not call the production distance/mapping functions.
PetscReal Distance(const Point& a, const Point& b, const Point& p)
{
    return ((b.p[0] - a.p[0]) * (p.p[1] - a.p[1]) -
            (b.p[1] - a.p[1]) * (p.p[0] - a.p[0])) / Length(a, b);
}
PetscReal Lambda(const QuadVertices& q, int e, const Point& p)
{
    return Distance(q[e], q[(e + 1) % 4], p);
}

enum class Query { Distance, Projection, Blend, Contrast, ScaledDifference,
                   Bubble, Polynomial, Diagonal };
constexpr std::array<Query, 8> queries{{Query::Distance, Query::Projection,
    Query::Blend, Query::Contrast, Query::ScaledDifference, Query::Bubble,
    Query::Polynomial, Query::Diagonal}};
int Count(Query kind)
{
    return kind == Query::Polynomial ? 1 : (kind == Query::Diagonal ? 2 : 4);
}
bool HasLengthUnits(Query kind)
{
    return kind == Query::Distance || kind == Query::Projection || kind == Query::Diagonal;
}
PetscErrorCode Evaluate(const QuadBasis& basis, Query kind, int e,
                        const Point& p, ScalarBasisValue& result)
{
    switch (kind) {
    case Query::Distance: return basis.EdgeDistance(Side(e), p, result);
    case Query::Projection: return basis.EdgeProjection(Side(e), p, result);
    case Query::Blend: return basis.EdgeBlend(Side(e), p, result);
    case Query::Contrast: return basis.OppositeEdgeContrast(Side(e), Side((e+2)%4), p, result);
    case Query::ScaledDifference: return basis.ScaledDistanceDifference(Side(e), Side((e+2)%4), p, result);
    case Query::Bubble: return basis.EdgeBubble(Side(e), p, result);
    case Query::Polynomial: return basis.AlternatingVertexPolynomial(p, result);
    case Query::Diagonal: return basis.DiagonalDistance(e, p, result);
    }
    return PETSC_ERR_ARG_WRONG;
}

PetscReal ReferenceValue(const QuadVertices& q, Query kind, int e, const Point& p)
{
    if (kind == Query::Diagonal) return Distance(q[e], q[e+2], p);
    if (kind == Query::Polynomial) {
        PetscReal sum = 0;
        for (int i = 0; i < 4; ++i)
            sum += (i%2 ? -1 : 1) * Lambda(q, (i+1)%4, p) / Lambda(q, (i+1)%4, q[i])
                                 * Lambda(q, (i+2)%4, p) / Lambda(q, (i+2)%4, q[i]);
        return sum;
    }
    const PetscReal a = Lambda(q, e, p), b = Lambda(q, (e+2)%4, p);
    switch (kind) {
    case Query::Distance: return a;
    case Query::Projection:
        return ((p.p[0] - q[e].p[0]) * (q[(e+1)%4].p[0] - q[e].p[0]) +
                (p.p[1] - q[e].p[1]) * (q[(e+1)%4].p[1] - q[e].p[1])) /
               Length(q[e], q[(e+1)%4]);
    case Query::Blend: return b / (a+b);
    case Query::Contrast: return (a-b) / (a+b);
    case Query::ScaledDifference:
        // END-corner diagonal: v[e+1] to v[e+3], not the other diagonal.
        return (a-b) / Length(q[(e+1)%4], q[(e+3)%4]);
    case Query::Bubble: {
        const Point m = Mix(q[e], q[(e+1)%4], PetscReal(0.5));
        // R_e=1 at its own midpoint. This trace fixes the normalization.
        return Lambda(q, (e+1)%4, p) / Lambda(q, (e+1)%4, m) *
               Lambda(q, (e+3)%4, p) / Lambda(q, (e+3)%4, m) * b / (a+b);
    }
    default: return 0; // Polynomial and diagonal were handled above.
    }
}

PetscErrorCode CheckGradient(const QuadBasis& basis, Query kind, int e,
                             const Point& p, const Point& extents)
{
    PetscFunctionBeginUser;
    const PetscReal step = std::cbrt(PetscReal(PETSC_MACHINE_EPSILON));
    const PetscReal tolerance = 256 * step * step;
    const PetscReal valueScale = HasLengthUnits(kind)
        ? std::max(extents.p[0], extents.p[1]) : PetscReal(1);
    ScalarBasisValue center;
    PetscCall(Evaluate(basis, kind, e, p, center));
    for (int d = 0; d < 2; ++d) {
        const PetscReal h = step * extents.p[d];
        Point plus = p, minus = p;
        plus.p[d] += h;
        minus.p[d] -= h;
        ScalarBasisValue a, b;
        PetscCall(Evaluate(basis, kind, e, plus, a));
        PetscCall(Evaluate(basis, kind, e, minus, b));
        PetscCall(Near(center.gradient.p[d], (a.value-b.value)/(2*h),
                        valueScale/extents.p[d], "Physical gradient vs finite difference", tolerance));
    }
    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode CheckSegment(const BasisEdgeGeometry& edge, const Point& a, const Point& b)
{
    PetscFunctionBeginUser;
    const PetscReal length = Length(a, b);
    PetscCall(Near(edge.length, length, length, "Segment length"));
    for (int d = 0; d < 2; ++d) {
        PetscCall(Require(edge.vertices[0].p[d] == a.p[d] && edge.vertices[1].p[d] == b.p[d],
                           "Directed segment endpoints"));
        PetscCall(Near(edge.midpoint.p[d], (a.p[d]+b.p[d])/2, length, "Segment midpoint"));
        PetscCall(Near(edge.tangent.p[d], (b.p[d]-a.p[d])/length, 1, "Unit tangent"));
    }
    PetscCall(Near(edge.normal.p[0], (b.p[1]-a.p[1])/length, 1, "Right normal x"));
    PetscCall(Near(edge.normal.p[1], -(b.p[0]-a.p[0])/length, 1, "Right normal y"));
    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode CheckCell(const QuadVertices& q)
{
    PetscFunctionBeginUser;
    QuadBasis basis;
    PetscCall(basis.Initialize(q));
    PetscCall(Require(basis.IsInitialized(), "Successful initialization must set state"));
    QuadVertices stored;
    PetscCall(basis.GetCorners(stored));
    for (int e = 0; e < 4; ++e) for (int d = 0; d < 2; ++d)
        PetscCall(Require(stored[e].p[d] == q[e].p[d], "Stored corner ordering"));
    const Point extents = Extents(q);
    const PetscReal size = std::max(extents.p[0], extents.p[1]);
    for (int e = 0; e < 4; ++e) {
        BasisEdgeGeometry edge;
        PetscCall(basis.GetEdge(Side(e), edge));
        PetscCall(CheckSegment(edge, q[e], q[(e+1)%4]));
        ScalarBasisValue corner;
        PetscCall(basis.AlternatingVertexPolynomial(q[e], corner));
        PetscCall(Near(corner.value, e%2 ? -1 : 1, 1, "Alternating corner values"));
        for (PetscReal s : {PetscReal(0), PetscReal(0.2), PetscReal(0.5), PetscReal(0.8), PetscReal(1)}) {
            const Point p = Mix(q[e], q[(e+1)%4], s);
            ScalarBasisValue value;
            PetscCall(basis.EdgeDistance(Side(e), p, value));
            PetscCall(Near(value.value, 0, size, "Zero distance on own edge"));
            PetscCall(basis.EdgeProjection(Side(e), p, value));
            PetscCall(Near(value.value, s*edge.length, size, "Physical arclength projection"));
            PetscCall(basis.EdgeBlend(Side(e), p, value));
            PetscCall(Near(value.value, 1, 1, "Blend on own edge"));
            PetscCall(basis.EdgeBlend(Side((e+2)%4), p, value));
            PetscCall(Near(value.value, 0, 1, "Blend on opposite edge"));
            for (int k = 0; k < 4; ++k) {
                PetscCall(basis.EdgeBubble(Side(k), p, value));
                PetscCall(Near(value.value, k == e ? 4*s*(1-s) : 0, 1, "Normalized bubble trace"));
                const PetscReal tangentDerivative = value.gradient.p[0]*edge.tangent.p[0] +
                                                     value.gradient.p[1]*edge.tangent.p[1];
                PetscCall(Near(tangentDerivative, k == e ? 4*(1-2*s)/edge.length : 0,
                                1/edge.length, "Bubble tangential derivative"));
            }
        }
    }
    for (int d = 0; d < 2; ++d) {
        BasisEdgeGeometry diagonal;
        PetscCall(basis.GetDiagonal(d, diagonal));
        PetscCall(CheckSegment(diagonal, q[d], q[d+2]));
    }
    for (PetscReal s : {PetscReal(0.19), PetscReal(0.43), PetscReal(0.81)})
        for (PetscReal t : {PetscReal(0.24), PetscReal(0.61), PetscReal(0.78)}) {
            const Point p = Interior(q, s, t);
            for (Query kind : queries) for (int e = 0; e < Count(kind); ++e) {
                ScalarBasisValue value;
                PetscCall(Evaluate(basis, kind, e, p, value));
                PetscCall(Near(value.value, ReferenceValue(q, kind, e, p),
                                HasLengthUnits(kind) ? size : PetscReal(1), "Independent value oracle"));
                if (kind == Query::Distance)
                    PetscCall(Require(value.value > 0, "Interior signed distance must be positive"));
                PetscCall(CheckGradient(basis, kind, e, p, extents));
            }
            for (int e = 0; e < 2; ++e) {
                ScalarBasisValue a, b;
                PetscCall(basis.EdgeBlend(Side(e), p, a));
                PetscCall(basis.EdgeBlend(Side(e+2), p, b));
                PetscCall(Near(a.value+b.value, 1, 1, "Opposite blends sum to one"));
                for (int d = 0; d < 2; ++d)
                    PetscCall(Near(a.gradient.p[d]+b.gradient.p[d], 0, 1/extents.p[d], "Blend gradient sum"));
            }
        }
    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode CheckRectangle(PetscReal width, PetscReal height)
{
    PetscFunctionBeginUser;
    const QuadVertices q{{{{0,0}}, {{width,0}}, {{width,height}}, {{0,height}}}};
    QuadBasis basis;
    PetscCall(basis.Initialize(q));
    // Closed forms independent of edge-distance implementation. Also use them
    // at extreme scales, where direct determinant products can overflow.
    for (PetscReal s : {PetscReal(0.17), PetscReal(0.5), PetscReal(0.83)})
        for (PetscReal t : {PetscReal(0.23), PetscReal(0.5), PetscReal(0.67)}) {
            const Point p{{s*width, t*height}};
            const PetscReal blends[4] = {1-t, s, t, 1-s};
            const Point gradients[4] = {{{0,-1/height}}, {{1/width,0}}, {{0,1/height}}, {{-1/width,0}}};
            for (int e = 0; e < 4; ++e) {
                ScalarBasisValue value;
                PetscCall(basis.EdgeBlend(Side(e), p, value));
                PetscCall(Near(value.value, blends[e], 1, "Rectangle blend"));
                for (int d = 0; d < 2; ++d)
                    PetscCall(Near(value.gradient.p[d], gradients[e].p[d],
                                    1/(d == 0 ? width : height), "Rectangle blend gradient"));
                const PetscReal u = e%2 == 0 ? s : t;
                const PetscReal factor = 4*u*(1-u);
                const Point df = e%2 == 0 ? Point{{4*(1-2*s)/width,0}} : Point{{0,4*(1-2*t)/height}};
                PetscCall(basis.EdgeBubble(Side(e), p, value));
                PetscCall(Near(value.value, factor*blends[e], 1, "Rectangle bubble"));
                for (int d = 0; d < 2; ++d)
                    PetscCall(Near(value.gradient.p[d], df.p[d]*blends[e] + factor*gradients[e].p[d],
                                    1/(d == 0 ? width : height), "Rectangle bubble gradient"));
            }
            ScalarBasisValue value;
            PetscCall(basis.AlternatingVertexPolynomial(p, value));
            PetscCall(Near(value.value, (1-2*s)*(1-2*t), 1, "Rectangle alternating polynomial"));
            PetscCall(Near(value.gradient.p[0], -2*(1-2*t)/width, 1/width, "Rectangle polynomial x gradient"));
            PetscCall(Near(value.gradient.p[1], -2*(1-2*s)/height, 1/height, "Rectangle polynomial y gradient"));
        }
    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode CheckTransformations(PetscReal large)
{
    PetscFunctionBeginUser;
    const QuadVertices q{{{{-0.3,0.1}}, {{1.8,0.2}}, {{2,1.4}}, {{-0.1,1.1}}}};
    QuadBasis original;
    PetscCall(original.Initialize(q));
    const Point p = Interior(q, PetscReal(0.37), PetscReal(0.58));
    const PetscReal c = std::cos(PetscReal(0.7)), s = std::sin(PetscReal(0.7));
    for (PetscReal scale : {1/large, PetscReal(1), large}) {
        const auto transform = [&](const Point& x) {
            return Point{{scale*(c*x.p[0]-s*x.p[1]+4), scale*(s*x.p[0]+c*x.p[1]-3)}};
        };
        QuadVertices moved;
        for (int i = 0; i < 4; ++i) moved[i] = transform(q[i]);
        QuadBasis changed;
        PetscCall(changed.Initialize(moved));
        for (Query kind : queries) for (int e = 0; e < Count(kind); ++e) {
            ScalarBasisValue a, b;
            PetscCall(Evaluate(original, kind, e, p, a));
            PetscCall(Evaluate(changed, kind, e, transform(p), b));
            const PetscReal valueScale = HasLengthUnits(kind) ? scale : PetscReal(1);
            const PetscReal gradientScale = HasLengthUnits(kind) ? PetscReal(1) : 1/scale;
            PetscCall(Near(b.value, a.value*valueScale, valueScale, "Value under rotation/translation/scaling"));
            PetscCall(Near(b.gradient.p[0], gradientScale*(c*a.gradient.p[0]-s*a.gradient.p[1]),
                            gradientScale, "Transformed physical gradient x"));
            PetscCall(Near(b.gradient.p[1], gradientScale*(s*a.gradient.p[0]+c*a.gradient.p[1]),
                            gradientScale, "Transformed physical gradient y"));
        }
    }
    PetscFunctionReturn(PETSC_SUCCESS);
}

template<class Call>
PetscErrorCode RejectedValue(Call call, PetscErrorCode expected)
{
    PetscFunctionBeginUser;
    ScalarBasisValue value{11, {{12,13}}};
    PetscCall(ExpectError([&] { return call(value); }, expected));
    PetscCall(Require(value.value == 11 && value.gradient.p[0] == 12 && value.gradient.p[1] == 13,
                       "Rejected evaluation modified output"));
    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode CheckFailuresAndOwnership()
{
    PetscFunctionBeginUser;
    const QuadVertices good{{{{0,0}}, {{2,0.1}}, {{1.8,1.4}}, {{-0.2,1}}}};
    const Point p = Interior(good, PetscReal(0.3), PetscReal(0.4));
    QuadBasis basis;
    PetscCall(Require(!basis.IsInitialized(), "Default basis state"));
    QuadVertices output = good;
    PetscCall(ExpectError([&] { return basis.GetCorners(output); }, PETSC_ERR_ARG_WRONG));
    for (int i = 0; i < 4; ++i) for (int d = 0; d < 2; ++d)
        PetscCall(Require(output[i].p[d] == good[i].p[d], "Rejected corner query modified output"));
    BasisEdgeGeometry edge;
    edge.length = -99;
    PetscCall(ExpectError([&] { return basis.GetEdge(CellSide::Bottom, edge); }, PETSC_ERR_ARG_WRONG));
    PetscCall(ExpectError([&] { return basis.GetDiagonal(0, edge); }, PETSC_ERR_ARG_WRONG));
    PetscCall(Require(edge.length == -99, "Rejected geometry query modified output"));
    for (Query kind : queries)
        PetscCall(RejectedValue([&](ScalarBasisValue& v) { return Evaluate(basis, kind, 0, p, v); }, PETSC_ERR_ARG_WRONG));

    QuadVertices input = good;
    PetscCall(basis.Initialize(input));
    input[0].p[0] = 100; // The basis must own its input geometry.
    ScalarBasisValue before, after;
    PetscCall(basis.EdgeBubble(CellSide::Bottom, p, before));
    for (int variant = 0; variant < 8; ++variant) {
        auto bad = good;
        if (variant == 0) std::swap(bad[1], bad[3]); // Clockwise.
        if (variant == 1) bad[1] = bad[0];          // Repeated vertex.
        if (variant == 2) bad[2] = {{0.1,0.1}};    // Concave.
        if (variant == 3) std::swap(bad[1], bad[2]); // Self-intersecting.
        if (variant == 4) bad[1] = Mix(bad[0], bad[2], PetscReal(0.5)); // Collinear.
        if (variant == 5) bad[0].p[0] = std::numeric_limits<PetscReal>::quiet_NaN();
        if (variant == 6) bad[0].p[1] = std::numeric_limits<PetscReal>::infinity();
        if (variant == 7) bad = {{{{0,0}}, {{1,0}}, {{1,1}}, {{0,PETSC_MACHINE_EPSILON}}}};
        QuadBasis empty;
        PetscCall(ExpectError([&] { return empty.Initialize(bad); }));
        PetscCall(Require(!empty.IsInitialized(), "Failed initialization published partial state"));
        PetscCall(ExpectError([&] { return basis.Initialize(bad); }));
        PetscCall(basis.GetCorners(output));
        for (int i = 0; i < 4; ++i) for (int d = 0; d < 2; ++d)
            PetscCall(Require(output[i].p[d] == good[i].p[d], "Failed reinitialization changed corners"));
        PetscCall(basis.EdgeBubble(CellSide::Bottom, p, after));
        PetscCall(Require(after.value == before.value && after.gradient.p[0] == before.gradient.p[0] &&
                           after.gradient.p[1] == before.gradient.p[1], "Failed reinitialization changed cached values"));
    }
    MeshInfo emptyMesh;
    PetscCall(ExpectError([&] { return basis.Initialize(emptyMesh, {0,0}); }, PETSC_ERR_ARG_WRONG));
    for (int invalid : {-1, 4}) {
        for (Query kind : queries) if (kind != Query::Polynomial)
            PetscCall(RejectedValue([&](ScalarBasisValue& v) { return Evaluate(basis, kind, invalid, p, v); }, PETSC_ERR_ARG_OUTOFRANGE));
        PetscCall(ExpectError([&] { return basis.GetEdge(Side(invalid), edge); }, PETSC_ERR_ARG_OUTOFRANGE));
    }
    PetscCall(ExpectError([&] { return basis.GetDiagonal(2, edge); }, PETSC_ERR_ARG_OUTOFRANGE));
    PetscCall(Require(edge.length == -99, "Rejected index modified geometry output"));
    for (int e = 0; e < 4; ++e) for (int other : {e, (e+1)%4, (e+3)%4}) {
        PetscCall(RejectedValue([&](ScalarBasisValue& v) { return basis.OppositeEdgeContrast(Side(e), Side(other), p, v); }, PETSC_ERR_ARG_WRONG));
        PetscCall(RejectedValue([&](ScalarBasisValue& v) { return basis.ScaledDistanceDifference(Side(e), Side(other), p, v); }, PETSC_ERR_ARG_WRONG));
    }
    for (PetscReal bad : {std::numeric_limits<PetscReal>::quiet_NaN(), std::numeric_limits<PetscReal>::infinity()})
        for (int d = 0; d < 2; ++d) for (Query kind : queries) {
            Point invalid = p;
            invalid.p[d] = bad;
            PetscCall(RejectedValue([&](ScalarBasisValue& v) { return Evaluate(basis, kind, 0, invalid, v); }, PETSC_ERR_ARG_OUTOFRANGE));
        }
    // Opposite signed distances cancel on an exterior line. Compute a point
    // on it using the independent determinant oracle, not production gradients.
    const Point origin{{0,0}};
    const auto sum = [&](const Point& x) { return Lambda(good, 0, x) + Lambda(good, 2, x); };
    const PetscReal dx = sum({{1,0}})-sum(origin), dy = sum({{0,1}})-sum(origin);
    Point singular = origin;
    const int axis = PetscAbsReal(dx) > PetscAbsReal(dy) ? 0 : 1;
    singular.p[axis] = -sum(origin)/(axis == 0 ? dx : dy);
    for (Query kind : {Query::Blend, Query::Contrast, Query::Bubble})
        PetscCall(RejectedValue([&](ScalarBasisValue& v) { return Evaluate(basis, kind, 0, singular, v); }, PETSC_ERR_ARG_OUTOFRANGE));
    // Finite exterior evaluation remains supported when the expression exists.
    const Point exterior{{-0.3,-0.2}};
    for (Query kind : queries) {
        PetscCall(Evaluate(basis, kind, 0, exterior, after));
        PetscCall(Near(after.value, ReferenceValue(good, kind, 0, exterior), 3, "Finite exterior evaluation"));
    }
    QuadBasis copy = basis;
    const QuadVertices replacement{{{{0,0}}, {{20,0}}, {{20,1}}, {{0,1}}}};
    PetscCall(basis.Initialize(replacement));
    PetscCall(copy.EdgeBubble(CellSide::Bottom, p, after));
    PetscCall(Require(after.value == before.value, "Copied basis shares mutable state"));
    PetscCall(basis.EdgeBlend(CellSide::Right, {{5,0.3}}, after));
    PetscCall(Near(after.value, PetscReal(0.25), 1, "Successful reinitialization replaces geometry"));
    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode RunUnitTests()
{
    PetscFunctionBeginUser;
    const std::array<QuadVertices, 6> fixtures{{
        {{{{0,0}}, {{2,0}}, {{2,1}}, {{0,1}}}},
        {{{{-0.7,0.2}}, {{1.7,0.2}}, {{1.7,1.3}}, {{-0.7,1.3}}}},
        {{{{0,0}}, {{1.6,1.2}}, {{1,2}}, {{-0.6,0.8}}}}, // Rotated rectangle.
        {{{{0,0}}, {{2,0}}, {{2.5,1}}, {{0.5,1}}}},      // Parallelogram.
        {{{{0,0}}, {{2,0}}, {{1.6,1.3}}, {{0.2,1.3}}}},  // Trapezoid.
        {{{{0,0}}, {{2,0.2}}, {{1.8,1.6}}, {{-0.15,1.3}}}} // General convex quad.
    }};
    for (std::size_t f = 0; f < fixtures.size(); ++f) {
        PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Checking basis fixture %d\n", static_cast<int>(f)));
        PetscCall(CheckCell(fixtures[f]));
    }
    // Cap the exponent for the configured PetscReal precision. In double
    // precision these include 1e+-150 scales and 1e300 aspect ratios.
    const int exponent = std::min(150, std::numeric_limits<PetscReal>::max_exponent10/2 - 4);
    const PetscReal large = std::pow(PetscReal(10), exponent), small = 1/large;
    PetscCall(CheckRectangle(2, 1));
    PetscCall(CheckRectangle(small, 2*small));
    PetscCall(CheckRectangle(2*large, large));
    PetscCall(CheckRectangle(large, small));
    PetscCall(CheckRectangle(small, large));
    PetscCall(CheckTransformations(large));
    PetscCall(CheckFailuresAndOwnership());
    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode MakeSnapshot(MPI_Comm comm, PetscInt nx, PetscInt ny,
                             PetscInt px, PetscInt py, PetscInt kind,
                             const MeshParam& parameters, MeshInfo& info)
{
    PetscFunctionBeginUser;
    DM dm = nullptr;
    Vec vertices = nullptr;
    PetscCall(DMDACreate2d(comm, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE, DMDA_STENCIL_BOX,
                          nx, ny, px, py, 2, 1, nullptr, nullptr, &dm));
    // Keep the test's grid fixed; do not read unrelated -da_* options.
    PetscCall(DMSetUp(dm));
    PetscCall(DMCreateGlobalVector(dm, &vertices));
    if (kind == 0) PetscCall(CreateFullMesh(dm, vertices, parameters));
    else if (kind == 1) PetscCall(LogicRectMesh(dm, vertices, parameters));
    else PetscCall(RefineMesh(dm, vertices, parameters));
    PetscCall(BuildMeshInfo(dm, vertices, info));
    PetscCall(VecDestroy(&vertices));
    PetscCall(DMDestroy(&dm));
    // Subsequent tests also verify independence from the original DM/Vec.
    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode CompareCell(const MeshInfo& distributed, const MeshInfo& serial, MeshIndex cell)
{
    PetscFunctionBeginUser;
    QuadBasis basis, reference;
    PetscCall(basis.Initialize(distributed, cell));
    PetscCall(reference.Initialize(serial, cell));
    QuadVertices q, expected;
    PetscCall(basis.GetCorners(q));
    PetscCall(reference.GetCorners(expected));
    for (int e = 0; e < 4; ++e) for (int d = 0; d < 2; ++d)
        PetscCall(Require(q[e].p[d] == expected[e].p[d], "MPI partition changed cell coordinates"));
    const Point extents = Extents(q);
    const PetscReal lengthScale = std::max(extents.p[0], extents.p[1]);
    for (const auto& st : {std::array<PetscReal,2>{{0.23,0.31}},
                           std::array<PetscReal,2>{{0.5,0.5}},
                           std::array<PetscReal,2>{{0.78,0.67}}}) {
        const Point p = Interior(q, st[0], st[1]);
        for (Query kind : queries) for (int e = 0; e < Count(kind); ++e) {
            ScalarBasisValue a, b;
            PetscCall(Evaluate(basis, kind, e, p, a));
            PetscCall(Evaluate(reference, kind, e, p, b));
            const PetscReal scale = HasLengthUnits(kind) ? lengthScale : PetscReal(1);
            PetscCall(Near(a.value, b.value, scale, "Distributed vs serial basis value"));
            PetscCall(Near(a.value, ReferenceValue(q, kind, e, p), scale, "Generated-cell independent oracle"));
            for (int d = 0; d < 2; ++d)
                PetscCall(Near(a.gradient.p[d], b.gradient.p[d], scale/extents.p[d], "Distributed vs serial gradient"));
        }
    }
    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode CheckMesh(PetscInt nx, PetscInt ny, PetscInt px, PetscInt py,
                         PetscInt kind, const MeshParam& parameters, PetscMPIInt ranks)
{
    PetscFunctionBeginUser;
    MeshInfo distributed, serial;
    PetscCall(MakeSnapshot(PETSC_COMM_WORLD, nx, ny, px, py, kind, parameters, distributed));
    PetscCall(MakeSnapshot(PETSC_COMM_SELF, nx, ny, 1, 1, kind, parameters, serial));
    const auto available = distributed.AvailableCells();
    int localGhostCells = 0;
    for (PetscInt j = available.begin.j; j < available.end.j; ++j)
        for (PetscInt i = available.begin.i; i < available.end.i; ++i) {
            PetscCall(CompareCell(distributed, serial, {i,j}));
            if (!distributed.OwnsCell({i,j})) ++localGhostCells;
        }

    std::vector<int> owners(static_cast<std::size_t>(serial.CellCount()), 0);
    std::vector<int> incidence(static_cast<std::size_t>(serial.EdgeCount()), 0);
    // Per-edge sums: signed scalar trace, signed canonical tangential derivative,
    // outward normal x, outward normal y. Interior sums must vanish, including
    // when the two cells belong to different MPI ranks.
    std::vector<PetscReal> jumps(4*incidence.size(), 0);
    const auto owned = distributed.OwnedCells();
    int localCells = 0;
    for (PetscInt j = owned.begin.j; j < owned.end.j; ++j)
        for (PetscInt i = owned.begin.i; i < owned.end.i; ++i) {
            ++localCells;
            PetscInt cellId;
            PetscCall(distributed.CellId({i,j}, cellId));
            ++owners[static_cast<std::size_t>(cellId)];
            QuadBasis basis;
            PetscCall(basis.Initialize(distributed, {i,j}));
            std::array<OrientedEdge,4> edges;
            PetscCall(distributed.GetCellEdges({i,j}, edges));
            for (int e = 0; e < 4; ++e) {
                const auto id = static_cast<std::size_t>(edges[e].id);
                const PetscReal sign = static_cast<PetscReal>(edges[e].direction);
                ++incidence[id];
                BasisEdgeGeometry geometry;
                PetscCall(basis.GetEdge(Side(e), geometry));
                EdgeVertices canonical;
                PetscCall(distributed.GetEdgeVertices(edges[e].id, canonical));
                const PetscReal length = Length(canonical[0], canonical[1]);
                for (int d = 0; d < 2; ++d) {
                    PetscCall(Near(geometry.tangent.p[d], sign*(canonical[1].p[d]-canonical[0].p[d])/length,
                                    1, "Cell vs canonical edge orientation"));
                    jumps[4*id+2+d] += geometry.normal.p[d];
                }
                const PetscReal s = PetscReal(0.27);
                const Point p = Mix(canonical[0], canonical[1], s);
                ScalarBasisValue bubble;
                PetscCall(basis.EdgeBubble(Side(e), p, bubble));
                const PetscReal tangential =
                    (bubble.gradient.p[0]*(canonical[1].p[0]-canonical[0].p[0]) +
                     bubble.gradient.p[1]*(canonical[1].p[1]-canonical[0].p[1]))/length;
                PetscCall(Near(bubble.value, 4*s*(1-s), 1, "Generated mesh bubble trace"));
                PetscCall(Near(tangential*length, 4*(1-2*s), 1, "Generated mesh tangential derivative"));
                jumps[4*id] += sign*bubble.value;
                jumps[4*id+1] += sign*tangential*length;
            }
        }
    // Collectives are outside all local-cell loops: empty owners participate.
    std::vector<int> allOwners(owners.size()), allIncidence(incidence.size());
    std::vector<PetscReal> allJumps(jumps.size());
    PetscCallMPI(MPI_Allreduce(owners.data(), allOwners.data(), static_cast<int>(owners.size()), MPI_INT, MPI_SUM, PETSC_COMM_WORLD));
    PetscCallMPI(MPI_Allreduce(incidence.data(), allIncidence.data(), static_cast<int>(incidence.size()), MPI_INT, MPI_SUM, PETSC_COMM_WORLD));
    PetscCallMPI(MPI_Allreduce(jumps.data(), allJumps.data(), static_cast<int>(jumps.size()), MPIU_REAL, MPI_SUM, PETSC_COMM_WORLD));
    for (int count : allOwners) PetscCall(Require(count == 1, "Each cell must be evaluated by exactly one owner"));
    for (PetscInt e = 0; e < serial.EdgeCount(); ++e) {
        EdgeTopology topology;
        PetscCall(serial.GetEdgeTopology(e, topology));
        const auto id = static_cast<std::size_t>(e);
        PetscCall(Require(allIncidence[id] == (topology.IsBoundary() ? 1 : 2), "Global edge incidence"));
        if (!topology.IsBoundary()) for (int k = 0; k < 4; ++k)
            PetscCall(Near(allJumps[4*id+k], 0, 1, "Interior edge trace/normal cancellation"));
    }
    int localEmpty = localCells == 0 ? 1 : 0, emptyRanks = 0, ghostCells = 0;
    PetscCallMPI(MPI_Allreduce(&localEmpty, &emptyRanks, 1, MPI_INT, MPI_SUM, PETSC_COMM_WORLD));
    PetscCallMPI(MPI_Allreduce(&localGhostCells, &ghostCells, 1, MPI_INT, MPI_SUM, PETSC_COMM_WORLD));
    if (nx == 2 && ny == 2)
        PetscCall(Require(emptyRanks == ranks-1, "Single-cell case must exercise empty cell owners"));
    if (ranks > 1) PetscCall(Require(ghostCells > 0, "MPI test did not exercise ghost cells"));

    QuadBasis retained;
    const MeshIndex probe = available.begin;
    PetscCall(retained.Initialize(distributed, probe));
    QuadVertices corners, after;
    PetscCall(retained.GetCorners(corners));
    PetscCall(ExpectError([&] { return retained.Initialize(distributed, {-1,0}); }));
    PetscCall(ExpectError([&] { return retained.Initialize(distributed, {nx-1,0}); }));
    // Also reject a globally valid cell whose four corners are unavailable here.
    bool foundUnavailable = false;
    for (PetscInt j = 0; j < ny-1 && !foundUnavailable; ++j)
        for (PetscInt i = 0; i < nx-1 && !foundUnavailable; ++i)
            if (!distributed.HasCell({i,j})) {
                PetscCall(ExpectError([&] { return retained.Initialize(distributed, {i,j}); }));
                foundUnavailable = true;
            }
    distributed = MeshInfo{}; // Basis retains no pointer into MeshInfo.
    PetscCall(retained.GetCorners(after));
    for (int i = 0; i < 4; ++i) for (int d = 0; d < 2; ++d)
        PetscCall(Require(after[i].p[d] == corners[i].p[d], "Basis did not preserve its geometry snapshot"));
    ScalarBasisValue value;
    const Point p = Interior(corners, PetscReal(0.4), PetscReal(0.6));
    PetscCall(retained.EdgeBubble(CellSide::Top, p, value));
    PetscCall(Near(value.value, ReferenceValue(corners, Query::Bubble, 2, p), 1, "Basis after MeshInfo destruction"));
    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode Run()
{
    PetscFunctionBeginUser;
    PetscInt kind = -1, expectedRanks = 1, px = 1, py = 1;
    PetscCall(PetscOptionsGetInt(nullptr, nullptr, "-basis_mesh_type", &kind, nullptr));
    PetscCall(PetscOptionsGetInt(nullptr, nullptr, "-expected_ranks", &expectedRanks, nullptr));
    PetscCall(PetscOptionsGetInt(nullptr, nullptr, "-mesh_px", &px, nullptr));
    PetscCall(PetscOptionsGetInt(nullptr, nullptr, "-mesh_py", &py, nullptr));
    PetscMPIInt ranks;
    PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD, &ranks));
    PetscCheck(expectedRanks == ranks, PETSC_COMM_WORLD, PETSC_ERR_ARG_WRONG,
               "Expected %d MPI ranks but PETSc sees %d. Check that the launcher matches PETSc's MPI.",
               static_cast<int>(expectedRanks), static_cast<int>(ranks));
    PetscCall(Require(kind >= -1 && kind <= 2, "basis_mesh_type must be -1 (unit), 0 (rectangle), 1 (quadrilateral), or 2 (stretched)"));
    if (kind == -1) {
        PetscCall(Require(ranks == 1, "Run the formula suite on one rank"));
        PetscCall(RunUnitTests());
        PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Basis unit tests passed\n"));
    } else {
        PetscCall(Require((px == 1 || px == 2) && (py == 1 || py == 2) && px*py == ranks,
                           "Use a matching 1x1, 2x1, 1x2 or 2x2 process grid"));
        MeshParam parameters;
        parameters.xstart = -0.75; parameters.ystart = 0.2;
        parameters.L = 2.5; parameters.H = 1.3;
        parameters.seed = 7; parameters.perturbation = 0.20;
        PetscCall(CheckMesh(13, 9, px, py, kind, parameters, ranks));
        parameters.xstart = 3.5; parameters.ystart = -2.25;
        parameters.L = 17; parameters.H = 0.08;
        parameters.seed = 991; parameters.perturbation = 0.249;
        PetscCall(CheckMesh(10, 7, px, py, kind, parameters, ranks));
        // 2x2 VERTICES is one cell, with enough vertices for all supported grids.
        PetscCall(CheckMesh(2, 2, px, py, kind, parameters, ranks));
        PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Basis mesh type %d passed on %d rank(s)\n",
                              static_cast<int>(kind), static_cast<int>(ranks)));
    }
    PetscFunctionReturn(PETSC_SUCCESS);
}

} // namespace

int main(int argc, char** argv)
{
    const PetscErrorCode error = PetscInitialize(&argc, &argv, nullptr,
        "Basis tests: default formula suite; -basis_mesh_type 0/1/2 selects generated mesh tests.\n");
    if (error) return static_cast<int>(error);
    // Abort WORLD on any unexpected rank-local error. Other ranks may already
    // be waiting in a collective; they must not continue or hang until timeout.
    PetscCallAbort(PETSC_COMM_WORLD, Run());
    return static_cast<int>(PetscFinalize());
}
