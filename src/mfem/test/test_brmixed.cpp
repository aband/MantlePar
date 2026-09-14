#include "brmixed.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <limits>
#include <vector>

namespace {

// Use PETSc checks, not assert(): every test remains active in Release builds.
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
    return {{a.p[0] + t*(b.p[0] - a.p[0]),
             a.p[1] + t*(b.p[1] - a.p[1])}};
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

PetscErrorCode Compare(const BRBasisValue& a, const BRBasisValue& b,
                       const Point& extents)
{
    PetscFunctionBeginUser;
    for (int c = 0; c < 2; ++c) {
        PetscCall(Near(a.value.p[c], b.value.p[c], 1, "Vector value comparison"));
        for (int d = 0; d < 2; ++d)
            PetscCall(Near(a.gradient[2*c+d], b.gradient[2*c+d], 1/extents.p[d],
                           "Physical gradient comparison"));
    }
    PetscFunctionReturn(PETSC_SUCCESS);
}

// Exact comparisons here check output preservation, not approximate mathematics.
PetscErrorCode Unchanged(const BRBasisValue& a, const BRBasisValue& b)
{
    PetscFunctionBeginUser;
    for (int c = 0; c < 2; ++c)
        PetscCall(Require(a.value.p[c] == b.value.p[c], "Rejected call changed vector output"));
    PetscCall(Require(a.gradient == b.gradient, "Rejected call changed gradient output"));
    PetscFunctionReturn(PETSC_SUCCESS);
}
PetscErrorCode Unchanged(const BRMixed::Values& a, const BRMixed::Values& b)
{
    PetscFunctionBeginUser;
    for (PetscInt k = 0; k < BRMixed::ElementDofs; ++k)
        PetscCall(Unchanged(a[k], b[k]));
    PetscFunctionReturn(PETSC_SUCCESS);
}
PetscErrorCode Unchanged(const ScalarBasisValue& a, const ScalarBasisValue& b)
{
    PetscFunctionBeginUser;
    PetscCall(Require(a.value == b.value && a.gradient.p[0] == b.gradient.p[0] &&
                       a.gradient.p[1] == b.gradient.p[1], "Rejected call changed scalar output"));
    PetscFunctionReturn(PETSC_SUCCESS);
}
PetscErrorCode SameCorners(const QuadVertices& a, const QuadVertices& b)
{
    PetscFunctionBeginUser;
    for (int i = 0; i < 4; ++i) for (int d = 0; d < 2; ++d)
        PetscCall(Require(a[i].p[d] == b[i].p[d], "Corner coordinates/order changed"));
    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode CheckPoint(const BRMixed& basis, const QuadVertices& q,
                          const Point& p, bool finiteDifference)
{
    PetscFunctionBeginUser;
    const Point extents = Extents(q);
    BRMixed::Values all;
    PetscCall(basis.EvaluateAll(p, all));
    // In particular, exercise every edge DOF through the single evaluator.
    // The legacy single/batch mismatch made two edge functions identically zero.
    for (PetscInt k = 0; k < BRMixed::ElementDofs; ++k) {
        BRBasisValue one;
        PetscCall(basis.Evaluate(p, k, one));
        PetscCall(Compare(one, all[k], extents));
    }

    PetscReal sum = 0;
    Point gradientSum{}, position{};
    std::array<PetscReal,4> coordinateGradient{};
    for (PetscInt i = 0; i < 4; ++i) {
        ScalarBasisValue vertex, bubble;
        Point normal;
        PetscCall(basis.EvaluateVertex(i, p, vertex));
        PetscCall(basis.EvaluateBubble(Side(i), p, bubble));
        PetscCall(basis.GetEdgeNormal(Side(i), normal));
        sum += vertex.value;
        for (int c = 0; c < 2; ++c) {
            // Subtract the origin before summing to avoid a translation-dependent
            // cancellation error in the test's coordinate reproduction oracle.
            const PetscReal coordinate = q[i].p[c] - q[0].p[c];
            position.p[c] += coordinate * vertex.value;
            gradientSum.p[c] += vertex.gradient.p[c];
            for (int d = 0; d < 2; ++d)
                coordinateGradient[2*c+d] += coordinate * vertex.gradient.p[d];
            for (int component = 0; component < 2; ++component) {
                PetscCall(Near(all[i+4*c].value.p[component], component == c ? vertex.value : 0,
                               1, "Vertex vector component ordering"));
                for (int d = 0; d < 2; ++d)
                    PetscCall(Near(all[i+4*c].gradient[2*component+d],
                                   component == c ? vertex.gradient.p[d] : 0,
                                   1/extents.p[d], "Vertex gradient component ordering"));
            }
            PetscCall(Near(all[i+8].value.p[c], bubble.value*normal.p[c], 1,
                           "Scalar bubble to vector normalization"));
            for (int d = 0; d < 2; ++d)
                PetscCall(Near(all[i+8].gradient[2*c+d], bubble.gradient.p[d]*normal.p[c],
                               1/extents.p[d], "Scalar bubble to vector gradient"));
        }
    }
    PetscCall(Near(sum, 1, 1, "Vertex partition of unity"));
    for (int c = 0; c < 2; ++c) {
        PetscCall(Near(gradientSum.p[c], 0, 1/extents.p[c], "Gradient sum must vanish"));
        PetscCall(Near(position.p[c], p.p[c]-q[0].p[c], extents.p[c],
                       "Physical affine coordinate reproduction"));
        for (int d = 0; d < 2; ++d)
            PetscCall(Near(coordinateGradient[2*c+d], c == d ? 1 : 0,
                           extents.p[c]/extents.p[d], "Affine coordinate derivative"));
    }

    if (finiteDifference) {
        const PetscReal step = std::cbrt(PetscReal(PETSC_MACHINE_EPSILON));
        const PetscReal tolerance = 256*step*step;
        ScalarBasisValue supplement;
        PetscCall(basis.EvaluateSupplement(p, supplement));
        for (int d = 0; d < 2; ++d) {
            Point plus = p, minus = p;
            plus.p[d] += step*extents.p[d];
            minus.p[d] -= step*extents.p[d];
            const PetscReal width = plus.p[d] - minus.p[d];
            PetscCall(Require(width > 0, "Finite-difference step was rounded away"));
            BRMixed::Values a, b;
            PetscCall(basis.EvaluateAll(plus, a));
            PetscCall(basis.EvaluateAll(minus, b));
            for (PetscInt k = 0; k < BRMixed::ElementDofs; ++k) for (int c = 0; c < 2; ++c)
                PetscCall(Near(all[k].gradient[2*c+d], (a[k].value.p[c]-b[k].value.p[c])/width,
                               1/extents.p[d], "Vector gradient vs finite difference", tolerance));
            ScalarBasisValue sp, sm;
            PetscCall(basis.EvaluateSupplement(plus, sp));
            PetscCall(basis.EvaluateSupplement(minus, sm));
            PetscCall(Near(supplement.gradient.p[d], (sp.value-sm.value)/width,
                           1/extents.p[d], "Supplement gradient vs finite difference", tolerance));
        }
    }
    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode CheckCell(const QuadVertices& q)
{
    PetscFunctionBeginUser;
    BRMixed basis, fromGeometry;
    QuadBasis geometry;
    PetscCall(basis.Initialize(q));
    PetscCall(Require(basis.IsInitialized(), "Successful initialization must set state"));
    PetscCall(geometry.Initialize(q));
    PetscCall(fromGeometry.Initialize(geometry));
    QuadVertices stored;
    PetscCall(basis.GetCorners(stored));
    PetscCall(SameCorners(stored, q));

    std::array<BRMixed::Values,4> atVertex;
    for (PetscInt i = 0; i < 4; ++i) {
        PetscCall(basis.EvaluateAll(q[i], atVertex[i]));
        for (PetscInt k = 0; k < BRMixed::ElementDofs; ++k) for (int c = 0; c < 2; ++c)
            PetscCall(Near(atVertex[i][k].value.p[c], k == i+4*c ? 1 : 0,
                           1, "Vertex DOF identity / zero bubbles at vertices"));
    }

    for (int e = 0; e < 4; ++e) {
        const int next = (e+1)%4;
        const PetscReal length = Length(q[e], q[next]);
        const Point tangent{{(q[next].p[0]-q[e].p[0])/length,
                              (q[next].p[1]-q[e].p[1])/length}};
        const PetscReal sign = (e == 0 || e == 3) ? -1 : 1;
        const Point expectedNormal{{sign*tangent.p[1], -sign*tangent.p[0]}};
        Point normal;
        PetscCall(basis.GetEdgeNormal(Side(e), normal));
        for (int c = 0; c < 2; ++c)
            PetscCall(Near(normal.p[c], expectedNormal.p[c], 1, "Oriented BR edge normal"));

        BRMixed::Values atMidpoint;
        PetscCall(basis.EvaluateAll(Mix(q[e], q[next], PetscReal(0.5)), atMidpoint));
        // Together with the eight vertex functionals above, these four rows
        // check the complete 12x12 DOF matrix against the identity.
        for (PetscInt k = 0; k < BRMixed::ElementDofs; ++k) {
            PetscReal functional = 0;
            for (int c = 0; c < 2; ++c)
                functional += expectedNormal.p[c] * (atMidpoint[k].value.p[c] -
                    PetscReal(0.5)*(atVertex[e][k].value.p[c]+atVertex[next][k].value.p[c]));
            PetscCall(Near(functional, k == 8+e ? 1 : 0, 1, "Edge DOF identity"));
        }

        for (PetscReal s : {PetscReal(0), PetscReal(0.13), PetscReal(0.5), PetscReal(0.76), PetscReal(1)}) {
            const Point p = Mix(q[e], q[next], s);
            BRMixed::Values values;
            ScalarBasisValue supplement;
            PetscCall(basis.EvaluateAll(p, values));
            PetscCall(basis.EvaluateSupplement(p, supplement));
            const PetscReal parity = e%2 ? -1 : 1;
            PetscCall(Near(supplement.value, parity*(1-2*s), 1, "Corrected supplement edge trace"));
            PetscCall(Near(supplement.gradient.p[0]*tangent.p[0] +
                           supplement.gradient.p[1]*tangent.p[1], -2*parity/length,
                           1/length, "Supplement tangential derivative"));
            for (int i = 0; i < 4; ++i) {
                const PetscReal value = i == e ? 1-s : (i == next ? s : 0);
                const PetscReal derivative = i == e ? -1 : (i == next ? 1 : 0);
                for (int c = 0; c < 2; ++c) {
                    PetscCall(Near(values[i+4*c].value.p[c], value, 1, "Linear vertex edge trace"));
                    PetscCall(Near(values[i+4*c].gradient[2*c]*tangent.p[0] +
                                   values[i+4*c].gradient[2*c+1]*tangent.p[1], derivative/length,
                                   1/length, "Vertex tangential derivative"));
                    PetscCall(Near(values[i+8].value.p[c], (i == e ? 4*s*(1-s) : 0)*expectedNormal.p[c],
                                   1, "Vector bubble edge trace"));
                    PetscCall(Near(values[i+8].gradient[2*c]*tangent.p[0] +
                                   values[i+8].gradient[2*c+1]*tangent.p[1],
                                   (i == e ? 4*(1-2*s)/length : 0)*expectedNormal.p[c],
                                   1/length, "Bubble tangential derivative"));
                }
                ScalarBasisValue bubble;
                PetscCall(basis.EvaluateBubble(Side(i), p, bubble));
                PetscCall(Near(bubble.value, i == e ? 4*s*(1-s) : 0, 1, "Scalar bubble normalization"));
            }
        }
    }

    for (PetscReal s : {PetscReal(0.18), PetscReal(0.44), PetscReal(0.77)})
        for (PetscReal t : {PetscReal(0.23), PetscReal(0.5), PetscReal(0.82)}) {
            const Point p = Interior(q, s, t);
            PetscCall(CheckPoint(basis, q, p, true));
            BRMixed::Values a, b;
            PetscCall(basis.EvaluateAll(p, a));
            PetscCall(fromGeometry.EvaluateAll(p, b));
            for (PetscInt k = 0; k < BRMixed::ElementDofs; ++k)
                PetscCall(Compare(a[k], b[k], Extents(q)));
        }
    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode CheckRectangle(PetscReal w, PetscReal h)
{
    PetscFunctionBeginUser;
    BRMixed basis;
    PetscCall(basis.Initialize(QuadVertices{{{{0,0}}, {{w,0}}, {{w,h}}, {{0,h}}}}));
    for (PetscReal s : {PetscReal(0.17), PetscReal(0.5), PetscReal(0.83)})
        for (PetscReal t : {PetscReal(0.21), PetscReal(0.5), PetscReal(0.67)}) {
            BRMixed::Values values;
            const Point p{{s*w, t*h}};
            PetscCall(basis.EvaluateAll(p, values));
            // Independent Q1 and edge-bubble formulas on [0,w] x [0,h].
            const std::array<PetscReal,4> nodes{{(1-s)*(1-t), s*(1-t), s*t, (1-s)*t}};
            const std::array<Point,4> dn{{{{-(1-t)/w, -(1-s)/h}}, {{(1-t)/w, -s/h}},
                                          {{t/w, s/h}}, {{-t/w, (1-s)/h}}}};
            const std::array<PetscReal,4> bubbles{{4*s*(1-s)*(1-t), 4*t*(1-t)*s,
                                                   4*s*(1-s)*t, 4*t*(1-t)*(1-s)}};
            const std::array<Point,4> db{{{{4*(1-2*s)*(1-t)/w, -4*s*(1-s)/h}},
                                          {{4*t*(1-t)/w, 4*(1-2*t)*s/h}},
                                          {{4*(1-2*s)*t/w, 4*s*(1-s)/h}},
                                          {{-4*t*(1-t)/w, 4*(1-2*t)*(1-s)/h}}}};
            for (int i = 0; i < 4; ++i) for (int c = 0; c < 2; ++c) {
                BRBasisValue expected;
                expected.value.p[c] = nodes[i];
                expected.gradient[2*c] = dn[i].p[0];
                expected.gradient[2*c+1] = dn[i].p[1];
                PetscCall(Compare(values[i+4*c], expected, Point{{w,h}}));
            }
            for (int e = 0; e < 4; ++e) {
                const int component = e%2 == 0 ? 1 : 0; // Horizontal up; vertical right.
                BRBasisValue expected, single;
                expected.value.p[component] = bubbles[e];
                expected.gradient[2*component] = db[e].p[0];
                expected.gradient[2*component+1] = db[e].p[1];
                PetscCall(Compare(values[8+e], expected, Point{{w,h}}));
                PetscCall(basis.Evaluate(p, 8+e, single));
                PetscCall(Compare(single, expected, Point{{w,h}}));
            }
            ScalarBasisValue supplement;
            PetscCall(basis.EvaluateSupplement(p, supplement));
            PetscCall(Near(supplement.value, (1-2*s)*(1-2*t), 1, "Rectangular supplement"));
            PetscCall(Near(supplement.gradient.p[0], -2*(1-2*t)/w, 1/w, "Supplement x gradient"));
            PetscCall(Near(supplement.gradient.p[1], -2*(1-2*s)/h, 1/h, "Supplement y gradient"));
        }
    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode CheckTransformations(PetscReal large)
{
    PetscFunctionBeginUser;
    const QuadVertices q{{{{-0.3,0.1}}, {{1.8,0.2}}, {{2,1.4}}, {{-0.1,1.1}}}};
    BRMixed original;
    PetscCall(original.Initialize(q));
    const Point p = Interior(q, PetscReal(0.37), PetscReal(0.58));
    BRMixed::Values a;
    PetscCall(original.EvaluateAll(p, a));
    const PetscReal cosine = std::cos(PetscReal(0.7)), sine = std::sin(PetscReal(0.7));
    const PetscReal rotation[2][2] = {{cosine, -sine}, {sine, cosine}};
    for (PetscReal scale : {1/large, PetscReal(1), large}) {
        const auto transform = [&](const Point& x) {
            return Point{{scale*(cosine*x.p[0]-sine*x.p[1]+4),
                           scale*(sine*x.p[0]+cosine*x.p[1]-3)}};
        };
        QuadVertices moved;
        for (int i = 0; i < 4; ++i) moved[i] = transform(q[i]);
        BRMixed transformed;
        PetscCall(transformed.Initialize(moved));
        BRMixed::Values b;
        PetscCall(transformed.EvaluateAll(transform(p), b));
        // Vertex functions are scalar functions times fixed global x/y axes.
        for (int i = 0; i < 4; ++i) for (int c = 0; c < 2; ++c) {
            PetscCall(Near(b[i+4*c].value.p[c], a[i+4*c].value.p[c], 1, "Transformed vertex value"));
            for (int d = 0; d < 2; ++d)
                PetscCall(Near(b[i+4*c].gradient[2*c+d],
                               (rotation[d][0]*a[i+4*c].gradient[2*c] +
                                rotation[d][1]*a[i+4*c].gradient[2*c+1])/scale,
                               1/scale, "Transformed scalar gradient"));
        }
        // Edge directions rotate with the cell: G' = R G R^T / scale.
        for (int i = 8; i < 12; ++i) for (int c = 0; c < 2; ++c) {
            PetscCall(Near(b[i].value.p[c], rotation[c][0]*a[i].value.p[0] +
                           rotation[c][1]*a[i].value.p[1], 1, "Rotated bubble vector"));
            for (int d = 0; d < 2; ++d) {
                PetscReal expected = 0;
                for (int j = 0; j < 2; ++j) for (int k = 0; k < 2; ++k)
                    expected += rotation[c][j]*a[i].gradient[2*j+k]*rotation[d][k];
                PetscCall(Near(b[i].gradient[2*c+d], expected/scale, 1/scale, "Rotated bubble gradient"));
            }
        }
    }
    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode CheckFailuresAndOwnership()
{
    PetscFunctionBeginUser;
    BRMixed basis;
    PetscCall(Require(!basis.IsInitialized(), "Default basis must be uninitialized"));
    PetscCall(Require(BRMixed::ElementDofs == 12 && BRMixed::PressureDofs == 1 &&
                       BRMixed::Pressure() == 1 && std::strcmp(BRMixed::Name(), "BR") == 0,
                       "BR metadata / constant P0 pressure"));
    const Point p{{0.4,0.6}};
    const BRBasisValue sentinel{{{11,12}}, {{13,14,15,16}}};
    const ScalarBasisValue scalarSentinel{11, {{12,13}}};
    BRMixed::Values sentinels;
    sentinels.fill(sentinel);
    auto all = sentinels;
    BRBasisValue one = sentinel;
    ScalarBasisValue scalar = scalarSentinel;
    Point normal{{11,12}};
    QuadVertices q{{{{0,0}}, {{2,0}}, {{1.8,1.4}}, {{-0.2,1}}}}, corners = q;
    PetscCall(ExpectError([&] { return basis.EvaluateAll(p, all); }, PETSC_ERR_ARG_WRONG));
    PetscCall(Unchanged(all, sentinels));
    PetscCall(ExpectError([&] { return basis.Evaluate(p, 0, one); }, PETSC_ERR_ARG_WRONG));
    PetscCall(Unchanged(one, sentinel));
    PetscCall(ExpectError([&] { return basis.EvaluateVertex(0, p, scalar); }, PETSC_ERR_ARG_WRONG));
    PetscCall(Unchanged(scalar, scalarSentinel));
    PetscCall(ExpectError([&] { return basis.EvaluateBubble(Side(0), p, scalar); }, PETSC_ERR_ARG_WRONG));
    PetscCall(Unchanged(scalar, scalarSentinel));
    PetscCall(ExpectError([&] { return basis.EvaluateSupplement(p, scalar); }, PETSC_ERR_ARG_WRONG));
    PetscCall(Unchanged(scalar, scalarSentinel));
    PetscCall(ExpectError([&] { return basis.GetEdgeNormal(Side(0), normal); }, PETSC_ERR_ARG_WRONG));
    PetscCall(Require(normal.p[0] == 11 && normal.p[1] == 12, "Uninitialized normal output changed"));
    PetscCall(ExpectError([&] { return basis.GetCorners(corners); }, PETSC_ERR_ARG_WRONG));
    PetscCall(SameCorners(corners, q));

    PetscCall(basis.Initialize(q));
    BRMixed::Values before, after;
    PetscCall(basis.EvaluateAll(p, before));
    for (int bad = 0; bad < 7; ++bad) {
        auto invalid = q;
        if (bad == 0) std::swap(invalid[1], invalid[3]); // Clockwise.
        if (bad == 1) invalid[1] = invalid[0];         // Repeated corner.
        if (bad == 2) invalid[2] = Point{{0.1,0.1}};  // Concave.
        if (bad == 3) std::swap(invalid[1], invalid[2]); // Bow tie.
        if (bad == 4) invalid[0].p[0] = std::numeric_limits<PetscReal>::quiet_NaN();
        if (bad == 5) invalid[0].p[1] = std::numeric_limits<PetscReal>::infinity();
        if (bad == 6) invalid[2] = Mix(invalid[1], invalid[3], PetscReal(0.5)); // Collinear.
        PetscCall(ExpectError([&] { return basis.Initialize(invalid); }));
        PetscCall(Require(basis.IsInitialized(), "Failed reinitialization cleared valid state"));
        PetscCall(basis.EvaluateAll(p, after));
        PetscCall(Unchanged(after, before));
        BRMixed empty;
        PetscCall(ExpectError([&] { return empty.Initialize(invalid); }));
        PetscCall(Require(!empty.IsInitialized(), "Failed initial construction set state"));
    }
    QuadBasis emptyGeometry;
    MeshInfo emptyMesh;
    PetscCall(ExpectError([&] { return basis.Initialize(emptyGeometry); }, PETSC_ERR_ARG_WRONG));
    PetscCall(ExpectError([&] { return basis.Initialize(emptyMesh, {0,0}); }, PETSC_ERR_ARG_WRONG));
    PetscCall(basis.EvaluateAll(p, after));
    PetscCall(Unchanged(after, before));

    for (PetscInt bad : {PetscInt(-1), PetscInt(12), PetscInt(100)}) {
        PetscCall(ExpectError([&] { return basis.Evaluate(p, bad, one); }, PETSC_ERR_ARG_OUTOFRANGE));
        PetscCall(Unchanged(one, sentinel));
    }
    for (int bad : {-1,4}) {
        PetscCall(ExpectError([&] { return basis.EvaluateVertex(bad, p, scalar); }, PETSC_ERR_ARG_OUTOFRANGE));
        PetscCall(Unchanged(scalar, scalarSentinel));
        PetscCall(ExpectError([&] { return basis.EvaluateBubble(Side(bad), p, scalar); }, PETSC_ERR_ARG_OUTOFRANGE));
        PetscCall(Unchanged(scalar, scalarSentinel));
        PetscCall(ExpectError([&] { return basis.GetEdgeNormal(Side(bad), normal); }, PETSC_ERR_ARG_OUTOFRANGE));
        PetscCall(Require(normal.p[0] == 11 && normal.p[1] == 12, "Invalid side changed normal output"));
    }
    for (PetscReal invalid : {std::numeric_limits<PetscReal>::quiet_NaN(),
                              std::numeric_limits<PetscReal>::infinity(),
                              -std::numeric_limits<PetscReal>::infinity()}) for (int d = 0; d < 2; ++d) {
        Point bad = p;
        bad.p[d] = invalid;
        PetscCall(ExpectError([&] { return basis.EvaluateAll(bad, all); }, PETSC_ERR_ARG_OUTOFRANGE));
        PetscCall(Unchanged(all, sentinels));
        for (PetscInt k = 0; k < BRMixed::ElementDofs; ++k) {
            PetscCall(ExpectError([&] { return basis.Evaluate(bad, k, one); }, PETSC_ERR_ARG_OUTOFRANGE));
            PetscCall(Unchanged(one, sentinel));
        }
        PetscCall(ExpectError([&] { return basis.EvaluateSupplement(bad, scalar); }, PETSC_ERR_ARG_OUTOFRANGE));
        PetscCall(Unchanged(scalar, scalarSentinel));
        for (int e = 0; e < 4; ++e) {
            PetscCall(ExpectError([&] { return basis.EvaluateVertex(e, bad, scalar); }, PETSC_ERR_ARG_OUTOFRANGE));
            PetscCall(Unchanged(scalar, scalarSentinel));
            PetscCall(ExpectError([&] { return basis.EvaluateBubble(Side(e), bad, scalar); }, PETSC_ERR_ARG_OUTOFRANGE));
            PetscCall(Unchanged(scalar, scalarSentinel));
        }
    }

    // Exterior points are not generally forbidden, but the rational basis must
    // reject a singular opposite-edge distance sum without publishing partial output.
    QuadBasis geometry;
    PetscCall(geometry.Initialize(q));
    ScalarBasisValue d0, d2;
    PetscCall(geometry.EdgeDistance(Side(0), Point{{0,0}}, d0));
    PetscCall(geometry.EdgeDistance(Side(2), Point{{0,0}}, d2));
    const Point gradient{{d0.gradient.p[0]+d2.gradient.p[0], d0.gradient.p[1]+d2.gradient.p[1]}};
    const int axis = PetscAbsReal(gradient.p[0]) > PetscAbsReal(gradient.p[1]) ? 0 : 1;
    PetscCall(Require(gradient.p[axis] != 0, "Singular-point fixture must have nonparallel edges"));
    Point singular{};
    singular.p[axis] = -(d0.value+d2.value)/gradient.p[axis];
    PetscCall(ExpectError([&] { return basis.EvaluateAll(singular, all); }, PETSC_ERR_ARG_OUTOFRANGE));
    PetscCall(Unchanged(all, sentinels));
    PetscCall(ExpectError([&] { return basis.EvaluateSupplement(singular, scalar); }, PETSC_ERR_ARG_OUTOFRANGE));
    PetscCall(Unchanged(scalar, scalarSentinel));

    BRMixed copied = basis, assigned, fromGeometry;
    assigned = basis;
    PetscCall(fromGeometry.Initialize(geometry));
    const QuadVertices replacement{{{{0,0}}, {{3,0}}, {{3,2}}, {{0,2}}}};
    PetscCall(basis.Initialize(replacement));
    PetscCall(geometry.Initialize(replacement));
    q[0].p[0] = 100; // Mutating all original geometry owners must not affect copies.
    for (const BRMixed* retained : {&copied, &assigned, &fromGeometry}) {
        PetscCall(retained->EvaluateAll(p, after));
        PetscCall(Unchanged(after, before));
    }
    PetscCall(basis.GetCorners(corners));
    PetscCall(SameCorners(corners, replacement));
    PetscCall(basis.EvaluateVertex(0, Point{{1.5,1}}, scalar));
    PetscCall(Near(scalar.value, PetscReal(0.25), 1, "Successful reinitialization replaced cached constants"));
    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode RunUnitTests()
{
    PetscFunctionBeginUser;
    const std::array<QuadVertices,6> fixtures{{
        {{{{0,0}}, {{2,0}}, {{2,1}}, {{0,1}}}},
        {{{{-0.7,0.2}}, {{1.7,0.2}}, {{1.7,1.3}}, {{-0.7,1.3}}}},
        {{{{0,0}}, {{1.6,1.2}}, {{1,2}}, {{-0.6,0.8}}}}, // Rotated rectangle.
        {{{{0,0}}, {{2,0}}, {{2.5,1}}, {{0.5,1}}}},      // Parallelogram.
        {{{{0,0}}, {{2,0}}, {{1.6,1.3}}, {{0.2,1.3}}}},  // Trapezoid.
        {{{{0,0}}, {{2,0.2}}, {{1.8,1.6}}, {{-0.15,1.3}}}} // General convex quad.
    }};
    for (std::size_t f = 0; f < fixtures.size(); ++f) {
        PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Checking BRMixed fixture %d\n", static_cast<int>(f)));
        PetscCall(CheckCell(fixtures[f]));
    }
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
    PetscCall(DMSetUp(dm)); // Fixed test grid; unrelated -da_* options must not alter it.
    PetscCall(DMCreateGlobalVector(dm, &vertices));
    if (kind == 0) PetscCall(CreateFullMesh(dm, vertices, parameters));
    else if (kind == 1) PetscCall(LogicRectMesh(dm, vertices, parameters));
    else PetscCall(RefineMesh(dm, vertices, parameters));
    PetscCall(BuildMeshInfo(dm, vertices, info));
    PetscCall(VecDestroy(&vertices));
    PetscCall(DMDestroy(&dm));
    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode CompareCell(const MeshInfo& distributed, const MeshInfo& serial, MeshIndex cell)
{
    PetscFunctionBeginUser;
    BRMixed basis, reference;
    PetscCall(basis.Initialize(distributed, cell));
    PetscCall(reference.Initialize(serial, cell));
    QuadVertices q, expected;
    PetscCall(basis.GetCorners(q));
    PetscCall(reference.GetCorners(expected));
    PetscCall(SameCorners(q, expected));
    for (const auto& st : {std::array<PetscReal,2>{{0.23,0.31}},
                           std::array<PetscReal,2>{{0.5,0.5}},
                           std::array<PetscReal,2>{{0.78,0.67}}}) {
        const Point p = Interior(q, st[0], st[1]);
        BRMixed::Values a, b;
        PetscCall(basis.EvaluateAll(p, a));
        PetscCall(reference.EvaluateAll(p, b));
        for (PetscInt k = 0; k < BRMixed::ElementDofs; ++k)
            PetscCall(Compare(a[k], b[k], Extents(q)));
        PetscCall(CheckPoint(basis, q, p, true));
    }
    PetscFunctionReturn(PETSC_SUCCESS);
}

// Bounded, non-affine test coefficients depend only on logical IDs. Thus both
// sides of an edge receive the same coefficients without any assembly code.
Point VertexCoefficient(MeshIndex v)
{
    return {{PetscReal((3*v.i+5*v.j)%17 - 8)/7,
             PetscReal((7*v.i+2*v.j)%19 - 9)/8}};
}
PetscReal EdgeCoefficient(PetscInt id) { return PetscReal((5*id+3)%23 - 11)/9; }

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
    constexpr std::array<PetscReal,3> samples{{PetscReal(0.23), PetscReal(0.5), PetscReal(0.81)}};
    // At each edge sample, accumulate signed jumps of u, length*du/ds and nBR.
    // Compare tangential derivatives only: normal derivatives need not be continuous.
    constexpr std::size_t stride = 6*samples.size();
    std::vector<PetscReal> jumps(stride*incidence.size(), 0);
    const auto owned = distributed.OwnedCells();
    int localCells = 0;
    for (PetscInt j = owned.begin.j; j < owned.end.j; ++j)
        for (PetscInt i = owned.begin.i; i < owned.end.i; ++i) {
            ++localCells;
            PetscInt cellId;
            PetscCall(distributed.CellId({i,j}, cellId));
            ++owners[static_cast<std::size_t>(cellId)];
            BRMixed basis;
            PetscCall(basis.Initialize(distributed, {i,j}));
            std::array<OrientedEdge,4> edges;
            PetscCall(distributed.GetCellEdges({i,j}, edges));
            const std::array<MeshIndex,4> vertices{{{i,j}, {i+1,j}, {i+1,j+1}, {i,j+1}}};
            std::array<PetscReal,BRMixed::ElementDofs> coefficients{};
            for (int v = 0; v < 4; ++v) {
                const Point value = VertexCoefficient(vertices[v]);
                coefficients[v] = value.p[0];
                coefficients[v+4] = value.p[1];
                coefficients[v+8] = EdgeCoefficient(edges[v].id);
            }
            for (int e = 0; e < 4; ++e) {
                const auto id = static_cast<std::size_t>(edges[e].id);
                const PetscReal sign = static_cast<PetscReal>(edges[e].direction);
                ++incidence[id];
                EdgeVertices canonical;
                EdgeTopology topology;
                PetscCall(distributed.GetEdgeVertices(edges[e].id, canonical));
                PetscCall(distributed.GetEdgeTopology(edges[e].id, topology));
                const PetscReal length = Length(canonical[0], canonical[1]);
                const Point tangent{{(canonical[1].p[0]-canonical[0].p[0])/length,
                                      (canonical[1].p[1]-canonical[0].p[1])/length}};
                // Independent orientation oracle from increasing logical i/j.
                const Point expectedNormal = topology.axis == EdgeAxis::AlongI
                    ? Point{{-tangent.p[1], tangent.p[0]}}
                    : Point{{tangent.p[1], -tangent.p[0]}};
                Point normal;
                PetscCall(basis.GetEdgeNormal(Side(e), normal));
                const Point a = VertexCoefficient(topology.vertices[0]);
                const Point b = VertexCoefficient(topology.vertices[1]);
                const PetscReal edgeCoefficient = EdgeCoefficient(edges[e].id);
                for (std::size_t sample = 0; sample < samples.size(); ++sample) {
                    const PetscReal s = samples[sample];
                    BRMixed::Values values;
                    PetscCall(basis.EvaluateAll(Mix(canonical[0], canonical[1], s), values));
                    for (int c = 0; c < 2; ++c) {
                        PetscReal velocity = 0, tangential = 0;
                        for (PetscInt k = 0; k < BRMixed::ElementDofs; ++k) {
                            velocity += coefficients[k]*values[k].value.p[c];
                            tangential += coefficients[k]*(values[k].gradient[2*c]*tangent.p[0] +
                                                            values[k].gradient[2*c+1]*tangent.p[1]);
                        }
                        const PetscReal expected = (1-s)*a.p[c]+s*b.p[c] +
                                                    edgeCoefficient*4*s*(1-s)*expectedNormal.p[c];
                        const PetscReal derivative = b.p[c]-a.p[c] +
                                                      edgeCoefficient*4*(1-2*s)*expectedNormal.p[c];
                        PetscCall(Near(normal.p[c], expectedNormal.p[c], 1, "Canonical shared-edge normal"));
                        PetscCall(Near(velocity, expected, 4, "Reconstructed velocity edge trace"));
                        PetscCall(Near(length*tangential, derivative, 12, "Reconstructed tangential derivative"));
                        const std::size_t offset = stride*id + 6*sample;
                        jumps[offset+c] += sign*velocity;
                        jumps[offset+2+c] += sign*length*tangential;
                        jumps[offset+4+c] += sign*normal.p[c];
                    }
                }
            }
        }

    // No collective inside a local-cell loop: ranks with zero owned cells must
    // participate in exactly the same reductions as all other ranks.
    std::vector<int> allOwners(owners.size()), allIncidence(incidence.size());
    std::vector<PetscReal> allJumps(jumps.size());
    PetscCallMPI(MPI_Allreduce(owners.data(), allOwners.data(), static_cast<int>(owners.size()), MPI_INT, MPI_SUM, PETSC_COMM_WORLD));
    PetscCallMPI(MPI_Allreduce(incidence.data(), allIncidence.data(), static_cast<int>(incidence.size()), MPI_INT, MPI_SUM, PETSC_COMM_WORLD));
    PetscCallMPI(MPI_Allreduce(jumps.data(), allJumps.data(), static_cast<int>(jumps.size()), MPIU_REAL, MPI_SUM, PETSC_COMM_WORLD));
    for (int count : allOwners)
        PetscCall(Require(count == 1, "Every cell must have exactly one evaluating owner"));
    for (PetscInt e = 0; e < serial.EdgeCount(); ++e) {
        EdgeTopology topology;
        PetscCall(serial.GetEdgeTopology(e, topology));
        const auto id = static_cast<std::size_t>(e);
        PetscCall(Require(allIncidence[id] == (topology.IsBoundary() ? 1 : 2), "Global edge incidence"));
        if (!topology.IsBoundary()) for (std::size_t k = 0; k < stride; ++k)
            PetscCall(Near(allJumps[stride*id+k], 0, 12, "Shared-edge trace/normal jump across owners"));
    }
    int localEmpty = localCells == 0 ? 1 : 0, emptyRanks = 0, ghostCells = 0;
    PetscCallMPI(MPI_Allreduce(&localEmpty, &emptyRanks, 1, MPI_INT, MPI_SUM, PETSC_COMM_WORLD));
    PetscCallMPI(MPI_Allreduce(&localGhostCells, &ghostCells, 1, MPI_INT, MPI_SUM, PETSC_COMM_WORLD));
    if (nx == 2 && ny == 2)
        PetscCall(Require(emptyRanks == ranks-1, "Single-cell case must exercise empty cell owners"));
    if (ranks > 1) PetscCall(Require(ghostCells > 0, "MPI case did not exercise ghost cells"));

    BRMixed retained;
    PetscCall(retained.Initialize(distributed, available.begin));
    QuadVertices q;
    PetscCall(retained.GetCorners(q));
    const Point p = Interior(q, PetscReal(0.3), PetscReal(0.6));
    BRMixed::Values before, after;
    PetscCall(retained.EvaluateAll(p, before));
    PetscCall(ExpectError([&] { return retained.Initialize(distributed, {-1,0}); }));
    PetscCall(ExpectError([&] { return retained.Initialize(distributed, {nx-1,0}); }));
    bool unavailable = false;
    for (PetscInt j = 0; j < ny-1 && !unavailable; ++j)
        for (PetscInt i = 0; i < nx-1 && !unavailable; ++i)
            if (!distributed.HasCell({i,j})) {
                PetscCall(ExpectError([&] { return retained.Initialize(distributed, {i,j}); }));
                unavailable = true;
            }
    distributed = MeshInfo{};
    PetscCall(retained.EvaluateAll(p, after));
    PetscCall(Unchanged(after, before)); // Geometry is owned; failed init also preserves caches.
    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode Run()
{
    PetscFunctionBeginUser;
    PetscInt kind = -1, expectedRanks = 1, px = 1, py = 1;
    PetscCall(PetscOptionsGetInt(nullptr, nullptr, "-brmixed_mesh_type", &kind, nullptr));
    PetscCall(PetscOptionsGetInt(nullptr, nullptr, "-expected_ranks", &expectedRanks, nullptr));
    PetscCall(PetscOptionsGetInt(nullptr, nullptr, "-mesh_px", &px, nullptr));
    PetscCall(PetscOptionsGetInt(nullptr, nullptr, "-mesh_py", &py, nullptr));
    PetscMPIInt ranks;
    PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD, &ranks));
    PetscCheck(expectedRanks == ranks, PETSC_COMM_WORLD, PETSC_ERR_ARG_WRONG,
               "Expected %d MPI ranks but PETSc sees %d. Check that the launcher matches PETSc's MPI.",
               static_cast<int>(expectedRanks), static_cast<int>(ranks));
    PetscCall(Require(kind >= -1 && kind <= 2,
                       "brmixed_mesh_type must be -1 (unit), 0 (rectangle), 1 (quadrilateral), or 2 (stretched)"));
    if (kind == -1) {
        PetscCall(Require(ranks == 1, "Run the formula suite on one rank"));
        PetscCall(RunUnitTests());
        PetscCall(PetscPrintf(PETSC_COMM_WORLD, "BRMixed unit tests passed\n"));
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
        // 2x2 VERTICES = one cell, with enough vertices for every supported grid.
        PetscCall(CheckMesh(2, 2, px, py, kind, parameters, ranks));
        PetscCall(PetscPrintf(PETSC_COMM_WORLD, "BRMixed mesh type %d passed on %d rank(s)\n",
                              static_cast<int>(kind), static_cast<int>(ranks)));
    }
    PetscFunctionReturn(PETSC_SUCCESS);
}

} // namespace

int main(int argc, char** argv)
{
    const PetscErrorCode error = PetscInitialize(&argc, &argv, nullptr,
        "BRMixed tests: default formula suite; -brmixed_mesh_type 0/1/2 selects generated mesh tests.\n");
    if (error) return static_cast<int>(error);
    // An unexpected local error must abort WORLD: another rank may already be
    // waiting in a collective. Expected error-contract tests use a return handler.
    PetscCallAbort(PETSC_COMM_WORLD, Run());
    return static_cast<int>(PetscFinalize());
}
