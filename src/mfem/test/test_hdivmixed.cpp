#include "hdivmixed.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <limits>
#include <vector>

namespace {

// Checks remain active in Release; none depend on assert() or legacy source.
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
                 PetscAbsReal(actual-expected) <= limit,
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
PetscReal Dot(const Point& a, const Point& b)
{ return a.p[0]*b.p[0] + a.p[1]*b.p[1]; }
PetscReal Length(const Point& a, const Point& b)
{ return std::hypot(b.p[0]-a.p[0], b.p[1]-a.p[1]); }
Point Mix(const Point& a, const Point& b, PetscReal s)
{
    if (s == 0) return a;
    if (s == 1) return b;
    return {{a.p[0]+s*(b.p[0]-a.p[0]), a.p[1]+s*(b.p[1]-a.p[1])}};
}
Point Interior(const QuadVertices& q, PetscReal s, PetscReal t)
{ return Mix(Mix(q[0],q[1],s), Mix(q[3],q[2],s), t); }
Point Extents(const QuadVertices& q)
{
    Point lo = q[0], hi = q[0];
    for (const auto& p : q) for (int d = 0; d < 2; ++d) {
        lo.p[d] = std::min(lo.p[d],p.p[d]);
        hi.p[d] = std::max(hi.p[d],p.p[d]);
    }
    return {{hi.p[0]-lo.p[0], hi.p[1]-lo.p[1]}};
}
// Independent polygon area: two triangles, translated to avoid large origins.
PetscReal Area(const QuadVertices& q)
{
    PetscReal twice = 0;
    for (int k = 1; k < 3; ++k)
        twice += (q[k].p[0]-q[0].p[0])*(q[k+1].p[1]-q[0].p[1]) -
                 (q[k].p[1]-q[0].p[1])*(q[k+1].p[0]-q[0].p[0]);
    return twice/2;
}
Point OutwardNormal(const QuadVertices& q, int e)
{
    const auto& a = q[e];
    const auto& b = q[(e+1)%4];
    const PetscReal length = Length(a,b);
    return {{(b.p[1]-a.p[1])/length, -(b.p[0]-a.p[0])/length}};
}
PetscReal Sign(int e) { return e == 0 || e == 3 ? PetscReal(-1) : PetscReal(1); }

// Thin cells amplify both arithmetic error and coordinate rounding. In
// particular, a rounded physical midpoint can perturb a normal trace by
// O(epsilon * aspect ratio * coordinate/height). Keep the ordinary-cell
// tolerance, with this additional allowance for translated, stretched cells.
PetscReal GeometryTolerance(const QuadVertices& q)
{
    const PetscReal area = Area(q);
    Point magnitude{};
    for (const auto& p : q) for (int d = 0; d < 2; ++d)
        magnitude.p[d] = std::max(magnitude.p[d],PetscAbsReal(p.p[d]));
    PetscReal aspect = 1, coordinateRatio = 0;
    for (int e = 0; e < 4; ++e) {
        const PetscReal length = Length(q[e],q[(e+1)%4]);
        const PetscReal height = area/length;
        const Point normal = OutwardNormal(q,e);
        aspect = std::max(aspect,length/height);
        coordinateRatio = std::max(coordinateRatio,
            (PetscAbsReal(normal.p[0])*magnitude.p[0] +
             PetscAbsReal(normal.p[1])*magnitude.p[1])/height);
    }
    return PETSC_MACHINE_EPSILON *
        std::max(PetscReal(4096),32*aspect*(1+coordinateRatio));
}

PetscErrorCode Compare(const HDivBasisValue& a, const HDivBasisValue& b,
                       PetscReal vectorScale, PetscReal divergenceScale)
{
    PetscFunctionBeginUser;
    for (int d = 0; d < 2; ++d)
        PetscCall(Near(a.value.p[d],b.value.p[d],vectorScale,"Vector value comparison"));
    PetscCall(Near(a.divergence,b.divergence,divergenceScale,"Divergence comparison"));
    PetscFunctionReturn(PETSC_SUCCESS);
}
PetscErrorCode Unchanged(const HDivBasisValue& a, const HDivBasisValue& b)
{
    PetscFunctionBeginUser;
    PetscCall(Require(a.value.p[0] == b.value.p[0] && a.value.p[1] == b.value.p[1] &&
                       a.divergence == b.divergence, "Rejected call changed its output"));
    PetscFunctionReturn(PETSC_SUCCESS);
}
template<std::size_t N>
PetscErrorCode Unchanged(const std::array<HDivBasisValue,N>& a,
                         const std::array<HDivBasisValue,N>& b)
{
    PetscFunctionBeginUser;
    for (std::size_t i = 0; i < N; ++i) PetscCall(Unchanged(a[i],b[i]));
    PetscFunctionReturn(PETSC_SUCCESS);
}
PetscErrorCode SameCorners(const QuadVertices& a, const QuadVertices& b)
{
    PetscFunctionBeginUser;
    for (int i = 0; i < 4; ++i) for (int d = 0; d < 2; ++d)
        PetscCall(Require(a[i].p[d] == b[i].p[d], "Corner coordinates/order changed"));
    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode CheckPoint(const HDivMixed& basis, const QuadVertices& q,
                          const Point& p, bool finiteDifference)
{
    PetscFunctionBeginUser;
    const Point extent = Extents(q);
    const PetscReal vectorScale = std::max({PetscReal(1),extent.p[0]/extent.p[1],extent.p[1]/extent.p[0]});
    const PetscReal divergenceScale = 1/extent.p[0] + 1/extent.p[1];
    const PetscReal area = Area(q);
    HDivMixed::Values all;
    PetscCall(basis.EvaluateAll(p,all));
    for (PetscInt k = 0; k < HDivMixed::ElementDofs; ++k) {
        HDivBasisValue one;
        PetscCall(basis.Evaluate(p,k,one));
        PetscCall(Compare(one,all[k],vectorScale,divergenceScale));
        const int e = static_cast<int>(k%4);
        const PetscReal expected = k < 4 ? 0 : Sign(e)*Length(q[e],q[(e+1)%4])/area;
        PetscCall(Near(all[k].divergence,expected,divergenceScale,"Signed physical divergence"));
    }
    for (int e = 0; e < 4; ++e) {
        HDivMixed::EdgeValues pair;
        PetscCall(basis.EvaluateEdge(p,Side(e),pair));
        PetscCall(Compare(pair[0],all[e],vectorScale,divergenceScale));
        PetscCall(Compare(pair[1],all[e+4],vectorScale,divergenceScale));
    }
    if (finiteDifference) {
        const PetscReal step = std::cbrt(PetscReal(PETSC_MACHINE_EPSILON));
        std::array<PetscReal,8> numerical{};
        for (int d = 0; d < 2; ++d) {
            Point plus = p, minus = p;
            plus.p[d] += step*extent.p[d]; minus.p[d] -= step*extent.p[d];
            const PetscReal width = plus.p[d]-minus.p[d];
            PetscCall(Require(width > 0,"Finite-difference step was rounded away"));
            HDivMixed::Values a,b;
            PetscCall(basis.EvaluateAll(plus,a)); PetscCall(basis.EvaluateAll(minus,b));
            for (int k = 0; k < 8; ++k)
                numerical[k] += (a[k].value.p[d]-b[k].value.p[d])/width;
        }
        for (int k = 0; k < 8; ++k)
            PetscCall(Near(all[k].divergence,numerical[k],divergenceScale,
                           "Analytic divergence vs finite difference",256*step*step));
    }
    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode CheckEdgeMoments(const HDivMixed& basis, const QuadVertices& q)
{
    PetscFunctionBeginUser;
    const PetscReal tolerance = GeometryTolerance(q);
    const PetscReal root = std::sqrt(PetscReal(3)/5);
    const std::array<PetscReal,3> nodes{{(1-root)/2,PetscReal(0.5),(1+root)/2}};
    const std::array<PetscReal,3> weights{{PetscReal(5)/18,PetscReal(4)/9,PetscReal(5)/18}};
    std::array<PetscReal,8> totalFlux{};
    PetscReal perimeter = 0;
    for (int e = 0; e < 4; ++e) {
        const Point outward = OutwardNormal(q,e);
        const PetscReal length = Length(q[e],q[(e+1)%4]);
        perimeter += length;
        Point shared;
        PetscCall(basis.GetEdgeNormal(Side(e),shared));
        for (int d = 0; d < 2; ++d)
            PetscCall(Near(shared.p[d],Sign(e)*outward.p[d],1,"Shared-normal convention"));
        std::array<PetscReal,8> mean{}, linear{};
        for (std::size_t g = 0; g < nodes.size(); ++g) {
            const PetscReal s = nodes[g];
            HDivMixed::Values values;
            PetscCall(basis.EvaluateAll(Mix(q[e],q[(e+1)%4],s),values));
            for (int k = 0; k < 8; ++k) {
                const PetscReal flux = Dot(values[k].value,outward);
                mean[k] += weights[g]*flux;
                linear[k] += 3*weights[g]*(1-2*s)*flux;
            }
        }
        for (int k = 0; k < 8; ++k) {
            // These eight functionals form the complete 8x8 identity matrix.
            PetscCall(Near(linear[k],k == e ? 1 : 0,1,"Linear normal moment / DOF identity",tolerance));
            PetscCall(Near(Sign(e)*mean[k],k == e+4 ? 1 : 0,1,"Mean normal moment / DOF identity",tolerance));
            totalFlux[k] += length*mean[k];
        }
        for (PetscReal s : {PetscReal(0),PetscReal(0.19),PetscReal(0.5),PetscReal(0.83),PetscReal(1)}) {
            HDivMixed::Values values;
            PetscCall(basis.EvaluateAll(Mix(q[e],q[(e+1)%4],s),values));
            for (int k = 0; k < 8; ++k) {
                const PetscReal expected = k == e ? 1-2*s : (k == e+4 ? Sign(e) : 0);
                PetscCall(Near(Dot(values[k].value,outward),expected,1,"Normal trace on every edge",tolerance));
            }
        }
    }
    HDivMixed::Values center;
    PetscCall(basis.EvaluateAll(Interior(q,PetscReal(0.5),PetscReal(0.5)),center));
    for (int k = 0; k < 8; ++k)
        PetscCall(Near(totalFlux[k],Area(q)*center[k].divergence,perimeter,
                       "Divergence theorem / local flux balance",tolerance));
    PetscFunctionReturn(PETSC_SUCCESS);
}

// Six independent physical affine vector fields span [P1]^2. Scaling and
// translation make their magnitudes comparable across the mesh fixtures.
Point Affine(int field, const Point& p, const Point& origin, const Point& extent)
{
    Point value{};
    const int component = field/3, polynomial = field%3;
    value.p[component] = polynomial == 0 ? 1 :
        (p.p[polynomial-1]-origin.p[polynomial-1])/extent.p[polynomial-1];
    return value;
}
PetscErrorCode CheckAffineReproduction(const HDivMixed& basis, const QuadVertices& q)
{
    PetscFunctionBeginUser;
    const Point extent = Extents(q);
    const PetscReal tolerance = GeometryTolerance(q);
    for (int field = 0; field < 6; ++field) {
        std::array<PetscReal,8> coefficients{};
        for (int e = 0; e < 4; ++e) {
            const Point a = Affine(field,q[e],q[0],extent);
            const Point b = Affine(field,q[(e+1)%4],q[0],extent);
            const Point outward = OutwardNormal(q,e);
            // Exact edge moments for an affine vector, using its endpoints.
            coefficients[e] = (Dot(a,outward)-Dot(b,outward))/2;
            coefficients[e+4] = Sign(e)*(Dot(a,outward)+Dot(b,outward))/2;
        }
        const int component = field/3, polynomial = field%3;
        const PetscReal expectedDivergence = polynomial == component+1 ? 1/extent.p[component] : 0;
        for (const auto& st : {std::array<PetscReal,2>{{PetscReal(0.17),PetscReal(0.29)}},
                               {{PetscReal(0.43),PetscReal(0.61)}}, {{PetscReal(0.83),PetscReal(0.74)}}}) {
            const Point p = Interior(q,st[0],st[1]), expected = Affine(field,p,q[0],extent);
            HDivMixed::Values values;
            PetscCall(basis.EvaluateAll(p,values));
            Point actual{};
            PetscReal divergence = 0;
            for (int k = 0; k < 8; ++k) {
                for (int d = 0; d < 2; ++d) actual.p[d] += coefficients[k]*values[k].value.p[d];
                divergence += coefficients[k]*values[k].divergence;
            }
            for (int d = 0; d < 2; ++d)
                PetscCall(Near(actual.p[d],expected.p[d],1,"Physical affine vector reproduction",tolerance));
            PetscCall(Near(divergence,expectedDivergence,1/extent.p[0]+1/extent.p[1],
                           "Affine divergence reproduction",tolerance));
        }
    }
    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode CheckCell(const QuadVertices& q)
{
    PetscFunctionBeginUser;
    HDivMixed basis, fromGeometry;
    QuadBasis geometry;
    PetscCall(basis.Initialize(q));
    PetscCall(geometry.Initialize(q));
    PetscCall(fromGeometry.Initialize(geometry));
    QuadVertices stored;
    PetscCall(basis.GetCorners(stored));
    PetscCall(SameCorners(stored,q));
    PetscCall(CheckEdgeMoments(basis,q));
    PetscCall(CheckAffineReproduction(basis,q));
    for (PetscReal s : {PetscReal(0.18),PetscReal(0.44),PetscReal(0.77)})
        for (PetscReal t : {PetscReal(0.23),PetscReal(0.5),PetscReal(0.82)}) {
            const Point p = Interior(q,s,t);
            PetscCall(CheckPoint(basis,q,p,true));
            HDivMixed::Values a,b;
            PetscCall(basis.EvaluateAll(p,a)); PetscCall(fromGeometry.EvaluateAll(p,b));
            PetscCall(Unchanged(a,b));
        }
    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode CheckRectangle(PetscReal w, PetscReal h)
{
    PetscFunctionBeginUser;
    HDivMixed basis;
    PetscCall(basis.Initialize(QuadVertices{{{{0,0}},{{w,0}},{{w,h}},{{0,h}}}}));
    for (const auto& st : {std::array<PetscReal,2>{{PetscReal(0.17),PetscReal(0.31)}},
                           {{PetscReal(0.5),PetscReal(0.5)}}, {{PetscReal(0.83),PetscReal(0.69)}},
                           {{PetscReal(-0.25),PetscReal(1.3)}}}) {
        const PetscReal s = st[0], t = st[1];
        HDivMixed::Values values;
        PetscCall(basis.EvaluateAll(Point{{s*w,t*h}},values));
        // Independent closed forms, without QuadBasis scalar helpers.
        const std::array<Point,8> expected{{
            {{-(w/h)*s*(1-s),-(1-2*s)*(1-t)}},
            {{s*(1-2*t),-(h/w)*t*(1-t)}},
            {{(w/h)*s*(1-s),-(1-2*s)*t}},
            {{(1-s)*(1-2*t),(h/w)*t*(1-t)}},
            {{0,1-t}}, {{s,0}}, {{0,t}}, {{1-s,0}}
        }};
        const std::array<PetscReal,8> divergence{{0,0,0,0,-1/h,1/w,1/h,-1/w}};
        const PetscReal scale = std::max({PetscReal(1),w/h,h/w});
        for (int k = 0; k < 8; ++k)
            PetscCall(Compare(values[k],HDivBasisValue{expected[k],divergence[k]},scale,1/w+1/h));
    }
    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode CheckTransformations()
{
    PetscFunctionBeginUser;
    const QuadVertices q{{{{-0.3,0.1}},{{1.8,0.2}},{{2,1.4}},{{-0.1,1.1}}}};
    const Point p = Interior(q,PetscReal(0.37),PetscReal(0.58));
    HDivMixed original;
    PetscCall(original.Initialize(q));
    HDivMixed::Values a;
    PetscCall(original.EvaluateAll(p,a));
    const PetscReal c = std::cos(PetscReal(0.7)), s = std::sin(PetscReal(0.7));
    const int exponent = std::min(150,std::numeric_limits<PetscReal>::max_exponent10/2-4);
    const PetscReal large = std::pow(PetscReal(10),exponent);
    for (PetscReal scale : {1/large,PetscReal(1),large}) {
        const auto transform = [&](const Point& x) {
            return Point{{scale*(c*x.p[0]-s*x.p[1]+4),scale*(s*x.p[0]+c*x.p[1]-3)}};
        };
        QuadVertices moved;
        for (int i = 0; i < 4; ++i) moved[i] = transform(q[i]);
        HDivMixed transformed;
        PetscCall(transformed.Initialize(moved));
        HDivMixed::Values b;
        PetscCall(transformed.EvaluateAll(transform(p),b));
        // Unit-normal-trace normalization: vector rotates, divergence scales by 1/scale.
        for (int k = 0; k < 8; ++k) {
            PetscCall(Near(b[k].value.p[0],c*a[k].value.p[0]-s*a[k].value.p[1],1,"Rotated/scaled vector x"));
            PetscCall(Near(b[k].value.p[1],s*a[k].value.p[0]+c*a[k].value.p[1],1,"Rotated/scaled vector y"));
            PetscCall(Near(scale*b[k].divergence,a[k].divergence,1,"Scaled physical divergence"));
        }
        PetscCall(CheckRectangle(2*scale,scale));
    }
    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode CheckFailuresAndOwnership()
{
    PetscFunctionBeginUser;
    HDivMixed basis;
    PetscCall(Require(!basis.IsInitialized(),"Default state"));
    PetscCall(Require(HDivMixed::ElementDofs == 8 && HDivMixed::PressureDofs == 1 &&
                       HDivMixed::Pressure() == 1 && std::strcmp(HDivMixed::Name(),"BDM") == 0,
                       "HDivMixed metadata / constant pressure"));
    const Point p{{0.4,0.6}};
    const HDivBasisValue sentinel{{{11,12}},13};
    HDivMixed::Values sentinels; sentinels.fill(sentinel);
    HDivMixed::EdgeValues pairSentinels; pairSentinels.fill(sentinel);
    auto all = sentinels;
    auto pair = pairSentinels;
    auto one = sentinel;
    Point normal{{11,12}};
    QuadVertices q{{{{0,0}},{{2,0}},{{1.8,1.4}},{{-0.2,1}}}}, stored = q;
    PetscCall(ExpectError([&] { return basis.EvaluateAll(p,all); },PETSC_ERR_ARG_WRONG));
    PetscCall(ExpectError([&] { return basis.Evaluate(p,0,one); },PETSC_ERR_ARG_WRONG));
    PetscCall(ExpectError([&] { return basis.EvaluateEdge(p,Side(0),pair); },PETSC_ERR_ARG_WRONG));
    PetscCall(Unchanged(all,sentinels)); PetscCall(Unchanged(one,sentinel)); PetscCall(Unchanged(pair,pairSentinels));
    PetscCall(ExpectError([&] { return basis.GetCorners(stored); },PETSC_ERR_ARG_WRONG));
    PetscCall(SameCorners(stored,q));
    PetscCall(ExpectError([&] { return basis.GetEdgeNormal(Side(0),normal); },PETSC_ERR_ARG_WRONG));
    PetscCall(Require(normal.p[0] == 11 && normal.p[1] == 12,"Rejected normal output changed"));
    PetscCall(basis.Initialize(q));
    HDivMixed::Values before,after;
    PetscCall(basis.EvaluateAll(p,before));
    for (int bad = 0; bad < 7; ++bad) {
        auto invalid = q;
        if (bad == 0) std::swap(invalid[1],invalid[3]);
        if (bad == 1) invalid[1] = invalid[0];
        if (bad == 2) invalid[2] = Point{{0.1,0.1}};
        if (bad == 3) std::swap(invalid[1],invalid[2]);
        if (bad == 4) invalid[0].p[0] = std::numeric_limits<PetscReal>::quiet_NaN();
        if (bad == 5) invalid[0].p[1] = std::numeric_limits<PetscReal>::infinity();
        if (bad == 6) invalid[2] = Mix(invalid[1],invalid[3],PetscReal(0.5));
        PetscCall(ExpectError([&] { return basis.Initialize(invalid); }));
        PetscCall(Require(basis.IsInitialized(),"Failed reinitialization cleared state"));
        PetscCall(basis.EvaluateAll(p,after)); PetscCall(Unchanged(after,before));
        HDivMixed empty;
        PetscCall(ExpectError([&] { return empty.Initialize(invalid); }));
        PetscCall(Require(!empty.IsInitialized(),"Failed initialization set state"));
    }
    QuadBasis emptyGeometry;
    MeshInfo emptyMesh;
    PetscCall(ExpectError([&] { return basis.Initialize(emptyGeometry); },PETSC_ERR_ARG_WRONG));
    PetscCall(ExpectError([&] { return basis.Initialize(emptyMesh,{0,0}); },PETSC_ERR_ARG_WRONG));
    PetscCall(basis.EvaluateAll(p,after)); PetscCall(Unchanged(after,before));
    for (PetscInt bad : {PetscInt(-1),PetscInt(8),PetscInt(100)}) {
        PetscCall(ExpectError([&] { return basis.Evaluate(p,bad,one); },PETSC_ERR_ARG_OUTOFRANGE));
        PetscCall(Unchanged(one,sentinel));
    }
    for (int bad : {-1,4}) {
        PetscCall(ExpectError([&] { return basis.EvaluateEdge(p,Side(bad),pair); },PETSC_ERR_ARG_OUTOFRANGE));
        PetscCall(Unchanged(pair,pairSentinels));
        PetscCall(ExpectError([&] { return basis.GetEdgeNormal(Side(bad),normal); },PETSC_ERR_ARG_OUTOFRANGE));
        PetscCall(Require(normal.p[0] == 11 && normal.p[1] == 12,"Invalid side changed normal"));
    }
    for (PetscReal invalid : {std::numeric_limits<PetscReal>::quiet_NaN(),
                              std::numeric_limits<PetscReal>::infinity(),
                              -std::numeric_limits<PetscReal>::infinity()}) for (int d = 0; d < 2; ++d) {
        Point bad = p; bad.p[d] = invalid;
        PetscCall(ExpectError([&] { return basis.EvaluateAll(bad,all); },PETSC_ERR_ARG_OUTOFRANGE));
        PetscCall(Unchanged(all,sentinels));
        for (PetscInt k = 0; k < 8; ++k) {
            PetscCall(ExpectError([&] { return basis.Evaluate(bad,k,one); },PETSC_ERR_ARG_OUTOFRANGE));
            PetscCall(Unchanged(one,sentinel));
        }
        for (int e = 0; e < 4; ++e) {
            PetscCall(ExpectError([&] { return basis.EvaluateEdge(bad,Side(e),pair); },PETSC_ERR_ARG_OUTOFRANGE));
            PetscCall(Unchanged(pair,pairSentinels));
        }
    }
    QuadBasis geometry;
    PetscCall(geometry.Initialize(q));
    ScalarBasisValue d0,d2;
    PetscCall(geometry.EdgeDistance(Side(0),Point{{0,0}},d0));
    PetscCall(geometry.EdgeDistance(Side(2),Point{{0,0}},d2));
    const Point gradient{{d0.gradient.p[0]+d2.gradient.p[0],d0.gradient.p[1]+d2.gradient.p[1]}};
    const int axis = PetscAbsReal(gradient.p[0]) > PetscAbsReal(gradient.p[1]) ? 0 : 1;
    PetscCall(Require(gradient.p[axis] != 0,"Singular-point fixture has parallel edges"));
    Point singular{}; singular.p[axis] = -(d0.value+d2.value)/gradient.p[axis];
    PetscCall(ExpectError([&] { return basis.EvaluateAll(singular,all); },PETSC_ERR_ARG_OUTOFRANGE));
    PetscCall(Unchanged(all,sentinels));
    PetscCall(ExpectError([&] { return basis.EvaluateEdge(singular,Side(0),pair); },PETSC_ERR_ARG_OUTOFRANGE));
    PetscCall(Unchanged(pair,pairSentinels));

    HDivMixed copied = basis, assigned, fromGeometry;
    assigned = basis;
    PetscCall(fromGeometry.Initialize(geometry));
    const QuadVertices replacement{{{{0,0}},{{3,0}},{{3,2}},{{0,2}}}};
    PetscCall(basis.Initialize(replacement));
    PetscCall(geometry.Initialize(replacement));
    q[0].p[0] = 100;
    for (const HDivMixed* retained : {&copied,&assigned,&fromGeometry}) {
        PetscCall(retained->EvaluateAll(p,after)); PetscCall(Unchanged(after,before));
    }
    PetscCall(basis.GetCorners(stored)); PetscCall(SameCorners(stored,replacement));
    PetscCall(basis.Evaluate(Point{{1.5,1}},4,one));
    PetscCall(Compare(one,HDivBasisValue{Point{{0,PetscReal(0.5)}},PetscReal(-0.5)},1,1));
    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode RunUnitTests()
{
    PetscFunctionBeginUser;
    const std::array<QuadVertices,6> fixtures{{
        {{{{0,0}},{{2,0}},{{2,1}},{{0,1}}}},
        {{{{-0.7,0.2}},{{1.7,0.2}},{{1.7,1.3}},{{-0.7,1.3}}}},
        {{{{0,0}},{{1.6,1.2}},{{1,2}},{{-0.6,0.8}}}},
        {{{{0,0}},{{2,0}},{{2.5,1}},{{0.5,1}}}},
        {{{{0,0}},{{2,0}},{{1.6,1.3}},{{0.2,1.3}}}},
        {{{{0,0}},{{2,0.2}},{{1.8,1.6}},{{-0.15,1.3}}}}
    }};
    for (const auto& q : fixtures) PetscCall(CheckCell(q));
    PetscCall(CheckRectangle(2,1));
    PetscCall(CheckRectangle(17,PetscReal(0.08)));
    PetscCall(CheckRectangle(PetscReal(0.08),17));
    PetscCall(CheckTransformations());
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
    PetscCall(DMDACreate2d(comm,DM_BOUNDARY_NONE,DM_BOUNDARY_NONE,DMDA_STENCIL_BOX,
                          nx,ny,px,py,2,1,nullptr,nullptr,&dm));
    PetscCall(DMSetUp(dm));
    PetscCall(DMCreateGlobalVector(dm,&vertices));
    if (kind == 0) PetscCall(CreateFullMesh(dm,vertices,parameters));
    else if (kind == 1) PetscCall(LogicRectMesh(dm,vertices,parameters));
    else PetscCall(RefineMesh(dm,vertices,parameters));
    PetscCall(BuildMeshInfo(dm,vertices,info));
    PetscCall(VecDestroy(&vertices)); PetscCall(DMDestroy(&dm));
    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode CompareCell(const MeshInfo& distributed, const MeshInfo& serial, MeshIndex cell)
{
    PetscFunctionBeginUser;
    HDivMixed a,b;
    PetscCall(a.Initialize(distributed,cell)); PetscCall(b.Initialize(serial,cell));
    QuadVertices q,reference;
    PetscCall(a.GetCorners(q)); PetscCall(b.GetCorners(reference));
    PetscCall(SameCorners(q,reference));
    PetscCall(CheckEdgeMoments(a,q));
    PetscCall(CheckAffineReproduction(a,q));
    const Point extent = Extents(q);
    for (const auto& st : {std::array<PetscReal,2>{{PetscReal(0.23),PetscReal(0.31)}},
                           {{PetscReal(0.5),PetscReal(0.5)}},{{PetscReal(0.78),PetscReal(0.67)}}}) {
        const Point p = Interior(q,st[0],st[1]);
        HDivMixed::Values actual,expected;
        PetscCall(a.EvaluateAll(p,actual)); PetscCall(b.EvaluateAll(p,expected));
        for (int k = 0; k < 8; ++k)
            PetscCall(Compare(actual[k],expected[k],1,1/extent.p[0]+1/extent.p[1]));
        PetscCall(CheckPoint(a,q,p,true));
    }
    PetscFunctionReturn(PETSC_SUCCESS);
}

// Two distinct coefficient families keyed by natural global edge ID.
PetscReal LinearCoefficient(PetscInt id) { return PetscReal((7*id+5)%19-9)/8; }
PetscReal ConstantCoefficient(PetscInt id) { return PetscReal((5*id+3)%23-11)/9; }

PetscErrorCode CheckMesh(PetscInt nx, PetscInt ny, PetscInt px, PetscInt py,
                         PetscInt kind, const MeshParam& parameters, PetscMPIInt ranks)
{
    PetscFunctionBeginUser;
    MeshInfo distributed,serial;
    PetscCall(MakeSnapshot(PETSC_COMM_WORLD,nx,ny,px,py,kind,parameters,distributed));
    PetscCall(MakeSnapshot(PETSC_COMM_SELF,nx,ny,1,1,kind,parameters,serial));
    const auto available = distributed.AvailableCells();
    int localGhosts = 0;
    for (PetscInt j = available.begin.j; j < available.end.j; ++j)
        for (PetscInt i = available.begin.i; i < available.end.i; ++i) {
            PetscCall(CompareCell(distributed,serial,{i,j}));
            if (!distributed.OwnsCell({i,j})) ++localGhosts;
        }
    std::vector<int> owners(static_cast<std::size_t>(serial.CellCount()),0);
    std::vector<int> incidence(static_cast<std::size_t>(serial.EdgeCount()),0);
    constexpr std::array<PetscReal,3> samples{{PetscReal(0.19),PetscReal(0.5),PetscReal(0.83)}};
    // Per sample: linear trace, constant trace, combined flux, shared normal x/y.
    constexpr std::size_t stride = 5*samples.size();
    std::vector<PetscReal> jumps(stride*incidence.size(),0);
    PetscReal localTolerance = 4096*PETSC_MACHINE_EPSILON;
    int localCells = 0;
    const auto owned = distributed.OwnedCells();
    for (PetscInt j = owned.begin.j; j < owned.end.j; ++j)
        for (PetscInt i = owned.begin.i; i < owned.end.i; ++i) {
            ++localCells;
            PetscInt cellId;
            PetscCall(distributed.CellId({i,j},cellId));
            ++owners[static_cast<std::size_t>(cellId)];
            HDivMixed basis;
            PetscCall(basis.Initialize(distributed,{i,j}));
            QuadVertices corners;
            PetscCall(basis.GetCorners(corners));
            const PetscReal tolerance = GeometryTolerance(corners);
            localTolerance = std::max(localTolerance,tolerance);
            std::array<OrientedEdge,4> edges;
            PetscCall(distributed.GetCellEdges({i,j},edges));
            std::array<PetscReal,8> coefficients{};
            for (int e = 0; e < 4; ++e) {
                coefficients[e] = LinearCoefficient(edges[e].id);
                coefficients[e+4] = ConstantCoefficient(edges[e].id);
            }
            for (int e = 0; e < 4; ++e) {
                const auto id = static_cast<std::size_t>(edges[e].id);
                ++incidence[id];
                EdgeVertices canonical;
                EdgeTopology topology;
                PetscCall(distributed.GetEdgeVertices(edges[e].id,canonical));
                PetscCall(distributed.GetEdgeTopology(edges[e].id,topology));
                const PetscReal length = Length(canonical[0],canonical[1]);
                const Point tangent{{(canonical[1].p[0]-canonical[0].p[0])/length,
                                      (canonical[1].p[1]-canonical[0].p[1])/length}};
                const Point normal = topology.axis == EdgeAxis::AlongI
                    ? Point{{-tangent.p[1],tangent.p[0]}}
                    : Point{{tangent.p[1],-tangent.p[0]}};
                Point actualNormal;
                PetscCall(basis.GetEdgeNormal(Side(e),actualNormal));
                for (int d = 0; d < 2; ++d)
                    PetscCall(Near(actualNormal.p[d],normal.p[d],1,"Canonical shared normal"));
                for (std::size_t g = 0; g < samples.size(); ++g) {
                    const PetscReal s = samples[g];
                    HDivMixed::Values values;
                    PetscCall(basis.EvaluateAll(Mix(canonical[0],canonical[1],s),values));
                    const PetscReal linear = Dot(values[e].value,normal);
                    const PetscReal constant = Dot(values[e+4].value,normal);
                    const PetscReal expectedLinear = topology.axis == EdgeAxis::AlongI ? 2*s-1 : 1-2*s;
                    PetscCall(Near(linear,expectedLinear,1,"Canonical linear trace orientation",tolerance));
                    PetscCall(Near(constant,1,1,"Canonical constant trace orientation",tolerance));
                    PetscReal flux = 0;
                    for (int k = 0; k < 8; ++k) flux += coefficients[k]*Dot(values[k].value,normal);
                    const PetscReal expected = coefficients[e]*expectedLinear+coefficients[e+4];
                    PetscCall(Near(flux,expected,4,"Combined normal trace including all eight modes",tolerance));
                    const std::array<PetscReal,5> data{{linear,constant,flux,actualNormal.p[0],actualNormal.p[1]}};
                    for (std::size_t k = 0; k < data.size(); ++k)
                        jumps[stride*id+5*g+k] += edges[e].direction*data[k];
                }
            }
        }
    // Collectives are outside cell loops, including for ranks owning no cells.
    std::vector<int> allOwners(owners.size()),allIncidence(incidence.size());
    std::vector<PetscReal> allJumps(jumps.size());
    PetscCallMPI(MPI_Allreduce(owners.data(),allOwners.data(),static_cast<int>(owners.size()),MPI_INT,MPI_SUM,PETSC_COMM_WORLD));
    PetscCallMPI(MPI_Allreduce(incidence.data(),allIncidence.data(),static_cast<int>(incidence.size()),MPI_INT,MPI_SUM,PETSC_COMM_WORLD));
    PetscCallMPI(MPI_Allreduce(jumps.data(),allJumps.data(),static_cast<int>(jumps.size()),MPIU_REAL,MPI_SUM,PETSC_COMM_WORLD));
    PetscReal globalTolerance = 0;
    PetscCallMPI(MPI_Allreduce(&localTolerance,&globalTolerance,1,MPIU_REAL,MPI_MAX,PETSC_COMM_WORLD));
    for (int count : allOwners) PetscCall(Require(count == 1,"Unique global cell ownership"));
    for (PetscInt e = 0; e < serial.EdgeCount(); ++e) {
        EdgeTopology topology;
        PetscCall(serial.GetEdgeTopology(e,topology));
        const auto id = static_cast<std::size_t>(e);
        PetscCall(Require(allIncidence[id] == (topology.IsBoundary() ? 1 : 2),"Global edge incidence"));
        if (!topology.IsBoundary()) for (std::size_t k = 0; k < stride; ++k)
            PetscCall(Near(allJumps[stride*id+k],0,4,"Normal-trace jump across cell owners",globalTolerance));
    }
    const int localEmpty = localCells == 0 ? 1 : 0;
    int emptyRanks = 0,ghosts = 0;
    PetscCallMPI(MPI_Allreduce(&localEmpty,&emptyRanks,1,MPI_INT,MPI_SUM,PETSC_COMM_WORLD));
    PetscCallMPI(MPI_Allreduce(&localGhosts,&ghosts,1,MPI_INT,MPI_SUM,PETSC_COMM_WORLD));
    if (nx == 2 && ny == 2) PetscCall(Require(emptyRanks == ranks-1,"Single-cell empty owners"));
    if (ranks > 1) PetscCall(Require(ghosts > 0,"MPI case did not exercise ghost cells"));

    if (available.begin.i < available.end.i && available.begin.j < available.end.j) {
        HDivMixed retained;
        PetscCall(retained.Initialize(distributed,available.begin));
        QuadVertices q;
        PetscCall(retained.GetCorners(q));
        const Point p = Interior(q,PetscReal(0.3),PetscReal(0.6));
        HDivMixed::Values before,after;
        PetscCall(retained.EvaluateAll(p,before));
        PetscCall(ExpectError([&] { return retained.Initialize(distributed,{-1,0}); }));
        PetscCall(ExpectError([&] { return retained.Initialize(distributed,{nx-1,0}); }));
        bool checkedUnavailable = false;
        for (PetscInt j = 0; j < ny-1 && !checkedUnavailable; ++j)
            for (PetscInt i = 0; i < nx-1 && !checkedUnavailable; ++i)
                if (!distributed.HasCell({i,j})) {
                    PetscCall(ExpectError([&] { return retained.Initialize(distributed,{i,j}); }));
                    checkedUnavailable = true;
                }
        distributed = MeshInfo{};
        PetscCall(retained.EvaluateAll(p,after)); PetscCall(Unchanged(after,before));
    }
    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode Run()
{
    PetscFunctionBeginUser;
    PetscInt kind = -1,expectedRanks = 1,px = 1,py = 1;
    PetscCall(PetscOptionsGetInt(nullptr,nullptr,"-hdivmixed_mesh_type",&kind,nullptr));
    PetscCall(PetscOptionsGetInt(nullptr,nullptr,"-expected_ranks",&expectedRanks,nullptr));
    PetscCall(PetscOptionsGetInt(nullptr,nullptr,"-mesh_px",&px,nullptr));
    PetscCall(PetscOptionsGetInt(nullptr,nullptr,"-mesh_py",&py,nullptr));
    PetscMPIInt ranks;
    PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD,&ranks));
    PetscCheck(expectedRanks == ranks,PETSC_COMM_WORLD,PETSC_ERR_ARG_WRONG,
               "Expected %d MPI ranks but PETSc sees %d; use the launcher matching PETSc's MPI",
               static_cast<int>(expectedRanks),static_cast<int>(ranks));
    PetscCall(Require(kind >= -1 && kind <= 2,"hdivmixed_mesh_type must be -1, 0, 1 or 2"));
    if (kind == -1) {
        PetscCall(Require(ranks == 1,"Run the formula suite on one rank"));
        PetscCall(RunUnitTests());
        PetscCall(PetscPrintf(PETSC_COMM_WORLD,"HDivMixed unit tests passed\n"));
    } else {
        PetscCall(Require((px == 1 || px == 2) && (py == 1 || py == 2) && px*py == ranks,
                           "Use a matching 1x1, 2x1, 1x2 or 2x2 process grid"));
        MeshParam parameters;
        parameters.xstart = -0.75; parameters.ystart = 0.2;
        parameters.L = 2.5; parameters.H = 1.3;
        parameters.seed = 7; parameters.perturbation = 0.20;
        PetscCall(CheckMesh(13,9,px,py,kind,parameters,ranks));
        parameters.xstart = 3.5; parameters.ystart = -2.25;
        parameters.L = 17; parameters.H = 0.08;
        parameters.seed = 991; parameters.perturbation = 0.249;
        PetscCall(CheckMesh(10,7,px,py,kind,parameters,ranks));
        // DMDA dimensions count vertices: one cell with valid 1/2/4-rank grids.
        PetscCall(CheckMesh(2,2,px,py,kind,parameters,ranks));
        PetscCall(PetscPrintf(PETSC_COMM_WORLD,"HDivMixed mesh type %d passed on %d rank(s)\n",
                              static_cast<int>(kind),static_cast<int>(ranks)));
    }
    PetscFunctionReturn(PETSC_SUCCESS);
}

} // namespace

int main(int argc, char** argv)
{
    const PetscErrorCode error = PetscInitialize(&argc,&argv,nullptr,
        "HDivMixed tests: default formula suite; -hdivmixed_mesh_type 0/1/2 selects mesh checks.\n");
    if (error) return static_cast<int>(error);
    // Abort WORLD on unexpected local failure so another rank cannot hang.
    PetscCallAbort(PETSC_COMM_WORLD,Run());
    return static_cast<int>(PetscFinalize());
}
