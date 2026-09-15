#include "local_matrix.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

namespace {

// Release builds run the same checks: no assert(), legacy source, or external
// test framework. Analytic polygon moments/edge traces are the primary oracles.
constexpr PetscReal tolerance = 65536 * PETSC_MACHINE_EPSILON;
constexpr PetscReal quadratureTolerance = PetscReal(4e-10);

PetscErrorCode Require(bool condition, const char* message)
{
    PetscFunctionBeginUser;
    PetscCheck(condition, PETSC_COMM_SELF, PETSC_ERR_PLIB, "%s", message);
    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode Near(PetscScalar actual, PetscReal expected, PetscReal scale,
                    const char* message, PetscReal relative = tolerance)
{
    PetscFunctionBeginUser;
    const PetscReal limit = relative * std::max(PetscAbsReal(expected), scale);
    PetscCheck(!PetscIsInfOrNanScalar(actual) && !PetscIsInfOrNanReal(expected) &&
                   PetscAbsScalar(actual - PetscScalar(expected)) <= limit,
               PETSC_COMM_SELF, PETSC_ERR_PLIB,
               "%s: got %.17g%+.17gi, expected %.17g, tolerance %.3g", message,
               static_cast<double>(PetscRealPart(actual)),
               static_cast<double>(PetscImaginaryPart(actual)),
               static_cast<double>(expected), static_cast<double>(limit));
    PetscFunctionReturn(PETSC_SUCCESS);
}

template<class Function>
PetscErrorCode ExpectError(Function&& function, PetscErrorCode expected)
{
    PetscFunctionBeginUser;
    PetscCall(PetscPushErrorHandler(PetscReturnErrorHandler, nullptr));
    // The lambdas used here forward a single call directly. They contain no
    // PetscCall/PetscCheck and therefore need no additional PETSc stack frame.
    const PetscErrorCode error = function();
    PetscCall(PetscPopErrorHandler());
    PetscCheck(error == expected, PETSC_COMM_SELF, PETSC_ERR_PLIB,
               "Expected error %d, got %d", static_cast<int>(expected), static_cast<int>(error));
    PetscFunctionReturn(PETSC_SUCCESS);
}

template<std::size_t N>
bool Same(const LocalMatrixBlock<N>& a, const LocalMatrixBlock<N>& b)
{ return a.A == b.A && a.B == b.B && a.C == b.C && a.f == b.f; }

template<std::size_t N, class Function>
PetscErrorCode RejectBlock(Function&& function, LocalMatrixBlock<N>& result,
                          PetscErrorCode expected)
{
    PetscFunctionBeginUser;
    const auto before = result;
    PetscCall(ExpectError(function, expected));
    PetscCall(Require(Same(result, before), "Rejected call changed its output block"));
    PetscFunctionReturn(PETSC_SUCCESS);
}

template<class Function>
PetscErrorCode RejectScalar(Function&& function, PetscScalar& result, PetscErrorCode expected)
{
    PetscFunctionBeginUser;
    const PetscScalar before = result;
    PetscCall(ExpectError(function, expected));
    PetscCall(Require(result == before, "Rejected call changed its output coupling"));
    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscReal Dot(const Point& a, const Point& b)
{ return a.p[0] * b.p[0] + a.p[1] * b.p[1]; }
Point Mix(const Point& a, const Point& b, PetscReal t)
{ return {{a.p[0] + t * (b.p[0] - a.p[0]), a.p[1] + t * (b.p[1] - a.p[1])}}; }
Point Interior(const QuadVertices& q, PetscReal s, PetscReal t)
{ return Mix(Mix(q[0], q[1], s), Mix(q[3], q[2], s), t); }
PetscReal Length(const QuadVertices& q, int e)
{ return std::hypot(q[(e+1)%4].p[0] - q[e].p[0], q[(e+1)%4].p[1] - q[e].p[1]); }
Point Outward(const QuadVertices& q, int e)
{
    const PetscReal length = Length(q, e);
    return {{(q[(e+1)%4].p[1]-q[e].p[1])/length,
             -(q[(e+1)%4].p[0]-q[e].p[0])/length}};
}
PetscReal Sign(int e) { return e == 0 || e == 3 ? PetscReal(-1) : PetscReal(1); }

// Moment matrix for [1,x,y], from two triangles and exact barycentric moments.
// Does not call MapCellPoint, CellJacobian, IntegrateCell, or GetCellArea.
using Moments = std::array<std::array<PetscReal, 3>, 3>;
Moments PolygonMoments(const QuadVertices& q)
{
    Moments m{};
    for (int k = 1; k < 3; ++k) {
        const std::array<Point,3> v{{q[0], q[k], q[k+1]}};
        const PetscReal area = ((v[1].p[0]-v[0].p[0])*(v[2].p[1]-v[0].p[1]) -
                                (v[1].p[1]-v[0].p[1])*(v[2].p[0]-v[0].p[0])) / 2;
        PetscReal sx = 0, sy = 0, xx = 0, yy = 0, xy = 0;
        for (const auto& p : v) {
            sx += p.p[0]; sy += p.p[1];
            xx += p.p[0]*p.p[0]; yy += p.p[1]*p.p[1]; xy += p.p[0]*p.p[1];
        }
        m[0][0] += area;
        m[0][1] += area*sx/3; m[0][2] += area*sy/3;
        m[1][1] += area*(sx*sx+xx)/12;
        m[1][2] += area*(sx*sy+xy)/12;
        m[2][2] += area*(sy*sy+yy)/12;
    }
    m[1][0] = m[0][1]; m[2][0] = m[0][2]; m[2][1] = m[1][2];
    return m;
}

using AffineVector = std::array<std::array<PetscReal,3>,2>;
Point Evaluate(const AffineVector& f, const Point& p)
{
    return {{f[0][0]+f[0][1]*p.p[0]+f[0][2]*p.p[1],
             f[1][0]+f[1][1]*p.p[0]+f[1][2]*p.p[1]}};
}
PetscReal ProductIntegral(const AffineVector& a, const AffineVector& b, const Moments& m)
{
    PetscReal value = 0;
    for (int d = 0; d < 2; ++d)
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j) value += a[d][i]*b[d][j]*m[i][j];
    return value;
}
std::array<PetscReal,12> BRDofs(const QuadVertices& q, const AffineVector& f)
{
    std::array<PetscReal,12> c{};
    for (int v = 0; v < 4; ++v) {
        const Point u = Evaluate(f, q[v]);
        c[v] = u.p[0]; c[v+4] = u.p[1];
    }
    // An affine function has no midpoint residual, so its BR edge DOFs are zero.
    return c;
}
std::array<PetscReal,8> HDivDofs(const QuadVertices& q, const AffineVector& f)
{
    std::array<PetscReal,8> c{};
    for (int e = 0; e < 4; ++e) {
        const Point normal = Outward(q, e);
        const PetscReal start = Dot(Evaluate(f, q[e]), normal);
        const PetscReal end = Dot(Evaluate(f, q[(e+1)%4]), normal);
        c[e] = (start-end)/2;
        c[e+4] = Sign(e)*(start+end)/2;
    }
    return c;
}
template<std::size_t N>
PetscScalar Action(const std::array<PetscReal,N>& a, const std::array<PetscScalar,N>& b)
{
    PetscScalar value{};
    for (std::size_t i = 0; i < N; ++i) value += a[i]*b[i];
    return value;
}
template<std::size_t N>
PetscScalar Bilinear(const LocalMatrixBlock<N>& block, const std::array<PetscReal,N>& a,
                     const std::array<PetscReal,N>& b)
{
    PetscScalar value{};
    for (std::size_t i = 0; i < N; ++i)
        for (std::size_t j = 0; j < N; ++j) value += a[i]*block.A[i*N+j]*b[j];
    return value;
}

struct Element {
    QuadBasis geometry;
    BRMixed br;
    HDivMixed hdiv;
    PetscErrorCode Initialize(const QuadVertices& q)
    {
        PetscFunctionBeginUser;
        PetscCall(geometry.Initialize(q));
        PetscCall(br.Initialize(geometry)); PetscCall(hdiv.Initialize(geometry));
        PetscFunctionReturn(PETSC_SUCCESS);
    }
    PetscErrorCode Initialize(const MeshInfo& mesh, MeshIndex cell)
    {
        PetscFunctionBeginUser;
        PetscCall(geometry.Initialize(mesh, cell));
        PetscCall(br.Initialize(geometry)); PetscCall(hdiv.Initialize(geometry));
        PetscFunctionReturn(PETSC_SUCCESS);
    }
};
struct Blocks { StokesLocalMatrix s; DarcyLocalMatrix d; PetscScalar k{}; };
PetscErrorCode Compute(const Element& e, const GaussRule1D& cell, const GaussRule1D& edge,
                       const LocalPorositySamples& p, const LocalMatrixParameters& parameters,
                       const LocalForceFunction& force, Blocks& b)
{
    PetscFunctionBeginUser;
    PetscCall(ComputeLocalStokes(e.br, cell, p, force, b.s));
    PetscCall(ComputeLocalDarcy(e.hdiv, cell, edge, p, parameters, force, b.d));
    PetscCall(ComputeLocalCoupling(e.geometry, cell, p, parameters, b.k));
    PetscFunctionReturn(PETSC_SUCCESS);
}
LocalPorositySamples ConstantPorosity(const GaussRule1D& cell, const GaussRule1D& edge,
                                     PetscReal phi)
{
    LocalPorositySamples p;
    p.cell.assign(cell.points.size()*cell.points.size(), phi);
    for (auto& side : p.edge) side.assign(edge.points.size(), phi);
    p.average = phi;
    return p;
}

template<std::size_t N>
PetscErrorCode CheckStructure(const LocalMatrixBlock<N>& b)
{
    PetscFunctionBeginUser;
    for (std::size_t i = 0; i < N; ++i) {
        for (std::size_t j = 0; j < N; ++j) {
            PetscCall(Require(!PetscIsInfOrNanScalar(b.A[i*N+j]), "Nonfinite A entry"));
            PetscCall(Require(b.A[i*N+j] == b.A[j*N+i], "A is not symmetric"));
            PetscCall(Require(PetscImaginaryPart(b.A[i*N+j]) == 0, "A has an imaginary part"));
        }
        PetscCall(Require(!PetscIsInfOrNanScalar(b.B[i]) && PetscImaginaryPart(b.B[i]) == 0,
                          "B must be finite and real"));
        PetscCall(Require(!PetscIsInfOrNanScalar(b.f[i]) && PetscImaginaryPart(b.f[i]) == 0,
                          "f must be finite and real"));
    }
    PetscCall(Require(!PetscIsInfOrNanScalar(b.C) && PetscImaginaryPart(b.C) == 0,
                      "C must be finite and real"));
    PetscFunctionReturn(PETSC_SUCCESS);
}

// Independent dense Cholesky of a diagonally scaled principal submatrix.
// Darcy mass is SPD. Stokes is SPD after pinning its three rigid motions.
template<std::size_t N>
PetscErrorCode CheckPositive(const LocalMatrixBlock<N>& block, const std::vector<int>& free)
{
    PetscFunctionBeginUser;
    std::array<PetscReal,N*N> lower{};
    for (std::size_t i = 0; i < free.size(); ++i) {
        const PetscReal diagonalI = PetscRealPart(block.A[free[i]*N+free[i]]);
        PetscCall(Require(diagonalI > 0, "Positive matrix has a nonpositive diagonal"));
        for (std::size_t j = 0; j <= i; ++j) {
            const PetscReal diagonalJ = PetscRealPart(block.A[free[j]*N+free[j]]);
            PetscCall(Require(diagonalJ > 0, "Positive matrix has a nonpositive diagonal"));
            PetscReal value = PetscRealPart(block.A[free[i]*N+free[j]]) /
                (PetscSqrtReal(diagonalI)*PetscSqrtReal(diagonalJ));
            for (std::size_t k = 0; k < j; ++k) value -= lower[i*N+k]*lower[j*N+k];
            if (i == j) {
                PetscCall(Require(value > 512*PETSC_MACHINE_EPSILON, "Cholesky found a nonpositive pivot"));
                lower[i*N+j] = PetscSqrtReal(value);
            } else lower[i*N+j] = value/lower[j*N+j];
        }
    }
    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode CheckAffineBlocks(const QuadVertices& q, const Blocks& b, PetscReal phi,
                                const AffineVector& force, bool exactMoments)
{
    PetscFunctionBeginUser;
    const Moments m = PolygonMoments(q);
    const PetscReal area = m[0][0], solid = 1-phi;
    PetscCall(CheckStructure(b.s)); PetscCall(CheckStructure(b.d));
    PetscCall(Near(b.s.C, area*phi/solid, area*phi, "Stokes compaction"));
    PetscCall(Near(b.d.C, area/solid, area, "Darcy compaction"));
    PetscCall(Near(b.k, -area*PetscSqrtReal(phi)/solid,
                   area*PetscSqrtReal(phi), "Negative pressure coupling"));
    std::array<AffineVector,6> fields{};
    for (int a = 0; a < 6; ++a) fields[a][a/3][a%3] = 1;
    for (const auto& a : fields) {
        const auto sa = BRDofs(q, a); const auto da = HDivDofs(q, a);
        const PetscReal divA = a[0][1]+a[1][2];
        PetscCall(Near(Action(sa,b.s.B), area*divA, area, "Stokes affine divergence sign"));
        // theta=1/2 and constant phi give phi^(3/2)/sqrt(phi)=phi.
        PetscCall(Near(Action(da,b.d.B), phi*area*divA, area, "Darcy affine boundary flux sign"));
        if (exactMoments) {
            PetscCall(Near(Action(sa,b.s.f), solid*ProductIntegral(a,force,m), area,
                           "Stokes affine load and solid fraction"));
            PetscCall(Near(Action(da,b.d.f), ProductIntegral(a,force,m), area,
                           "Darcy affine load without porosity factor"));
        }
        for (const auto& c : fields) {
            const PetscReal divC = c[0][1]+c[1][2];
            const PetscReal strain = a[0][1]*c[0][1] + a[1][2]*c[1][2] +
                (a[0][2]+a[1][1])*(c[0][2]+c[1][1])/2;
            PetscCall(Near(Bilinear(b.s,sa,BRDofs(q,c)),
                           2*solid*area*(strain-divA*divC/3), area,
                           "Stokes affine bilinear form (including 1/3 correction)"));
            if (exactMoments)
                PetscCall(Near(Bilinear(b.d,da,HDivDofs(q,c)), ProductIntegral(a,c,m), area,
                               "Unweighted Darcy affine mass and physical area"));
        }
    }
    // Stronger than a zero energy check: each rigid motion is in ker(A).
    std::array<AffineVector,3> rigid{};
    rigid[0][0][0] = 1; rigid[1][1][0] = 1;
    rigid[2][0][2] = -1; rigid[2][1][1] = 1;
    for (const auto& field : rigid) {
        const auto c = BRDofs(q,field);
        for (int row = 0; row < 12; ++row) {
            PetscScalar value{}; PetscReal scale = 0;
            for (int j = 0; j < 12; ++j) {
                value += b.s.A[12*row+j]*c[j];
                scale += PetscAbsScalar(b.s.A[12*row+j])*PetscAbsReal(c[j]);
            }
            PetscCall(Near(value,0,std::max(scale,area),"Stokes rigid-motion nullspace"));
        }
    }
    for (int e = 0; e < 4; ++e) {
        PetscCall(Near(b.d.B[e],0,Length(q,e),"Constant porosity: zero linear flux moment"));
        PetscCall(Near(b.d.B[e+4],Sign(e)*Length(q,e)*phi,Length(q,e),
                       "Constant porosity: signed constant flux moment"));
    }
    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode CheckCell(const QuadVertices& q)
{
    PetscFunctionBeginUser;
    Element e; PetscCall(e.Initialize(q));
    const AffineVector f{{{{0.7,-0.1,0.4}},{{-0.2,0.3,0.8}}}};
    const LocalForceFunction force = [f](const Point& p) { return Evaluate(f,p); };
    LocalMatrixParameters parameters; parameters.theta = PetscReal(0.5);
    for (PetscInt n : {1,2,3,5,8}) {
        GaussRule1D cell,edge;
        PetscCall(CreateGaussRule(n,cell)); PetscCall(CreateGaussRule(n == 1 ? 1 : n+1,edge));
        const auto p = ConstantPorosity(cell,edge,PetscReal(0.23));
        Blocks b; PetscCall(Compute(e,cell,edge,p,parameters,force,b));
        PetscCall(CheckAffineBlocks(q,b,p.average,f,n >= 2));
        if (n == 8) {
            PetscCall(CheckPositive(b.d,{0,1,2,3,4,5,6,7}));
            const Point tangent{{q[1].p[0]-q[0].p[0],q[1].p[1]-q[0].p[1]}};
            const int thirdPin = PetscAbsReal(tangent.p[0]) >= PetscAbsReal(tangent.p[1]) ? 5 : 1;
            std::vector<int> free;
            for (int k = 0; k < 12; ++k) if (k != 0 && k != 4 && k != thirdPin) free.push_back(k);
            PetscCall(CheckPositive(b.s,free));
        }
    }
    // Individual BR divergence entries from boundary traces, independently of
    // the volume gradients used by ComputeLocalStokes. A bubble has integral 2L/3.
    GaussRule1D fine; PetscCall(CreateGaussRule(16,fine));
    auto p = ConstantPorosity(fine,fine,PetscReal(0.23));
    StokesLocalMatrix s;
    PetscCall(ComputeLocalStokes(e.br,fine,p,{},s));
    for (int v = 0; v < 4; ++v) {
        const int previous = (v+3)%4;
        for (int d = 0; d < 2; ++d) {
            const PetscReal flux = (Length(q,previous)*Outward(q,previous).p[d] +
                                     Length(q,v)*Outward(q,v).p[d])/2;
            PetscCall(Near(s.B[v+4*d],flux,Length(q,v),"BR vertex boundary flux",quadratureTolerance));
        }
        PetscCall(Near(s.B[8+v],Sign(v)*2*Length(q,v)/3,Length(q,v),
                       "BR edge-bubble boundary flux",quadratureTolerance));
    }
    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscReal RatioProfile(const Point& p) { return PetscReal(0.3)+PetscReal(0.035)*p.p[0]+PetscReal(0.021)*p.p[1]; }
PetscReal CellProfile(const Point& p) { const PetscReal r = RatioProfile(p); return r/(1+r); }
Point ForceProfile(const Point& p)
{ return {{1+PetscReal(0.2)*p.p[0]+p.p[0]*p.p[1], PetscReal(0.3)+PetscReal(0.1)*p.p[0]-PetscReal(0.2)*p.p[1]*p.p[1]}}; }
LocalPorositySamples VariablePorosity(const QuadVertices& q, const GaussRule1D& cell,
                                     const GaussRule1D& edge)
{
    auto p = ConstantPorosity(cell,edge,PetscReal(0.27));
    const std::size_t n = cell.points.size();
    for (std::size_t j = 0; j < n; ++j) for (std::size_t i = 0; i < n; ++i)
        p.cell[j*n+i] = CellProfile(Interior(q,(1+cell.points[i])/2,(1+cell.points[j])/2));
    // Deliberately supplied independently: tests must detect recomputing average
    // inside the kernel. It need not equal this rule's sampled mean.
    p.average = PetscReal(0.31);
    for (int e = 0; e < 4; ++e) for (std::size_t g = 0; g < edge.points.size(); ++g)
        p.edge[e][g] = PetscReal(0.12)+PetscReal(0.02)*e+
            (e%2 == 0 ? PetscReal(0.07) : PetscReal(-0.04))*(1+edge.points[g])/2;
    return p;
}

PetscErrorCode CheckVariablePorosity(const QuadVertices& q)
{
    PetscFunctionBeginUser;
    Element e; PetscCall(e.Initialize(q));
    const auto m = PolygonMoments(q);
    const PetscReal ratioIntegral = PetscReal(0.3)*m[0][0]+PetscReal(0.035)*m[0][1]+PetscReal(0.021)*m[0][2];
    for (PetscInt n : {2,3,6}) {
        GaussRule1D cell,edge;
        PetscCall(CreateGaussRule(n,cell)); PetscCall(CreateGaussRule(n+1,edge));
        auto p = VariablePorosity(q,cell,edge);
        LocalMatrixParameters parameters;
        for (int theta : {0,1}) {
            parameters.theta = static_cast<PetscReal>(theta);
            Blocks b; PetscCall(Compute(e,cell,edge,p,parameters,{},b));
            PetscCall(Near(b.s.C,ratioIntegral,m[0][0],"Variable cell samples: integral phi/(1-phi)"));
            PetscCall(Near(b.d.C,ratioIntegral/p.average,m[0][0],"Supplied average in Darcy compaction"));
            PetscCall(Near(b.k,-ratioIntegral/PetscSqrtReal(p.average),m[0][0],"Supplied average in coupling"));
            for (int side = 0; side < 4; ++side) {
                const PetscReal a = PetscReal(0.12)+PetscReal(0.02)*side;
                const PetscReal slope = side%2 == 0 ? PetscReal(0.07) : PetscReal(-0.04);
                const PetscReal zeroth = theta == 0 ? a+slope/2 : a*a+a*slope+slope*slope/3;
                const PetscReal linear = theta == 0 ? -slope/6 : -a*slope/3-slope*slope/6;
                const PetscReal scale = Length(q,side)/PetscSqrtReal(p.average);
                PetscCall(Near(b.d.B[side],scale*linear,scale,"Variable edge linear moment and CCW sample order"));
                PetscCall(Near(b.d.B[side+4],Sign(side)*scale*zeroth,scale,"Variable edge constant moment"));
            }
        }
    }
    PetscFunctionReturn(PETSC_SUCCESS);
}

// Non-affine fields exercise edge-enrichment terms absent from affine patch
// tests. Reference integration uses a Duffy map on two physical triangles,
// independently of the production quadrilateral mapping/Jacobian and rule.
constexpr std::array<PetscReal,12> brCoefficients{{0.2,-0.3,0.7,0.1,-0.5,0.8,-0.1,0.4,0.6,-0.7,0.9,0.3}};
constexpr std::array<PetscReal,8> hdCoefficients{{0.3,-0.2,0.8,0.5,-0.6,0.7,0.4,-0.9}};
using Actions = std::array<PetscReal,5>; // Stokes energy, Darcy mass, both loads, Stokes divergence.
PetscErrorCode TriangleReference(const Element& e, const QuadVertices& q, Actions& result)
{
    PetscFunctionBeginUser;
    GaussRule1D rule; PetscCall(CreateGaussRule(24,rule));
    Actions work{};
    for (int triangle = 1; triangle < 3; ++triangle) {
        const Point a = q[0], b = q[triangle], c = q[triangle+1];
        const PetscReal determinant = (b.p[0]-a.p[0])*(c.p[1]-a.p[1]) -
                                       (b.p[1]-a.p[1])*(c.p[0]-a.p[0]);
        for (std::size_t i = 0; i < rule.points.size(); ++i)
            for (std::size_t j = 0; j < rule.points.size(); ++j) {
                const PetscReal u = (1+rule.points[i])/2, v = (1+rule.points[j])/2;
                Point point{};
                for (int d = 0; d < 2; ++d)
                    point.p[d] = a.p[d]+u*(b.p[d]-a.p[d])+(1-u)*v*(c.p[d]-a.p[d]);
                const PetscReal weight = determinant*(1-u)*rule.weights[i]*rule.weights[j]/4;
                BRMixed::Values br; HDivMixed::Values hd;
                PetscCall(e.br.EvaluateAll(point,br)); PetscCall(e.hdiv.EvaluateAll(point,hd));
                Point velocity{}, darcy{}; std::array<PetscReal,4> g{};
                for (int k = 0; k < 12; ++k) {
                    for (int d = 0; d < 2; ++d) velocity.p[d] += brCoefficients[k]*br[k].value.p[d];
                    for (int d = 0; d < 4; ++d) g[d] += brCoefficients[k]*br[k].gradient[d];
                }
                for (int k = 0; k < 8; ++k)
                    for (int d = 0; d < 2; ++d) darcy.p[d] += hdCoefficients[k]*hd[k].value.p[d];
                const PetscReal traceThird = (g[0]+g[3])/3, shear = (g[1]+g[2])/2;
                // Norm of the 3-D trace-free strain, with epsilon_zz=0.
                const PetscReal devnorm = (g[0]-traceThird)*(g[0]-traceThird) +
                    (g[3]-traceThird)*(g[3]-traceThird)+traceThird*traceThird+2*shear*shear;
                const PetscReal solid = 1-CellProfile(point);
                const Point force = ForceProfile(point);
                const Actions density{{2*solid*devnorm,Dot(darcy,darcy),
                                        solid*Dot(force,velocity),Dot(force,darcy),g[0]+g[3]}};
                for (std::size_t k = 0; k < work.size(); ++k) work[k] += weight*density[k];
            }
    }
    result = work;
    PetscFunctionReturn(PETSC_SUCCESS);
}
Actions BlockActions(const Blocks& b)
{
    return {{PetscRealPart(Bilinear(b.s,brCoefficients,brCoefficients)),
             PetscRealPart(Bilinear(b.d,hdCoefficients,hdCoefficients)),
             PetscRealPart(Action(brCoefficients,b.s.f)),PetscRealPart(Action(hdCoefficients,b.d.f)),
             PetscRealPart(Action(brCoefficients,b.s.B))}};
}
PetscErrorCode CheckQuadrature(const QuadVertices& q)
{
    PetscFunctionBeginUser;
    Element e; PetscCall(e.Initialize(q));
    Actions reference; PetscCall(TriangleReference(e,q,reference));
    PetscReal coarseError = 0, fineError = 0;
    for (PetscInt n : {2,8,16}) {
        GaussRule1D cell,edge;
        PetscCall(CreateGaussRule(n,cell)); PetscCall(CreateGaussRule(5,edge));
        auto p = VariablePorosity(q,cell,edge);
        LocalMatrixParameters parameters;
        std::size_t calls = 0;
        const LocalForceFunction force = [&calls](const Point& point) { ++calls; return ForceProfile(point); };
        Blocks b; PetscCall(Compute(e,cell,edge,p,parameters,force,b));
        PetscCall(Require(calls == 2*cell.points.size()*cell.points.size(), "Cell quadrature point count is not adjustable"));
        const Actions actual = BlockActions(b);
        PetscReal error = 0;
        for (std::size_t k = 0; k < actual.size(); ++k) {
            const PetscReal scale = std::max(PetscReal(1),PetscAbsReal(reference[k]));
            error = std::max(error,PetscAbsReal(actual[k]-reference[k])/scale);
            if (n == 16) PetscCall(Near(actual[k],reference[k],scale,"Independent triangle-integration reference",quadratureTolerance));
        }
        if (n == 2) coarseError = error;
        if (n == 16) fineError = error;
    }
    PetscCall(Require(coarseError > 100*quadratureTolerance, "Convergence fixture failed to expose underintegration"));
    PetscCall(Require(fineError < coarseError/100, "Increasing quadrature did not resolve underintegration"));
    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode CheckBranches(const QuadVertices& q)
{
    PetscFunctionBeginUser;
    Element e; PetscCall(e.Initialize(q));
    GaussRule1D cell,edge;
    PetscCall(CreateGaussRule(4,cell)); PetscCall(CreateGaussRule(3,edge));
    const PetscReal area = PolygonMoments(q)[0][0];
    LocalMatrixParameters parameters;
    auto p = ConstantPorosity(cell,edge,PetscReal(0.2));
    // These fields are intentionally distinct to detect recomputation of the
    // supplied average and conflation of the three degeneracy branches.
    for (auto& side : p.edge) std::fill(side.begin(),side.end(),PetscReal(0.09));
    const auto check = [&](PetscReal average, bool darcyScaled, bool coupled) -> PetscErrorCode {
        PetscFunctionBeginUser;
        p.average = average;
        Blocks b; PetscCall(Compute(e,cell,edge,p,parameters,{},b));
        const PetscReal expectedD = darcyScaled ? area*PetscReal(0.25)/average : area/PetscReal(0.8);
        const PetscReal expectedK = coupled ? -area*PetscReal(0.25)/PetscSqrtReal(average) : 0;
        PetscCall(Near(b.s.C,area*PetscReal(0.25),area,"Stokes must ignore average"));
        PetscCall(Near(b.d.C,expectedD,PetscAbsReal(expectedD),"Darcy cutoff branch"));
        PetscCall(Near(b.k,expectedK,PetscAbsReal(expectedK),"Coupling cutoff branch"));
        if (!coupled) PetscCall(Require(b.k == PetscScalar(0),"Inactive coupling is not exactly zero"));
        const PetscReal factor = average == 0 ? PetscReal(0.09) : PetscReal(0.09)/PetscSqrtReal(average);
        for (int side = 0; side < 4; ++side)
            PetscCall(Near(b.d.B[side+4],Sign(side)*Length(q,side)*factor,Length(q,side)*factor,
                           "Darcy edge scaling must use exact-zero substitution only"));
        PetscFunctionReturn(PETSC_SUCCESS);
    };
    PetscCall(check(0,false,false));
    const PetscReal cc = parameters.couplingAverageCutoff;
    const PetscReal dc = parameters.darcyCompactionAverageCutoff;
    PetscCall(check(std::nextafter(cc,PetscReal(0)),false,false));
    PetscCall(check(cc,false,false));
    PetscCall(check(std::nextafter(cc,PetscReal(1)),false,true));
    PetscCall(check(dc,false,true));
    PetscCall(check(std::nextafter(dc,PetscReal(1)),true,true));
    PetscCall(check(PetscReal(0.2),true,true));
    parameters.darcyCompactionAverageCutoff = PetscReal(0.3);
    parameters.couplingAverageCutoff = PetscReal(0.1);
    PetscCall(check(PetscReal(0.2),false,true));
    parameters.darcyCompactionAverageCutoff = PetscReal(0.1);
    parameters.couplingAverageCutoff = PetscReal(0.3);
    PetscCall(check(PetscReal(0.2),true,false));
    parameters.darcyCompactionAverageCutoff = 0; parameters.couplingAverageCutoff = 0;
    PetscCall(check(0,false,false)); PetscCall(check(PetscReal(1e-18),true,true));

    // Fully dry cell, including zero edge porosity, with the ordinary theta=0.
    parameters = LocalMatrixParameters{}; p = ConstantPorosity(cell,edge,0);
    Blocks dry; PetscCall(Compute(e,cell,edge,p,parameters,{},dry));
    PetscCall(Near(dry.s.C,0,0,"Dry Stokes C")); PetscCall(Near(dry.k,0,0,"Dry coupling"));
    PetscCall(Near(dry.d.C,area,area,"Dry Darcy C must retain its fallback value"));
    for (const auto& value : dry.d.B) PetscCall(Near(value,0,0,"Dry Darcy edge flux"));
    parameters.theta = -1; // Explicit legacy pow(0,0)=1 behavior.
    PetscCall(Compute(e,cell,edge,p,parameters,{},dry));
    for (int side = 0; side < 4; ++side)
        PetscCall(Near(dry.d.B[side+4],Sign(side)*Length(q,side),Length(q,side),"Zero exponent at zero edge porosity"));
    // Edge phi=1 is admissible; only CELL phi=1 has a singular denominator.
    p = ConstantPorosity(cell,edge,PetscReal(0.2)); parameters.theta = 0;
    for (auto& side : p.edge) std::fill(side.begin(),side.end(),PetscReal(1));
    PetscCall(Compute(e,cell,edge,p,parameters,{},dry));
    for (int side = 0; side < 4; ++side)
        PetscCall(Near(dry.d.B[side+4],Sign(side)*Length(q,side)/PetscSqrtReal(p.average),Length(q,side),"Unit edge porosity"));
    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode CheckReuse(const QuadVertices& q)
{
    PetscFunctionBeginUser;
    Element e; PetscCall(e.Initialize(q));
    GaussRule1D cell,edge;
    PetscCall(CreateGaussRule(4,cell)); PetscCall(CreateGaussRule(5,edge));
    auto p = ConstantPorosity(cell,edge,PetscReal(0.2));
    LocalMatrixParameters parameters;
    PetscReal amplitude = 1;
    const LocalForceFunction force = [&amplitude](const Point&) { return Point{{amplitude,-2*amplitude}}; };
    Blocks a; PetscCall(Compute(e,cell,edge,p,parameters,force,a));
    const Blocks first = a;
    PetscCall(Compute(e,cell,edge,p,parameters,force,a));
    PetscCall(Require(Same(a.s,first.s) && Same(a.d,first.d) && a.k == first.k,"Repeated call accumulated old results"));
    amplitude = -3;
    PetscCall(Compute(e,cell,edge,p,parameters,force,a));
    for (int k = 0; k < 12; ++k) PetscCall(Near(a.s.f[k],-3*PetscRealPart(first.s.f[k]),1,"Runtime Stokes force update"));
    for (int k = 0; k < 8; ++k) PetscCall(Near(a.d.f[k],-3*PetscRealPart(first.d.f[k]),1,"Runtime Darcy force update"));
    PetscCall(Require(a.s.A == first.s.A && a.s.B == first.s.B && a.s.C == first.s.C &&
                      a.d.A == first.d.A && a.d.B == first.d.B && a.d.C == first.d.C,"Force update changed a matrix block"));
    PetscCall(Compute(e,cell,edge,p,parameters,{},a));
    for (const auto& v : a.s.f) PetscCall(Near(v,0,0,"Empty Stokes force must clear f"));
    for (const auto& v : a.d.f) PetscCall(Near(v,0,0,"Empty Darcy force must clear f"));
    p = ConstantPorosity(cell,edge,PetscReal(0.4)); parameters.theta = 1;
    PetscCall(Compute(e,cell,edge,p,parameters,{},a));
    PetscCall(Require(a.d.A == first.d.A,"Darcy mass acquired a porosity/material prefactor"));
    for (std::size_t k = 0; k < a.s.A.size(); ++k)
        PetscCall(Near(a.s.A[k],PetscReal(0.75)*PetscRealPart(first.s.A[k]),1,"Runtime Stokes porosity update"));
    for (int side = 0; side < 4; ++side)
        PetscCall(Near(a.d.B[side+4],Sign(side)*Length(q,side)*PetscReal(0.16)/PetscSqrtReal(PetscReal(0.4)),
                       Length(q,side),"Runtime theta and edge porosity update"));
    // Reusing an output on different geometry must match a fresh result.
    QuadVertices scaled = q;
    for (auto& v : scaled) { v.p[0] = 2*v.p[0]+1; v.p[1] = 3*v.p[1]-2; }
    PetscCall(e.Initialize(scaled));
    PetscCall(Compute(e,cell,edge,p,parameters,force,a));
    Blocks fresh; PetscCall(Compute(e,cell,edge,p,parameters,force,fresh));
    PetscCall(Require(Same(a.s,fresh.s) && Same(a.d,fresh.d) && a.k == fresh.k,"Geometry reuse retained stale output"));
    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode CheckErrors(const QuadVertices& q)
{
    PetscFunctionBeginUser;
    Element e, empty; PetscCall(e.Initialize(q));
    GaussRule1D cell,edge;
    PetscCall(CreateGaussRule(3,cell)); PetscCall(CreateGaussRule(4,edge));
    auto p = ConstantPorosity(cell,edge,PetscReal(0.2));
    LocalMatrixParameters parameters;
    Blocks output; PetscCall(Compute(e,cell,edge,p,parameters,ForceProfile,output));
    const auto stokes = [&]() { return ComputeLocalStokes(e.br,cell,p,{},output.s); };
    const auto darcy = [&]() { return ComputeLocalDarcy(e.hdiv,cell,edge,p,parameters,{},output.d); };
    const auto coupling = [&]() { return ComputeLocalCoupling(e.geometry,cell,p,parameters,output.k); };
    PetscCall(RejectBlock([&] { return ComputeLocalStokes(empty.br,cell,p,{},output.s); },output.s,PETSC_ERR_ARG_WRONG));
    PetscCall(RejectBlock([&] { return ComputeLocalDarcy(empty.hdiv,cell,edge,p,parameters,{},output.d); },output.d,PETSC_ERR_ARG_WRONG));
    PetscCall(RejectScalar([&] { return ComputeLocalCoupling(empty.geometry,cell,p,parameters,output.k); },output.k,PETSC_ERR_ARG_WRONG));
    const auto valid = p;
    p.cell.pop_back();
    PetscCall(RejectBlock(stokes,output.s,PETSC_ERR_ARG_SIZ));
    PetscCall(RejectBlock(darcy,output.d,PETSC_ERR_ARG_SIZ));
    PetscCall(RejectScalar(coupling,output.k,PETSC_ERR_ARG_SIZ));
    p = valid; p.edge[3].pop_back();
    PetscCall(RejectBlock(darcy,output.d,PETSC_ERR_ARG_SIZ));
    PetscCall(stokes()); PetscCall(coupling()); // Unused edge inputs must be ignored.
    const PetscReal nan = std::numeric_limits<PetscReal>::quiet_NaN();
    const PetscReal infinity = std::numeric_limits<PetscReal>::infinity();
    for (PetscReal invalid : {PetscReal(-0.01),PetscReal(1),PetscReal(1.1),nan,infinity}) {
        p = valid; p.cell.back() = invalid;
        PetscCall(RejectBlock(stokes,output.s,PETSC_ERR_ARG_OUTOFRANGE));
        PetscCall(RejectBlock(darcy,output.d,PETSC_ERR_ARG_OUTOFRANGE));
        PetscCall(RejectScalar(coupling,output.k,PETSC_ERR_ARG_OUTOFRANGE));
        p = valid; p.average = invalid;
        PetscCall(RejectBlock(darcy,output.d,PETSC_ERR_ARG_OUTOFRANGE));
        PetscCall(RejectScalar(coupling,output.k,PETSC_ERR_ARG_OUTOFRANGE));
        PetscCall(stokes()); // Average is not a Stokes input.
    }
    for (PetscReal invalid : {PetscReal(-0.01),PetscReal(1.1),nan,infinity}) {
        p = valid; p.edge[3].back() = invalid;
        PetscCall(RejectBlock(darcy,output.d,PETSC_ERR_ARG_OUTOFRANGE));
    }
    p = valid;
    for (PetscReal invalid : {PetscReal(-1),PetscReal(1),nan,infinity}) {
        parameters.darcyCompactionAverageCutoff = invalid;
        PetscCall(RejectBlock(darcy,output.d,PETSC_ERR_ARG_OUTOFRANGE));
        parameters = LocalMatrixParameters{}; parameters.couplingAverageCutoff = invalid;
        PetscCall(RejectScalar(coupling,output.k,PETSC_ERR_ARG_OUTOFRANGE));
        PetscCall(darcy()); // Coupling cutoff is not used by Darcy.
        parameters = LocalMatrixParameters{};
    }
    parameters.theta = nan; parameters.darcyCompactionAverageCutoff = nan;
    PetscCall(coupling());
    parameters = LocalMatrixParameters{};
    for (PetscReal invalid : {nan,infinity}) {
        parameters.theta = invalid;
        PetscCall(RejectBlock(darcy,output.d,PETSC_ERR_ARG_OUTOFRANGE));
    }
    parameters.theta = -2; p.edge[3].back() = 0;
    PetscCall(RejectBlock(darcy,output.d,PETSC_ERR_ARG_OUTOFRANGE)); // After volume integration.
    p = valid; parameters = LocalMatrixParameters{};
    const auto validCell = cell, validEdge = edge;
    cell.points.clear();
    PetscCall(RejectBlock(stokes,output.s,PETSC_ERR_ARG_WRONG));
    PetscCall(RejectScalar(coupling,output.k,PETSC_ERR_ARG_WRONG));
    cell = validCell; cell.weights[0] = -1;
    PetscCall(RejectBlock(stokes,output.s,PETSC_ERR_ARG_OUTOFRANGE));
    cell = validCell; cell.weights[0] *= 2;
    PetscCall(RejectBlock(stokes,output.s,PETSC_ERR_ARG_WRONG));
    cell = validCell; cell.points[0] = 2;
    PetscCall(RejectBlock(stokes,output.s,PETSC_ERR_ARG_OUTOFRANGE));
    cell = validCell; edge.weights.pop_back();
    PetscCall(RejectBlock(darcy,output.d,PETSC_ERR_ARG_WRONG));
    edge = validEdge; edge.points[0] = nan;
    PetscCall(RejectBlock(darcy,output.d,PETSC_ERR_ARG_OUTOFRANGE));
    edge = validEdge;
    // Fail on the third callback to exercise transactional output after partial
    // integration. Both standard and nonstandard C++ exceptions are covered.
    for (int failure = 0; failure < 4; ++failure) {
        int calls = 0;
        const LocalForceFunction force = [&](const Point&) -> Point {
            if (++calls == 3) {
                if (failure == 0) return {{nan,0}};
                if (failure == 1) return {{0,infinity}};
                if (failure == 2) throw std::runtime_error("intentional force failure");
                throw 17;
            }
            return {{1,-2}};
        };
        const PetscErrorCode expected = failure < 2 ? PETSC_ERR_FP : PETSC_ERR_USER;
        PetscCall(RejectBlock([&] { return ComputeLocalStokes(e.br,cell,p,force,output.s); },output.s,expected));
        calls = 0;
        PetscCall(RejectBlock([&] { return ComputeLocalDarcy(e.hdiv,cell,edge,p,parameters,force,output.d); },output.d,expected));
    }
    // Successful calls after all expected failures also exercise PETSc's stack.
    PetscCall(Compute(e,cell,edge,p,parameters,{},output));
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
        {{{{0,0}},{{2,0.2}},{{1.8,1.6}},{{-0.15,1.3}}}},
        {{{{-1,0.2}},{{3,0.2}},{{3,0.4}},{{-1,0.4}}}}
    }};
    for (const auto& q : fixtures) {
        PetscCall(CheckCell(q)); PetscCall(CheckVariablePorosity(q));
    }
    for (int i : {0,4}) {
        PetscCall(CheckQuadrature(fixtures[i])); PetscCall(CheckBranches(fixtures[i]));
        PetscCall(CheckReuse(fixtures[i])); PetscCall(CheckErrors(fixtures[i]));
    }
    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode MakeSnapshot(MPI_Comm comm, PetscInt nx, PetscInt ny, PetscInt px,
                            PetscInt py, PetscInt kind, const MeshParam& p, MeshInfo& mesh)
{
    PetscFunctionBeginUser;
    DM dm = nullptr; Vec coordinates = nullptr;
    PetscCall(DMDACreate2d(comm,DM_BOUNDARY_NONE,DM_BOUNDARY_NONE,DMDA_STENCIL_BOX,
                          nx,ny,px,py,2,1,nullptr,nullptr,&dm));
    PetscCall(DMSetUp(dm)); PetscCall(DMCreateGlobalVector(dm,&coordinates));
    if (kind == 0) PetscCall(CreateFullMesh(dm,coordinates,p));
    else if (kind == 1) PetscCall(LogicRectMesh(dm,coordinates,p));
    else PetscCall(RefineMesh(dm,coordinates,p));
    PetscCall(BuildMeshInfo(dm,coordinates,mesh));
    PetscCall(VecDestroy(&coordinates)); PetscCall(DMDestroy(&dm));
    PetscFunctionReturn(PETSC_SUCCESS);
}

template<std::size_t N>
PetscErrorCode CompareBlocks(const LocalMatrixBlock<N>& actual, const LocalMatrixBlock<N>& expected)
{
    PetscFunctionBeginUser;
    for (std::size_t i = 0; i < N*N; ++i)
        PetscCall(Near(actual.A[i],PetscRealPart(expected.A[i]),1,"Distributed/serial local A"));
    for (std::size_t i = 0; i < N; ++i) {
        PetscCall(Near(actual.B[i],PetscRealPart(expected.B[i]),1,"Distributed/serial local B"));
        PetscCall(Near(actual.f[i],PetscRealPart(expected.f[i]),1,"Distributed/serial local f"));
    }
    PetscCall(Near(actual.C,PetscRealPart(expected.C),1,"Distributed/serial local C"));
    PetscFunctionReturn(PETSC_SUCCESS);
}
PetscReal MeshEdgeProfile(const Point& p)
{ return PetscReal(0.22)+PetscReal(0.015)*p.p[0]-PetscReal(0.02)*p.p[1]; }

PetscErrorCode CheckMesh(PetscInt nx, PetscInt ny, PetscInt px, PetscInt py,
                         PetscInt kind, PetscMPIInt ranks)
{
    PetscFunctionBeginUser;
    MeshParam parameters;
    parameters.xstart = -0.75; parameters.ystart = 0.25;
    parameters.L = 2.7; parameters.H = 1.3;
    parameters.seed = 79; parameters.perturbation = 0.22;
    MeshInfo distributed,serial;
    PetscCall(MakeSnapshot(PETSC_COMM_WORLD,nx,ny,px,py,kind,parameters,distributed));
    PetscCall(MakeSnapshot(PETSC_COMM_SELF,nx,ny,1,1,kind,parameters,serial));
    GaussRule1D cell,edge;
    PetscCall(CreateGaussRule(4,cell)); PetscCall(CreateGaussRule(3,edge));
    const AffineVector f{{{{0.7,-0.1,0.4}},{{-0.2,0.3,0.8}}}};
    const LocalForceFunction force = [f](const Point& point) { return Evaluate(f,point); };
    LocalMatrixParameters flow; flow.theta = PetscReal(0.5);
    std::vector<int> owners(static_cast<std::size_t>(serial.CellCount()),0);
    std::vector<int> incidence(static_cast<std::size_t>(serial.EdgeCount()),0);
    std::vector<PetscReal> flux(2*incidence.size(),0);
    std::array<PetscReal,4> totals{}; // Area, Stokes C, Darcy C, coupling.
    int localCells = 0;
    const auto owned = distributed.OwnedCells();
    for (PetscInt j = owned.begin.j; j < owned.end.j; ++j)
        for (PetscInt i = owned.begin.i; i < owned.end.i; ++i) {
            const MeshIndex index{i,j}; ++localCells;
            PetscInt id = 0; PetscCall(distributed.CellId(index,id));
            ++owners[static_cast<std::size_t>(id)];
            QuadVertices q,referenceCorners;
            PetscCall(distributed.GetCellCorners(index,q)); PetscCall(serial.GetCellCorners(index,referenceCorners));
            for (int v = 0; v < 4; ++v) for (int d = 0; d < 2; ++d)
                PetscCall(Near(q[v].p[d],referenceCorners[v].p[d],1,"Distributed geometry including ghost corners"));
            Element actual,reference;
            PetscCall(actual.Initialize(distributed,index)); PetscCall(reference.Initialize(serial,index));
            const PetscReal phi = PetscReal(0.12)+PetscReal(0.005)*static_cast<PetscReal>(id%17);
            auto p = ConstantPorosity(cell,edge,phi);
            Blocks a,b;
            PetscCall(Compute(actual,cell,edge,p,flow,force,a));
            PetscCall(Compute(reference,cell,edge,p,flow,force,b));
            PetscCall(CompareBlocks(a.s,b.s)); PetscCall(CompareBlocks(a.d,b.d));
            PetscCall(Near(a.k,PetscRealPart(b.k),1,"Distributed/serial coupling"));
            PetscCall(CheckAffineBlocks(q,a,phi,f,true));
            const PetscReal area = PolygonMoments(q)[0][0];
            totals[0] += area; totals[1] += PetscRealPart(a.s.C);
            totals[2] += PetscRealPart(a.d.C); totals[3] += PetscRealPart(a.k);

            // Equal physical edge data on neighbors, differing cell averages.
            // Undo only the scalar 1/sqrt(average), not a basis orientation sign.
            for (int side = 0; side < 4; ++side)
                for (std::size_t g = 0; g < edge.points.size(); ++g)
                    p.edge[side][g] = MeshEdgeProfile(Mix(q[side],q[(side+1)%4],(1+edge.points[g])/2));
            LocalMatrixParameters edgeFlow; edgeFlow.theta = 0;
            DarcyLocalMatrix boundary;
            PetscCall(ComputeLocalDarcy(actual.hdiv,cell,edge,p,edgeFlow,{},boundary));
            std::array<OrientedEdge,4> edges;
            PetscCall(distributed.GetCellEdges(index,edges));
            for (int side = 0; side < 4; ++side) {
                const std::size_t edgeId = static_cast<std::size_t>(edges[side].id);
                const PetscReal start = MeshEdgeProfile(q[side]), end = MeshEdgeProfile(q[(side+1)%4]);
                const PetscReal scale = Length(q,side)/PetscSqrtReal(phi);
                PetscCall(Near(boundary.B[side],-scale*(end-start)/6,scale,"MPI linear edge moment"));
                PetscCall(Near(boundary.B[side+4],Sign(side)*scale*(start+end)/2,scale,"MPI constant edge moment"));
                ++incidence[edgeId];
                flux[2*edgeId] += PetscSqrtReal(phi)*PetscRealPart(boundary.B[side]);
                flux[2*edgeId+1] += PetscSqrtReal(phi)*PetscRealPart(boundary.B[side+4]);
            }
        }
    // All collectives are outside cell loops, so empty cell owners participate.
    std::vector<int> globalOwners(owners.size()),globalIncidence(incidence.size());
    std::vector<PetscReal> globalFlux(flux.size());
    std::array<PetscReal,4> globalTotals{},expectedTotals{};
    PetscCallMPI(MPI_Allreduce(owners.data(),globalOwners.data(),static_cast<int>(owners.size()),MPI_INT,MPI_SUM,PETSC_COMM_WORLD));
    PetscCallMPI(MPI_Allreduce(incidence.data(),globalIncidence.data(),static_cast<int>(incidence.size()),MPI_INT,MPI_SUM,PETSC_COMM_WORLD));
    PetscCallMPI(MPI_Allreduce(flux.data(),globalFlux.data(),static_cast<int>(flux.size()),MPIU_REAL,MPI_SUM,PETSC_COMM_WORLD));
    PetscCallMPI(MPI_Allreduce(totals.data(),globalTotals.data(),4,MPIU_REAL,MPI_SUM,PETSC_COMM_WORLD));
    for (int count : globalOwners) PetscCall(Require(count == 1,"Global cell was omitted or counted more than once"));
    for (PetscInt id = 0; id < serial.CellCount(); ++id) {
        MeshIndex index; QuadVertices q;
        PetscCall(serial.CellIndex(id,index)); PetscCall(serial.GetCellCorners(index,q));
        const PetscReal area = PolygonMoments(q)[0][0];
        const PetscReal phi = PetscReal(0.12)+PetscReal(0.005)*static_cast<PetscReal>(id%17);
        expectedTotals[0] += area; expectedTotals[1] += area*phi/(1-phi);
        expectedTotals[2] += area/(1-phi); expectedTotals[3] -= area*PetscSqrtReal(phi)/(1-phi);
    }
    PetscCall(Near(globalTotals[0],parameters.L*parameters.H,parameters.L*parameters.H,"Global physical domain area"));
    for (int k = 0; k < 4; ++k)
        PetscCall(Near(globalTotals[k],expectedTotals[k],PetscAbsReal(expectedTotals[k]),"Global analytic integral over owned cells"));
    for (PetscInt id = 0; id < serial.EdgeCount(); ++id) {
        EdgeTopology topology; PetscCall(serial.GetEdgeTopology(id,topology));
        PetscCall(Require(globalIncidence[static_cast<std::size_t>(id)] == (topology.IsBoundary() ? 1 : 2),"Global edge incidence"));
        if (!topology.IsBoundary()) {
            PetscCall(Near(globalFlux[2*static_cast<std::size_t>(id)],0,1,"Interior linear edge flux cancellation"));
            PetscCall(Near(globalFlux[2*static_cast<std::size_t>(id)+1],0,1,"Interior constant edge flux cancellation"));
        }
    }
    const int empty = localCells == 0 ? 1 : 0; int emptyRanks = 0;
    PetscCallMPI(MPI_Allreduce(&empty,&emptyRanks,1,MPI_INT,MPI_SUM,PETSC_COMM_WORLD));
    if (nx == 2 && ny == 2) PetscCall(Require(emptyRanks == ranks-1,"Single-cell case did not exercise empty owners"));
    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode Run()
{
    PetscFunctionBeginUser;
    PetscInt kind = -1,expectedRanks = 1,px = 1,py = 1,nx = 9,ny = 7;
    PetscCall(PetscOptionsGetInt(nullptr,nullptr,"-local_matrix_mesh_type",&kind,nullptr));
    PetscCall(PetscOptionsGetInt(nullptr,nullptr,"-expected_ranks",&expectedRanks,nullptr));
    PetscCall(PetscOptionsGetInt(nullptr,nullptr,"-mesh_px",&px,nullptr));
    PetscCall(PetscOptionsGetInt(nullptr,nullptr,"-mesh_py",&py,nullptr));
    PetscCall(PetscOptionsGetInt(nullptr,nullptr,"-mesh_nx",&nx,nullptr));
    PetscCall(PetscOptionsGetInt(nullptr,nullptr,"-mesh_ny",&ny,nullptr));
    PetscMPIInt ranks = 0; PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD,&ranks));
    PetscCheck(expectedRanks == ranks,PETSC_COMM_WORLD,PETSC_ERR_ARG_WRONG,
               "Expected %d ranks but PETSc sees %d; use the MPICH launcher matching PETSc",
               static_cast<int>(expectedRanks),static_cast<int>(ranks));
    PetscCall(Require(kind >= -1 && kind <= 2,"local_matrix_mesh_type must be -1, 0, 1 or 2"));
    if (kind == -1) {
        PetscCall(Require(ranks == 1,"Run the unit suite on one rank"));
        PetscCall(RunUnitTests());
        PetscCall(PetscPrintf(PETSC_COMM_WORLD,"Local matrix unit tests passed\n"));
    } else {
        PetscCall(Require((px == 1 || px == 2) && (py == 1 || py == 2) && px*py == ranks,
                          "Use a matching 1x1, 2x1, 1x2 or 2x2 process grid"));
        PetscCall(Require(nx >= px && ny >= py && nx >= 2 && ny >= 2 && nx <= 64 && ny <= 64,
                          "Test mesh requires 2..64 vertices per direction and enough vertices for the process grid"));
        PetscCall(CheckMesh(nx,ny,px,py,kind,ranks));
        PetscCall(PetscPrintf(PETSC_COMM_WORLD,"Local matrix mesh type %d passed on %d rank(s)\n",
                              static_cast<int>(kind),static_cast<int>(ranks)));
    }
    PetscFunctionReturn(PETSC_SUCCESS);
}

} // namespace

int main(int argc, char** argv)
{
    const PetscErrorCode error = PetscInitialize(&argc,&argv,nullptr,
        "Local matrix tests: default unit suite; -local_matrix_mesh_type 0/1/2 selects mesh checks.\n");
    if (error) return static_cast<int>(error);
    // Abort WORLD on an unexpected local failure, so other ranks cannot hang
    // waiting at a later collective. Expected errors are handled inside the suite.
    PetscCallAbort(PETSC_COMM_WORLD,Run());
    return static_cast<int>(PetscFinalize());
}
