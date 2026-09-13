#include "integral.h"

#include <algorithm>
#include <array>
#include <limits>
#include <vector>

namespace {

struct Power { int x, y; };
constexpr std::array<Power, 15> powers{{
    {0,0}, {1,0}, {0,1}, {2,0}, {1,1}, {0,2},
    {3,0}, {2,1}, {1,2}, {0,3},
    {4,0}, {3,1}, {2,2}, {1,3}, {0,4}
}};
constexpr int fieldCount = static_cast<int>(powers.size());
using Generator = PetscErrorCode (*)(DM, Vec, const MeshParam&);

Point P(PetscReal x, PetscReal y) { return Point{{x, y}}; }
PetscReal One(const Point&) { return 1.0; }

PetscReal IntegerPower(PetscReal x, int n)
{
    PetscReal result = 1.0;
    for (int i = 0; i < n; ++i) result *= x;
    return result;
}

PetscReal Binomial(int n, int k)
{
    PetscReal result = 1.0;
    for (int i = 1; i <= k; ++i) result *= PetscReal(n - i + 1) / i;
    return result;
}

PetscErrorCode Require(bool condition, const char* message)
{
    PetscFunctionBeginUser;
    PetscCheck(condition, PETSC_COMM_SELF, PETSC_ERR_PLIB, "%s", message);
    PetscFunctionReturn(PETSC_SUCCESS);
}

// An explicit scale makes small-cell comparisons relative to their area,
// instead of silently accepting a large absolute error on thin meshes.
PetscErrorCode Near(PetscReal actual, PetscReal expected, PetscReal scale,
                     const char* message)
{
    PetscFunctionBeginUser;
    const PetscReal tolerance = 4096.0 * PETSC_MACHINE_EPSILON
                             * std::max(PetscAbsReal(expected), PetscAbsReal(scale));
    PetscCheck(!PetscIsInfOrNanReal(actual) && !PetscIsInfOrNanReal(expected) &&
               PetscAbsReal(actual - expected) <= tolerance,
               PETSC_COMM_SELF, PETSC_ERR_PLIB,
               "%s: got %.17g, expected %.17g, tolerance %.3g", message,
               static_cast<double>(actual), static_cast<double>(expected),
               static_cast<double>(tolerance));
    PetscFunctionReturn(PETSC_SUCCESS);
}

template <typename Call>
PetscErrorCode ExpectError(Call&& call, const char* message)
{
    PetscFunctionBeginUser;
    PetscCall(PetscPushErrorHandler(PetscReturnErrorHandler, nullptr));
    const PetscErrorCode error = call();
    PetscCall(PetscPopErrorHandler());
    PetscCall(Require(error != PETSC_SUCCESS, message));
    PetscFunctionReturn(PETSC_SUCCESS);
}

QuadVertices Rectangle(PetscReal x0, PetscReal x1, PetscReal y0, PetscReal y1)
{
    return {{P(x0,y0), P(x1,y0), P(x1,y1), P(x0,y1)}};
}

Point NormalizedPoint(const Point& point, const MeshParam& box)
{
    return P((point.p[0] - box.xstart) / box.L,
             (point.p[1] - box.ystart) / box.H);
}

QuadVertices NormalizedCorners(const QuadVertices& corners, const MeshParam& box)
{
    QuadVertices result{};
    for (std::size_t k = 0; k < corners.size(); ++k)
        result[k] = NormalizedPoint(corners[k], box);
    return result;
}

MeshParam BoundingBox(const QuadVertices& corners)
{
    MeshParam box;
    box.xstart = corners[0].p[0];
    box.ystart = corners[0].p[1];
    PetscReal xmax = box.xstart, ymax = box.ystart;
    for (const Point& p : corners) {
        box.xstart = std::min(box.xstart, p.p[0]);
        box.ystart = std::min(box.ystart, p.p[1]);
        xmax = std::max(xmax, p.p[0]);
        ymax = std::max(ymax, p.p[1]);
    }
    box.L = xmax - box.xstart;
    box.H = ymax - box.ystart;
    return box;
}

// Independent exact polynomial oracle. Green's theorem gives
// integral_K x^a y^b dA = sum_edges integral x^(a+1)y^b dy / (a+1).
// Expand each straight edge with the binomial theorem and integrate powers of t.
// No Gauss rule, bilinear map, or production Jacobian is used here.
PetscReal PolygonMoment(const QuadVertices& corners, int a, int b)
{
    PetscReal result = 0.0;
    for (std::size_t e = 0; e < corners.size(); ++e) {
        const Point& p = corners[e];
        const Point& q = corners[(e + 1) % corners.size()];
        const PetscReal dx = q.p[0] - p.p[0], dy = q.p[1] - p.p[1];
        for (int i = 0; i <= a + 1; ++i) {
            for (int j = 0; j <= b; ++j) {
                result += dy / (a + 1) * Binomial(a + 1, i) * Binomial(b, j)
                        * IntegerPower(p.p[0], a + 1 - i) * IntegerPower(dx, i)
                        * IntegerPower(p.p[1], b - j) * IntegerPower(dy, j)
                        / (i + j + 1);
            }
        }
    }
    return result;
}

PetscReal Evaluate(const Point& point, const MeshParam& box, Power power)
{
    const Point p = NormalizedPoint(point, box);
    return IntegerPower(p.p[0], power.x) * IntegerPower(p.p[1], power.y);
}

PetscErrorCode CheckRules()
{
    PetscFunctionBeginUser;
    const QuadVertices square = Rectangle(0,1,0,1);
    for (PetscInt n = 1; n <= 8; ++n) {
        GaussRule1D rule;
        PetscCall(CreateGaussRule(n, rule));
        PetscCall(Require(rule.points.size() == static_cast<std::size_t>(n) &&
                           rule.weights.size() == static_cast<std::size_t>(n),
                           "Wrong number of Gauss points or weights"));
        for (int degree = 0; degree <= 2*n - 1; ++degree) {
            PetscReal moment = 0.0;
            for (PetscInt i = 0; i < n; ++i)
                moment += rule.weights[i] * IntegerPower(rule.points[i], degree);
            const PetscReal exact = degree % 2 ? 0.0 : 2.0 / (degree + 1);
            PetscCall(Near(moment, exact, 2.0, "Reference polynomial exactness"));
        }
        int evaluations = 0;
        PetscReal result = 0.0;
        PetscCall(IntegrateCell(square, rule, [&](const Point&) {
            ++evaluations; return PetscReal(1.0);
        }, result));
        PetscCall(Require(evaluations == n*n, "Cell loop does not use n*n points"));
        PetscCall(Near(result, 1.0, 1.0, "Reference square area"));
        evaluations = 0;
        PetscCall(IntegrateEdge(EdgeVertices{{P(0,0),P(1,0)}}, rule,
            [&](const Point&) { ++evaluations; return PetscReal(1.0); }, result));
        PetscCall(Require(evaluations == n, "Edge loop does not use n points"));

        // Includes x^5*y^5 for n=3 and x^7*y^7 for n>=4.
        const int maximum = std::min(2*static_cast<int>(n) - 1, 7);
        for (int a = 0; a <= maximum; ++a) {
            for (int b = 0; b <= maximum; ++b) {
                PetscCall(IntegrateCell(square, rule, [=](const Point& p) {
                    return IntegerPower(p.p[0], a) * IntegerPower(p.p[1], b);
                }, result));
                PetscCall(Near(result, 1.0 / ((a + 1)*(b + 1)), 1.0,
                                "Tensor-product polynomial exactness"));
            }
        }
    }
    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode CheckCellShapes()
{
    PetscFunctionBeginUser;
    const std::array<QuadVertices, 6> shapes{{
        Rectangle(-0.7,1.3,-0.2,0.8),
        {{P(0,0),P(2,1),P(1.5,2),P(-0.5,1)}}, // rotated rectangle
        {{P(0,0),P(2,0.2),P(2.6,1.3),P(0.6,1.1)}}, // parallelogram
        {{P(0,0),P(2,0),P(1.5,1),P(0.2,1)}}, // trapezoid
        {{P(0,0),P(2,0),P(1.8,1.1),P(0.2,1)}}, // general convex quad
        Rectangle(0,1e-10,0,2) // high aspect ratio
    }};
    const std::array<Point,4> references{{P(-1,-1),P(1,-1),P(1,1),P(-1,1)}};
    for (const QuadVertices& cell : shapes) {
        const MeshParam box = BoundingBox(cell);
        const QuadVertices normalized = NormalizedCorners(cell, box);
        const PetscReal area = box.L * box.H * PolygonMoment(normalized, 0, 0);
        for (std::size_t k = 0; k < cell.size(); ++k) {
            const Point mapped = MapCellPoint(references[k], cell);
            PetscCall(Near(mapped.p[0], cell[k].p[0], box.L, "Cell corner map x"));
            PetscCall(Near(mapped.p[1], cell[k].p[1], box.H, "Cell corner map y"));
        }
        for (PetscInt n : {3,5}) {
            GaussRule1D rule;
            PetscCall(CreateGaussRule(n, rule));
            for (Power power : powers) {
                PetscReal result = 0.0;
                PetscCall(IntegrateCell(cell, rule, [&](const Point& p) {
                    return Evaluate(p, box, power);
                }, result));
                const PetscReal exact = box.L * box.H
                                      * PolygonMoment(normalized, power.x, power.y);
                PetscCall(Near(result, exact, area, "Physical cell polynomial moment"));
            }
        }
    }
    GaussRule1D rule;
    PetscCall(CreateGaussRule(3, rule));
    PetscReal result = 0.0;
    PetscCall(IntegrateCell(Rectangle(1e6,1e6+3,-1e6,-1e6+1.5), rule, One, result));
    PetscCall(Near(result, 4.5, 4.5, "Jacobian under a large translation"));
    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode CheckEdgesAndScaling()
{
    PetscFunctionBeginUser;
    GaussRule1D rule;
    PetscCall(CreateGaussRule(3, rule));
    const std::array<EdgeVertices,3> edges{{
        {{P(0,0),P(2,0)}}, {{P(0,0),P(0,1)}}, {{P(0,0),P(2,1)}}
    }};
    for (const EdgeVertices& edge : edges) {
        const EdgeVertices reverse{{edge[1],edge[0]}};
        const PetscReal dx=edge[1].p[0]-edge[0].p[0], dy=edge[1].p[1]-edge[0].p[1];
        const PetscReal exactLength=PetscSqrtReal(dx*dx+dy*dy);
        PetscReal length=0.0, result=0.0;
        Point normal{};
        PetscCall(GetEdgeGeometry(edge,length,normal));
        PetscCall(Near(length,exactLength,exactLength,"Edge length"));
        PetscCall(Near(normal.p[0],dy/exactLength,1.0,"Right normal x"));
        PetscCall(Near(normal.p[1],-dx/exactLength,1.0,"Right normal y"));
        for (int d=0; d<2; ++d) {
            PetscCall(Near(MapEdgePoint(-1,edge).p[d],edge[0].p[d],exactLength,"Edge start"));
            PetscCall(Near(MapEdgePoint(1,edge).p[d],edge[1].p[d],exactLength,"Edge end"));
        }
        const auto quadratic=[](const Point& p) { return p.p[0]*p.p[0]+p.p[1]*p.p[1]; };
        const PetscReal exactScalar=exactLength*(dx*dx+dy*dy)/3;
        PetscCall(IntegrateEdge(edge,rule,One,result));
        PetscCall(Near(result,exactLength,exactLength,"Constant line integral"));
        PetscCall(IntegrateEdge(edge,rule,quadratic,result));
        PetscCall(Near(result,exactScalar,exactLength,"Quadratic line integral"));
        PetscCall(IntegrateEdge(reverse,rule,quadratic,result));
        PetscCall(Near(result,exactScalar,exactLength,"Reversed scalar line integral"));
        const auto flux=[](const Point& p) { return P(p.p[0]*p.p[0],p.p[1]*p.p[1]); };
        const PetscReal exactFlux=(dx*dx*dy-dy*dy*dx)/3;
        PetscCall(IntegrateNormalFlux(edge,rule,flux,result));
        PetscCall(Near(result,exactFlux,exactLength,"Quadratic normal flux"));
        PetscCall(IntegrateNormalFlux(reverse,rule,flux,result));
        PetscCall(Near(result,-exactFlux,exactLength,"Reversed normal flux"));
    }
    const QuadVertices cell=Rectangle(-1,2,0.25,1.75);
    const Point center=P(0.5,1.0);
    const auto squareX=[](const Point& p) { return p.p[0]*p.p[0]; };
    PetscReal result=0.0;
    PetscCall(IntegrateCellNormalized(cell,rule,center,2.0,One,result));
    PetscCall(Near(result,4.5/4,4.5/4,"Normalized area"));
    const PetscReal exact=2*IntegerPower(PetscReal(0.75),3)/3*PetscReal(0.75);
    PetscCall(IntegrateCellNormalized(cell,rule,center,2.0,squareX,result));
    PetscCall(Near(result,exact,4.5/4,"Normalized coordinates and measure"));
    PetscCall(IntegrateCell(cell,rule,[&](const Point& p) {
        const PetscReal x=(p.p[0]-center.p[0])/2; return x*x;
    },result));
    PetscCall(Near(result,4*exact,4.5,"Scaled callback with physical measure"));
    PetscFunctionReturn(PETSC_SUCCESS);
}

// Closed-form boundary integral for exp(x+y); independent of Gauss quadrature.
PetscReal ExponentialIntegral(const QuadVertices& cell)
{
    PetscReal result=0.0;
    for (std::size_t e=0; e<cell.size(); ++e) {
        const Point& a=cell[e];
        const Point& b=cell[(e+1)%cell.size()];
        const PetscReal z=b.p[0]-a.p[0]+b.p[1]-a.p[1];
        const PetscReal factor=z == 0.0 ? 1.0 : (PetscExpReal(z)-1.0)/z;
        result+=(b.p[1]-a.p[1])*PetscExpReal(a.p[0]+a.p[1])*factor;
    }
    return result;
}

PetscErrorCode CheckNonPolynomial()
{
    PetscFunctionBeginUser;
    const std::array<QuadVertices,2> shapes{{
        Rectangle(0,1,0,1), {{P(0,0),P(2,0),P(1.8,1.1),P(0.2,1)}}
    }};
    for (const auto& cell : shapes) {
        const PetscReal exact=ExponentialIntegral(cell);
        PetscReal low=0.0, high=0.0;
        GaussRule1D coarse, fine;
        PetscCall(CreateGaussRule(2,coarse));
        PetscCall(CreateGaussRule(8,fine));
        const auto function=[](const Point& p) { return PetscExpReal(p.p[0]+p.p[1]); };
        PetscCall(IntegrateCell(cell,coarse,function,low));
        PetscCall(IntegrateCell(cell,fine,function,high));
        PetscCall(Near(high,exact,exact,"High-order non-polynomial integral"));
        const PetscReal floor=4096.0*PETSC_MACHINE_EPSILON*PetscAbsReal(exact);
        PetscCall(Require(PetscAbsReal(high-exact)<=std::max(floor,PetscAbsReal(low-exact)),
                           "Higher order did not improve the smooth integral"));
    }
    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode CheckErrors()
{
    PetscFunctionBeginUser;
    GaussRule1D rule;
    PetscCall(CreateGaussRule(3,rule));
    const auto originalPoints=rule.points;
    const auto originalWeights=rule.weights;
    PetscCall(ExpectError([&]{return CreateGaussRule(0,rule);},"Accepted zero Gauss points"));
    PetscCall(ExpectError([&]{return CreateGaussRule(-2,rule);},"Accepted negative Gauss points"));
    PetscCall(Require(rule.points==originalPoints && rule.weights==originalWeights,
                       "Failed creation changed the existing rule"));
    const PetscReal nan=std::numeric_limits<PetscReal>::quiet_NaN();
    const PetscReal inf=std::numeric_limits<PetscReal>::infinity();
    const QuadVertices cell=Rectangle(0,1,0,1);
    const EdgeVertices edge{{P(0,0),P(1,1)}};
    PetscReal result=123.0;
    const std::array<QuadVertices,6> invalid{{
        {{cell[0],cell[3],cell[2],cell[1]}}, // clockwise
        {{cell[0],cell[2],cell[1],cell[3]}}, // folded
        {{cell[0],cell[0],cell[2],cell[3]}}, // repeated corner
        {{P(0,0),P(2,0),P(0.2,0.1),P(0,1)}}, // concave
        Rectangle(0,1,0,0), // collapsed
        {{P(nan,0),cell[1],cell[2],cell[3]}}
    }};
    for (const auto& bad : invalid)
        PetscCall(ExpectError([&]{return IntegrateCell(bad,rule,One,result);},"Accepted invalid cell"));
    const std::array<PetscReal,4> scales{{0,-1,nan,inf}};
    for (PetscReal h : scales)
        PetscCall(ExpectError([&]{return IntegrateCellNormalized(cell,rule,P(0,0),h,One,result);},
                               "Accepted invalid normalization scale"));
    PetscCall(ExpectError([&]{return IntegrateCellNormalized(cell,rule,P(nan,0),1,One,result);},
                           "Accepted invalid normalization center"));
    PetscCall(ExpectError([&]{return IntegrateEdge(EdgeVertices{{P(0,0),P(0,0)}},rule,One,result);},
                           "Accepted zero-length edge"));
    PetscCall(ExpectError([&]{return IntegrateEdge(EdgeVertices{{P(inf,0),P(1,1)}},rule,One,result);},
                           "Accepted non-finite edge"));
    PetscCall(ExpectError([&]{return IntegrateCell(cell,rule,[&](const Point&){return nan;},result);},
                           "Accepted non-finite cell integrand"));
    PetscCall(ExpectError([&]{return IntegrateEdge(edge,rule,[&](const Point&){return inf;},result);},
                           "Accepted non-finite edge integrand"));
    PetscCall(ExpectError([&]{return IntegrateNormalFlux(edge,rule,[&](const Point&){return P(nan,1);},result);},
                           "Accepted non-finite vector integrand"));
    for (int kind=0; kind<6; ++kind) {
        GaussRule1D bad=rule;
        if (kind==0) bad=GaussRule1D{};
        if (kind==1) bad.weights.pop_back();
        if (kind==2) bad.points[0]=2;
        if (kind==3) bad.weights[0]=-1;
        if (kind==4) bad.weights[0]=nan;
        if (kind==5) bad.weights[0]*=2;
        PetscCall(ExpectError([&]{return IntegrateCell(cell,bad,One,result);},"Accepted invalid rule"));
    }
    PetscCall(Require(result==123.0,"Error paths changed the output value"));
    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode MakeMesh(MPI_Comm comm, PetscInt M, PetscInt N, PetscInt px, PetscInt py,
                        const MeshParam& mp, Generator generator, DM& dm, Vec& vertices)
{
    PetscFunctionBeginUser;
    PetscCall(DMDACreate2d(comm,DM_BOUNDARY_NONE,DM_BOUNDARY_NONE,DMDA_STENCIL_BOX,
                          M,N,px,py,2,1,nullptr,nullptr,&dm));
    PetscCall(DMSetUp(dm));
    PetscCall(DMCreateGlobalVector(dm,&vertices));
    PetscCall(generator(dm,vertices,mp));
    PetscFunctionReturn(PETSC_SUCCESS);
}

struct MeshReport {
    std::array<PetscReal,fieldCount+2> totals{}; // moments, all-cell flux, boundary flux
    std::vector<PetscReal> cellMoments;
    std::vector<int> coverage;
    int sloped=0;
    int localCells=0;
};

PetscErrorCode MeasureMesh(DM dm, Vec vertices, const MeshParam& mp,
                           const GaussRule1D& rule, const GaussRule1D& higher,
                           MeshReport& report)
{
    PetscFunctionBeginUser;
    PetscInt M,N,xs,ys,xm,ym;
    PetscCall(DMDAGetInfo(dm,nullptr,&M,&N,nullptr,nullptr,nullptr,nullptr,
                          nullptr,nullptr,nullptr,nullptr,nullptr,nullptr));
    PetscCall(DMDAGetCorners(dm,&xs,&ys,nullptr,&xm,&ym,nullptr));
    const std::size_t cells=static_cast<std::size_t>(M-1)*(N-1);
    report=MeshReport{};
    report.cellMoments.assign(cells*fieldCount,0.0);
    report.coverage.assign(cells,0);
    Vec local=nullptr;
    const PetscScalar*** a=nullptr;
    PetscCall(DMGetLocalVector(dm,&local));
    // Prevent a stale cached local vector from hiding a missing ghost update.
    PetscCall(VecSet(local,std::numeric_limits<PetscReal>::quiet_NaN()));
    PetscCall(DMGlobalToLocalBegin(dm,vertices,INSERT_VALUES,local));
    PetscCall(DMGlobalToLocalEnd(dm,vertices,INSERT_VALUES,local));
    PetscCall(DMDAVecGetArrayDOFRead(dm,local,&a));
    for (PetscInt j=ys; j<std::min(ys+ym,N-1); ++j) {
        for (PetscInt i=xs; i<std::min(xs+xm,M-1); ++i) {
            const QuadVertices cell{{
                P(PetscRealPart(a[j][i][0]),PetscRealPart(a[j][i][1])),
                P(PetscRealPart(a[j][i+1][0]),PetscRealPart(a[j][i+1][1])),
                P(PetscRealPart(a[j+1][i+1][0]),PetscRealPart(a[j+1][i+1][1])),
                P(PetscRealPart(a[j+1][i][0]),PetscRealPart(a[j+1][i][1]))
            }};
            const QuadVertices norm=NormalizedCorners(cell,mp);
            const PetscReal area=mp.L*mp.H*PolygonMoment(norm,0,0);
            PetscCall(Require(area>0.0,"Non-positive independent polygon area"));
            const std::size_t id=static_cast<std::size_t>(j)*(M-1)+i;
            ++report.coverage[id];
            ++report.localCells;
            for (int k=0; k<fieldCount; ++k) {
                const Power power=powers[k];
                const auto function=[&](const Point& point){return Evaluate(point,mp,power);};
                PetscReal value=0.0, reference=0.0;
                PetscCall(IntegrateCell(cell,rule,function,value));
                PetscCall(IntegrateCell(cell,higher,function,reference));
                const PetscReal exact=mp.L*mp.H*PolygonMoment(norm,power.x,power.y);
                PetscCall(Near(value,exact,area,"Owned-cell polynomial moment"));
                PetscCall(Near(reference,exact,area,"Higher-order cell polynomial moment"));
                report.totals[k]+=value;
                report.cellMoments[id*fieldCount+k]=value;
            }
            Point center{};
            for (const Point& point : cell)
                for (int d=0; d<2; ++d) center.p[d]+=point.p[d]/4;
            PetscReal normalizedArea=0.0;
            PetscCall(IntegrateCellNormalized(cell,rule,center,2*PetscSqrtReal(area),One,normalizedArea));
            PetscCall(Near(normalizedArea,0.25,0.25,"Owned-cell normalized area"));

            // div(F)=X+Y, for X=(x-xstart)/L and Y=(y-ystart)/H.
            const auto flux=[&](const Point& point) {
                const Point p=NormalizedPoint(point,mp);
                return P(mp.L*p.p[0]*p.p[0]/2,mp.H*p.p[1]*p.p[1]/2);
            };
            PetscReal cellFlux=0.0;
            for (int e=0; e<4; ++e) {
                const EdgeVertices edge{{cell[e],cell[(e+1)%4]}};
                PetscReal value=0.0;
                PetscCall(IntegrateNormalFlux(edge,rule,flux,value));
                cellFlux+=value;
                const bool boundary=(e==0 && j==0) || (e==1 && i==M-2)
                                  || (e==2 && j==N-2) || (e==3 && i==0);
                if (boundary) report.totals[fieldCount+1]+=value;
                const Point p=norm[e], q=norm[(e+1)%4];
                if (PetscAbsReal(p.p[0]-q.p[0])>64*PETSC_MACHINE_EPSILON &&
                    PetscAbsReal(p.p[1]-q.p[1])>64*PETSC_MACHINE_EPSILON) report.sloped=1;
            }
            const PetscReal exactFlux=mp.L*mp.H*(PolygonMoment(norm,1,0)+PolygonMoment(norm,0,1));
            PetscCall(Near(cellFlux,exactFlux,area,"Cell divergence theorem on generated mesh"));
            report.totals[fieldCount]+=cellFlux;
        }
    }
    PetscCall(DMDAVecRestoreArrayDOFRead(dm,local,&a));
    PetscCall(DMRestoreLocalVector(dm,&local));
    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode CheckMeshCase(PetscInt M, PetscInt N, PetscInt px, PetscInt py,
                             const MeshParam& mp, Generator generator, bool expectSloped)
{
    PetscFunctionBeginUser;
    DM dm=nullptr, serialDM=nullptr;
    Vec vertices=nullptr, serialVertices=nullptr;
    GaussRule1D rule,higher;
    PetscCall(CreateGaussRule(3,rule));
    PetscCall(CreateGaussRule(5,higher));
    PetscCall(MakeMesh(PETSC_COMM_WORLD,M,N,px,py,mp,generator,dm,vertices));
    MeshReport distributed,serial;
    PetscCall(MeasureMesh(dm,vertices,mp,rule,higher,distributed));
    // A complete serial mesh on every rank permits per-cell comparison, not
    // merely comparison of totals that could hide a rank-dependent perturbation.
    PetscCall(MakeMesh(PETSC_COMM_SELF,M,N,1,1,mp,generator,serialDM,serialVertices));
    PetscCall(MeasureMesh(serialDM,serialVertices,mp,rule,higher,serial));
    std::array<PetscReal,fieldCount+2> totals{};
    std::vector<PetscReal> cellMoments(distributed.cellMoments.size());
    std::vector<int> coverage(distributed.coverage.size());
    // All reductions are outside the cell loop, including on empty ranks.
    PetscCallMPI(MPI_Allreduce(distributed.totals.data(),totals.data(),fieldCount+2,
                               MPIU_REAL,MPI_SUM,PETSC_COMM_WORLD));
    PetscCallMPI(MPI_Allreduce(distributed.cellMoments.data(),cellMoments.data(),
                               static_cast<int>(cellMoments.size()),MPIU_REAL,MPI_SUM,PETSC_COMM_WORLD));
    PetscCallMPI(MPI_Allreduce(distributed.coverage.data(),coverage.data(),
                               static_cast<int>(coverage.size()),MPI_INT,MPI_SUM,PETSC_COMM_WORLD));
    int sloped=0,empty=distributed.localCells==0 ? 1 : 0,emptyRanks=0;
    PetscCallMPI(MPI_Allreduce(&distributed.sloped,&sloped,1,MPI_INT,MPI_MAX,PETSC_COMM_WORLD));
    PetscCallMPI(MPI_Allreduce(&empty,&emptyRanks,1,MPI_INT,MPI_SUM,PETSC_COMM_WORLD));
    for (std::size_t id=0; id<coverage.size(); ++id) {
        PetscCall(Require(coverage[id]==1,"Physical cell was omitted or counted more than once"));
        const PetscReal area=serial.cellMoments[id*fieldCount];
        for (int k=0; k<fieldCount; ++k)
            PetscCall(Near(cellMoments[id*fieldCount+k],serial.cellMoments[id*fieldCount+k],
                            area,"Distributed cell differs from its serial counterpart"));
    }
    const PetscReal area=mp.L*mp.H;
    for (int k=0; k<fieldCount; ++k) {
        const Power power=powers[k];
        const PetscReal exact=area/((power.x+1)*(power.y+1));
        PetscCall(Near(totals[k],exact,area,"Global analytic polynomial integral"));
        PetscCall(Near(totals[k],serial.totals[k],area,"Global serial/MPI agreement"));
    }
    PetscCall(Near(totals[fieldCount],area,area,"Sum of cell fluxes"));
    PetscCall(Near(totals[fieldCount+1],area,area,"Physical-boundary flux"));
    PetscCall(Near(totals[fieldCount],totals[fieldCount+1],area,"Internal-edge flux cancellation"));
    if (expectSloped) PetscCall(Require(sloped!=0,"Quadrilateral test did not contain slanted edges"));
    if (M==2 && N==2) {
        PetscMPIInt size;
        PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD,&size));
        PetscCall(Require(emptyRanks==size-1,"Single-cell case did not exercise empty ranks correctly"));
    }
    PetscCall(VecDestroy(&serialVertices));
    PetscCall(VecDestroy(&vertices));
    PetscCall(DMDestroy(&serialDM));
    PetscCall(DMDestroy(&dm));
    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode RunTests()
{
    PetscFunctionBeginUser;
    PetscInt type=-1,px=PETSC_DECIDE,py=PETSC_DECIDE;
    PetscCall(PetscOptionsGetInt(nullptr,nullptr,"-integral_mesh_type",&type,nullptr));
    PetscCall(PetscOptionsGetInt(nullptr,nullptr,"-mesh_px",&px,nullptr));
    PetscCall(PetscOptionsGetInt(nullptr,nullptr,"-mesh_py",&py,nullptr));
    PetscCheck(type>=-1 && type<=2,PETSC_COMM_WORLD,PETSC_ERR_ARG_OUTOFRANGE,
               "integral_mesh_type: -1=unit tests, 0=rectangular, 1=quadrilateral, 2=stretched");
    if (type==-1) {
        PetscCall(CheckRules());
        PetscCall(CheckCellShapes());
        PetscCall(CheckEdgesAndScaling());
        PetscCall(CheckNonPolynomial());
        PetscCall(CheckErrors());
        PetscCall(PetscPrintf(PETSC_COMM_WORLD,"integral unit tests passed\n"));
        PetscFunctionReturn(PETSC_SUCCESS);
    }
    PetscMPIInt size;
    PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD,&size));
    PetscCheck((size==1 || size==2 || size==4) &&
               (px==PETSC_DECIDE || px==1 || px==2) &&
               (py==PETSC_DECIDE || py==1 || py==2),
               PETSC_COMM_WORLD,PETSC_ERR_ARG_OUTOFRANGE,
               "These mesh tests use 1x1, 2x1, 1x2, or 2x2 process grids");
    PetscCheck(px==PETSC_DECIDE || py==PETSC_DECIDE || px*py==size,
               PETSC_COMM_WORLD,PETSC_ERR_ARG_WRONG,
               "Process grid does not match MPI_Comm_size; check the MPI launcher");
    const std::array<Generator,3> generators{{CreateFullMesh,LogicRectMesh,RefineMesh}};
    const std::array<const char*,3> names{{"rectangular","quadrilateral","stretched"}};
    const Generator generator=generators[static_cast<std::size_t>(type)];
    MeshParam mp;
    mp.xstart=-0.75; mp.ystart=0.2; mp.L=2.5; mp.H=1.3;
    mp.seed=7; mp.perturbation=0.15;
    PetscCall(CheckMeshCase(13,9,px,py,mp,generator,type==1));
    if (type==1) {
        mp.seed=991; mp.perturbation=0.249;
        PetscCall(CheckMeshCase(10,7,px,py,mp,generator,true));
        mp.perturbation=0.0;
        PetscCall(CheckMeshCase(13,9,px,py,mp,generator,false));
    }
    mp.xstart=3.5; mp.ystart=-2.25; mp.L=17.0; mp.H=0.08;
    mp.seed=42; mp.perturbation=0.15;
    PetscCall(CheckMeshCase(8,6,px,py,mp,generator,type==1));
    // One physical cell; all other ranks must contribute zero and still reduce.
    PetscCall(CheckMeshCase(2,2,px,py,mp,generator,false));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD,"integral %s mesh tests passed on %d rank(s)\n",
                          names[static_cast<std::size_t>(type)],static_cast<int>(size)));
    PetscFunctionReturn(PETSC_SUCCESS);
}

} // namespace

int main(int argc, char** argv)
{
    const PetscErrorCode initialized=PetscInitialize(&argc,&argv,nullptr,
        "Integral verification: no options runs unit tests; -integral_mesh_type 0,1,2 runs mesh tests.\n");
    if (initialized) return static_cast<int>(initialized);
    // A rank-local test failure must terminate peers that may be in a collective.
    PetscCallAbort(PETSC_COMM_WORLD,RunTests());
    return static_cast<int>(PetscFinalize());
}
