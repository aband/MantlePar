// Spatial-derivative convergence and cell-average smoothness gradients.
// -tensor_test: convergence (default), gradient, gradient_pseudo1d,
//               gradient_contract. Gradient cases produce no CSV files.
#include "tensorstencilpoly.h"
#include <petscsys.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string>
#include <utility>
#include <vector>

namespace {

constexpr PetscReal L = 2.0, H = 1.0;
const PetscReal pi = std::acos(PetscReal(-1));
using Values = std::array<PetscReal, 3>; // u, du/dx, du/dy

struct Options {
    PetscInt mesh = 0, degree = 1, n0 = 16, levels = 4, seed = 7;
    PetscReal perturbation = 0.15, rateTolerance = 0.35;
    std::string output, test = "convergence";
};

struct Grid {
    PetscInt n = 0;
    std::vector<QuadVertices> cells;
    std::vector<PetscReal> areas;
};

struct Row {
    PetscInt n = 0;
    PetscReal h = 0;
    Values l2{}, maximum{};
};

const char* MeshName(PetscInt mesh)
{ return mesh == 0 ? "rectangular" : "quadrilateral"; }

std::size_t CellId(PetscInt n, PetscInt i, PetscInt j)
{ return static_cast<std::size_t>(j) * static_cast<std::size_t>(n) + i; }

// Index-based perturbations: repeatable without random-engine/library state.
std::uint64_t Mix64(std::uint64_t value)
{
    value += UINT64_C(0x9e3779b97f4a7c15);
    value = (value ^ (value >> 30)) * UINT64_C(0xbf58476d1ce4e5b9);
    value = (value ^ (value >> 27)) * UINT64_C(0x94d049bb133111eb);
    return value ^ (value >> 31);
}

PetscReal SignedRandom(std::uint64_t key)
{
    return 2.0 * static_cast<PetscReal>(Mix64(key) >> 11)
         * (1.0 / 9007199254740992.0) - 1.0;
}

Values SmoothField(const Point& point)
{
    const PetscReal x = point.p[0], y = point.p[1];
    const PetscReal e = std::exp(x + 2*y);
    const PetscReal sx = std::sin(pi*x), cx = std::cos(pi*x);
    const PetscReal sy = std::sin(2*pi*y), cy = std::cos(2*pi*y);
    return {{e + sx*cy, e + pi*cx*cy, 2*e - 2*pi*sx*sy}};
}

// A full Q_p polynomial with mixed terms, independent analytic derivatives:
// (sum_i x^i) * (sum_j (j+1)*y^j).
Values PolynomialField(const Point& point, PetscInt degree)
{
    PetscReal x = 0, y = 0, dx = 0, dy = 0;
    for (PetscInt k = 0; k <= degree; ++k) {
        x += std::pow(point.p[0], k);
        y += (k+1) * std::pow(point.p[1], k);
        if (k > 0) {
            dx += k * std::pow(point.p[0], k-1);
            dy += k * (k+1) * std::pow(point.p[1], k-1);
        }
    }
    return {{x*y, dx*y, x*dy}};
}

PetscReal Sinc(PetscReal z)
{
    if (std::abs(z) < 1e-4) {
        const PetscReal z2 = z*z;
        return 1 - z2/6 + z2*z2/120;
    }
    return std::sin(z)/z;
}

PetscReal Sinhc(PetscReal z)
{
    if (std::abs(z) < 1e-4) {
        const PetscReal z2 = z*z;
        return 1 + z2/6 + z2*z2/120;
    }
    return std::sinh(z)/z;
}

// Closed-form average, avoiding subtraction of nearly equal exponentials.
PetscReal RectangularAverage(const QuadVertices& cell)
{
    const PetscReal dx = cell[1].p[0] - cell[0].p[0];
    const PetscReal dy = cell[3].p[1] - cell[0].p[1];
    const PetscReal x = cell[0].p[0] + dx/2;
    const PetscReal y = cell[0].p[1] + dy/2;
    return std::exp(x + 2*y) * Sinhc(dx/2) * Sinhc(dy)
         + std::sin(pi*x) * Sinc(pi*dx/2)
         * std::cos(2*pi*y) * Sinc(pi*dy);
}

PetscErrorCode ReadOptions(Options& options)
{
    PetscFunctionBeginUser;
    char test[64] = "convergence";
    PetscCall(PetscOptionsGetString(nullptr,nullptr,"-tensor_test",
                                    test,sizeof(test),nullptr));
    options.test = test;
    PetscCheck(options.test == "convergence" || options.test == "gradient" ||
               options.test == "gradient_pseudo1d" || options.test == "gradient_contract",
               PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,
               "tensor_test: convergence, gradient, gradient_pseudo1d, or gradient_contract");
    PetscCall(PetscOptionsGetInt(nullptr,nullptr,"-tensor_mesh_type",&options.mesh,nullptr));
    PetscCheck(options.mesh == 0 || options.mesh == 1, PETSC_COMM_SELF,
               PETSC_ERR_ARG_OUTOFRANGE, "tensor_mesh_type: 0=rectangular, 1=quadrilateral");
    if (options.test != "convergence") PetscFunctionReturn(PETSC_SUCCESS);
    PetscCall(PetscOptionsGetInt(nullptr,nullptr,"-tensor_degree",&options.degree,nullptr));
    PetscCall(PetscOptionsGetInt(nullptr,nullptr,"-tensor_n0",&options.n0,nullptr));
    PetscCall(PetscOptionsGetInt(nullptr,nullptr,"-tensor_levels",&options.levels,nullptr));
    PetscCall(PetscOptionsGetInt(nullptr,nullptr,"-tensor_seed",&options.seed,nullptr));
    PetscCall(PetscOptionsGetReal(nullptr,nullptr,"-tensor_perturbation",
                                 &options.perturbation,nullptr));
    PetscCall(PetscOptionsGetReal(nullptr,nullptr,"-tensor_rate_tolerance",
                                 &options.rateTolerance,nullptr));
    char output[4096] = "";
    PetscCall(PetscOptionsGetString(nullptr,nullptr,"-tensor_output",
                                    output,sizeof(output),nullptr));
    PetscCheck(options.degree == 1 || options.degree == 2, PETSC_COMM_SELF,
               PETSC_ERR_ARG_OUTOFRANGE, "This test covers Q1 and Q2");
    PetscCheck(options.n0 >= 8 && options.levels >= 3 && options.seed >= 0,
               PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE,
               "Require tensor_n0 >= 8, tensor_levels >= 3, and a nonnegative seed");
    PetscCheck(std::isfinite(options.perturbation) &&
               options.perturbation > 0 && options.perturbation < 0.25,
               PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE,
               "Require 0 < tensor_perturbation < 0.25 for the quadrilateral mesh");
    PetscCheck(std::isfinite(options.rateTolerance) &&
               options.rateTolerance >= 0 && options.rateTolerance < 1,
               PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE,
               "Require 0 <= tensor_rate_tolerance < 1");
    options.output = output;
    if (options.output.empty())
        options.output = std::string("tensorstencilpoly_") + MeshName(options.mesh)
                       + "_q" + std::to_string(options.degree) + ".csv";
    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode MakeGrid(PetscInt n, const Options& options, Grid& grid)
{
    PetscFunctionBeginUser;
    PetscCheck(n > 0 && n < std::numeric_limits<PetscInt>::max(),
               PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Invalid mesh size");
    const auto width = static_cast<std::size_t>(n) + 1;
    PetscCheck(width <= std::vector<Point>().max_size()/width &&
               width <= std::vector<QuadVertices>().max_size()/width,
               PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Mesh size is too large");
    std::vector<Point> vertices(width*width);
    const PetscReal hx = L/n, hy = H/n;
    bool perturbed = false;
    for (PetscInt j = 0; j <= n; ++j) {
        for (PetscInt i = 0; i <= n; ++i) {
            Point point{{hx*i,hy*j}};
            if (options.mesh == 1 && i > 0 && i < n && j > 0 && j < n) {
                const std::uint64_t key = static_cast<std::uint64_t>(options.seed)
                    ^ Mix64(static_cast<std::uint64_t>(i))
                    ^ Mix64(static_cast<std::uint64_t>(j) + UINT64_C(0xd1b54a32d192ed03));
                point.p[0] += options.perturbation*hx*SignedRandom(key);
                point.p[1] += options.perturbation*hy*
                    SignedRandom(key ^ UINT64_C(0x94d049bb133111eb));
                perturbed = true;
            }
            vertices[static_cast<std::size_t>(j)*width+i] = point;
        }
    }
    Grid work;
    work.n = n;
    const auto count = static_cast<std::size_t>(n)*n;
    work.cells.resize(count);
    work.areas.resize(count);
    GaussRule1D areaRule;
    PetscCall(CreateGaussRule(1,areaRule));
    PetscReal totalArea = 0;
    bool slanted = false;
    for (PetscInt j = 0; j < n; ++j) {
        for (PetscInt i = 0; i < n; ++i) {
            const auto id = CellId(n,i,j);
            const auto v = static_cast<std::size_t>(j)*width+i;
            const QuadVertices cell{{vertices[v],vertices[v+1],
                                     vertices[v+width+1],vertices[v+width]}};
            PetscCall(ValidateQuad(cell));
            work.cells[id] = cell;
            PetscCall(IntegrateCell(cell,areaRule,
                [](const Point&) { return PetscReal(1); },work.areas[id]));
            PetscCheck(work.areas[id] > 0, PETSC_COMM_SELF, PETSC_ERR_PLIB,
                       "Mesh contains a nonpositive cell area");
            totalArea += work.areas[id];
            for (std::size_t e = 0; e < 4; ++e) {
                const Point& a = cell[e];
                const Point& b = cell[(e+1)%4];
                slanted = slanted ||
                    (std::abs(a.p[0]-b.p[0]) > 1e-12*hx &&
                     std::abs(a.p[1]-b.p[1]) > 1e-12*hy);
            }
        }
    }
    PetscCheck(std::abs(totalArea-L*H) <= 1e-10*L*H,
               PETSC_COMM_SELF, PETSC_ERR_PLIB, "Mesh does not cover the stated domain");
    PetscCheck(options.mesh == 0 || (perturbed && slanted),
               PETSC_COMM_SELF, PETSC_ERR_PLIB,
               "Quadrilateral case must contain actual slanted edges");
    grid = std::move(work);
    PetscFunctionReturn(PETSC_SUCCESS);
}

template <class Function>
PetscErrorCode CellAverages(const Grid& grid, const GaussRule1D& rule,
                            Function&& function, std::vector<PetscReal>& averages)
{
    PetscFunctionBeginUser;
    averages.resize(grid.cells.size());
    for (std::size_t k = 0; k < grid.cells.size(); ++k) {
        PetscReal integral;
        PetscCall(IntegrateCell(grid.cells[k],rule,function,integral));
        averages[k] = integral/grid.areas[k];
    }
    PetscFunctionReturn(PETSC_SUCCESS);
}

MeshIndex StencilStart(PetscInt i, PetscInt j, PetscInt n, PetscInt degree)
{
    const PetscInt last = n-(degree+1);
    // Q2 is centered in the interior. Shift a full stencil at boundaries.
    // Q1 uses a fixed neighboring 2x2 stencil, also shifted as needed.
    return {std::clamp(i-degree/2,PetscInt(0),last),
            std::clamp(j-degree/2,PetscInt(0),last)};
}

PetscErrorCode PrepareStencil(
    const Grid& grid, const std::vector<PetscReal>& averages,
    PetscInt i, PetscInt j, PetscInt degree,
    TensorStencilPoly& polynomial, std::vector<PetscReal>& local)
{
    PetscFunctionBeginUser;
    const MeshIndex start = StencilStart(i,j,grid.n,degree);
    const PetscInt side = degree+1;
    std::vector<QuadVertices> corners;
    corners.reserve(static_cast<std::size_t>(side)*side);
    local.clear();
    local.reserve(static_cast<std::size_t>(side)*side);
    for (PetscInt dj = 0; dj < side; ++dj) {
        for (PetscInt di = 0; di < side; ++di) {
            const auto id = CellId(grid.n,start.i+di,start.j+dj);
            corners.push_back(grid.cells[id]);
            local.push_back(averages[id]);
        }
    }
    PetscCall(polynomial.Initialize(corners,start,{side,side},
                                    degree,std::sqrt(L*H)/grid.n));
    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode Evaluate(const TensorStencilPoly& polynomial,
                        const std::vector<PetscReal>& averages,
                        const Point& point, Values& values)
{
    PetscFunctionBeginUser;
    PetscCall(polynomial.Evaluate(averages,point,values[0]));
    PetscCall(polynomial.EvaluateDerivative(averages,point,1,0,values[1]));
    PetscCall(polynomial.EvaluateDerivative(averages,point,0,1,values[2]));
    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode CheckExactness(const Options& options, const GaussRule1D& rule)
{
    PetscFunctionBeginUser;
    Grid grid;
    PetscCall(MakeGrid(8,options,grid));
    std::vector<PetscReal> averages, local;
    PetscCall(CellAverages(grid,rule,[&](const Point& p) {
        return PolynomialField(p,options.degree)[0];
    },averages));
    Values maximum{};
    PetscReal maximumAverage = 0;
    TensorStencilPoly polynomial;
    for (PetscInt j = 0; j < grid.n; ++j) {
        for (PetscInt i = 0; i < grid.n; ++i) {
            PetscCall(PrepareStencil(grid,averages,i,j,options.degree,polynomial,local));
            const auto id = CellId(grid.n,i,j);
            const auto& cell = grid.cells[id];
            PetscReal reconstructedIntegral = 0;
            for (std::size_t qy = 0; qy < rule.points.size(); ++qy) {
                for (std::size_t qx = 0; qx < rule.points.size(); ++qx) {
                    const Point ref{{rule.points[qx],rule.points[qy]}};
                    const Point point = MapCellPoint(ref,cell);
                    Values result;
                    PetscCall(Evaluate(polynomial,local,point,result));
                    const Values exact = PolynomialField(point,options.degree);
                    for (std::size_t d = 0; d < 3; ++d) {
                        const PetscReal error = std::abs(result[d]-exact[d]);
                        maximum[d] = std::max(maximum[d],error);
                        PetscCheck(error <= 5e-11*(1+std::abs(exact[d])),
                                   PETSC_COMM_SELF,PETSC_ERR_PLIB,
                                   "Q%lld polynomial reproduction failed for component %zu",
                                   static_cast<long long>(options.degree),d);
                    }
                    reconstructedIntegral += rule.weights[qx]*rule.weights[qy]
                        *CellJacobian(ref,cell)*result[0];
                }
            }
            const PetscReal error =
                std::abs(reconstructedIntegral/grid.areas[id]-averages[id]);
            maximumAverage = std::max(maximumAverage,error);
            PetscCheck(error <= 5e-11*(1+std::abs(averages[id])),
                       PETSC_COMM_SELF,PETSC_ERR_PLIB,"Cell-average reproduction failed");
        }
    }
    PetscCall(PetscPrintf(PETSC_COMM_SELF,
        "Exactness Q%lld (%s): max errors u=%.3e ux=%.3e uy=%.3e average=%.3e\n",
        static_cast<long long>(options.degree),MeshName(options.mesh),
        static_cast<double>(maximum[0]),static_cast<double>(maximum[1]),
        static_cast<double>(maximum[2]),static_cast<double>(maximumAverage)));
    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode Measure(PetscInt n, const Options& options,
                       const GaussRule1D& averageRule,
                       const GaussRule1D& errorRule, Row& row)
{
    PetscFunctionBeginUser;
    Grid grid;
    PetscCall(MakeGrid(n,options,grid));
    std::vector<PetscReal> averages(grid.cells.size()), local;
    if (options.mesh == 0) {
        for (std::size_t k = 0; k < grid.cells.size(); ++k)
            averages[k] = RectangularAverage(grid.cells[k]);
    } else {
        PetscCall(CellAverages(grid,averageRule,
            [](const Point& p) { return SmoothField(p)[0]; },averages));
    }
    std::array<long double,3> squared{};
    Values maximum{};
    TensorStencilPoly polynomial;
    for (PetscInt j = 0; j < n; ++j) {
        for (PetscInt i = 0; i < n; ++i) {
            PetscCall(PrepareStencil(grid,averages,i,j,options.degree,polynomial,local));
            const auto& cell = grid.cells[CellId(n,i,j)];
            for (std::size_t qy = 0; qy < errorRule.points.size(); ++qy) {
                for (std::size_t qx = 0; qx < errorRule.points.size(); ++qx) {
                    const Point ref{{errorRule.points[qx],errorRule.points[qy]}};
                    const Point point = MapCellPoint(ref,cell);
                    Values result;
                    PetscCall(Evaluate(polynomial,local,point,result));
                    const Values exact = SmoothField(point);
                    const PetscReal weight = errorRule.weights[qx]*errorRule.weights[qy]
                                           * CellJacobian(ref,cell);
                    PetscCheck(weight > 0 && std::isfinite(weight),
                               PETSC_COMM_SELF,PETSC_ERR_PLIB,"Invalid physical error weight");
                    for (std::size_t d = 0; d < 3; ++d) {
                        const PetscReal error = std::abs(result[d]-exact[d]);
                        squared[d] += static_cast<long double>(weight)*error*error;
                        maximum[d] = std::max(maximum[d],error);
                    }
                }
            }
            // Include corners, edge midpoints and cell center in the sampled max.
            for (PetscReal y : {-1.0,0.0,1.0}) {
                for (PetscReal x : {-1.0,0.0,1.0}) {
                    const Point point = MapCellPoint(Point{{x,y}},cell);
                    Values result;
                    PetscCall(Evaluate(polynomial,local,point,result));
                    const Values exact = SmoothField(point);
                    for (std::size_t d = 0; d < 3; ++d)
                        maximum[d] = std::max(maximum[d],std::abs(result[d]-exact[d]));
                }
            }
        }
    }
    Row work;
    work.n = n;
    work.h = std::sqrt(L*H)/n;
    work.maximum = maximum;
    for (std::size_t d = 0; d < 3; ++d) {
        work.l2[d] = static_cast<PetscReal>(std::sqrt(squared[d]));
        PetscCheck(std::isfinite(work.l2[d]) && work.l2[d] > 0 &&
                   std::isfinite(work.maximum[d]) && work.maximum[d] > 0,
                   PETSC_COMM_SELF,PETSC_ERR_PLIB,"Convergence error must be finite and positive");
    }
    row = work;
    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscReal Rate(PetscReal previous, PetscReal current, PetscReal ratio)
{ return std::log(previous/current)/std::log(ratio); }

void WriteHeader(std::ostream& csv)
{
    csv << "mesh,degree,N,h,perturbation,seed";
    for (const char* quantity : {"u","ux","uy"})
        csv << ',' << quantity << "_l2," << quantity << "_l2_rate,"
            << quantity << "_linf," << quantity << "_linf_rate";
    csv << '\n';
}

void WriteRow(std::ostream& csv, const Options& options,
              const Row& row, const Row* previous)
{
    csv << MeshName(options.mesh) << ',' << options.degree << ',' << row.n << ','
        << row.h << ',' << (options.mesh == 0 ? 0.0 : options.perturbation)
        << ',' << options.seed;
    for (std::size_t d = 0; d < 3; ++d) {
        csv << ',' << row.l2[d] << ',';
        if (previous) csv << Rate(previous->l2[d],row.l2[d],previous->h/row.h);
        csv << ',' << row.maximum[d] << ',';
        if (previous) csv << Rate(previous->maximum[d],row.maximum[d],previous->h/row.h);
    }
    csv << '\n';
}

PetscReal FittedRate(const std::vector<Row>& rows, std::size_t component, bool maxNorm)
{
    std::array<PetscReal,3> x{}, y{};
    PetscReal mx = 0, my = 0;
    for (std::size_t k = 0; k < 3; ++k) {
        const Row& row = rows[rows.size()-3+k];
        x[k] = std::log(row.h);
        y[k] = std::log(maxNorm ? row.maximum[component] : row.l2[component]);
        mx += x[k]/3;
        my += y[k]/3;
    }
    PetscReal numerator = 0, denominator = 0;
    for (std::size_t k = 0; k < 3; ++k) {
        numerator += (x[k]-mx)*(y[k]-my);
        denominator += (x[k]-mx)*(x[k]-mx);
    }
    return numerator/denominator;
}

PetscErrorCode CheckRates(const std::vector<Row>& rows, const Options& options)
{
    PetscFunctionBeginUser;
    const char* names[3] = {"u","ux","uy"};
    bool passed = true;
    PetscCall(PetscPrintf(PETSC_COMM_SELF,
                          "Rates fitted over the finest three meshes:\n"));
    for (std::size_t d = 0; d < 3; ++d) {
        const PetscReal expected = d == 0 ? options.degree+1 : options.degree;
        for (bool maxNorm : {false,true}) {
            const PetscReal rate = FittedRate(rows,d,maxNorm);
            bool decreasing = true;
            for (std::size_t k = rows.size()-2; k < rows.size(); ++k) {
                const PetscReal oldError = maxNorm ? rows[k-1].maximum[d] : rows[k-1].l2[d];
                const PetscReal newError = maxNorm ? rows[k].maximum[d] : rows[k].l2[d];
                decreasing = decreasing && newError < oldError;
            }
            const bool ok = decreasing && std::isfinite(rate) &&
                            rate >= expected-options.rateTolerance;
            passed = passed && ok;
            PetscCall(PetscPrintf(PETSC_COMM_SELF,
                "  %-2s %-11s rate=%.4f expected=%.0f minimum=%.2f %s\n",
                names[d],maxNorm ? "sampled-max" : "L2",
                static_cast<double>(rate),static_cast<double>(expected),
                static_cast<double>(expected-options.rateTolerance),ok ? "PASS" : "FAIL"));
        }
    }
    PetscCheck(passed,PETSC_COMM_SELF,PETSC_ERR_PLIB,
               "Function or derivative convergence did not reach the expected rate");
    PetscFunctionReturn(PETSC_SUCCESS);
}

// These checks differentiate sigma with respect to the stencil cell averages.
// Geometry, h and smoothness order remain fixed during each difference check.
struct GradientStats {
    std::size_t regions = 0, components = 0;
    PetscReal maximumError = 0;
};

std::vector<QuadVertices> GradientGeometry(MeshIndex size, bool quadrilateral)
{
    const auto vertex = [quadrilateral](PetscInt i, PetscInt j) {
        return Point{{.3+.13*i+(quadrilateral ? .012*std::sin(.7*i+1.1*j) : 0),
                      -.2+.19*j+(quadrilateral ? .014*std::cos(1.2*i-.4*j) : 0)}};
    };
    std::vector<QuadVertices> cells;
    for (PetscInt j = 0; j < size.j; ++j)
        for (PetscInt i = 0; i < size.i; ++i)
            cells.push_back({{vertex(i,j),vertex(i+1,j),vertex(i+1,j+1),vertex(i,j+1)}});
    return cells;
}

PetscErrorCode GradientNear(PetscReal actual, PetscReal expected,
                             PetscReal tolerance, const char* message)
{
    PetscFunctionBeginUser;
    PetscCheck(std::isfinite(actual) && std::isfinite(expected) &&
               std::abs(actual-expected) <= tolerance*(1+std::abs(expected)),
               PETSC_COMM_SELF,PETSC_ERR_PLIB,"%s: %.17g versus %.17g",message,
               static_cast<double>(actual),static_cast<double>(expected));
    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode CheckGradientRegion(const TensorStencilPoly& polynomial,
                                    const std::vector<PetscReal>& averages,
                                    const MeshIndex* target, GradientStats& stats)
{
    PetscFunctionBeginUser;
    const auto value = [&](const std::vector<PetscReal>& input, PetscReal& sigma) {
        return target ? polynomial.Smoothness(input,*target,sigma)
                      : polynomial.Smoothness(input,sigma);
    };
    const auto evaluate = [&](const std::vector<PetscReal>& input, PetscReal& sigma,
                              std::vector<PetscReal>& gradient) {
        return target ? polynomial.SmoothnessWithGradient(input,*target,sigma,gradient)
                      : polynomial.SmoothnessWithGradient(input,sigma,gradient);
    };
    PetscReal sigma, direct;
    std::vector<PetscReal> gradient;
    PetscCall(evaluate(averages,sigma,gradient));
    PetscCall(value(averages,direct));
    PetscCheck(std::isfinite(sigma) && sigma >= 0 && sigma == direct &&
               gradient.size() == averages.size(),PETSC_COMM_SELF,PETSC_ERR_PLIB,
               "SmoothnessWithGradient must preserve the indicator and return one entry per cell");

    for (std::size_t k = 0; k < averages.size(); ++k) {
        auto plus = averages, minus = averages;
        // sigma is quadratic: central differences have no truncation term in
        // exact arithmetic. A moderate step limits subtractive cancellation.
        const PetscReal step = 1e-4*std::max(PetscReal(1),std::abs(averages[k]));
        plus[k] += step;
        minus[k] -= step;
        PetscReal plusValue, minusValue;
        PetscCall(value(plus,plusValue));
        PetscCall(value(minus,minusValue));
        const PetscReal difference = (plusValue-minusValue)/(2*step);
        const PetscReal error = std::abs(difference-gradient[k])/(1+std::abs(gradient[k]));
        PetscCheck(std::isfinite(difference) && std::isfinite(gradient[k]) && error <= 2e-7,
                   PETSC_COMM_SELF,PETSC_ERR_PLIB,
                   "Gradient FD failed: stencil=(%lld,%lld), target=(%lld,%lld), "
                   "entry=%zu, analytic=%.17g, FD=%.17g, scaled error=%.3e "
                   "(target -1,-1 denotes the reference square)",
                   static_cast<long long>(polynomial.Size().i),
                   static_cast<long long>(polynomial.Size().j),
                   target ? static_cast<long long>(target->i) : -1LL,
                   target ? static_cast<long long>(target->j) : -1LL,k,
                   static_cast<double>(gradient[k]),static_cast<double>(difference),
                   static_cast<double>(error));
        stats.maximumError = std::max(stats.maximumError,error);
        ++stats.components;
    }

    long double sum = 0, norm = 0, euler = 0;
    for (std::size_t k = 0; k < averages.size(); ++k) {
        sum += gradient[k];
        norm += std::abs(gradient[k]);
        euler += static_cast<long double>(gradient[k])*(averages[k]-averages[0]);
    }
    PetscCall(GradientNear(static_cast<PetscReal>(sum/(1+norm)),0,5e-14,
                            "Gradient must annihilate a constant direction"));
    PetscCall(GradientNear(static_cast<PetscReal>(euler),2*sigma,5e-12,
                            "Quadratic Euler identity g dot (u-u0) = 2 sigma"));

    auto shifted = averages, scaled = averages;
    for (std::size_t k = 0; k < averages.size(); ++k) {
        // Data are multiples of 1/8, so this common shift is exact in double.
        shifted[k] += 1048576.;
        scaled[k] *= -2;
    }
    PetscReal otherSigma;
    std::vector<PetscReal> otherGradient;
    PetscCall(evaluate(shifted,otherSigma,otherGradient));
    PetscCheck(otherSigma == sigma && otherGradient == gradient,
               PETSC_COMM_SELF,PETSC_ERR_PLIB,
               "A representable constant shift changed smoothness or its gradient");
    PetscCall(evaluate(scaled,otherSigma,otherGradient));
    PetscCall(GradientNear(otherSigma,4*sigma,2e-13,"sigma(-2u) = 4 sigma(u)"));
    PetscCheck(otherGradient.size() == gradient.size(),PETSC_COMM_SELF,PETSC_ERR_PLIB,
               "Scaling changed the gradient size");
    for (std::size_t k = 0; k < gradient.size(); ++k)
        PetscCall(GradientNear(otherGradient[k],-2*gradient[k],2e-13,
                                "g(-2u) = -2 g(u)"));

    PetscCall(evaluate(std::vector<PetscReal>(averages.size(),1e100),otherSigma,otherGradient));
    PetscCheck(otherSigma == 0 && otherGradient.size() == averages.size() &&
               std::all_of(otherGradient.begin(),otherGradient.end(),
                           [](PetscReal entry) { return entry == 0; }),
               PETSC_COMM_SELF,PETSC_ERR_PLIB,
               "Constant data must give exactly zero smoothness and gradient");
    auto aliased = averages;
    PetscCall(evaluate(aliased,otherSigma,aliased));
    PetscCheck(otherSigma == sigma && aliased == gradient,PETSC_COMM_SELF,PETSC_ERR_PLIB,
               "Using the input vector as the gradient output changed the result");
    ++stats.regions;
    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode CheckGradients(const Options& options)
{
    PetscFunctionBeginUser;
    const bool pseudo1d = options.test == "gradient_pseudo1d";
    const std::vector<MeshIndex> sizes = pseudo1d
        ? std::vector<MeshIndex>{{1,1},{4,1},{1,4}}
        : std::vector<MeshIndex>{{2,2},{3,2},{2,3},{3,3},{4,3}};
    GradientStats stats;
    for (const auto size : sizes) {
        TensorStencilPoly polynomial;
        PetscCall(polynomial.Initialize(GradientGeometry(size,options.mesh == 1),
                                        {2,4},size,size.i+size.j-2,std::sqrt(.13*.19)));
        std::vector<PetscReal> averages(static_cast<std::size_t>(size.i)*size.j);
        for (PetscInt k = 0; k < polynomial.CellCount(); ++k)
            averages[static_cast<std::size_t>(k)] = ((13*k)%19-9)/8.0;
        std::vector<MeshIndex> targets{{0,0}};
        if (size.i > 1 || size.j > 1) targets.push_back({size.i-1,size.j-1});
        // Prepare multiple targets before querying either, to exercise cache isolation.
        for (const auto target : targets) PetscCall(polynomial.SetTargetSmoothness(target));
        PetscCall(CheckGradientRegion(polynomial,averages,nullptr,stats));
        for (const auto& target : targets)
            PetscCall(CheckGradientRegion(polynomial,averages,&target,stats));
    }
    PetscCall(PetscPrintf(PETSC_COMM_SELF,
        "PASS: smoothness gradients (%s, %s): %zu reference/target cases, "
        "%zu FD entries; max |FD-g|/(1+|g|)=%.3e\n"
        "  constant shifts, zero constants, scaling, Euler identity, "
        "zero gradient sum and input/output aliasing passed.\n",
        MeshName(options.mesh),pseudo1d ? "constant and pseudo-1D stencils" : "2D stencils",
        stats.regions,stats.components,static_cast<double>(stats.maximumError)));
    PetscFunctionReturn(PETSC_SUCCESS);
}

template<class Function>
PetscErrorCode ExpectGradientFailure(Function&& function, const char* message,
                                      std::size_t& failures)
{
    PetscFunctionBeginUser;
    PetscReal sigma = 123;
    const std::vector<PetscReal> sentinel{8,9};
    auto gradient = sentinel;
    PetscCall(PetscPushErrorHandler(PetscReturnErrorHandler,nullptr));
    const PetscErrorCode error = function(sigma,gradient);
    PetscCall(PetscPopErrorHandler());
    PetscCheck(error != PETSC_SUCCESS && sigma == 123 && gradient == sentinel,
               PETSC_COMM_SELF,PETSC_ERR_PLIB,
               "%s: expected an error with both outputs unchanged",message);
    ++failures;
    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode CheckGradientContract()
{
    PetscFunctionBeginUser;
    TensorStencilPoly polynomial;
    const std::vector<PetscReal> averages{1,2,3,4};
    std::size_t failures = 0;
    PetscCall(ExpectGradientFailure([&](PetscReal& sigma, std::vector<PetscReal>& gradient) {
        return polynomial.SmoothnessWithGradient(averages,sigma,gradient);
    },"Uninitialized reference indicator",failures));
    PetscCall(ExpectGradientFailure([&](PetscReal& sigma, std::vector<PetscReal>& gradient) {
        return polynomial.SmoothnessWithGradient(averages,{0,0},sigma,gradient);
    },"Uninitialized target indicator",failures));
    PetscCall(polynomial.Initialize(GradientGeometry({2,2},true),{2,4},{2,2},2,.2));
    PetscCall(ExpectGradientFailure([&](PetscReal& sigma, std::vector<PetscReal>& gradient) {
        return polynomial.SmoothnessWithGradient(averages,{0,0},sigma,gradient);
    },"Missing target cache",failures));
    for (const MeshIndex target : {MeshIndex{-1,0},MeshIndex{2,4}}) {
        PetscCall(ExpectGradientFailure([&](PetscReal& sigma, std::vector<PetscReal>& gradient) {
            return polynomial.SmoothnessWithGradient(averages,target,sigma,gradient);
        },"Target must be a valid local stencil offset",failures));
    }
    PetscCall(polynomial.SetTargetSmoothness({0,0}));
    auto nanInput = averages, infInput = averages;
    nanInput[1] = std::numeric_limits<PetscReal>::quiet_NaN();
    infInput[1] = std::numeric_limits<PetscReal>::infinity();
    const PetscReal huge = std::numeric_limits<PetscReal>::max()/16;
    const std::vector<std::vector<PetscReal>> invalid{
        {1},nanInput,infInput,{huge,-huge,huge,-huge}};
    const char* messages[] = {"Wrong average count","NaN input","Infinite input",
                              "Indicator overflow"};
    for (bool target : {false,true}) {
        for (std::size_t k = 0; k < invalid.size(); ++k) {
            PetscCall(ExpectGradientFailure([&](PetscReal& sigma, std::vector<PetscReal>& gradient) {
                return target ? polynomial.SmoothnessWithGradient(invalid[k],{0,0},sigma,gradient)
                              : polynomial.SmoothnessWithGradient(invalid[k],sigma,gradient);
            },messages[k],failures));
        }
    }
    // Reinitialization must discard the old target cache. Order zero must
    // produce zero for nonconstant data too, through both indicator overloads.
    PetscCall(polynomial.Initialize(GradientGeometry({2,2},true),{2,4},{2,2},0,.2));
    PetscCall(ExpectGradientFailure([&](PetscReal& sigma, std::vector<PetscReal>& gradient) {
        return polynomial.SmoothnessWithGradient(averages,{0,0},sigma,gradient);
    },"Reinitialization retained a stale target cache",failures));
    PetscCall(polynomial.SetTargetSmoothness({0,0}));
    for (bool target : {false,true}) {
        PetscReal sigma = -1, direct = -2;
        std::vector<PetscReal> gradient;
        PetscCall(target ? polynomial.SmoothnessWithGradient(averages,{0,0},sigma,gradient)
                         : polynomial.SmoothnessWithGradient(averages,sigma,gradient));
        PetscCall(target ? polynomial.Smoothness(averages,{0,0},direct)
                         : polynomial.Smoothness(averages,direct));
        PetscCheck(sigma == 0 && direct == 0 && gradient.size() == averages.size() &&
                   std::all_of(gradient.begin(),gradient.end(),
                               [](PetscReal entry) { return entry == 0; }),
                   PETSC_COMM_SELF,PETSC_ERR_PLIB,"Order-zero indicator or gradient is nonzero");
    }
    PetscCall(PetscPrintf(PETSC_COMM_SELF,
        "PASS: gradient API: %zu rejected inputs preserve both outputs; "
        "cache invalidation and both order-zero indicators passed.\n",failures));
    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode Run()
{
    PetscFunctionBeginUser;
    PetscMPIInt size;
    PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD,&size));
    PetscCheck(size == 1,PETSC_COMM_WORLD,PETSC_ERR_ARG_WRONG,
               "This local tensor polynomial test requires one MPI rank");
    Options options;
    PetscCall(ReadOptions(options));
    if (options.test == "gradient" || options.test == "gradient_pseudo1d") {
        PetscCall(CheckGradients(options));
        PetscFunctionReturn(PETSC_SUCCESS);
    }
    if (options.test == "gradient_contract") {
        PetscCall(CheckGradientContract());
        PetscFunctionReturn(PETSC_SUCCESS);
    }
    GaussRule1D averageRule, errorRule;
    PetscCall(CreateGaussRule(10,averageRule));
    PetscCall(CreateGaussRule(8,errorRule));
    PetscCall(CheckExactness(options,errorRule));

    const std::filesystem::path output(options.output);
    std::error_code ioError;
    if (!output.parent_path().empty())
        std::filesystem::create_directories(output.parent_path(),ioError);
    PetscCheck(!ioError,PETSC_COMM_SELF,PETSC_ERR_FILE_OPEN,
               "Cannot create output directory: %s",ioError.message().c_str());
    std::ofstream csv(output);
    PetscCheck(csv.is_open(),PETSC_COMM_SELF,PETSC_ERR_FILE_OPEN,
               "Cannot open convergence CSV: %s",options.output.c_str());
    csv << std::scientific << std::setprecision(17);
    WriteHeader(csv);
    PetscCall(PetscPrintf(PETSC_COMM_SELF,
        "Tensor polynomial convergence: %s Q%lld, domain [0,2] x [0,1]\n"
        "Cell averages: %s; errors: physical 8x8 quadrature plus boundary samples.\n",
        MeshName(options.mesh),static_cast<long long>(options.degree),
        options.mesh == 0 ? "analytic" : "10x10 quadrature"));
    WriteHeader(std::cout);
    std::cout << std::scientific << std::setprecision(6);
    std::vector<Row> rows;
    PetscInt n = options.n0;
    for (PetscInt level = 0; level < options.levels; ++level) {
        Row row;
        PetscCall(Measure(n,options,averageRule,errorRule,row));
        const Row* previous = rows.empty() ? nullptr : &rows.back();
        WriteRow(csv,options,row,previous);
        WriteRow(std::cout,options,row,previous);
        csv.flush();
        std::cout.flush();
        PetscCheck(csv.good(),PETSC_COMM_SELF,PETSC_ERR_FILE_WRITE,
                   "Failed to write convergence CSV");
        rows.push_back(row);
        if (level+1 < options.levels) {
            PetscCheck(n <= (std::numeric_limits<PetscInt>::max()-1)/2,
                       PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Mesh refinement overflows PetscInt");
            n *= 2;
        }
    }
    csv.close();
    PetscCheck(!csv.fail(),PETSC_COMM_SELF,PETSC_ERR_FILE_WRITE,
               "Failed to close convergence CSV");
    PetscCall(CheckRates(rows,options));
    PetscCall(PetscPrintf(PETSC_COMM_SELF,"PASS; CSV: %s\n",options.output.c_str()));
    PetscFunctionReturn(PETSC_SUCCESS);
}

} // namespace

int main(int argc, char** argv)
{
    PetscErrorCode error = PetscInitialize(&argc,&argv,nullptr,
        "Tensor polynomial convergence and smoothness-gradient verification.\n"
        "-tensor_test convergence|gradient|gradient_pseudo1d|gradient_contract\n"
        "Gradient cases: -tensor_mesh_type 0|1; no CSV or plotting needed.\n"
        "-tensor_mesh_type 0|1 -tensor_degree 1|2 -tensor_n0 16 -tensor_levels 4\n"
        "-tensor_seed 7 -tensor_perturbation 0.15 -tensor_rate_tolerance 0.35\n"
        "-tensor_output path/to/table.csv\n");
    if (error) return static_cast<int>(error);
    error = Run();
    const PetscErrorCode finalizeError = PetscFinalize();
    return static_cast<int>(error ? error : finalizeError);
}
