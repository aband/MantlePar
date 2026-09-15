#ifndef MANTLE_MFEM_LOCAL_MATRIX_H
#define MANTLE_MFEM_LOCAL_MATRIX_H

#include "brmixed.h"
#include "hdivmixed.h"
#include "integral.h"

#include <array>
#include <cstddef>
#include <functional>
#include <limits>
#include <vector>

// Element blocks in the CURRENT BRMixed/HDivMixed local DOF order.
// A is a full, symmetric, ROW-MAJOR velocity matrix: A[row*N + column].
// B[j] couples velocity DOF j to the single constant pressure DOF.
// C is the pressure-compaction scalar; f is the velocity load vector.
// These are LOCAL signs. Global assembly is responsible for any block signs
// required by the full system (the legacy solver later negated B and C).
// PetscScalar storage can be passed directly to PETSc matrix/vector insertion;
// the present real formulation produces zero imaginary parts in complex PETSc.
template <std::size_t N>
struct LocalMatrixBlock {
    static constexpr PetscInt VelocityDofs = static_cast<PetscInt>(N);
    static constexpr PetscInt PressureDofs = 1;

    std::array<PetscScalar, N * N> A{};
    std::array<PetscScalar, N> B{};
    PetscScalar C{};
    std::array<PetscScalar, N> f{};
};

using StokesLocalMatrix = LocalMatrixBlock<BRMixed::ElementDofs>;
using DarcyLocalMatrix = LocalMatrixBlock<HDivMixed::ElementDofs>;

// Reconstructed porosity at the quadrature points of ONE physical cell.
// The kernel neither reconstructs nor averages porosity. Phase evaluation and
// any intercell averaging/upwinding of the edge data belong to the caller.
//
// Let nc = cellRule.points.size() and ne = edgeRule.points.size().
//   cell[j*nc+i] is at MapCellPoint({{cellRule.points[i],
//                                    cellRule.points[j]}}, corners).
//   edge[e][g] is at MapEdgePoint(edgeRule.points[g],
//                                {corners[e], corners[(e+1)%4]}).
// Thus i varies fastest in the nc*nc cell samples. Each edge has ne samples.
// Edges are Bottom=0, Right=1, Top=2, Left=3, directed COUNTERCLOCKWISE.
// In particular, Top runs right-to-left and Left runs top-to-bottom. Convert
// canonical/global edge samples to this direction before calling the kernel.
// Legacy side order was Left,Bottom,Right,Top: old_edge=(new_edge+1)%4.
// Regenerate these samples whenever geometry, porosity, or the rule changes.
//
// Require 0 <= cell[g] < 1, 0 <= edge[e][g] <= 1, and 0 <= average < 1,
// all finite. Cell phi=1 is singular in this formulation and is rejected;
// no implicit clipping, regularization, or recomputation of average occurs.
// The average is an independent, externally supplied physical cell average.
// It starts unset so that forgetting to supply it cannot select a dry branch.
struct LocalPorositySamples {
    std::vector<PetscReal> cell;
    std::array<std::vector<PetscReal>, 4> edge;
    PetscReal average = std::numeric_limits<PetscReal>::quiet_NaN();
};

// Runtime inputs, read on every call. These deliberately preserve the different
// legacy degeneracy branches; changing a cutoff changes the numerical model.
// Both average cutoffs must be finite and in [0,1).
// No viscosity/permeability prefactor is added: the existing nondimensional,
// rescaled formulation already accounts for those reference scales externally.
struct LocalMatrixParameters {
    // Darcy boundary weight uses phi_edge^(1+theta). The default gives power 1;
    // set theta explicitly to the value used by your flow/material model.
    PetscReal theta = 0;

    // Darcy C uses phi/average only when average is STRICTLY above this value;
    // otherwise that ratio is replaced by 1. Does not affect the edge weight.
    PetscReal darcyCompactionAverageCutoff = PetscReal(1e-15);

    // Pressure coupling uses phi/sqrt(average) only when average is STRICTLY
    // above this value; otherwise the coupling is zero.
    PetscReal couplingAverageCutoff = PetscReal(1e-16);
};

// Real forcing at PHYSICAL coordinates, in the same nondimensional scaling as
// the legacy stokesForce/darcyForce. Captures can carry material parameters and
// time. An empty function ({}) means zero force. Return finite components;
// thrown C++ exceptions are converted to PETSc errors. Make no MPI collectives
// and do not mutate the basis, rules, porosity, parameters, or output here.
// Stokes multiplies this force by (1-phi); Darcy adds NO porosity multiplier.
using LocalForceFunction = std::function<Point(const Point&)>;

// All functions are LOCAL, without MPI communication or global numbering.
// Call after PetscInitialize, for owned cells whose geometry/field data and
// required ghosts are available. A basis is an owned geometry snapshot: rebuild
// it after mesh motion. Rectangles and valid convex logical quads use the same
// physical-coordinate integration with a point-dependent positive Jacobian.
// BRMixed/HDivMixed already contain the shared-edge orientation signs; none are
// applied again here. Darcy fluxes use the OUTWARD normal of the CCW boundary.
//
// Create GaussRule1D objects with CreateGaussRule and reuse them. The number of
// points, not polynomial degree, is adjustable; Darcy may use different cell
// and edge rules. Higher orders may be needed for rational quad basis functions
// and nonpolynomial coefficients. No quadrature order is hard-coded here.
//
// Every call computes a fresh zeroed work result and assigns output only after
// success. Validation remains active in Release. Only inputs used by the
// particular routine are checked. No global Mat/Vec, boundary conditions,
// phase model, DofMap, or material object is required or retained.

// Legacy AssignLocMatStokes, with 12 velocity DOFs and pressure pS=1:
//   A_ij = integral_K 2(1-phi)[eps_i:eps_j - div_i*div_j/3] dA
//   B_j  = integral_K div(v_j)*pS dA
//   C    = integral_K phi/(1-phi)*pS^2 dA
//   f_j  = integral_K (1-phi)*force.v_j dA
// eps=(grad(v)+grad(v)^T)/2. Keep the legacy 1/3 deviatoric correction.
// Only porosity.cell is used; porosity.edge/average are not required.
PetscErrorCode ComputeLocalStokes(
    const BRMixed& basis, const GaussRule1D& cellRule,
    const LocalPorositySamples& porosity, const LocalForceFunction& force,
    StokesLocalMatrix& result);

// Legacy AssignLocMatDarcy, with 8 velocity DOFs and pressure pD=1:
//   A_ij = integral_K u_i.u_j dA                    (UNWEIGHTED mass)
//   B_j  = integral_boundaryK phi_edge^(1+theta)/sqrt(phi_hat)
//                              *u_j.n_out*pD ds
//   C    = integral_K r/(1-phi)*pD^2 dA
//   f_j  = integral_K force.u_j dA
// phi_hat = (average == 0 ? 1 : average), using EXACT zero independently of
// the C cutoff; r = (average > darcyCompactionAverageCutoff ? phi/average : 1).
// B is a weighted BOUNDARY integral, not a volume divergence approximation.
// theta must be finite and the edge power must be finite. For phi_edge=0,
// require 1+theta >= 0; the exponent-zero case gives 1, as in legacy pow(0,0).
// Both sample arrays and the externally supplied average are required.
PetscErrorCode ComputeLocalDarcy(
    const HDivMixed& basis, const GaussRule1D& cellRule,
    const GaussRule1D& edgeRule, const LocalPorositySamples& porosity,
    const LocalMatrixParameters& parameters, const LocalForceFunction& force,
    DarcyLocalMatrix& result);

// Legacy AssignLocMatCouple, with the same physical cell and cell quadrature:
//   k = -integral_K t/(1-phi)*pS*pD dA
//   t = (average > couplingAverageCutoff ? phi/sqrt(average) : 0).
// The negative local sign and unnormalized constant pressure bases are retained.
// Uses porosity.cell, porosity.average and parameters.couplingAverageCutoff only.
PetscErrorCode ComputeLocalCoupling(
    const QuadBasis& geometry, const GaussRule1D& cellRule,
    const LocalPorositySamples& porosity,
    const LocalMatrixParameters& parameters, PetscScalar& result);

#endif

