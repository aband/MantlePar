#ifndef MANTLE_MFEM_BOUNDARY_CONDITIONS_H
#define MANTLE_MFEM_BOUNDARY_CONDITIONS_H

#include "assembly.h"

#include <array>
#include <functional>
#include <vector>

// Legacy mixed-FEM naming:
// Stokes Dirichlet = velocity; Neumann = total traction (stress * n_out).
// Darcy Dirichlet = normal component of the ASSEMBLED velocity unknown;
//       Neumann = pressure data for its natural boundary integral.
// This Darcy naming differs from a pressure-only primal formulation.
enum class BoundaryType { Unspecified, Dirichlet, Neumann };

struct BoundaryPoint {
    Point position{};
    Point outwardNormal{};
    CellSide side = CellSide::Bottom;
    MeshIndex cell{};              // Adjacent physical cell.
    PetscInt edgeId = 0;           // MeshInfo natural edge ID.
    PetscInt sideEdge = 0;         // Increasing i on bottom/top, j on left/right.
    PetscReal time = 0;
};

// A future input reader can resolve a named function and its parameters into
// this callback. If function is empty, constant is used. Values are REAL and
// must be finite. Capture runtime material parameters as needed; time is also
// passed explicitly in BoundaryPoint. Callbacks must be deterministic across
// ranks, contain no MPI collectives, and not modify assembly inputs.
// C++ exceptions are converted into collective PETSc errors. A callback using
// PETSc stack macros must catch exceptions before unwinding its PETSc frame.
struct BoundaryValue {
    PetscReal constant = 0;
    std::function<PetscReal(const BoundaryPoint&)> function;
};

struct BoundaryCondition {
    BoundaryType type = BoundaryType::Unspecified;
    BoundaryValue value{};
};

// Select WHOLE boundary edges in [firstEdge,endEdge). -1 means the end of the
// side. Segments must follow the mesh edges; changing type within one edge
// would require subdividing the edge/mesh. Top/left indices still increase in
// global i/j, independently of their reversed CCW quadrature direction.
// Larger priority overrides smaller priority ON AN EDGE. A tie between two
// matching, specified rules is an error. Ordering the input list has no effect.
struct BoundaryRegion {
    CellSide side = CellSide::Bottom;
    PetscInt firstEdge = 0;
    PetscInt endEdge = -1;
    int priority = 0;
};

struct StokesBoundaryRule {
    BoundaryRegion region{};
    // Cartesian x and y, separately: velocity component for Dirichlet,
    // traction component for Neumann. Unspecified lets another rule provide
    // that component, e.g. a high-priority override of only the x condition.
    std::array<BoundaryCondition, 2> component{};
};

struct StokesBoundarySpecification {
    std::vector<StokesBoundaryRule> rules;
    // All incident Dirichlet values for a vertex component must agree within
    // absTol + relTol*max(|a|,|b|). Dirichlet takes precedence over Neumann at
    // shared vertices. Rule priority does NOT silently overwrite incompatible
    // vertex data from another edge; specify consistent corner/segment values.
    PetscReal absoluteTolerance = PetscReal(1e-12);
    PetscReal relativeTolerance = PetscReal(1e-10);
};

enum class DarcyNeumannVariable {
    // Pressure DOF variable used by AssembleDarcyBlocks:
    // h = phi_edge^(1+theta) / sqrt(phi_hat) * pressure,
    // phi_hat = (cell_average == 0 ? 1 : cell_average).
    AssembledPressure,
    // Potential BEFORE the cell-average pressure rescaling:
    // h = phi_edge^(1+theta) * potential.
    PressurePotential,
    // Value already IS h. This explicitly bypasses porosity/model weighting,
    // useful for another weak formulation or an externally prepared load.
    WeightedNormalLoad
};

struct DarcyBoundaryRule {
    BoundaryRegion region{};
    BoundaryCondition condition{};
};

struct DarcyBoundarySpecification {
    std::vector<DarcyBoundaryRule> rules;
    DarcyNeumannVariable neumannVariable = DarcyNeumannVariable::AssembledPressure;
    // Natural contribution to momentum RHS = pressureSign * integral h*v.n.
    // For A*u - B^T*p = f (legacy sign), pressureSign=-1. For the opposite
    // pressure-gradient sign use +1. This is unrelated to the sign of the
    // PRESSURE ROW used later in BoundaryApplicationOptions.
    PetscReal pressureSign = -1;
};

// Evaluated boundary data for ONE velocity space and ONE MPI partition.
// essentialDofs contains sorted, unique, OWNED algebraic IDs. naturalLoad has
// the complete assembled boundary load, including entries later constrained.
// No ghost values or communicator are retained. Treat all members as read-only
// between construction and application. Rebuild after changing data, time,
// rules, mesh coordinates, partition, or model parameters.
// Owns its Vec: explicit collective cleanup before PetscFinalize, as with
// MixedBlocks. There is no collective work in a C++ destructor.
struct BoundaryData {
    std::vector<PetscInt> essentialDofs;
    std::vector<PetscScalar> essentialValues;
    Vec naturalLoad = nullptr;
    DofSpace space = DofSpace::CellPressure;
    PetscInt globalDofs = 0;
    PetscInt ownedBegin = 0, ownedEnd = 0;

    BoundaryData() = default;
    BoundaryData(const BoundaryData&) = delete;
    BoundaryData& operator=(const BoundaryData&) = delete;
    BoundaryData(BoundaryData&&) = delete;
    BoundaryData& operator=(BoundaryData&&) = delete;
    bool IsEmpty() const noexcept
    { return !naturalLoad && essentialDofs.empty() && essentialValues.empty(); }
};

PetscErrorCode DestroyBoundaryData(BoundaryData& data);

// COLLECTIVE on the SAME communicator/rank ordering as velocityMap.
// Each boundary edge/component must be covered by a specified rule. Both
// zero and nonzero Neumann data must be explicit; missing data are errors.
// edgeRule must have at least two points (bubble moment / linear projection).
// Empty cell/DOF owners participate. Geometry must have valid MPI ghosts.
// Supply the same specification, time, quadrature and model parameters on
// every rank. Callback captures may contain local, consistently ghosted fields.
// Essential values are evaluated using available boundary cells and kept by
// the DOF owner; natural integrals are contributed only by the cell owner.
// Output must be empty; it is published only after all stages succeed.
//
// Stokes: interpolate prescribed vertex components and match their boundary
// flux with the BR bubble. Full velocity data preserve the legacy normal-flux
// moment (NOT midpoint interpolation). Cartesian mixed conditions also work
// on slanted edges; normal/tangential conditions on a slanted outer boundary
// would require a separate linear-constraint representation, not x/y flags.
PetscErrorCode BuildStokesBoundaryData(
    MPI_Comm comm, const MeshInfo& mesh, const DofMap& velocityMap,
    const GaussRule1D& edgeRule, const StokesBoundarySpecification& specification,
    PetscReal time, BoundaryData& result);

// Darcy: L2-project OUTWARD normal velocity onto both H(div) edge traces.
// Dirichlet data refer to the assembled velocity unknown, not an implicitly
// converted physical mass flux. Perform any model-specific conversion in the
// callback. Natural pressure data use neumannVariable and pressureSign above.
// For weighted pressure modes, porosity must supply edge samples in the SAME
// CCW order and edge rule as assembly. Only owned cells with Neumann edges
// call it, once each, with fresh samples; cell[] is unused here. The average
// is required only for AssembledPressure. Refresh field ghosts beforehand.
// An empty porosity callback is allowed for all-Dirichlet conditions or for
// WeightedNormalLoad. Never divide prescribed Dirichlet flow by porosity here.
PetscErrorCode BuildDarcyBoundaryData(
    MPI_Comm comm, const MeshInfo& mesh, const DofMap& velocityMap,
    const GaussRule1D& edgeRule, const DarcyBoundarySpecification& specification,
    PetscReal time, const CellPorosityFunction& porosity,
    const LocalMatrixParameters& parameters, BoundaryData& result);

struct BoundaryApplicationOptions {
    // REQUIRED: set +1 or -1 explicitly. If the eventual pressure equation is
    // beta*B*u + (signed pressure blocks)*p = g, set pressureRowSign=beta.
    // The legacy solver negated B, hence beta=-1. B and C in the returned
    // MixedBlocks RETAIN their raw assembly signs; compose signs afterwards.
    PetscReal pressureRowSign = 0;
    PetscReal diagonal = 1; // Positive identity-row scale.
};

// COLLECTIVE. original is a fresh, unconstrained assembly with full numbering.
// Creates an INDEPENDENT constrained copy in empty result; original and data
// are unchanged, so changing boundary values/time cannot accumulate old loads.
// f <- f + naturalLoad; g <- g - beta*B*u_D, using the ORIGINAL B.
// A rows/columns are eliminated with RHS lifting, A_DD=diagonal*I,
// f_D=diagonal*u_D. Constrained columns of B are zeroed. C is copied unchanged.
// When composing a system, form its signed transpose coupling FROM THIS B;
// do not retain a transpose of the unconstrained B. Velocity Dirichlet data
// do not alter separate pressure-pressure coupling matrices such as K_sd.
// Only the velocity-boundary operation is performed: no pressure gauge,
// pressure pinning, solver, or nullspace decision is made here.
// Requires the assembled AIJ matrices produced by assembly.cpp.
PetscErrorCode ApplyBoundaryConditions(
    MPI_Comm comm, const MixedBlocks& original, const BoundaryData& data,
    const BoundaryApplicationOptions& options, MixedBlocks& result);

#endif

