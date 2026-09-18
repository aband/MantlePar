#ifndef MANTLE_COUPLE_INITIALIZATION_H
#define MANTLE_COUPLE_INITIALIZATION_H

#include "input.h"
#include "field_initialization.h"
#include "eutectic_rescaled.h"
#include "boundary_conditions.h"
#include "linear_solver.h"
#include "advectiveflux.h"
#include "diffusiveflux.h"

#include <functional>
#include <string>
#include <vector>

namespace mantle::couple {

// Positions and time have the simulation's nondimensional units. Callbacks own
// their parameters, make no collective calls, and do not depend on YAML lifetime.
using ScalarProfile = std::function<PetscReal(const Point&, PetscReal)>;

enum class AdvectionPolicy { InflowOutflow, Outflow, ZeroFlux };
struct AdvectionCondition {
    AdvectionPolicy policy = AdvectionPolicy::ZeroFlux;
    ScalarProfile inflow;
};
enum class ThermalPolicy { Temperature, OutwardFlux, ZeroFlux };
struct ThermalCondition {
    ThermalPolicy policy = ThermalPolicy::ZeroFlux;
    ScalarProfile value;
};
template<class Condition> struct BoundaryRule {
    BoundaryRegion region;
    Condition condition;
};
struct TransportBoundarySpecification {
    std::vector<BoundaryRule<AdvectionCondition>> enthalpy, composition;
    std::vector<BoundaryRule<ThermalCondition>> temperature;
};

struct FlowSetup {
    // Explicit legacy preheat mode: equilibrium phase is diagnostic, while
    // every coefficient supplied to the flow operator has zero porosity.
    bool dryPorosity = false;
    StokesBoundarySpecification stokes;
    DarcyBoundarySpecification darcy;
    LocalMatrixParameters material;
    LinearSolverOptions solver;
    LocalForceFunction stokesForce, darcyForce;
    ScalarProfile stokesPressureSource, darcyPressureSource;
};

struct Configuration {
    input::InputConfig input;
    phase::EutecticModel phase;
    // Pressure for the phase law is dimensional Pa, not a flow pressure DOF.
    ScalarProfile pressure;
    ScalarProfile initialEnthalpy, initialComposition;
    FlowSetup flow;
    TransportBoundarySpecification boundary;
    LaxFriedrichsOptions stabilizer;
    DiffusiveSamplingOptions sampling;
    PetscReal thermalDiffusivity = 0;
};

// Local parsers throw input::InputError. The core reader retains its strict
// schema_version=2 and owns YAML/MPI handling; no second YAML reader is added.
input::ReadInputOptions InitializationReaderOptions();
Configuration MakeInitializationConfiguration(const input::InputConfig& input);
PetscErrorCode ReadInitializationInput(
    MPI_Comm comm, const std::string& filename, Configuration& output);

// Noncollective adapters to the existing flux boundary types. Supply the
// OUTWARD normal speed for the relevant advected variable at this point.
// InflowOutflow uses prescribed data only when speed<0, extrapolates at speed>0,
// and returns ZeroFlux at speed=0. Outflow rejects backflow. These policies are
// for scalar characteristic transport; the future coupled law chooses speeds.
PetscErrorCode EvaluateAdvectionBoundary(
    const std::vector<BoundaryRule<AdvectionCondition>>& rules,
    const BoundaryPoint& point, PetscReal normalSpeed, AdvectiveBoundaryValue& value);
// PrescribedTemperature -> PrescribedQuantity, NOT PrescribedState for H.
// PrescribedFlux means physical OUTWARD -kappa*grad(T).n per unit edge length.
PetscErrorCode EvaluateThermalBoundary(
    const std::vector<BoundaryRule<ThermalCondition>>& rules,
    const BoundaryPoint& point, DiffusiveBoundaryValue& value);

// Owns PETSc resources; explicit collective cleanup before PetscFinalize.
// No collective operations in C++ destruction. Fields are separate scalar Vecs
// on one cell DMDA. Geometry ownership need not match field ownership.
struct InitialState {
    DM vertexDM = nullptr, cellDM = nullptr;
    Vec vertices = nullptr;
    Vec enthalpy = nullptr, composition = nullptr;
    Vec temperature = nullptr, porosity = nullptr;
    MeshInfo mesh;
    PetscReal time = 0;
    // Independent MFEM numbering, not the scalar cell DMDA numbering.
    DofMap stokesVelocityMap, darcyVelocityMap, pressureMap;
    LinearSystem flowSystem;
    LinearSolveReport flowReport;
    // Actual assembly coefficients, row-major over mesh.OwnedCells().
    std::vector<LocalPorositySamples> flowPorosity;

    InitialState() = default;
    InitialState(const InitialState&) = delete;
    InitialState& operator=(const InitialState&) = delete;
    InitialState(InitialState&&) = delete;
    InitialState& operator=(InitialState&&) = delete;
    bool IsEmpty() const noexcept {
        return !vertexDM && !cellDM && !vertices && !enthalpy && !composition
            && !temperature && !porosity && flowSystem.IsEmpty();
    }
};

// Collective. Pass identical validated configuration on all ranks and an empty
// result. Builds mesh and volume averages of H,C,T,phi at time.start. T,phi are
// quadrature averages of phase(H_initial(x),C_initial(x),P(x)); they are startup
// diagnostics, not phase evaluated on averaged H,C. Assembles and solves the
// coupled Darcy-Stokes system using phase porosity sampled from those same
// profiles. Shared edges on aligned profile jumps use harmonic phase traces.
// Retains the system, DOF maps, coefficients and accepted solve report; retrieve
// borrowed solution fields with GetCoupledLinearSystemSolution(flowSystem,...).
// No reconstruction or time step occurs. On failure, result remains empty.
PetscErrorCode Initialize(
    MPI_Comm comm, const Configuration& configuration, InitialState& result);
PetscErrorCode DestroyInitialState(InitialState& state);

// Collective. Writes cell averages per GEOMETRY owner, flow coefficients per
// MFEM DOF owner, rank-zero setup.txt and optional input_used.yaml.
// CSV values come from the actual distributed Vecs,
// with an AO/scatter mapping when geometry and field ownership differ.
// Output directory is relative to the YAML file, as in the existing examples.
PetscErrorCode WriteInitialState(
    MPI_Comm comm, const Configuration& configuration, const InitialState& state);

} // namespace mantle::couple
#endif
