#ifndef MANTLE_CORE_INPUT_H
#define MANTLE_CORE_INPUT_H

#include <petscsys.h>

#include <array>
#include <cstdint>
#include <functional>
#include <map>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace mantle::input {

class InputError : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

// Plain owning data; no YAML, MFEM, phase, transport or preCICE types escape
// the reader. Scalars retain their text; accessors perform strict conversion.
// An extensible parameter map can contain nested maps, lists and nulls.
struct Value {
    enum class Kind { Null, Scalar, Sequence, Mapping };
    Kind kind = Kind::Null;
    std::string scalar;
    std::vector<Value> sequence;
    std::map<std::string, Value> mapping;
    std::string path;               // e.g. flow.solver.relative_tolerance
    int line = 0, column = 0;       // one-based source locations; 0 = generated

    const Value* Find(const std::string& key) const;
    const Value& At(const std::string& key) const;
    bool IsNull() const noexcept { return kind == Kind::Null; }
    std::string AsString() const;
    bool AsBool() const;            // true/false only
    double AsReal() const;          // finite decimal value
    std::int64_t AsInteger() const; // exact decimal conversion, no truncation
    std::uint64_t AsUnsigned() const;
    [[noreturn]] void Fail(const std::string& message) const;
};

struct FunctionInput {
    std::string name;
    Value parameters;
};

struct SimulationInput { std::string name, mode = "steady"; };
struct ParallelInput {
    std::string launcher;
    int ranks = 1;
    std::array<int, 2> processGrid{{1, 1}};
};
struct MeshInput {
    std::string family = "rectangular";
    std::array<double, 2> x{{0, 1}}, y{{0, 1}};
    std::array<std::int64_t, 2> cells{{1, 1}};
    double perturbation = 0.15;
    std::uint64_t seed = 1;
    std::string boundaryVertices = "fixed";
};
struct QuadratureInput { int cellPointsPerAxis = 5, edgePoints = 5; };
struct PorosityInput {
    std::string source = "prescribed_function";
    FunctionInput prescribedFunction;
    std::string provider = "hdf5";
    std::string file, group = "/porosity", runtimeField;
    std::map<std::string, std::string> datasets;
    std::string cellAverage = "physical_quadrature";
};

enum class Side { Left, Right, Bottom, Top };
enum class Selection { WholeSide, PhysicalInterval, BoundaryCells };
struct BoundaryRegionInput {
    Side side = Side::Bottom;
    Selection selection = Selection::WholeSide;
    std::string coordinate;         // x on horizontal sides, y on vertical sides
    std::array<double, 2> interval{{0, 1}};
    std::int64_t start = 0, count = 0;
    bool indicesOnBaseMesh = true;
};
struct BoundaryRegionsInput {
    double coordinateTolerance = 1e-12;
    std::map<std::string, BoundaryRegionInput> definitions;
};
struct BoundaryEdgeRange {
    Side side = Side::Bottom;
    std::int64_t first = 0, end = 0; // global half-open side-edge range
};
enum class BoundaryKind { Dirichlet, Neumann };
struct BoundaryConditionInput {
    BoundaryKind type = BoundaryKind::Dirichlet;
    FunctionInput value;
};
struct StokesSegmentInput {
    std::string region;
    int priority = 10;
    std::array<std::optional<BoundaryConditionInput>, 2> component;
};
struct DarcySegmentInput {
    std::string region;
    int priority = 10;
    BoundaryConditionInput condition;
};
struct BoundaryInput {
    double absoluteTolerance = 1e-12, relativeTolerance = 1e-10;
    std::string stokesComponents = "cartesian";
    // [x,y] or [normal,tangent], as selected by stokesComponents.
    // Current mesh families keep the outer rectangle fixed. Conversion of
    // normal/tangent data to Cartesian conditions belongs to the driver.
    std::array<BoundaryConditionInput, 2> stokesDefault;
    std::vector<StokesSegmentInput> stokesSegments;
    std::string darcyDirichletVariable = "assembled_normal_velocity";
    std::string darcyNeumannVariable = "pressure_potential";
    BoundaryConditionInput darcyDefault;
    std::vector<DarcySegmentInput> darcySegments;
};
struct BlockSolverInput {
    std::string ksp = "gmres", pc = "none";
    double relativeTolerance = 1e-12, absoluteTolerance = 1e-14;
    double divergenceTolerance = 1e8;
    int maximumIterations = 2000;
};
struct SolverInput {
    std::string optionsPrefix = "coupled_", ksp = "fgmres";
    std::string preconditioner = "schur";
    double relativeTolerance = 1e-10, absoluteTolerance = 1e-12;
    double divergenceTolerance = 1e8;
    int maximumIterations = 2000;
    bool initialGuessNonzero = false;
    BlockSolverInput velocity, pressure;
    bool errorIfNotConverged = true, requireSubsolverConvergence = true;
    bool requireTrueResidual = true, removePressureNullspace = false;
};
struct FlowInput {
    bool enabled = false;
    std::string system = "coupled_stokes_darcy", porosity = "porosity";
    std::string formulation = "legacy_rescaled";
    double theta = 0, darcyCompactionAverageCutoff = 1e-15;
    double couplingAverageCutoff = 1e-16;
    FunctionInput stokesForce, darcyForce, stokesPressureSource, darcyPressureSource;
    BoundaryInput boundary;
    std::string signConvention = "legacy", pressureNullspace = "none";
    Value pressureModes;            // driver-specific description when Provided
    bool projectRhs = false;
    SolverInput solver;
};

// Model-specific schemas live with their modules. Preserve every setting,
// even when disabled; an enabled module requires a registered validator.
struct ModuleInput { bool enabled = false; Value settings; };
struct TimeInput {
    double start = 0, end = 1;
    std::int64_t maximumSteps = 100000;
    std::string control = "fixed";
    double initialStep = 1e-3, minimumStep = 1e-8, maximumStep = 1e-2, cfl = 0.5;
};
struct CouplingInput {
    std::string backend = "internal", internalScheme = "single_pass";
    std::vector<std::string> updateOrder;
    int maximumIterations = 30;
    double relativeTolerance = 1e-8, absoluteTolerance = 1e-12, relaxation = 1;
    std::vector<std::string> monitoredFields;
    std::string checkpointStorage = "memory", checkpointScope = "complete_state";
    Value precice;                  // native definitions retained for the adapter
};
struct RestartInput {
    bool enabled = false;
    std::string file;
    bool writeCheckpoints = false;
    std::int64_t everyAcceptedSteps = 20;
    std::string checkpointFile = "checkpoints/state_{step}.h5";
};
struct ConvergenceInput {
    bool enabled = false;
    std::vector<std::string> meshFamilies;
    int levels = 1, additionalReferenceLevels = 1;
    std::array<int, 2> refinementFactor{{2, 2}};
    std::string hMeasure = "maximum_cell_diameter", referenceSource = "finer_mesh";
    Value reference;                // also permits registered reference providers
    std::vector<std::string> fields;
    std::string norm = "l2", evaluation = "common_physical_points";
    int errorPointsPerAxis = 7;
};
struct OutputInput {
    std::string directory, format = "hdf5_xdmf", fieldFile = "fields_{step}.h5";
    std::string xdmfSeries = "fields.xdmf";
    bool parallelHdf5 = true;
    std::vector<std::string> quantities;
    bool profilesEnabled = false;
    std::string profileFile = "profiles_{step}.csv";
    std::array<double, 2> profileStart{{0, 0}}, profileEnd{{0, 1}};
    int profilePoints = 201;
    std::vector<std::string> profileFields;
    std::string convergenceTable = "csv", solverReport = "csv";
    std::vector<std::string> diagnostics, plotFormats;
    bool plotConvergence = false, plotProfiles = false;
    std::string steadySchedule = "after_solve", transientSchedule = "accepted_steps";
    std::int64_t everyAcceptedSteps = 1;
    bool saveResolvedInput = true;
};

struct InputConfig {
    int schemaVersion = 2;
    // Root's absolute, lexical source path is authoritative in collective use.
    std::string sourceFile, baseDirectory, originalYaml;
    SimulationInput simulation;
    ParallelInput parallel;
    MeshInput mesh;
    QuadratureInput quadrature;
    PorosityInput porosity;
    BoundaryRegionsInput boundaryRegions;
    FlowInput flow;
    ModuleInput transport, phase;
    // Optional root `extensions: {module_name: {enabled: ..., ...}}`.
    std::map<std::string, ModuleInput> extensions;
    TimeInput time;
    CouplingInput coupling;
    RestartInput restart;
    ConvergenceInput convergence;
    OutputInput output;
};

struct InputExtensions {
    // Throw InputError on invalid settings. Validators are LOCAL, pure,
    // deterministic across ranks, and must NOT call MPI/PETSc collectives.
    // Built-in current-schema keys remain strict; open maps are only the
    // function/model/adapter/reference settings explicitly documented above.
    using Validator = std::function<void(const Value&, const InputConfig&)>;
    std::map<std::string, Validator> modules; // transport, phase, extension names
    Validator precice;
    // Custom function names; components is 1 for scalars or 2 for vector loads.
    std::map<std::string, std::function<void(const FunctionInput&, int)>> functions;
    // Custom convergence reference source -> schema validator.
    std::map<std::string, Validator> references;
};
struct ReadInputOptions {
    bool requireMatchingMpiSize = true;
    InputExtensions extensions;
};

// Local, noncollective C++ APIs. Throw InputError (including field location)
// on errors. Do not initialize PETSc/MPI or access any coefficient data file.
// sourceFile is used for provenance and resolving relative filenames.
InputConfig ParseInput(const std::string& yaml, const std::string& sourceFile,
                       const InputExtensions& extensions = {});
InputConfig ReadInputFile(const std::string& filename,
                         const InputExtensions& extensions = {});

// Collective after PetscInitialize, on a valid participant communicator.
// Only communicator rank 0 opens filename; it broadcasts the original YAML
// and its absolute path. Every rank parses/validates identical bytes locally.
// Read/parse/validation failures are agreed collectively before return.
// Output is unchanged on failure and committed on EVERY rank only on success.
// Supply matching options/validators on every rank; rank-0 filename is used.
// Does not launch MPI, register callbacks, load HDF5, build a solver, write
// preCICE XML, insert PETSc options, or create output directories.
PetscErrorCode ReadInput(MPI_Comm comm, const std::string& filename,
                        InputConfig& output, const ReadInputOptions& options = {});

// Names/parameters remain data; callback construction belongs to the driver.
// This helper resolves a configured boundary region on the current supported
// fixed rectangular outer boundary, including an optional refinement level.
// Physical endpoints must coincide with boundary vertices within tolerance.
// Shared-vertex Dirichlet value agreement is checked by the MFEM BC builder.
BoundaryEdgeRange ResolveBoundaryRegion(const InputConfig& input,
                                       const std::string& region, int level = 0);
std::array<std::int64_t, 2> CellsAtLevel(const InputConfig& input, int level);

// Expand {mesh_family}, {level}, {step}; unknown tokens are errors. Output
// paths/other input paths are not opened here. No shell or environment expansion.
std::string ExpandInputPath(const std::string& pattern, const std::string& meshFamily,
                            int level, std::int64_t step);
std::string ResolveInputPath(const InputConfig& input, const std::string& path);
std::string ResolveOutputPath(const InputConfig& input, const std::string& filename,
                             const std::string& meshFamily, int level = 0,
                             std::int64_t step = 0);

} // namespace mantle::input
#endif
