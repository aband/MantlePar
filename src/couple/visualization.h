#ifndef MANTLE_COUPLE_VISUALIZATION_H
#define MANTLE_COUPLE_VISUALIZATION_H

#include "initialization.h"

namespace mantle::couple {

struct FlowPointSample {
    PetscInt cellId=0, edgeId=-1, side=-1, q=0;
    Point position{}, stokesVelocity{}, darcyVelocity{};
    PetscReal weight=0, stokesPressure=0, darcyPressure=0, flowPorosity=0;
};
struct InitialFlowSamples {
    std::vector<FlowPointSample> cells, edges;
};
// Collective, with local output over geometry-owned cells. Reconstructs the
// solved BR/H(div) fields using ghosted MFEM coefficients. Edge samples retain
// BOTH cell traces: H(div) tangential velocity and P0 pressure may jump. Edge q
// follows canonical edge orientation; side remains the cell's CCW side number.
// Darcy velocity is the assembled rescaled variable, without sign conversion.
PetscErrorCode SampleInitialFlow(MPI_Comm comm, const Configuration& configuration,
                                const InitialState& state, InitialFlowSamples& samples);

// Collective startup diagnostic. Phase is sampled at mapped cell centers;
// physical BR/H(div) velocities are reconstructed at cell/edge Gauss points.
// Original phase Gauss samples remain available for integration checks. CSV
// includes physical values and explicitly named raw nondimensional diagnostics.
// Edge flow output retains both cell traces. The manifest (schema 2) enumerates
// current files; plot_initialization.py displays physical units only.
PetscErrorCode WriteInitializationVisualization(
    MPI_Comm comm, const Configuration& configuration, const InitialState& state);

} // namespace mantle::couple
#endif
