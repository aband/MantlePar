#ifndef MANTLE_COUPLE_PREHEAT_H
#define MANTLE_COUPLE_PREHEAT_H

#include "initialization.h"

namespace mantle::couple {
struct PreheatStep {
    PetscInt step = 0;
    PetscReal time = 0, dt = 0, courant = 0;
    PetscReal integral = 0, minimum = 0, maximum = 0;
    PetscReal boundaryFlux = 0, balanceError = 0;
};
struct PreheatReport {
    PetscReal initialTime = 0, finalTime = 0, stepLimit = 0;
    PetscReal initialIntegral = 0, finalIntegral = 0, boundaryIntegral = 0;
    std::vector<PreheatStep> history;
};

// Collective legacy dry preheat: H_t + div(us*H - kappa*grad(H)) = 0,
// C held fixed, us frozen at the accepted initial flow. Forward Euler with
// ML-WENO (3,2) advection and constant-cell normal-line diffusion, as in the
// legacy advdiff driver. Equilibrium phase is diagnostic, not the diffused
// quantity in this specific dry thermal model. Values are nondimensional.
// Fixed steps exceeding the advective/diffusive rate bound are rejected;
// cfl control selects a bounded step. The final step lands on time.end.
// On failure H/C/time retain the last accepted step. No rejected candidate is
// published. Phase Vecs are refreshed at successful completion.
PetscErrorCode AdvanceLegacyPreheat(MPI_Comm, const Configuration&, InitialState&, PreheatReport&);
PetscErrorCode WritePreheatHistory(MPI_Comm, const Configuration&, const PreheatReport&);
}
#endif
