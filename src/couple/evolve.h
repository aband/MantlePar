#ifndef MANTLE_COUPLE_EVOLVE_H
#define MANTLE_COUPLE_EVOLVE_H
#include "initialization.h"
#include "state_reconstruction.h"

namespace mantle::couple {
// reviewed.pdf: (2.26), (2.81), (4.1), (4.11)-(4.14), SSPRK2 (3.210).
struct TransportVelocities { Point mixture{}, effective{}; };
TransportVelocities PhaseTransportVelocities(Point solid, Point scaledRelative,
    PetscReal porosity, PetscReal theta, const phase::PhaseState& left,
    const phase::PhaseState& right);
PetscErrorCode SolveReconstructedFlow(MPI_Comm, const Configuration&, InitialState&,
                                    const StateReconstruction&);
// Imports exact cell averages, verifies mesh/scales/units and resets the clock
// to time.start. Source directory is supplied explicitly by the driver.
PetscErrorCode LoadPreheatState(MPI_Comm,const Configuration&,const std::string&,InitialState&);
PetscErrorCode AdvancePhaseCoupled(MPI_Comm,const Configuration&,InitialState&);
}
#endif
