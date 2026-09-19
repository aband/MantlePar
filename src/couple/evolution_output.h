#ifndef MANTLE_COUPLE_EVOLUTION_OUTPUT_H
#define MANTLE_COUPLE_EVOLUTION_OUTPUT_H
#include "visualization.h"
#include "state_reconstruction.h"
#include <fstream>

namespace mantle::couple {
// Optional compact, per-rank movie data. No additional mechanics solves.
class EvolutionOutput {
public:
    PetscErrorCode Open(MPI_Comm,const Configuration&);
    PetscErrorCode Write(MPI_Comm,const Configuration&,const InitialState&,
                         const StateReconstruction&,const InitialFlowSamples&);
    PetscErrorCode Finish(MPI_Comm,const Configuration&,const InitialState&);
    bool Enabled() const { return enabled_; }
private:
    PetscBool enabled_=PETSC_FALSE;
    PetscInt frames_=0,centers_=0,gauss_=0;
    std::ofstream centersFile_,gaussFile_,timesFile_;
};
}
#endif
