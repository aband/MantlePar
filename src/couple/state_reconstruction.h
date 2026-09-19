#ifndef MANTLE_COUPLE_STATE_RECONSTRUCTION_H
#define MANTLE_COUPLE_STATE_RECONSTRUCTION_H

#include "initialization.h"
#include "reconstruction.h"

namespace mantle::couple {
// Local reconstructions plus an explicit MPI scatter for their scalar halo.
// Geometry ownership matches InitialState; field ownership may differ.
// Explicit collective Destroy(), including after a failed Initialize().
class StateReconstruction {
public:
    StateReconstruction() = default;
    StateReconstruction(const StateReconstruction&) = delete;
    StateReconstruction& operator=(const StateReconstruction&) = delete;
    PetscErrorCode Initialize(MPI_Comm, const InitialState&);
    PetscErrorCode Update(const InitialState&, bool updateComposition = true);
    // Mean-preserving local bounds at phase, flux and diffusion sampling points.
    // Enabled only for the fully coupled solver; preheat behavior is unchanged.
    PetscErrorCode Limit(const Configuration&);
    PetscErrorCode Destroy();
    PetscErrorCode Evaluate(MeshIndex, const Point&, PetscReal& H, PetscReal& C) const;
    PetscErrorCode Enthalpy(MeshIndex, const Point&, PetscReal&) const;
    PetscReal AverageH(MeshIndex cell) const;
    PetscReal AverageC(MeshIndex cell) const;
    const MeshInfo& Mesh() const { return mesh_; }
private:
    std::size_t Index(MeshIndex) const;
    MPI_Comm comm_ = MPI_COMM_NULL;
    DM geometryDM_ = nullptr;
    Vec coordinates_ = nullptr, local_ = nullptr;
    VecScatter scatter_ = nullptr;
    MeshInfo mesh_;
    MeshRange targets_, valuesRange_;
    std::vector<Reconstruction> h_, c_;
    std::vector<PetscReal> averagesH_, averagesC_, areas_;
    std::vector<PetscReal> limitH_, limitC_;
};
}
#endif
