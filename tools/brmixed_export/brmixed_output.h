#ifndef MANTLE_BRMIXED_OUTPUT_H
#define MANTLE_BRMIXED_OUTPUT_H

#include "brmixed.h"
#include <string>

struct BRPlotOptions {
    PetscInt subdivisions = 36;       // Display cells along each reference axis.
    PetscInt trace_segments = 200;    // Common-edge sampling intervals.
    PetscReal perturbation = 0.18;    // Fraction of the rectangular spacing, [0,0.25).
    PetscReal tolerance = 4096 * PETSC_MACHINE_EPSILON;
};

// Collective on comm, with identical options and stem on every rank.
// Samples the actual BRMixed class on two adjacent rectangular/perturbed cells,
// in both logical neighbor directions. Cell A belongs to rank 0 and cell B to
// rank 1 (both to rank 0 in serial); additional ranks use empty HDF5 selections.
//
// Writes/replaces stem.h5, four stem_<case>.xdmf files, stem_summary.csv and
// stem_traces.csv. Open an XDMF file in ParaView; keep it beside the HDF5 file.
// Requires parallel HDF5 built with the same MPI as PETSc.
//
// Matched vertex/edge modes and a nontrivial combined velocity are evaluated
// independently on either side of the same physical edge points. Reports value
// and tangential-derivative jumps; normal-derivative jumps are not H1 criteria.
// Output is written even if the sampled conformity check fails, then a nonzero
// PETSc error is returned. This diagnostic does not replace convergence tests.
PetscErrorCode WriteBRMixedXdmf(MPI_Comm comm, const BRPlotOptions& options,
                               const std::string& stem);

#endif

