#ifndef MANTLE_HDIVMIXED_OUTPUT_H
#define MANTLE_HDIVMIXED_OUTPUT_H

#include "hdivmixed.h"
#include <algorithm>
#include <limits>
#include <string>

struct HDivPlotOptions {
    PetscInt subdivisions = 36;
    PetscInt trace_segments = 200;
    PetscReal perturbation = 0.18; // Fraction of rectangular spacing, [0,0.25).
    // Exported data are Float64, including with other PETSc real precisions.
    PetscReal tolerance = 4096 * std::max(PetscReal(PETSC_MACHINE_EPSILON),
        PetscReal(std::numeric_limits<double>::epsilon()));
};

// Collective on comm; options and stem must agree on every rank.
// Samples the actual HDivMixed basis on two neighboring rectangular/perturbed
// cells, in both logical directions. A is sampled on rank 0, B on rank 1
// (both on rank 0 in serial); other ranks use empty collective HDF5 selections.
//
// Writes/replaces stem.h5, four stem_<case>.xdmf files, stem_summary.csv and
// stem_traces.csv. XDMF and HDF5 must stay together for ParaView.
// Requires parallel HDF5 built with PETSc's MPI.
//
// Checks the two shared modes and a combined field at identical physical edge
// points, using one canonical normal. Reports normal-trace discrepancies.
// Tangential-component and divergence jumps are recorded and are allowed.
// Preserves separate cell samples: never averages values across the interface.
// Writes the diagnostic output before returning an error for failed checks.
PetscErrorCode WriteHDivMixedXdmf(MPI_Comm comm, const HDivPlotOptions& options,
                                 const std::string& stem);

#endif
