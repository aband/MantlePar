#ifndef MANTLE_PHASE_OUTPUT_H
#define MANTLE_PHASE_OUTPUT_H

#include "eutectic_rescaled.h"
#include <petscsys.h>
#include <optional>
#include <string>

struct PhasePlotOptions {
    PetscInt composition_cells = 200;
    PetscInt enthalpy_cells = 400;
    double pressure_pa = 0.0;
    double composition_min = 0.0;
    std::optional<double> composition_max; // Default: current material's Xe.
    std::optional<double> enthalpy_min;    // Default: Tep(P) - 0.2.
    std::optional<double> enthalpy_max;    // Default: Tmp(P) + LD + 0.2.
    bool normalized_axes = true;          // x=CD/Xe, y=HD-Tep(P); otherwise x=CD,y=HD.
    bool write_boundaries = true;
};

// Collective on comm. All ranks must supply identical material parameters,
// options, and output stem, and see the same output filesystem.
// Creates parent directories, then writes/replaces stem.h5 and stem.xdmf.
// Open stem.xdmf in ParaView. The two files must remain together.
//
// The H-C grid is a phase diagram, independent of the simulation's spatial mesh.
// Rows are distributed among ranks, including ranks with no rows. Every dataset
// write is collective; bulk field arrays are never gathered onto one rank.
// HDF5 must support parallel MPI I/O using the same MPI installation as PETSc.
//
// Singular dTD/dCD samples are exported as zero with DerivativeFinite=0;
// threshold on DerivativeFinite before interpreting the derivative field.
// Returned-state presence flags similarly identify absent solid/liquid phases.
//
// Reuse with runtime updates: call model.setParameters(updated), then call this
// function with a new stem. Automatic plot bounds use the updated parameters.
PetscErrorCode WritePhaseXdmf(MPI_Comm comm,
                              const mantle::phase::EutecticModel& model,
                              const PhasePlotOptions& options,
                              const std::string& stem);

#endif

