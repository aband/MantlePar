#include "brmixed_output.h"
#include <exception>

namespace {
PetscErrorCode Run()
{
    PetscFunctionBeginUser;
    BRPlotOptions options;
    PetscInt expected = 0;
    PetscBool supplied = PETSC_FALSE;
    PetscMPIInt ranks;
    char output[PETSC_MAX_PATH_LEN] = "output/brmixed/brmixed";
    PetscCall(PetscOptionsGetInt(nullptr, nullptr, "-br_n", &options.subdivisions, nullptr));
    PetscCall(PetscOptionsGetInt(nullptr, nullptr, "-br_trace_n", &options.trace_segments, nullptr));
    PetscCall(PetscOptionsGetReal(nullptr, nullptr, "-br_perturb", &options.perturbation, nullptr));
    PetscCall(PetscOptionsGetReal(nullptr, nullptr, "-br_tolerance", &options.tolerance, nullptr));
    PetscCall(PetscOptionsGetString(nullptr, nullptr, "-output", output, sizeof(output), nullptr));
    PetscCall(PetscOptionsGetInt(nullptr, nullptr, "-expected_ranks", &expected, &supplied));
    PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD, &ranks));
    PetscCheck(!supplied || expected == ranks, PETSC_COMM_WORLD, PETSC_ERR_ARG_WRONG,
               "Unexpected MPI rank count; use the launcher matching PETSc's MPI");
    PetscCall(WriteBRMixedXdmf(PETSC_COMM_WORLD, options, output));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD,
        "Wrote %s.h5, XDMF files and trace reports.\n"
        "Run plot_brmixed.py on the HDF5 file to generate the figures.\n", output));
    PetscFunctionReturn(PETSC_SUCCESS);
}
}

int main(int argc, char** argv)
{
    const PetscErrorCode error = PetscInitialize(&argc, &argv, nullptr,
        "BRMixed basis and two-cell conformity visualizer.\n"
        "  -br_n 36          : display subdivisions per reference axis\n"
        "  -br_trace_n 200   : sampling intervals on the common edge\n"
        "  -br_perturb 0.18  : quadrilateral perturbation in [0,0.25)\n"
        "  -br_tolerance T  : normalized jump tolerance\n"
        "  -output output/brmixed/brmixed : output stem without extension\n");
    if (error) return static_cast<int>(error);
    try {
        PetscCallAbort(PETSC_COMM_WORLD, Run());
    } catch (const std::exception& exception) {
        PetscCallAbort(PETSC_COMM_WORLD, PetscPrintf(PETSC_COMM_SELF, "%s\n", exception.what()));
        MPI_Abort(PETSC_COMM_WORLD, PETSC_ERR_LIB);
        return static_cast<int>(PETSC_ERR_LIB);
    }
    return static_cast<int>(PetscFinalize());
}
