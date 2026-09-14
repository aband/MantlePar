#include "hdivmixed_output.h"
#include <exception>

namespace {
PetscErrorCode Run()
{
    PetscFunctionBeginUser;
    HDivPlotOptions options;
    PetscInt expected = 0;
    PetscBool supplied = PETSC_FALSE;
    PetscMPIInt ranks;
    char output[PETSC_MAX_PATH_LEN] = "output/hdivmixed/hdivmixed";
    PetscCall(PetscOptionsGetInt(nullptr,nullptr,"-hdiv_n",&options.subdivisions,nullptr));
    PetscCall(PetscOptionsGetInt(nullptr,nullptr,"-hdiv_trace_n",&options.trace_segments,nullptr));
    PetscCall(PetscOptionsGetReal(nullptr,nullptr,"-hdiv_perturb",&options.perturbation,nullptr));
    PetscCall(PetscOptionsGetReal(nullptr,nullptr,"-hdiv_tolerance",&options.tolerance,nullptr));
    PetscCall(PetscOptionsGetString(nullptr,nullptr,"-output",output,sizeof(output),nullptr));
    PetscCall(PetscOptionsGetInt(nullptr,nullptr,"-expected_ranks",&expected,&supplied));
    PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD,&ranks));
    PetscCheck(!supplied || expected == ranks,PETSC_COMM_WORLD,PETSC_ERR_ARG_WRONG,
               "Unexpected MPI rank count; use the launcher matching PETSc's MPI");
    PetscCall(WriteHDivMixedXdmf(PETSC_COMM_WORLD,options,output));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD,
        "Wrote %s.h5, XDMF files and trace reports.\n"
        "Run plot_hdivmixed.py on the HDF5 file to generate the figures.\n",output));
    PetscFunctionReturn(PETSC_SUCCESS);
}
}

int main(int argc, char** argv)
{
    const PetscErrorCode error = PetscInitialize(&argc,&argv,nullptr,
        "HDivMixed basis and two-cell normal-conformity visualizer.\n"
        "  -hdiv_n 36          : subdivisions per reference axis\n"
        "  -hdiv_trace_n 200   : intervals on the shared edge\n"
        "  -hdiv_perturb 0.18  : quadrilateral perturbation in [0,0.25)\n"
        "  -hdiv_tolerance T  : normalized normal-trace tolerance\n"
        "  -output output/hdivmixed/hdivmixed : output stem without extension\n");
    if (error) return static_cast<int>(error);
    try {
        PetscCallAbort(PETSC_COMM_WORLD,Run());
    } catch (const std::exception& exception) {
        PetscCallAbort(PETSC_COMM_WORLD,PetscPrintf(PETSC_COMM_SELF,"%s\n",exception.what()));
        MPI_Abort(PETSC_COMM_WORLD,PETSC_ERR_LIB);
        return static_cast<int>(PETSC_ERR_LIB);
    }
    return static_cast<int>(PetscFinalize());
}
