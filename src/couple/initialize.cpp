#include "initialization.h"
#include "visualization.h"

namespace {
PetscErrorCode Run(mantle::couple::InitialState& state)
{
    PetscFunctionBeginUser;
    char filename[PETSC_MAX_PATH_LEN]="input.yaml";
    PetscCall(PetscOptionsGetString(nullptr,nullptr,"-input",filename,sizeof(filename),nullptr));
    mantle::couple::Configuration configuration;
    PetscCall(mantle::couple::ReadInitializationInput(PETSC_COMM_WORLD,filename,configuration));
    PetscCall(mantle::couple::Initialize(PETSC_COMM_WORLD,configuration,state));
    PetscCall(mantle::couple::WriteInitialState(PETSC_COMM_WORLD,configuration,state));
    PetscCall(mantle::couple::WriteInitializationVisualization(PETSC_COMM_WORLD,configuration,state));
    PetscFunctionReturn(PETSC_SUCCESS);
}
}

int main(int argc,char** argv)
{
    PetscErrorCode error=PetscInitialize(&argc,&argv,nullptr,
        "Initialize H/C and phase, then solve coupled Darcy-Stokes flow.\n"
        "  -input path/to/input.yaml\n"
        "Writes cell averages, Gauss-point visualization data and setup.txt; performs no time stepping.\n");
    if (error) return static_cast<int>(error);
    mantle::couple::InitialState state;
    error=Run(state);
    const auto cleanup=mantle::couple::DestroyInitialState(state);
    const auto finalize=PetscFinalize();
    return static_cast<int>(error ? error : cleanup ? cleanup : finalize);
}
