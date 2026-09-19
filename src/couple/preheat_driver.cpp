#include "preheat.h"
#include "visualization.h"

namespace {
PetscErrorCode Run(mantle::couple::InitialState& state)
{
    PetscFunctionBeginUser;
    char filename[PETSC_MAX_PATH_LEN]="input.yaml";
    PetscCall(PetscOptionsGetString(nullptr,nullptr,"-input",filename,sizeof(filename),nullptr));
    mantle::couple::Configuration c;
    PetscCall(mantle::couple::ReadInitializationInput(PETSC_COMM_WORLD,filename,c));
    PetscCall(mantle::couple::Initialize(PETSC_COMM_WORLD,c,state));
    mantle::couple::PreheatReport report;
    PetscCall(mantle::couple::AdvanceLegacyPreheat(PETSC_COMM_WORLD,c,state,report));
    PetscCall(mantle::couple::WriteInitialState(PETSC_COMM_WORLD,c,state));
    PetscCall(mantle::couple::WritePreheatHistory(PETSC_COMM_WORLD,c,report));
    PetscCall(mantle::couple::WriteInitializationVisualization(PETSC_COMM_WORLD,c,state));
    PetscFunctionReturn(PETSC_SUCCESS);
}
}
int main(int argc,char** argv)
{
    auto error=PetscInitialize(&argc,&argv,nullptr,
        "Legacy dry preheat: initialize flow once, advance H, hold C fixed.\n"
        "  -input path/to/input.yaml\nUses time.start/end/step and writes evolved H/C and visualization.\n");
    if (error) return static_cast<int>(error);
    mantle::couple::InitialState state;
    error=Run(state);
    const auto cleanup=mantle::couple::DestroyInitialState(state), finalize=PetscFinalize();
    return static_cast<int>(error?error:cleanup?cleanup:finalize);
}
