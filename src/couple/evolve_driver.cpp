#include "evolve.h"
#include "visualization.h"

namespace {
PetscErrorCode Run(mantle::couple::InitialState& state)
{
    PetscFunctionBeginUser;
    char filename[PETSC_MAX_PATH_LEN]="input.yaml",source[PETSC_MAX_PATH_LEN]="";
    PetscBool supplied=PETSC_FALSE;
    PetscCall(PetscOptionsGetString(nullptr,nullptr,"-input",filename,sizeof(filename),nullptr));
    PetscCall(PetscOptionsGetString(nullptr,nullptr,"-preheat",source,sizeof(source),&supplied));
    PetscCheck(supplied,PETSC_COMM_WORLD,PETSC_ERR_USER_INPUT,"Supply -preheat <completed preheat output directory>");
    mantle::couple::Configuration c;
    PetscCall(mantle::couple::ReadInitializationInput(PETSC_COMM_WORLD,filename,c));
    PetscCheck(!c.flow.dryPorosity,PETSC_COMM_WORLD,PETSC_ERR_USER_INPUT,"Full evolution requires runtime phase_porosity");
    PetscCall(mantle::couple::Initialize(PETSC_COMM_WORLD,c,state,false));
    PetscCall(mantle::couple::LoadPreheatState(PETSC_COMM_WORLD,c,source,state));
    PetscCall(mantle::couple::AdvancePhaseCoupled(PETSC_COMM_WORLD,c,state));
    PetscCall(mantle::couple::WriteInitialState(PETSC_COMM_WORLD,c,state));
    PetscCall(mantle::couple::WriteInitializationVisualization(PETSC_COMM_WORLD,c,state));
    PetscFunctionReturn(PETSC_SUCCESS);
}
}
int main(int argc,char** argv)
{
    auto error=PetscInitialize(&argc,&argv,nullptr,
        "Full phase-coupled H/C evolution, SSPRK2 with mechanics at each stage.\n"
        "  -input input.yaml -preheat path/to/completed/preheat/output\n"
        "  -couple_porosity_snapshots saves cell-center porosity at every accepted step.\n"
        "  -couple_movie_snapshots saves center phase/temperature and Gauss velocity movies.\n");
    if (error) return static_cast<int>(error);
    mantle::couple::InitialState state;
    error=Run(state);
    const auto cleanup=mantle::couple::DestroyInitialState(state),finalize=PetscFinalize();
    return static_cast<int>(error?error:cleanup?cleanup:finalize);
}
