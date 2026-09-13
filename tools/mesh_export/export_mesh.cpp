#include "mesh.h"
#include "mesh_output.h"

namespace {

PetscErrorCode GenerateAndExport()
{
    DM dm = nullptr;
    Vec vertices = nullptr;
    MeshParam mp;
    PetscInt nx = 32, ny = 16, px = PETSC_DECIDE, py = PETSC_DECIDE;
    PetscInt type = 1, seed = 42;
    char output[PETSC_MAX_PATH_LEN] = "mesh";
    PetscFunctionBeginUser;
    PetscCall(PetscOptionsGetInt(nullptr, nullptr, "-mesh_nx", &nx, nullptr));
    PetscCall(PetscOptionsGetInt(nullptr, nullptr, "-mesh_ny", &ny, nullptr));
    PetscCall(PetscOptionsGetInt(nullptr, nullptr, "-mesh_px", &px, nullptr));
    PetscCall(PetscOptionsGetInt(nullptr, nullptr, "-mesh_py", &py, nullptr));
    PetscCall(PetscOptionsGetInt(nullptr, nullptr, "-mesh_type", &type, nullptr));
    PetscCall(PetscOptionsGetInt(nullptr, nullptr, "-mesh_seed", &seed, nullptr));
    PetscCall(PetscOptionsGetReal(nullptr, nullptr, "-mesh_xstart", &mp.xstart, nullptr));
    PetscCall(PetscOptionsGetReal(nullptr, nullptr, "-mesh_ystart", &mp.ystart, nullptr));
    PetscCall(PetscOptionsGetReal(nullptr, nullptr, "-mesh_length", &mp.L, nullptr));
    PetscCall(PetscOptionsGetReal(nullptr, nullptr, "-mesh_height", &mp.H, nullptr));
    PetscCall(PetscOptionsGetReal(nullptr, nullptr, "-mesh_perturbation", &mp.perturbation, nullptr));
    PetscCall(PetscOptionsGetString(nullptr, nullptr, "-output", output, sizeof(output), nullptr));
    PetscCheck(nx >= 1 && ny >= 1 && nx < PETSC_MAX_INT && ny < PETSC_MAX_INT,
               PETSC_COMM_WORLD, PETSC_ERR_ARG_OUTOFRANGE, "Invalid cell counts");
    PetscCheck(type >= 0 && type <= 2 && seed >= 0, PETSC_COMM_WORLD,
               PETSC_ERR_ARG_OUTOFRANGE, "Use mesh_type 0, 1, or 2 and a nonnegative seed");
    mp.seed = static_cast<std::uint64_t>(seed);

    PetscCall(DMDACreate2d(PETSC_COMM_WORLD, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE,
                          DMDA_STENCIL_BOX, nx + 1, ny + 1, px, py, 2, 1,
                          nullptr, nullptr, &dm));
    PetscCall(DMSetUp(dm));
    PetscCall(DMCreateGlobalVector(dm, &vertices));
    if (type == 0) PetscCall(CreateFullMesh(dm, vertices, mp));
    else if (type == 1) PetscCall(LogicRectMesh(dm, vertices, mp));
    else PetscCall(RefineMesh(dm, vertices, mp));
    PetscCall(WriteMeshXdmf(dm, vertices, output));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Wrote %s.h5 and %s.xdmf\n", output, output));
    PetscCall(VecDestroy(&vertices));
    PetscCall(DMDestroy(&dm));
    PetscFunctionReturn(PETSC_SUCCESS);
}

} // namespace

int main(int argc, char** argv)
{
    const PetscErrorCode init = PetscInitialize(&argc, &argv, nullptr,
        "Export a mesh as parallel HDF5 + XDMF. mesh_type: 0=rectangle, 1=perturbed, 2=sine.\n");
    if (init) return static_cast<int>(init);
    PetscCallAbort(PETSC_COMM_WORLD, GenerateAndExport());
    return static_cast<int>(PetscFinalize());
}
