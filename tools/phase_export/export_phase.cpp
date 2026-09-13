#include "phase_output.h"

#include <array>
#include <exception>

namespace {

PetscErrorCode ReadDouble(const char* name, double& value) {
    PetscReal parsed = static_cast<PetscReal>(value);
    PetscFunctionBeginUser;
    PetscCall(PetscOptionsGetReal(nullptr, nullptr, name, &parsed, nullptr));
    value = static_cast<double>(parsed);
    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode ReadOptionalDouble(const char* name, std::optional<double>& value) {
    PetscReal parsed = 0;
    PetscBool supplied = PETSC_FALSE;
    PetscFunctionBeginUser;
    PetscCall(PetscOptionsGetReal(nullptr, nullptr, name, &parsed, &supplied));
    if (supplied) value = static_cast<double>(parsed);
    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode GenerateAndExport() {
    using mantle::phase::MaterialParameters;
    MaterialParameters material;
    PhasePlotOptions plot;
    PetscBool normalized = PETSC_TRUE, boundaries = PETSC_TRUE;
    char output[PETSC_MAX_PATH_LEN] = "output/phase/diagram";
    PetscFunctionBeginUser;
    PetscCall(PetscOptionsGetInt(nullptr, nullptr, "-phase_nc", &plot.composition_cells, nullptr));
    PetscCall(PetscOptionsGetInt(nullptr, nullptr, "-phase_nh", &plot.enthalpy_cells, nullptr));
    PetscCall(ReadDouble("-phase_pressure", plot.pressure_pa));
    PetscCall(ReadDouble("-phase_c_min", plot.composition_min));
    PetscCall(ReadOptionalDouble("-phase_c_max", plot.composition_max));
    PetscCall(ReadOptionalDouble("-phase_h_min", plot.enthalpy_min));
    PetscCall(ReadOptionalDouble("-phase_h_max", plot.enthalpy_max));
    PetscCall(PetscOptionsGetBool(nullptr, nullptr, "-phase_normalized_axes", &normalized, nullptr));
    PetscCall(PetscOptionsGetBool(nullptr, nullptr, "-phase_boundaries", &boundaries, nullptr));
    PetscCall(PetscOptionsGetString(nullptr, nullptr, "-output", output, sizeof(output), nullptr));
    plot.normalized_axes = normalized == PETSC_TRUE;
    plot.write_boundaries = boundaries == PETSC_TRUE;

    struct Option { const char* name; double MaterialParameters::*member; };
    const std::array<Option, 13> options{{
        {"-phase_tm0", &MaterialParameters::Tm0}, {"-phase_te0", &MaterialParameters::Te0},
        {"-phase_nu", &MaterialParameters::nu}, {"-phase_l", &MaterialParameters::L},
        {"-phase_cp", &MaterialParameters::cp}, {"-phase_xe", &MaterialParameters::Xe},
        {"-phase_rho", &MaterialParameters::rho}, {"-phase_rhor", &MaterialParameters::rhor},
        {"-phase_mus", &MaterialParameters::mus}, {"-phase_mul", &MaterialParameters::mul},
        {"-phase_k0", &MaterialParameters::k0}, {"-phase_g", &MaterialParameters::g},
        {"-phase_alpha0", &MaterialParameters::alpha0}
    }};
    for (const auto& option : options) PetscCall(ReadDouble(option.name, material.*(option.member)));
    mantle::phase::EutecticModel model;
    try {
        model.setParameters(material);
    } catch (const std::exception& error) {
        SETERRQ(PETSC_COMM_WORLD, PETSC_ERR_ARG_WRONG, "Invalid material parameters: %s", error.what());
    }
    PetscCall(WritePhaseXdmf(PETSC_COMM_WORLD, model, plot, output));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Wrote %s.h5 and %s.xdmf\nOpen the XDMF file in ParaView.\n",
                          output, output));
    PetscFunctionReturn(PETSC_SUCCESS);
}

} // namespace

int main(int argc, char** argv) {
    const PetscErrorCode init = PetscInitialize(&argc, &argv, nullptr,
        "Export the eutectic phase diagram using parallel HDF5 and XDMF.\n"
        "  -phase_nc 200 -phase_nh 400 : composition/enthalpy cell counts\n"
        "  -phase_pressure 0           : dimensional pressure in Pa\n"
        "  -phase_c_min/-phase_c_max   : bulk composition range (default 0 to Xe)\n"
        "  -phase_h_min/-phase_h_max   : dimensionless enthalpy range\n"
        "  -phase_normalized_axes true : x=CD/Xe, y=HD-Tep(P)\n"
        "  -phase_boundaries true      : include boundary curves\n"
        "  -output output/phase/diagram : output stem, without an extension\n"
        "Material overrides (SI units): -phase_tm0, -phase_te0, -phase_nu,\n"
        "  -phase_l, -phase_cp, -phase_xe, -phase_rho, -phase_rhor,\n"
        "  -phase_mus, -phase_mul, -phase_k0, -phase_g, -phase_alpha0.\n");
    if (init) return static_cast<int>(init);
    PetscCallAbort(PETSC_COMM_WORLD, GenerateAndExport());
    return static_cast<int>(PetscFinalize());
}
