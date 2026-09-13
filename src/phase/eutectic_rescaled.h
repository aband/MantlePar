#ifndef MANTLE_PHASE_EUTECTIC_RESCALED_H
#define MANTLE_PHASE_EUTECTIC_RESCALED_H

#include <iosfwd>

// Standalone C++17: no PETSc, MPI, or mesh dependency.
namespace mantle::phase {

// Dimensional material inputs. Defaults preserve the original model.
struct MaterialParameters {
    double Tm0 = 2000.0;  // Pure-component melting temperature at P=0 [K].
    double Te0 = 1480.0;  // Eutectic temperature at P=0 [K].
    double nu = 6.5e6;    // dP/dT [Pa/K], finite and nonzero.
    double L = 5.0e5;     // Latent heat [J/kg].
    double cp = 1200.0;   // Specific heat capacity [J/(kg K)].
    double Xe = 0.25;     // Eutectic liquid composition, strictly between 0 and 1.

    // Properties used by the reference scales and lithostatic pressure.
    double rho = 3000.0;  // Reference density [kg/m^3].
    double rhor = 500.0;  // Positive reference density contrast [kg/m^3].
    double mus = 1.0e19;  // Solid viscosity [Pa s].
    double mul = 1.0;     // Liquid viscosity [Pa s].
    double k0 = 1.0e-8;   // Reference permeability [m^2].
    double g = 10.0;      // Positive gravity magnitude [m/s^2].
    double alpha0 = 3.0e-5; // Thermal expansion coefficient [1/K].
};

// Recomputed together whenever setParameters() succeeds; read-only to callers.
struct DerivedParameters {
    double dT = 0.0;      // Tm0 - Te0 [K].
    double h0 = 0.0;      // cp*dT, enthalpy scale [J/kg].
    double gamma = 0.0;   // 1/nu [K/Pa].
    double LD = 0.0;      // L/h0.
    double TDm0 = 0.0;    // Tm0/dT; no temperature offset is subtracted.
    double TDe0 = 0.0;    // Te0/dT.
    double invk0 = 0.0;   // 1/k0 [1/m^2].
    double l0 = 0.0;      // sqrt(mus*k0/mul) [m].
    double u0 = 0.0;      // k0*rhor*g/mul [m/s].
    double p0 = 0.0;      // rhor*g*l0 [Pa].
    double t0 = 0.0;      // l0/u0 [s].
};

enum class PhaseRegion {
    PureSolid = 1,
    TwoSolids = 2,
    Eutectic = 3,         // Two solids plus liquid, at the eutectic temperature.
    SolidLiquid = 4,      // Solid 1 plus liquid, above the eutectic temperature.
    Liquid = 5,
    PureMelting = 6       // C=0, isothermal melting of component 1.
};

struct PhaseBoundaries {
    double Tep = 0.0;     // Pressure-corrected eutectic temperature, dimensionless.
    double Tmp = 0.0;     // Pressure-corrected pure melting temperature.
    double H_eutectic_end = 0.0; // Tep + LD*(C/Xe); unused when C=0.
    double H_liquidus = 0.0;    // Tmp - C/Xe + LD.
};

struct PhaseState {
    double phi1 = 0.0;
    double phi2 = 0.0;
    double phil = 0.0;
    double cl = 0.0;
    double cs = 0.0;
    PhaseRegion region = PhaseRegion::PureSolid;
    double TDp = 0.0;
    double Tep = 0.0;
    double dTD_dCD = 0.0;
    double dTD_dHD = 0.0;
    bool has_solid = false;
    bool has_liquid = false;
    bool finite_temperature_derivatives = true;

    // When a phase is absent, its composition is a zero placeholder:
    // consult has_solid/has_liquid before using cs/cl as phase properties.
};

// Binary eutectic equilibrium on the component-1-rich side: 0 <= CD <= Xe.
//
// Units and conserved quantities:
//   TD = T/dT, HD = h/h0, P in Pa (not P/p0), CD is NOT divided by Xe.
//   phi1 + phi2 + phil = 1; HD = TD + LD*phil; CD = phi2 + phil*cl.
// The fraction/composition balance retains the original equal-density mixing
// approximation. rho and rhor do not introduce density weighting into it.
//
// Runtime updates:
//   auto p = model.parameters();
//   p.L = 6.0e5;
//   model.setParameters(p);
// Input parameters and all derived values are committed only after validation.
// A rejected update leaves the model unchanged. Updates do not modify previously
// returned PhaseState values or any external solution fields.
//
// If h0 changes, preserving a stored dimensional enthalpy requires
//   HD_new = HD_old * h0_old / h0_new.
// Similarly, changes to dT, l0, u0, p0, or t0 require the caller to convert
// affected dimensionless fields if their dimensional values are to be retained.
// What is conserved during a material change is the solver's responsibility.
//
// evaluate() is const and returns a fresh value. Concurrent evaluations are safe
// provided no thread calls setParameters() concurrently. MPI callers update the
// model on each participating rank; the model itself performs no communication.
class EutecticModel {
public:
    explicit EutecticModel(const MaterialParameters& parameters = {});

    const MaterialParameters& parameters() const noexcept;
    const DerivedParameters& derived() const noexcept;
    void setParameters(const MaterialParameters& parameters);

    // Invalid/nonfinite inputs throw std::invalid_argument. Inputs outside the
    // composition interval are rejected, not clipped. Unrepresentable computed
    // scales/temperatures throw std::overflow_error.
    PhaseState evaluate(double HD, double CD, double P) const;
    PhaseRegion classify(double HD, double CD, double P) const;
    PhaseBoundaries boundaries(double CD, double P) const;

    // Boundary convention: at the solidus choose the solid branch; at the end
    // of eutectic melting choose Eutectic unless already fully liquid; at the
    // liquidus choose Liquid. At C=0, H=Tmp belongs to PureSolid.
    // Derivatives at these interfaces use the selected branch and need not be
    // two-sided derivatives. On C=0 below the liquidus, dTD/dCD is the limit
    // from C>0. At (C=0,H=Tmp) this limit is -infinity, explicitly reported with
    // finite_temperature_derivatives=false. Do not use it in a Jacobian without
    // a solver-level choice of regularization. Very large derivatives can also
    // exceed the range of double and set this flag to false.

    double pressureCorrectedTemperature(double TD, double P) const;
    double staticPressure(double zD) const; // zD is depth/l0, positive downward.
    double staticPressure(double zD, double length_scale) const;
    void printInfo(std::ostream& out) const;

private:
    MaterialParameters parameters_{};
    DerivedParameters derived_{};
};

} // namespace mantle::phase

#endif

