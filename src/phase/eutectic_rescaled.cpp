#include "eutectic_rescaled.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <ostream>
#include <stdexcept>
#include <string>

namespace mantle::phase {
namespace {

void requireFinite(double value, const char* name) {
    if (!std::isfinite(value)) {
        throw std::invalid_argument(std::string(name) + " must be finite");
    }
}

void requirePositive(double value, const char* name) {
    requireFinite(value, name);
    if (value <= 0.0) {
        throw std::invalid_argument(std::string(name) + " must be positive");
    }
}

double checkedDouble(long double value, const char* name) {
    const long double limit = std::numeric_limits<double>::max();
    if (!std::isfinite(value) || std::abs(value) > limit) {
        throw std::overflow_error(std::string(name) + " is not representable");
    }
    return static_cast<double>(value);
}

double positiveScale(long double value, const char* name) {
    const double result = checkedDouble(value, name);
    if (result <= 0.0) {
        throw std::overflow_error(std::string(name) + " is not a positive scale");
    }
    return result;
}

DerivedParameters derive(const MaterialParameters& p) {
    requirePositive(p.Tm0, "Tm0");
    requirePositive(p.Te0, "Te0");
    if (p.Tm0 <= p.Te0) {
        throw std::invalid_argument("Tm0 must be greater than Te0");
    }
    requireFinite(p.nu, "nu");
    if (p.nu == 0.0) {
        throw std::invalid_argument("nu must be nonzero");
    }
    requirePositive(p.L, "L");
    requirePositive(p.cp, "cp");
    requireFinite(p.Xe, "Xe");
    if (p.Xe <= 0.0 || p.Xe >= 1.0) {
        throw std::invalid_argument("Xe must lie strictly between 0 and 1");
    }
    requirePositive(p.rho, "rho");
    requirePositive(p.rhor, "rhor");
    requirePositive(p.mus, "mus");
    requirePositive(p.mul, "mul");
    requirePositive(p.k0, "k0");
    requirePositive(p.g, "g");
    requireFinite(p.alpha0, "alpha0");

    DerivedParameters d;
    const long double deltaT = static_cast<long double>(p.Tm0) - p.Te0;
    const long double h0 = static_cast<long double>(p.cp) * deltaT;
    d.dT = positiveScale(deltaT, "dT");
    d.h0 = positiveScale(h0, "h0");
    d.gamma = checkedDouble(1.0L / p.nu, "gamma");
    if (d.gamma == 0.0) {
        throw std::overflow_error("gamma underflows to zero");
    }
    d.LD = positiveScale(p.L / h0, "LD");
    d.TDm0 = positiveScale(p.Tm0 / deltaT, "TDm0");
    d.TDe0 = positiveScale(p.Te0 / deltaT, "TDe0");
    d.invk0 = positiveScale(1.0L / p.k0, "invk0");
    const long double l0 = std::sqrt(static_cast<long double>(p.mus) * p.k0 / p.mul);
    const long double u0 = static_cast<long double>(p.k0) * p.rhor * p.g / p.mul;
    d.l0 = positiveScale(l0, "l0");
    d.u0 = positiveScale(u0, "u0");
    d.p0 = positiveScale(static_cast<long double>(p.rhor) * p.g * l0, "p0");
    d.t0 = positiveScale(l0 / u0, "t0");
    return d;
}

void validateComposition(double CD, double Xe) {
    requireFinite(CD, "CD");
    if (CD < 0.0 || CD > Xe) {
        throw std::invalid_argument("CD must lie in [0, Xe]");
    }
}

PhaseRegion selectRegion(double HD, double CD, const PhaseBoundaries& b) {
    if (CD == 0.0) {
        if (HD <= b.Tmp) return PhaseRegion::PureSolid;
        if (HD < b.H_liquidus) return PhaseRegion::PureMelting;
        return PhaseRegion::Liquid;
    }
    if (HD <= b.Tep) return PhaseRegion::TwoSolids;
    if (HD >= b.H_liquidus) return PhaseRegion::Liquid;
    if (HD <= b.H_eutectic_end) return PhaseRegion::Eutectic;
    return PhaseRegion::SolidLiquid;
}

// Correct only floating-point excursions, never a substantial violation.
double unitFraction(long double value) {
    constexpr long double tolerance = 64.0L * std::numeric_limits<double>::epsilon();
    if (!std::isfinite(value) || value < -tolerance || value > 1.0L + tolerance) {
        throw std::runtime_error("Computed phase fraction is outside [0, 1]");
    }
    return static_cast<double>(std::clamp(value, 0.0L, 1.0L));
}

double negativeDerivative(long double magnitude) {
    if (magnitude > std::numeric_limits<double>::max()) {
        return -std::numeric_limits<double>::infinity();
    }
    return -static_cast<double>(magnitude);
}

} // namespace

EutecticModel::EutecticModel(const MaterialParameters& parameters) {
    setParameters(parameters);
}

const MaterialParameters& EutecticModel::parameters() const noexcept {
    return parameters_;
}

const DerivedParameters& EutecticModel::derived() const noexcept {
    return derived_;
}

void EutecticModel::setParameters(const MaterialParameters& parameters) {
    const DerivedParameters next = derive(parameters);
    parameters_ = parameters;
    derived_ = next;
}

double EutecticModel::pressureCorrectedTemperature(double TD, double P) const {
    requireFinite(TD, "TD");
    requireFinite(P, "P");
    const long double shift = static_cast<long double>(P) / parameters_.nu / derived_.dT;
    return checkedDouble(static_cast<long double>(TD) + shift,
                         "pressure-corrected temperature");
}

double EutecticModel::staticPressure(double zD) const {
    return staticPressure(zD, derived_.l0);
}

double EutecticModel::staticPressure(double zD, double length_scale) const {
    requireFinite(zD, "zD");
    requirePositive(length_scale, "length_scale");
    return checkedDouble(static_cast<long double>(parameters_.rho) * parameters_.g *
                             zD * length_scale,
                         "static pressure");
}

PhaseBoundaries EutecticModel::boundaries(double CD, double P) const {
    validateComposition(CD, parameters_.Xe);
    PhaseBoundaries b;
    b.Tep = pressureCorrectedTemperature(derived_.TDe0, P);
    // Both temperatures share the same pressure shift and differ by one in
    // this nondimensionalization. Use a common origin for the boundary lines.
    b.Tmp = checkedDouble(static_cast<long double>(b.Tep) + 1.0L, "Tmp");
    const long double ratio = static_cast<long double>(CD) / parameters_.Xe;
    b.H_eutectic_end = checkedDouble(static_cast<long double>(b.Tep) + derived_.LD * ratio,
                                     "H_eutectic_end");
    b.H_liquidus = checkedDouble(static_cast<long double>(b.Tep) + (1.0L - ratio) +
                                    derived_.LD,
                                "H_liquidus");
    if (CD == 0.0) {
        b.H_liquidus = checkedDouble(static_cast<long double>(b.Tmp) + derived_.LD,
                                    "pure-component liquidus");
    }
    if (b.Tmp <= b.Tep || b.H_liquidus <= b.Tep ||
        (CD == 0.0 && b.H_liquidus <= b.Tmp)) {
        throw std::overflow_error("Phase temperature/enthalpy intervals are not resolvable");
    }
    return b;
}

PhaseRegion EutecticModel::classify(double HD, double CD, double P) const {
    requireFinite(HD, "HD");
    return selectRegion(HD, CD, boundaries(CD, P));
}

PhaseState EutecticModel::evaluate(double HD, double CD, double P) const {
    requireFinite(HD, "HD");
    const PhaseBoundaries b = boundaries(CD, P);
    const long double latent = derived_.LD;
    const long double Xe = parameters_.Xe;
    PhaseState s;
    s.region = selectRegion(HD, CD, b);
    s.Tep = b.Tep;

    switch (s.region) {
    case PhaseRegion::PureSolid:
        s.phi1 = 1.0;
        s.TDp = HD;
        s.dTD_dHD = 1.0;
        break;

    case PhaseRegion::PureMelting:
        s.phil = unitFraction((static_cast<long double>(HD) - b.Tmp) / latent);
        s.phi1 = 1.0 - s.phil;
        s.TDp = b.Tmp;
        break;

    case PhaseRegion::TwoSolids:
        s.phi1 = 1.0 - CD;
        s.phi2 = CD;
        s.TDp = HD;
        s.dTD_dHD = 1.0;
        break;

    case PhaseRegion::Eutectic: {
        const long double maximum_liquid = static_cast<long double>(CD) / Xe;
        // The public boundary is rounded to double. At that boundary the
        // second solid is exhausted exactly; immediately below, cap only the
        // possible rounding overshoot in the enthalpy-derived liquid fraction.
        const long double liquid = HD == b.H_eutectic_end ? maximum_liquid :
            std::min(maximum_liquid, (static_cast<long double>(HD) - b.Tep) / latent);
        const long double solid2 = std::max(0.0L, static_cast<long double>(CD) - liquid * Xe);
        s.phil = unitFraction(liquid);
        s.phi2 = unitFraction(solid2);
        s.phi1 = unitFraction(1.0L - liquid - solid2);
        s.cl = parameters_.Xe;
        s.TDp = b.Tep;
        break;
    }

    case PhaseRegion::SolidLiquid: {
        // Solve LD*phi_l^2 - (HD-Tmp)*phi_l - CD/Xe = 0.
        // Rationalize the small root when HD<Tmp to avoid cancellation.
        const long double a = static_cast<long double>(HD) - b.Tmp;
        const long double ratio = static_cast<long double>(CD) / Xe;
        const long double root = std::hypot(a, 2.0L * std::sqrt(latent) * std::sqrt(ratio));
        const long double liquid = a >= 0.0L ?
            (a + root) / (2.0L * latent) : 2.0L * ratio / (root - a);
        s.phil = unitFraction(liquid);
        s.phi1 = unitFraction(1.0L - liquid);
        s.cl = unitFraction(static_cast<long double>(CD) / liquid);
        // Choose the temperature expression that avoids subtracting nearly
        // equal large enthalpy and latent-heat contributions.
        s.TDp = checkedDouble(a >= 0.0L ? static_cast<long double>(b.Tmp) - ratio / liquid :
                                         static_cast<long double>(HD) - latent * liquid,
                              "temperature");
        s.dTD_dHD = unitFraction(a >= 0.0L ?
            2.0L * (latent / root) * (ratio / (root + a)) : 0.5L * (1.0L - a / root));
        s.dTD_dCD = negativeDerivative((latent / root) / Xe);
        break;
    }

    case PhaseRegion::Liquid:
        s.phil = 1.0;
        s.cl = CD;
        s.TDp = checkedDouble(static_cast<long double>(HD) - latent, "temperature");
        s.dTD_dHD = 1.0;
        break;
    }

    if (CD == 0.0 && HD > b.Tep && s.region != PhaseRegion::Liquid) {
        // Composition derivatives at the pure-component edge are taken from
        // C>0, where an arbitrarily small admixture enters the liquidus branch.
        const long double gap = std::abs(static_cast<long double>(HD) - b.Tmp);
        s.dTD_dCD = gap == 0.0L ? -std::numeric_limits<double>::infinity() :
                                   negativeDerivative((latent / gap) / Xe);
    }

    const double solid = s.phi1 + s.phi2;
    s.has_solid = solid > 0.0;
    s.has_liquid = s.phil > 0.0;
    s.cs = s.has_solid ? s.phi2 / solid : 0.0;
    if (!s.has_liquid) s.cl = 0.0;
    s.finite_temperature_derivatives = std::isfinite(s.dTD_dCD) && std::isfinite(s.dTD_dHD);
    return s;
}

void EutecticModel::printInfo(std::ostream& out) const {
    out << "Dimensionless melting temperature: " << derived_.TDm0 << '\n'
        << "Dimensionless eutectic temperature: " << derived_.TDe0 << '\n'
        << "Dimensionless latent heat: " << derived_.LD << '\n'
        << "Temperature scale [K]: " << derived_.dT << '\n'
        << "Enthalpy scale [J/kg]: " << derived_.h0 << '\n'
        << "Length scale [m]: " << derived_.l0 << '\n'
        << "Velocity scale [m/s]: " << derived_.u0 << '\n'
        << "Pressure scale [Pa]: " << derived_.p0 << '\n'
        << "Time scale [s]: " << derived_.t0 << '\n';
}

} // namespace mantle::phase
