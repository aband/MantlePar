// Run without arguments for all groups, or pass one group name used by CTest.
// Checks deliberately remain active when NDEBUG is defined (Release builds).
#include "eutectic_rescaled.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>

using mantle::phase::DerivedParameters;
using mantle::phase::EutecticModel;
using mantle::phase::MaterialParameters;
using mantle::phase::PhaseRegion;
using mantle::phase::PhaseState;

namespace {

std::size_t checks = 0;
std::size_t states = 0;
std::string context;
constexpr std::array<double, 4> pressures{0.0, 1.0e5, 1.0e9, -1.0e7};

[[noreturn]] void fail(const std::string& message) {
    throw std::runtime_error(message + (context.empty() ? "" : "\n  " + context));
}

void require(bool condition, const std::string& message) {
    ++checks;
    if (!condition) fail(message);
}

void near(double actual, double expected, const char* name,
          double rtol = 3.0e-12, double atol = 3.0e-13) {
    ++checks;
    const double tolerance = atol + rtol * std::max(std::abs(actual), std::abs(expected));
    if (!std::isfinite(actual) || !std::isfinite(expected) ||
        std::abs(actual - expected) > tolerance) {
        std::ostringstream out;
        out << std::setprecision(17) << name << ": got " << actual
            << ", expected " << expected << ", tolerance " << tolerance;
        fail(out.str());
    }
}

template <class Exception, class Function>
void expectThrow(Function function, const char* name) {
    ++checks;
    try {
        function();
    } catch (const Exception&) {
        return;
    } catch (const std::exception& error) {
        fail(std::string(name) + ": wrong exception: " + error.what());
    }
    fail(std::string(name) + ": expected an exception");
}

MaterialParameters alternativeParameters() {
    MaterialParameters p;
    p.Tm0 = 2100.0; p.Te0 = 1500.0; p.cp = 1350.0; p.L = 6.0e5;
    p.Xe = 0.3; p.nu = 8.0e6; p.rho = 3100.0; p.rhor = 450.0;
    p.mus = 3.0e19; p.mul = 2.0; p.k0 = 2.0e-8; p.g = 9.81;
    p.alpha0 = 4.0e-5;
    return p;
}

std::array<MaterialParameters, 3> parameterSets() {
    const auto alternate = alternativeParameters();
    auto negative_slope = alternate;
    negative_slope.nu = -negative_slope.nu;
    return {MaterialParameters{}, alternate, negative_slope};
}

void pointContext(const EutecticModel& model, double H, double C, double P) {
    std::ostringstream out;
    out << std::setprecision(17) << "H=" << H << ", C=" << C << ", P=" << P
        << ", Xe=" << model.parameters().Xe << ", LD=" << model.derived().LD
        << ", nu=" << model.parameters().nu;
    context = out.str();
}

PhaseState checkState(const EutecticModel& model, double H, double C, double P) {
    pointContext(model, H, C, P);
    const auto s = model.evaluate(H, C, P);
    ++states;
    for (const double value : {s.phi1, s.phi2, s.phil, s.cs, s.cl}) {
        require(std::isfinite(value) && value >= 0.0 && value <= 1.0,
                "Fraction/composition must be finite and in [0,1]");
    }
    near(s.phi1 + s.phi2 + s.phil, 1.0, "Total phase fraction");
    near(s.TDp + model.derived().LD * s.phil, H, "Enthalpy conservation");
    // Use relative accuracy even for trace concentrations: a unit-sized
    // absolute tolerance would hide complete loss of a small component.
    near(s.phi2 + s.phil * s.cl, C, "Composition conservation", 3.0e-12, 0.0);
    require(s.has_solid == (s.phi1 + s.phi2 > 0.0), "Solid-presence flag");
    require(s.has_liquid == (s.phil > 0.0), "Liquid-presence flag");
    if (s.has_solid) {
        near(s.cs * (s.phi1 + s.phi2), s.phi2, "Solid composition balance");
    } else {
        near(s.cs, 0.0, "Absent-solid composition", 0.0, 0.0);
    }
    if (!s.has_liquid) near(s.cl, 0.0, "Absent-liquid composition", 0.0, 0.0);
    near(s.Tep, (model.parameters().Te0 + P / model.parameters().nu) /
                    model.derived().dT, "Pressure-corrected eutectic temperature");
    require(s.region == model.classify(H, C, P), "classify/evaluate disagreement");
    require(std::isfinite(s.dTD_dHD) && s.dTD_dHD >= 0.0 && s.dTD_dHD <= 1.0,
            "Enthalpy derivative must be finite and in [0,1]");
    require(!std::isnan(s.dTD_dCD) && s.dTD_dCD <= 0.0, "Composition derivative sign");
    require(s.finite_temperature_derivatives == std::isfinite(s.dTD_dCD),
            "Derivative-finiteness flag");
    return s;
}

void regions() {
    for (const auto& p : parameterSets()) {
        const EutecticModel model(p);
        const auto d = model.derived();
        for (const double P : pressures) {
            const double C = 0.4 * p.Xe;
            const auto b = model.boundaries(C, P);
            auto s = checkState(model, b.Tep - 0.25, C, P);
            require(s.region == PhaseRegion::TwoSolids, "Two-solids region");
            near(s.phi1, 1.0 - C, "Solid 1 fraction");
            near(s.phi2, C, "Solid 2 fraction");
            near(s.phil, 0.0, "No sub-eutectic liquid");

            for (const double liquid : {0.1, 0.2, 0.3}) {
                s = checkState(model, b.Tep + d.LD * liquid, C, P);
                require(s.region == PhaseRegion::Eutectic, "Eutectic region");
                near(s.TDp, b.Tep, "Eutectic plateau");
                near(s.phil, liquid, "Eutectic liquid fraction");
                near(s.phi2, C - liquid * p.Xe, "Eutectic solid 2");
                near(s.cl, p.Xe, "Eutectic liquid composition");
            }

            // Forward construction is independent of the quadratic inversion:
            // choose temperature and liquid fraction, then construct H and C.
            for (const double offset : {0.15, 0.5, 0.85}) {
                for (const double liquid : {0.05, 0.3, 0.8, 0.99}) {
                    const double T = b.Tep + offset;
                    const double cl = p.Xe * (1.0 - offset);
                    s = checkState(model, T + d.LD * liquid, liquid * cl, P);
                    require(s.region == PhaseRegion::SolidLiquid, "Solid-liquid region");
                    near(s.TDp, T, "Recovered liquidus temperature");
                    near(s.phil, liquid, "Recovered liquid fraction");
                    near(s.cl, cl, "Recovered liquid composition");
                    near(s.phi2, 0.0, "Exhausted solid 2");
                }
            }

            s = checkState(model, b.H_liquidus + 0.3, C, P);
            require(s.region == PhaseRegion::Liquid, "Fully liquid region");
            near(s.phil, 1.0, "Fully molten fraction");
            near(s.cl, C, "Fully molten composition");

            const auto pure = model.boundaries(0.0, P);
            s = checkState(model, pure.Tmp - 0.3, 0.0, P);
            require(s.region == PhaseRegion::PureSolid, "Pure-solid region");
            near(s.phi1, 1.0, "Pure solid 1 fraction");
            for (const double liquid : {0.1, 0.5, 0.9}) {
                s = checkState(model, pure.Tmp + d.LD * liquid, 0.0, P);
                require(s.region == PhaseRegion::PureMelting, "Pure-component melting region");
                near(s.phil, liquid, "Pure-component liquid fraction");
                near(s.TDp, pure.Tmp, "Pure-component melting plateau");
            }
            s = checkState(model, pure.H_liquidus + 0.2, 0.0, P);
            require(s.region == PhaseRegion::Liquid, "Pure-component fully liquid region");

            const auto eutectic = model.boundaries(p.Xe, P);
            s = checkState(model, eutectic.Tep + 0.5 * d.LD, p.Xe, P);
            require(s.region == PhaseRegion::Eutectic, "Eutectic-composition melting");
            near(s.phil, 0.5, "Eutectic-composition liquid fraction");
        }
    }
}

void boundaries() {
    const double infinity = std::numeric_limits<double>::infinity();
    for (const auto& p : parameterSets()) {
        const EutecticModel model(p);
        for (const double P : pressures) {
            for (const double C : {0.0, 1.0e-300, 1.0e-18, 0.4 * p.Xe,
                                   std::nextafter(p.Xe, 0.0), p.Xe}) {
                const auto b = model.boundaries(C, P);
                // Include exact boundaries and the immediately adjacent doubles.
                for (const double H : {b.Tep, b.Tmp, b.H_eutectic_end, b.H_liquidus}) {
                    const auto left = checkState(model, std::nextafter(H, -infinity), C, P);
                    const auto at = checkState(model, H, C, P);
                    const auto right = checkState(model, std::nextafter(H, infinity), C, P);
                    near(left.TDp, at.TDp, "Temperature continuity from below");
                    near(right.TDp, at.TDp, "Temperature continuity from above");
                    near(left.phil, at.phil, "Liquid-fraction continuity from below");
                    near(right.phil, at.phil, "Liquid-fraction continuity from above");
                }
                require(model.classify(b.H_liquidus, C, P) == PhaseRegion::Liquid,
                        "Exact liquidus belongs to Liquid");
            }

            // Branch ownership on both sides of non-collapsed interfaces.
            const double C = 0.4 * p.Xe;
            const auto b = model.boundaries(C, P);
            require(model.classify(b.Tep, C, P) == PhaseRegion::TwoSolids, "Solidus ownership");
            require(model.classify(std::nextafter(b.Tep, infinity), C, P) == PhaseRegion::Eutectic,
                    "Immediately above solidus");
            require(model.classify(b.H_eutectic_end, C, P) == PhaseRegion::Eutectic,
                    "Eutectic endpoint ownership");
            require(model.classify(std::nextafter(b.H_eutectic_end, infinity), C, P) ==
                        PhaseRegion::SolidLiquid, "Immediately above eutectic endpoint");
            require(model.classify(std::nextafter(b.H_liquidus, -infinity), C, P) ==
                        PhaseRegion::SolidLiquid, "Immediately below liquidus");

            const auto pure = model.boundaries(0.0, P);
            const auto corner = checkState(model, pure.Tmp, 0.0, P);
            require(corner.region == PhaseRegion::PureSolid, "Pure solidus ownership");
            require(model.classify(std::nextafter(pure.Tmp, infinity), 0.0, P) ==
                        PhaseRegion::PureMelting, "Immediately above pure solidus");
            require(model.classify(std::nextafter(pure.H_liquidus, -infinity), 0.0, P) ==
                        PhaseRegion::PureMelting, "Immediately below pure liquidus");
            const auto eutectic = model.boundaries(p.Xe, P);
            near(eutectic.H_eutectic_end, eutectic.H_liquidus,
                 "Collapsed eutectic/liquidus endpoints", 0.0, 0.0);
            const auto last = checkState(model, eutectic.H_liquidus, p.Xe, P);
            require(last.region == PhaseRegion::Liquid && !last.has_solid,
                    "Fully molten eutectic endpoint");
        }
    }
}

void conservation() {
    std::mt19937_64 random(13579);
    std::uniform_real_distribution<double> unit(0.0, 1.0);
    for (const auto& p : parameterSets()) {
        const EutecticModel model(p);
        const auto d = model.derived();
        for (const double P : pressures) {
            const auto b = model.boundaries(0.4 * p.Xe, P);
            for (int i = 0; i < 10000; ++i) {
                // Alternate ordinary mixtures and concentrations down to ~1e-301.
                const double C = i % 2 == 0 ? p.Xe * unit(random) :
                                            p.Xe * std::pow(10.0, -300.0 * unit(random));
                const double H = b.Tep - 0.5 + (2.0 + d.LD) * unit(random);
                checkState(model, H, C, P);
            }
        }
    }
}

void derivatives() {
    for (const auto& p : parameterSets()) {
        const EutecticModel model(p);
        const auto d = model.derived();
        for (const double P : pressures) {
            const double C = 0.4 * p.Xe;
            const auto b = model.boundaries(C, P);
            for (const double H : {b.Tep - 0.2, (b.Tep + b.H_eutectic_end) / 2.0,
                                   (b.H_eutectic_end + b.H_liquidus) / 2.0,
                                   b.H_liquidus + 0.2}) {
                const auto s = checkState(model, H, C, P);
                const double step = 1.0e-6;
                require(s.finite_temperature_derivatives, "Interior derivatives must be finite");
                // Check that finite differences remain inside the same region.
                require(model.classify(H + step, C, P) == s.region &&
                        model.classify(H - step, C, P) == s.region &&
                        model.classify(H, C + step, P) == s.region &&
                        model.classify(H, C - step, P) == s.region,
                        "Finite-difference stencil crossed a phase boundary");
                near(s.dTD_dHD, (model.evaluate(H + step, C, P).TDp -
                                 model.evaluate(H - step, C, P).TDp) / (2.0 * step),
                     "dT/dH finite difference", 2.0e-8, 2.0e-8);
                near(s.dTD_dCD, (model.evaluate(H, C + step, P).TDp -
                                 model.evaluate(H, C - step, P).TDp) / (2.0 * step),
                     "dT/dC finite difference", 2.0e-8, 2.0e-8);
            }

            const auto pure = model.boundaries(0.0, P);
            for (const double H : {pure.Tep - 0.1, pure.Tmp - 0.4,
                                   pure.Tmp + 0.4 * d.LD, pure.H_liquidus + 0.2}) {
                const auto s = checkState(model, H, 0.0, P);
                const double stepC = 1.0e-8;
                const double stepH = 1.0e-6;
                near(s.dTD_dCD, (model.evaluate(H, stepC, P).TDp - s.TDp) / stepC,
                     "Pure-component one-sided dT/dC", 2.0e-6, 2.0e-6);
                near(s.dTD_dHD, (model.evaluate(H + stepH, 0.0, P).TDp -
                                 model.evaluate(H - stepH, 0.0, P).TDp) / (2.0 * stepH),
                     "Pure-component dT/dH", 2.0e-8, 2.0e-8);
            }
            const auto corner = checkState(model, pure.Tmp, 0.0, P);
            require(!corner.finite_temperature_derivatives &&
                    std::isinf(corner.dTD_dCD) && corner.dTD_dCD < 0.0,
                    "Pure melting corner must report its unbounded composition derivative");
            // At interfaces, test the documented branch convention, not a
            // centered difference across a nondifferentiable transition.
            near(model.evaluate(b.Tep, C, P).dTD_dHD, 1.0, "Solidus branch derivative");
            near(model.evaluate(b.H_eutectic_end, C, P).dTD_dHD, 0.0,
                 "Eutectic endpoint branch derivative");
            near(model.evaluate(b.H_liquidus, C, P).dTD_dHD, 1.0,
                 "Liquidus branch derivative");
        }
    }
}

void checkScales(const EutecticModel& model) {
    const auto& p = model.parameters();
    const auto& d = model.derived();
    near(d.dT, p.Tm0 - p.Te0, "dT");
    near(d.h0, p.cp * (p.Tm0 - p.Te0), "h0");
    near(d.LD, p.L / (p.cp * (p.Tm0 - p.Te0)), "LD");
    near(d.TDm0, p.Tm0 / (p.Tm0 - p.Te0), "TDm0");
    near(d.TDe0, p.Te0 / (p.Tm0 - p.Te0), "TDe0");
    near(d.gamma, 1.0 / p.nu, "gamma", 3.0e-12, 0.0);
    near(d.invk0, 1.0 / p.k0, "invk0");
    near(d.l0, std::sqrt(p.mus * p.k0 / p.mul), "l0");
    near(d.u0, p.k0 * p.rhor * p.g / p.mul, "u0", 3.0e-12, 0.0);
    near(d.p0, p.rhor * p.g * d.l0, "p0");
    near(d.t0, d.l0 / d.u0, "t0");
}

void scales() {
    const EutecticModel defaults;
    near(defaults.derived().dT, 520.0, "Default temperature scale");
    near(defaults.derived().h0, 624000.0, "Default enthalpy scale");
    near(defaults.derived().TDm0, 50.0 / 13.0, "Default melting temperature");
    near(defaults.derived().TDe0, 37.0 / 13.0, "Default eutectic temperature");
    near(defaults.derived().LD, 125.0 / 156.0, "Default latent heat");
    for (const auto& p : parameterSets()) {
        const EutecticModel model(p);
        context = "Reference scales and dimensional pressure";
        checkScales(model);
        near(model.staticPressure(0.0), 0.0, "Surface static pressure");
        near(model.staticPressure(0.5), p.rho * p.g * 0.5 * model.derived().l0,
             "Static pressure with internal length scale");
        near(model.staticPressure(0.5, 1000.0), p.rho * p.g * 500.0,
             "Static pressure with explicit length scale");
        const double C = 0.4 * p.Xe;
        const auto b = model.boundaries(C, 0.0);
        const double H = (b.H_eutectic_end + b.H_liquidus) / 2.0;
        const auto base = checkState(model, H, C, 0.0);
        for (const double P : pressures) {
            const double shift = P / p.nu / model.derived().dT;
            near(model.pressureCorrectedTemperature(2.0, P), 2.0 + shift,
                 "Pressure input is dimensional Pa");
            const auto shifted = checkState(model, H + shift, C, P);
            near(shifted.TDp, base.TDp + shift, "Pressure translation of temperature");
            near(shifted.phil, base.phil, "Pressure translation preserves liquid fraction");
            near(shifted.cl, base.cl, "Pressure translation preserves composition");
            near(shifted.dTD_dHD, base.dTD_dHD, "Pressure translation preserves dT/dH");
            near(shifted.dTD_dCD, base.dTD_dCD, "Pressure translation preserves dT/dC");
        }
    }
}

constexpr std::array<double MaterialParameters::*, 13> materialMembers{
    &MaterialParameters::Tm0, &MaterialParameters::Te0, &MaterialParameters::nu,
    &MaterialParameters::L, &MaterialParameters::cp, &MaterialParameters::Xe,
    &MaterialParameters::rho, &MaterialParameters::rhor, &MaterialParameters::mus,
    &MaterialParameters::mul, &MaterialParameters::k0, &MaterialParameters::g,
    &MaterialParameters::alpha0
};

void sameConfiguration(const EutecticModel& model, const MaterialParameters& p,
                       const DerivedParameters& d) {
    for (const auto member : materialMembers) {
        near(model.parameters().*member, p.*member,
             "Parameter unchanged after rejected update", 0.0, 0.0);
    }
    for (double DerivedParameters::*member : {&DerivedParameters::dT, &DerivedParameters::h0,
            &DerivedParameters::gamma, &DerivedParameters::LD, &DerivedParameters::TDm0,
            &DerivedParameters::TDe0, &DerivedParameters::invk0, &DerivedParameters::l0,
            &DerivedParameters::u0, &DerivedParameters::p0, &DerivedParameters::t0}) {
        near(model.derived().*member, d.*member, "Derived quantity unchanged after rejected update",
             0.0, 0.0);
    }
}

void parameterUpdates() {
    EutecticModel model;
    const auto original = model.parameters();
    const auto old = model.derived();
    const double dimensional_enthalpy = 4.0e6; // Fully molten for both parameter sets.
    const double H_old = dimensional_enthalpy / old.h0;
    const auto before = checkState(model, H_old, 0.1, 0.0);
    auto p = model.parameters();
    p = alternativeParameters();
    near(model.parameters().L, original.L, "Editing a parameter copy leaves the model unchanged");
    model.setParameters(p);
    checkScales(model);
    const double H_new = H_old * old.h0 / model.derived().h0;
    const auto after = checkState(model, H_new, 0.1, 0.0);
    require(after.region == PhaseRegion::Liquid, "Updated fully molten state");
    near(after.TDp * model.derived().dT, (dimensional_enthalpy - p.L) / p.cp,
         "Updated temperature at fixed dimensional enthalpy");
    near(p.cp * model.derived().dT * after.TDp + p.L * after.phil,
         dimensional_enthalpy, "Dimensional enthalpy after runtime rescaling");
    near(before.TDp * old.dT, (dimensional_enthalpy - original.L) / original.cp,
         "Previously returned state retains its original values");

    // Change latent heat alone: its new value must be used immediately.
    const double unchanged_h0 = model.derived().h0;
    p.L *= 1.7;
    model.setParameters(p);
    near(model.derived().h0, unchanged_h0, "Changing L preserves enthalpy scale", 0.0, 0.0);
    checkScales(model);
    const auto pure = model.boundaries(0.0, 0.0);
    const auto half = checkState(model, pure.Tmp + 0.5 * model.derived().LD, 0.0, 0.0);
    near(half.phil, 0.5, "Updated latent heat controls pure melting");

    // Composition bounds and the Clapeyron slope must also update immediately.
    p.Xe = 0.2;
    p.nu = -p.nu;
    model.setParameters(p);
    expectThrow<std::invalid_argument>([&] { model.evaluate(4.0, 0.25, 0.0); }, "Updated Xe bound");
    checkScales(model);
    require(model.pressureCorrectedTemperature(3.0, 1.0e8) < 3.0,
            "Updated negative pressure slope");

    // A failing update must preserve every input and derived parameter.
    const auto saved = model.derived();
    for (double MaterialParameters::*member : {&MaterialParameters::L, &MaterialParameters::cp,
            &MaterialParameters::rho, &MaterialParameters::rhor, &MaterialParameters::mus,
            &MaterialParameters::mul, &MaterialParameters::k0, &MaterialParameters::g,
            &MaterialParameters::nu, &MaterialParameters::Xe}) {
        context = "Rejected zero-valued material parameter update";
        auto bad = p;
        bad.*member = 0.0;
        expectThrow<std::invalid_argument>([&] { model.setParameters(bad); }, "Invalid update");
        sameConfiguration(model, p, saved);
    }
    auto bad = p;
    bad.cp = std::numeric_limits<double>::max();
    expectThrow<std::overflow_error>([&] { model.setParameters(bad); }, "Overflowing scale update");
    sameConfiguration(model, p, saved);
    checkState(model, 4.0, 0.1, 0.0);
}

void invalidInputs() {
    const EutecticModel model;
    const double infinity = std::numeric_limits<double>::infinity();
    const double nan = std::numeric_limits<double>::quiet_NaN();
    context = "Invalid phase inputs";
    for (const double C : {-0.1, std::nextafter(0.0, -infinity),
                           std::nextafter(model.parameters().Xe, infinity), infinity, -infinity, nan}) {
        expectThrow<std::invalid_argument>([&] { model.evaluate(4.0, C, 0.0); }, "evaluate CD");
        expectThrow<std::invalid_argument>([&] { model.classify(4.0, C, 0.0); }, "classify CD");
        expectThrow<std::invalid_argument>([&] { model.boundaries(C, 0.0); }, "boundaries CD");
    }
    for (const double value : {infinity, -infinity, nan}) {
        expectThrow<std::invalid_argument>([&] { model.evaluate(value, 0.1, 0.0); }, "evaluate HD");
        expectThrow<std::invalid_argument>([&] { model.classify(value, 0.1, 0.0); }, "classify HD");
        expectThrow<std::invalid_argument>([&] { model.evaluate(4.0, 0.1, value); }, "evaluate P");
        expectThrow<std::invalid_argument>([&] { model.boundaries(0.1, value); }, "boundaries P");
        expectThrow<std::invalid_argument>([&] { model.pressureCorrectedTemperature(value, 0.0); }, "TD");
        expectThrow<std::invalid_argument>([&] { model.staticPressure(value); }, "Depth");
    }
    for (const double length : {0.0, -1.0, infinity, nan}) {
        expectThrow<std::invalid_argument>([&] { model.staticPressure(0.5, length); }, "Length scale");
    }
    expectThrow<std::overflow_error>([&] {
        model.boundaries(0.1, std::numeric_limits<double>::max());
    }, "Unresolvable pressure-shifted intervals");

    context = "Invalid material parameters";
    for (const auto member : materialMembers) {
        for (const double value : {nan, infinity, -infinity}) {
            MaterialParameters bad;
            bad.*member = value;
            expectThrow<std::invalid_argument>([&] { const EutecticModel invalid(bad); },
                                               "Nonfinite material parameter");
        }
    }
    for (double MaterialParameters::*member : {&MaterialParameters::Tm0, &MaterialParameters::Te0,
            &MaterialParameters::cp, &MaterialParameters::L, &MaterialParameters::rho,
            &MaterialParameters::rhor, &MaterialParameters::mus, &MaterialParameters::mul,
            &MaterialParameters::k0, &MaterialParameters::g}) {
        for (const double value : {0.0, -1.0}) {
            MaterialParameters bad;
            bad.*member = value;
            expectThrow<std::invalid_argument>([&] { const EutecticModel invalid(bad); },
                                               "Nonpositive material parameter");
        }
    }
    for (const double Xe : {0.0, -0.1, 1.0, 1.1}) {
        MaterialParameters bad;
        bad.Xe = Xe;
        expectThrow<std::invalid_argument>([&] { const EutecticModel invalid(bad); }, "Xe interval");
    }
    MaterialParameters bad;
    bad.Te0 = bad.Tm0;
    expectThrow<std::invalid_argument>([&] { const EutecticModel invalid(bad); }, "Zero temperature gap");
    bad.Te0 += 1.0;
    expectThrow<std::invalid_argument>([&] { const EutecticModel invalid(bad); }, "Reversed temperatures");
    bad = MaterialParameters{}; bad.nu = 0.0;
    expectThrow<std::invalid_argument>([&] { const EutecticModel invalid(bad); }, "Zero Clapeyron constant");
}

void sameState(const PhaseState& actual, const PhaseState& expected) {
    for (double PhaseState::*member : {&PhaseState::phi1, &PhaseState::phi2, &PhaseState::phil,
            &PhaseState::cl, &PhaseState::cs, &PhaseState::TDp, &PhaseState::Tep,
            &PhaseState::dTD_dHD, &PhaseState::dTD_dCD}) {
        require(actual.*member == expected.*member, "A returned state changed or copied incompletely");
    }
    require(actual.region == expected.region && actual.has_solid == expected.has_solid &&
            actual.has_liquid == expected.has_liquid &&
            actual.finite_temperature_derivatives == expected.finite_temperature_derivatives,
            "State metadata changed or copied incompletely");
}

void stateValues() {
    EutecticModel model;
    const auto b = model.boundaries(0.1, 1.0e8);
    const double H = (b.H_eutectic_end + b.H_liquidus) / 2.0;
    const auto first = checkState(model, H, 0.1, 1.0e8);
    const auto snapshot = first;
    checkState(model, b.H_liquidus + 0.3, 0.1, 1.0e8);
    checkState(model, b.Tep - 0.2, 0.1, 1.0e8);
    expectThrow<std::invalid_argument>([&] { model.evaluate(H, -0.1, 1.0e8); }, "Failed evaluation");
    sameState(first, snapshot);
    sameState(model.evaluate(H, 0.1, 1.0e8), snapshot);
    model.setParameters(alternativeParameters());
    sameState(first, snapshot);
    auto copy = first;
    copy.Tep += 1.0;
    near(first.Tep, snapshot.Tep, "Returned states are independent values", 0.0, 0.0);
    require(copy.Tep != first.Tep, "State copy must be independently mutable");
    const EutecticModel constant_model;
    checkState(constant_model, 4.0, 0.1, 0.0);
}

struct TestCase {
    const char* name;
    void (*run)();
};

const std::array<TestCase, 8> testCases{{
    {"regions", regions},
    {"boundaries", boundaries},
    {"conservation", conservation},
    {"derivatives", derivatives},
    {"scales", scales},
    {"parameter_updates", parameterUpdates},
    {"invalid_inputs", invalidInputs},
    {"state", stateValues}
}};

} // namespace

int main(int argc, char* argv[]) {
    const std::string requested = argc == 1 ? "all" : argv[1];
    if (argc > 2) {
        std::cerr << "Usage: test_eutectic [all|group]\n";
        return 2;
    }
    bool matched = false;
    int failures = 0;
    for (const auto& test : testCases) {
        if (requested != "all" && requested != test.name) continue;
        matched = true;
        checks = 0;
        states = 0;
        context.clear();
        try {
            test.run();
            std::cout << "PASS " << test.name << ": " << checks << " checks, "
                      << states << " equilibrium states\n";
        } catch (const std::exception& error) {
            ++failures;
            std::cerr << "FAIL " << test.name << ": " << error.what() << '\n';
        }
    }
    if (!matched) {
        std::cerr << "Unknown test group: " << requested << "\nAvailable: all";
        for (const auto& test : testCases) std::cerr << ' ' << test.name;
        std::cerr << '\n';
        return 2;
    }
    return failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
