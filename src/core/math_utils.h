#ifndef MANTLE_CORE_MATH_UTILS_H
#define MANTLE_CORE_MATH_UTILS_H

#include <cstdint>
#include <vector>

// Pure C++ helpers: no PETSc initialization or MPI dependency.
// Invalid arguments throw std::invalid_argument; unrepresentable results throw
// std::overflow_error (including overflowing intermediate values). Floating-point
// underflow follows normal double arithmetic.
namespace mantle::math {

std::uint64_t Factorial(int n);
// top!/bottom!, including 1 when top==bottom. Both arguments must be nonnegative.
std::uint64_t FactorialRatio(int top, int bottom);

// Nonnegative integer powers; x^0 and y^0 are 1, including at zero.
double Monomial(double x, double y, int xPower, int yPower);

// Coefficients are [constant, x, x^2, ...]. An empty polynomial represents zero.
double EvaluatePolynomial(double x, const std::vector<double>& coefficients);
// Includes the highest-degree term; orders above the degree return {0.0}.
std::vector<double> DifferentiatePolynomial(const std::vector<double>& coefficients,
                                           int order = 1);

// Finite nonnegative physical coefficients. Zero on either side gives zero.
// Avoids the intermediate overflow in 2*a*b/(a+b).
double HarmonicMean(double a, double b);

} // namespace mantle::math

#endif
