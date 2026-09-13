#include "math_utils.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace mantle::math {
namespace {
void CheckFinite(double value)
{
    if (!std::isfinite(value)) throw std::invalid_argument("Expected a finite number");
}
double CheckedResult(double value)
{
    if (!std::isfinite(value)) throw std::overflow_error("Mathematical result exceeds double range");
    return value;
}
void CheckCoefficients(const std::vector<double>& coefficients)
{ for (double coefficient : coefficients) CheckFinite(coefficient); }
double IntegerPower(double x, unsigned exponent)
{
    double result = 1;
    while (exponent) {
        if (exponent & 1U) result *= x;
        exponent >>= 1U;
        if (exponent) x *= x;
    }
    return CheckedResult(result);
}
} // namespace

std::uint64_t FactorialRatio(int top, int bottom)
{
    if (bottom < 0 || top < bottom)
        throw std::invalid_argument("Require 0 <= bottom <= top");
    std::uint64_t result = 1;
    for (int factor = bottom; factor < top;) {
        ++factor;
        if (result > std::numeric_limits<std::uint64_t>::max() / static_cast<unsigned>(factor))
            throw std::overflow_error("Factorial ratio exceeds uint64_t");
        result *= static_cast<unsigned>(factor);
    }
    return result;
}

std::uint64_t Factorial(int n) { return FactorialRatio(n, 0); }

double Monomial(double x, double y, int xPower, int yPower)
{
    CheckFinite(x);
    CheckFinite(y);
    if (xPower < 0 || yPower < 0) throw std::invalid_argument("Powers must be nonnegative");
    if ((x == 0 && xPower > 0) || (y == 0 && yPower > 0)) return 0;
    return CheckedResult(IntegerPower(x, static_cast<unsigned>(xPower))
                         * IntegerPower(y, static_cast<unsigned>(yPower)));
}

double EvaluatePolynomial(double x, const std::vector<double>& coefficients)
{
    CheckFinite(x);
    CheckCoefficients(coefficients);
    double result = 0;
    for (auto it = coefficients.rbegin(); it != coefficients.rend(); ++it)
        result = std::fma(result, x, *it);
    return CheckedResult(result);
}

std::vector<double> DifferentiatePolynomial(const std::vector<double>& coefficients, int order)
{
    if (order < 0) throw std::invalid_argument("Derivative order must be nonnegative");
    CheckCoefficients(coefficients);
    if (coefficients.empty() || static_cast<std::size_t>(order) >= coefficients.size()) return {0};
    const auto derivative = static_cast<std::size_t>(order);
    std::vector<double> result(coefficients.size() - derivative);
    for (std::size_t k = derivative; k < coefficients.size(); ++k) {
        double coefficient = coefficients[k];
        for (std::size_t d = 0; d < derivative; ++d)
            coefficient = CheckedResult(coefficient * static_cast<double>(k-d));
        result[k-derivative] = coefficient;
    }
    return result;
}

double HarmonicMean(double a, double b)
{
    CheckFinite(a);
    CheckFinite(b);
    if (a < 0 || b < 0) throw std::invalid_argument("Harmonic mean requires nonnegative inputs");
    if (a == 0 || b == 0) return 0;
    const double low = std::min(a,b), high = std::max(a,b);
    return CheckedResult(low / (0.5 + 0.5 * (low / high)));
}

} // namespace mantle::math
