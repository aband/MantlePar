#include "math_utils.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>

namespace {
void Require(bool passed, const char* message)
{ if (!passed) throw std::runtime_error(message); }
void Near(double actual, double exact)
{
    Require(std::isfinite(actual), "Non-finite result");
    const double scale = std::max(std::abs(actual),std::abs(exact));
    Require(std::abs(actual-exact) <= 32*std::numeric_limits<double>::epsilon()*scale,
            "Incorrect numerical result");
}
template<class Error, class Call>
void Throws(Call call)
{
    bool caught = false;
    try { call(); } catch (const Error&) { caught = true; }
    Require(caught, "Expected exception was not thrown");
}
void Run()
{
    using namespace mantle::math;
    Require(Factorial(0)==1 && Factorial(1)==1 && Factorial(5)==120, "Factorial");
    Require(Factorial(20)==UINT64_C(2432902008176640000), "20 factorial");
    Require(FactorialRatio(7,4)==210 && FactorialRatio(50,49)==50, "Factorial ratio");
    Require(FactorialRatio(std::numeric_limits<int>::max(),std::numeric_limits<int>::max())==1,
            "Empty factorial product");
    Throws<std::invalid_argument>([]{ Factorial(-1); });
    Throws<std::invalid_argument>([]{ FactorialRatio(3,4); });
    Throws<std::invalid_argument>([]{ FactorialRatio(3,-1); });
    Throws<std::overflow_error>([]{ Factorial(21); });
    Near(Monomial(-2,3,3,2),-72);
    Near(Monomial(0,0,0,0),1);
    Near(Monomial(1e300,0,4,1),0);
    Throws<std::invalid_argument>([]{ Monomial(1,2,-1,0); });
    Throws<std::overflow_error>([]{ Monomial(1e200,1,2,0); });

    // p(x) = 2 - 3*x + 4*x^2 - 5*x^3.
    const std::vector<double> coefficients{2,-3,4,-5};
    for (double x : {-3.0,-1.0,0.0,0.25,1.0,2.0}) {
        Near(EvaluatePolynomial(x,coefficients),2-3*x+4*x*x-5*x*x*x);
        Near(EvaluatePolynomial(x,DifferentiatePolynomial(coefficients)), -3+8*x-15*x*x);
        Near(EvaluatePolynomial(x,DifferentiatePolynomial(coefficients,2)), 8-30*x);
        Near(EvaluatePolynomial(x,DifferentiatePolynomial(coefficients,3)), -30);
    }
    Require(DifferentiatePolynomial(coefficients,0)==coefficients,"Zeroth derivative");
    Require(DifferentiatePolynomial(coefficients,4)==std::vector<double>{0},"Derivative above degree");
    Require(DifferentiatePolynomial({})==std::vector<double>{0},"Empty derivative");
    Require(DifferentiatePolynomial({1e308,1e308,1e308,1},3)==std::vector<double>{6},
            "Discarded low-degree terms must not overflow");
    Near(EvaluatePolynomial(2,{}),0);
    Near(EvaluatePolynomial(1e308,{-1e308,2}),1e308); // FMA avoids a spurious overflow.
    Throws<std::invalid_argument>([]{ DifferentiatePolynomial({1},-1); });
    Throws<std::overflow_error>([]{ DifferentiatePolynomial({0,0,1e308}); });
    Throws<std::overflow_error>([]{ EvaluatePolynomial(1e200,{0,0,1}); });

    Near(HarmonicMean(2,6),3);
    Near(HarmonicMean(6,2),3);
    Near(HarmonicMean(0,0),0);
    Near(HarmonicMean(0,6),0);
    Near(HarmonicMean(1e308,1e308),1e308);
    Near(HarmonicMean(1e-300,1e300),2e-300);
    Near(HarmonicMean(1e-300,3e-300),1.5e-300);
    Throws<std::invalid_argument>([]{ HarmonicMean(-1,2); });
    const double nan = std::numeric_limits<double>::quiet_NaN();
    const double inf = std::numeric_limits<double>::infinity();
    Throws<std::invalid_argument>([&]{ HarmonicMean(nan,2); });
    Throws<std::invalid_argument>([&]{ EvaluatePolynomial(inf,{1}); });
    Throws<std::invalid_argument>([&]{ DifferentiatePolynomial({nan},0); });
    Throws<std::invalid_argument>([&]{ Monomial(nan,0,0,0); });
}
} // namespace

int main()
{
    try {
        Run();
        std::cout << "math_utils tests passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "math_utils test failed: " << error.what() << '\n';
        return 1;
    }
}
