#include "evolve.h"
#include <cmath>
#include <iostream>

int main() {
    using namespace mantle::couple;
    mantle::phase::PhaseState a,b;
    a.cl=.2; a.cs=.03; b.cl=.24; b.cs=.05;
    const Point solid{{.01,.02}},darcy{{.3,-.4}};
    const auto check=[](double a,double b) { return std::abs(a-b)<1e-14; };
    const auto wet=PhaseTransportVelocities(solid,darcy,.1,.5,a,b);
    const double q=std::pow(.1,1.5),bulk=.1*.22+.9*.04;
    for (int k=0;k<2;++k) {
        if (!check(wet.mixture.p[k],solid.p[k]+q*darcy.p[k])) return 1;
        if (!check(wet.effective.p[k],solid.p[k]+.22/bulk*q*darcy.p[k])) return 2;
    }
    const auto dry=PhaseTransportVelocities(solid,darcy,0,.5,a,b);
    for (int k=0;k<2;++k) if (!check(dry.mixture.p[k],solid.p[k]) || !check(dry.effective.p[k],solid.p[k])) return 3;
    a.cl=a.cs=b.cl=b.cs=0;
    const auto pure=PhaseTransportVelocities(solid,darcy,.1,0,a,b);
    for (int k=0;k<2;++k) if (!std::isfinite(pure.effective.p[k]) || !check(pure.effective.p[k],solid.p[k])) return 4;
    // Equation (4.13) reduces to v_l when only liquid carries the species.
    a.cl=b.cl=.2;
    const auto incompatible=PhaseTransportVelocities(solid,darcy,.1,0,a,b);
    for (int k=0;k<2;++k) if (!check(incompatible.effective.p[k],solid.p[k]+darcy.p[k])) return 5;
    std::cout<<"Passed scaled Darcy velocity, arithmetic phase composition, dry and pure limits\n";
}
