// Diffusion kernel properties, analytical accuracy, geometry and derivatives.
// Run via diffusiveflux_tests.cmake; requires one PETSc rank.
#include "transport_test_support.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <random>
#include <string>

namespace {
using transport_test::Call;
using transport_test::Require;
using transport_test::Near;
int differences = 0;
double maxFdError = 0;
void FD(double a, double b, const std::string& label)
{
    ++differences;
    maxFdError = std::max(maxFdError, std::abs(a-b)/(1+std::abs(a)+std::abs(b)));
    Near(a,b,label,2e-8);
}
const GaussRule1D one{{0},{2}};
const GaussRule1D three{{-std::sqrt(3.0/5),0,std::sqrt(3.0/5)}, {5.0/9,8.0/9,5.0/9}};
const GaussRule1D five{{-.9061798459386639928,-.5384693101056830910,0,
                       .5384693101056830910,.9061798459386639928},
                      {.2369268850561890875,.4786286704993664680,.5688888888888888889,
                       .4786286704993664680,.2369268850561890875}};
const QuadVertices rectangleLeft{{{{-1,-.5}},{{0,-.5}},{{0,.5}},{{-1,.5}}}};
const QuadVertices rectangleRight{{{{0,-.5}},{{1.5,-.5}},{{1.5,.5}},{{0,.5}}}};
const QuadVertices quadLeft{{{{-1.3,-.8}},{{.1,-.6}},{{.3,1}},{{-1.1,.7}}}};
const QuadVertices quadRight{{{{.1,-.6}},{{1.2,-.9}},{{1.4,.8}},{{.3,1}}}};

double Polynomial(const Point& p)
{ double x=p.p[0],y=p.p[1]; return 2+.7*x-.3*y+.2*x*x+.4*x*y+.5*y*y; }
Point PolynomialGradient(const Point& p)
{ return {{{.7+.4*p.p[0]+.4*p.p[1]}, {-.3+.4*p.p[0]+p.p[1]}}}; }
double Linear(const Point& p) { return 2+.2*p.p[0]-.3*p.p[1]; }
double CubicPotential(double u) { return u*u*u/3; }
DiffusiveFluxLaw CubicLaw()
{
    DiffusiveFluxLaw law;
    law.evaluate=[](double u,const DiffusiveFluxPoint&,DiffusiveQuantity& r) -> int {
        r={CubicPotential(u),u*u};return 0;
    };
    return law;
}
std::vector<double> Samples(const DiffusiveSampling& s, double (*f)(const Point&))
{
    std::vector<double> u;
    for (const auto& p:s.SamplePoints()) u.push_back(f(p));
    return u;
}
void Inside(const QuadVertices& cell,const Point& p)
{
    for (std::size_t j=0;j<4;++j) {
        const auto& a=cell[j]; const auto& b=cell[(j+1)%4];
        const long double x=b.p[0]-a.p[0],y=b.p[1]-a.p[1];
        const long double xx=p.p[0]-a.p[0],yy=p.p[1]-a.p[1];
        Require(x*yy-y*xx >= -2e-14*(std::abs(x*yy)+std::abs(y*xx)),"sample stays in its cell");
    }
}
double Exact(const DiffusiveSampling& s, Point (*gradient)(const Point&))
{
    double result=0;
    for (std::size_t q=0;q<s.QuadraturePoints().size();++q) {
        const auto g=gradient(s.QuadraturePoints()[q]);
        result-=s.QuadratureWeights()[q]*(g.p[0]*s.Normal().p[0]+g.p[1]*s.Normal().p[1]);
    }
    return result;
}

void Coefficients()
{
    for (int npts=2;npts<=16;++npts) {
        for (auto location:{LagrangeDerivativeLocation::Middle,LagrangeDerivativeLocation::EndLo,
                           LagrangeDerivativeLocation::EndHi}) {
            if (location==LagrangeDerivativeLocation::Middle && npts%2) continue;
            std::vector<double> w;
            Call(CreateLagrangeDerivativeWeights(npts,location,w));
            const int n=npts-1;
            const double x=location==LagrangeDerivativeLocation::Middle ? n/2.0 :
                           location==LagrangeDerivativeLocation::EndLo ? 0 : n;
            if (npts==2 && location==LagrangeDerivativeLocation::Middle) {
                Near(w[0],-1,"two-point left weight");Near(w[1],1,"two-point right weight");
            }
            if (npts==4 && location==LagrangeDerivativeLocation::Middle) {
                Near(w[0],1.0/24,"four-point outer weight");Near(w[1],-9.0/8,"four-point inner weight");
            }
            // Scaled monomials keep the endpoint moment check well conditioned.
            for (int p=0;p<=n;++p) {
                long double computed=0,absolute=0;
                for (int i=0;i<npts;++i) {
                    const long double term=w[i]*std::pow((static_cast<long double>(i)-x)/n,p);
                    computed+=term; absolute+=std::abs(term);
                }
                const long double target=p==1 ? 1.0L/n : 0;
                Require(std::abs(computed-target)<3e-13L*(1+absolute),"polynomial derivative moments");
            }
        }
    }
}

void GeometryAndFluxes()
{
    for (int mesh=0;mesh<4;++mesh) {
        auto left=mesh%2 ? quadLeft:rectangleLeft;
        auto right=mesh%2 ? quadRight:rectangleRight;
        if (mesh>=2) {
            for (auto* cell:{&left,&right}) for(auto& p:*cell) p.p[0]*=1e-6;
        }
        const EdgeVertices edge{{left[1],left[2]}};
        for (int n:{2,4,6,8}) for (const auto* rule:{&one,&three,&five}) {
            DiffusiveSampling sampling,reversed;
            Call(CreateInteriorDiffusiveSampling(edge,left,right,*rule,{n,.9},sampling));
            const EdgeVertices reverse{{edge[1],edge[0]}};
            Call(CreateInteriorDiffusiveSampling(reverse,right,left,*rule,{n,.9},reversed));
            Require(sampling.SamplesPerPoint()==static_cast<std::size_t>(n),"config controls sample count");
            std::vector<double> coefficient(rule->points.size(),1);
            for (std::size_t a=0;a<sampling.SamplePoints().size();++a) {
                const bool isLeft=a%n < static_cast<std::size_t>(n/2);
                Require(sampling.SampleSides()[a]==(isLeft?DiffusiveSampleSide::Left:DiffusiveSampleSide::Right),
                        "geometric sample side");
                Inside(isLeft?left:right,sampling.SamplePoints()[a]);
            }
            double f,rf;
            auto u=Samples(sampling,Polynomial),ru=Samples(reversed,Polynomial);
            Call(IntegrateDiffusiveFlux(sampling,u,coefficient,0,{},f));
            Call(IntegrateDiffusiveFlux(reversed,ru,coefficient,0,{},rf));
            // The most stretched cases amplify state-roundoff by 1/ds.
            Near(f,Exact(sampling,PolynomialGradient),"quadratic physical flux",mesh<2?3e-11:2e-8);
            Near(f,-rf,"orientation reversal conservation",mesh<2?3e-11:2e-8);
            std::fill(u.begin(),u.end(),1e7);
            Call(IntegrateDiffusiveFlux(sampling,u,coefficient,0,{},f));
            Require(f==0,"constant field exactly zero");
            for(std::size_t a=0;a<u.size();++a) u[a]=sampling.SampleSides()[a]==DiffusiveSampleSide::Left ? 1:3;
            Call(IntegrateDiffusiveFlux(sampling,u,coefficient,0,{},f));
            Require(f<0,"constant-reconstruction jump diffuses down gradient");
        }
    }
    // Thin horizontal/vertical strips: every outer face, including the two
    // opposite faces of a single cell. This detects index-based boundary logic.
    for (const Point scale: {Point{{1,1}},Point{{1e-5,2}},Point{{2,1e-5}}}) {
        auto cell=quadLeft;
        for(auto& p:cell) {p.p[0]*=scale.p[0];p.p[1]*=scale.p[1];}
        for (int side=0;side<4;++side) for(int n:{2,4,6}) {
            DiffusiveSampling s;
            Call(CreateBoundaryDiffusiveSampling({{cell[side],cell[(side+1)%4]}},cell,three,{n,.9},s));
            Require(s.SamplesPerPoint()==static_cast<std::size_t>(n+1),"boundary extra node");
            auto u=Samples(s,Polynomial);
            std::vector<DiffusiveBoundaryValue> data(3);
            for(std::size_t q=0;q<3;++q) {
                data[q]={DiffusiveBoundaryType::PrescribedState,Polynomial(s.QuadraturePoints()[q])};
                Require(s.SampleSides()[q*(n+1)+n]==DiffusiveSampleSide::Boundary,"boundary face slot");
                u[q*(n+1)+n]=NAN; // Deliberately ignored.
            }
            for(std::size_t a=0;a<u.size();++a)
                if(s.SampleSides()[a]!=DiffusiveSampleSide::Boundary) Inside(cell,s.SamplePoints()[a]);
            double f;
            Call(IntegrateDiffusiveBoundaryFlux(s,u,{1,1,1},data,0,{},f));
            Near(f,Exact(s,PolynomialGradient),"Dirichlet quadratic flux on all faces",2e-8);
        }
    }
}

void Jacobians()
{
    std::mt19937 generator(173);
    std::uniform_real_distribution<double> random(.2,1.9);
    const double h=2e-6;
    const auto law=CubicLaw();
    for (int n:{2,4,6,8}) for (bool boundary:{false,true}) {
        DiffusiveSampling s;
        const EdgeVertices edge{{quadLeft[1],quadLeft[2]}};
        if (boundary) Call(CreateBoundaryDiffusiveSampling(edge,quadLeft,three,{n,.83},s));
        else Call(CreateInteriorDiffusiveSampling(edge,quadLeft,quadRight,three,{n,.83},s));
        for(int trial=0;trial<5;++trial) {
            std::vector<double> u(s.SamplePoints().size()),kappa(3);
            for(auto& a:u)a=random(generator);
            for(auto& a:kappa)a=random(generator);
            std::vector<DiffusiveBoundaryValue> data(3);
            for(auto& a:data)a={static_cast<DiffusiveBoundaryType>(trial%4),random(generator)};
            if(trial==4) {data[0].type=DiffusiveBoundaryType::PrescribedState;
                          data[1].type=DiffusiveBoundaryType::PrescribedFlux;
                          data[2].type=DiffusiveBoundaryType::ZeroFlux;}
            const auto evaluate=[&](const std::vector<double>& values,const std::vector<double>& kk,
                                     const std::vector<DiffusiveBoundaryValue>& bc) {
                double f;
                if(boundary) Call(IntegrateDiffusiveBoundaryFlux(s,values,kk,bc,.2,law,f));
                else Call(IntegrateDiffusiveFlux(s,values,kk,.2,law,f));
                return f;
            };
            DiffusiveEdgeFluxResult r;
            if(boundary)Call(IntegrateDiffusiveBoundaryFluxWithDerivatives(s,u,kappa,data,.2,law,r));
            else Call(IntegrateDiffusiveFluxWithDerivatives(s,u,kappa,.2,law,r));
            Require(evaluate(u,kappa,data)==r.flux,"value/Jacobian agreement");
            for(std::size_t a=0;a<u.size();++a) {
                auto up=u,um=u;up[a]+=h;um[a]-=h;
                FD((evaluate(up,kappa,data)-evaluate(um,kappa,data))/(2*h),r.derivativeSamples[a],"sample partial");
            }
            for(std::size_t q=0;q<3;++q) {
                auto kp=kappa,km=kappa;kp[q]+=h;km[q]-=h;
                FD((evaluate(u,kp,data)-evaluate(u,km,data))/(2*h),r.derivativeDiffusivity[q],"diffusivity partial");
                if(boundary) {
                    auto dp=data,dm=data;dp[q].value+=h;dm[q].value-=h;
                    FD((evaluate(u,kappa,dp)-evaluate(u,kappa,dm))/(2*h),r.derivativeBoundaryData[q],"boundary partial");
                }
            }
            // A nonlinear cell-variable-to-sample map and state-dependent kappa:
            // verify the assembled chain rule, including shared sample support.
            auto chainValue=[&](double z) {
                auto uu=u,kk=kappa;
                auto dd=data;
                for(std::size_t a=0;a<u.size();++a)uu[a]+=z*.03*(a+1)+z*z*.07;
                for(std::size_t q=0;q<3;++q) {kk[q]+=z*.1*(q+1)+z*z;
                                            dd[q].value+=z*.08*(q+1);}
                return evaluate(uu,kk,dd);
            };
            double analytic=0;
            for(std::size_t a=0;a<u.size();++a)analytic+=r.derivativeSamples[a]*.03*(a+1);
            for(std::size_t q=0;q<3;++q) {
                analytic+=r.derivativeDiffusivity[q]*.1*(q+1);
                if(boundary)analytic+=r.derivativeBoundaryData[q]*.08*(q+1);
            }
            FD((chainValue(h)-chainValue(-h))/(2*h),analytic,"full chained Jacobian");
        }
    }
}

void PhysicsAndFailures()
{
    DiffusiveSampling s;
    const EdgeVertices edge{{quadLeft[1],quadLeft[2]}};
    Call(CreateBoundaryDiffusiveSampling(edge,quadLeft,three,{},s));
    auto u=Samples(s,Linear);
    std::vector<double> coefficient(3,1);
    std::vector<DiffusiveBoundaryValue> data(3);
    for(std::size_t q=0;q<3;++q)data[q]={DiffusiveBoundaryType::PrescribedState,Linear(s.QuadraturePoints()[q])};
    double f,g;
    Call(IntegrateDiffusiveBoundaryFlux(s,u,coefficient,data,.5,CubicLaw(),f));
    for(auto& a:data){a.type=DiffusiveBoundaryType::PrescribedQuantity;a.value=CubicPotential(a.value);}
    Call(IntegrateDiffusiveBoundaryFlux(s,u,coefficient,data,.5,CubicLaw(),g));
    Near(f,g,"state and quantity Dirichlet agree");
    double exact=0;
    for(std::size_t q=0;q<3;++q) {
        const double v=Linear(s.QuadraturePoints()[q]);
        exact-=s.QuadratureWeights()[q]*v*v*(.2*s.Normal().p[0]-.3*s.Normal().p[1]);
    }
    Near(f,exact,"nonlinear transformed quantity flux");
    DiffusiveFluxLaw spatial;
    spatial.evaluate=[](double v,const DiffusiveFluxPoint& p,DiffusiveQuantity& r)->int {
        r={v+p.position.p[0]*p.time,1};return 0;
    };
    for(std::size_t q=0;q<3;++q)data[q]={DiffusiveBoundaryType::PrescribedState,Linear(s.QuadraturePoints()[q])};
    Call(IntegrateDiffusiveBoundaryFlux(s,u,coefficient,data,.5,spatial,f));
    double length;Point normal;Call(GetEdgeGeometry(edge,length,normal));
    Near(f,-length*(.7*normal.p[0]-.3*normal.p[1]),"position and time reach callback");
    for(auto& a:data)a={DiffusiveBoundaryType::PrescribedFlux,-.7};
    std::fill(u.begin(),u.end(),NAN);std::fill(coefficient.begin(),coefficient.end(),NAN);
    DiffusiveFluxLaw empty;empty.evaluate={};DiffusiveEdgeFluxResult r;
    Call(IntegrateDiffusiveBoundaryFluxWithDerivatives(s,u,coefficient,data,0,empty,r));
    Near(r.flux,-.7*length,"Neumann data are physical outward flux");
    for(auto a:r.derivativeSamples)Require(a==0,"Neumann sample derivative zero");
    for(auto a:r.derivativeDiffusivity)Require(a==0,"Neumann coefficient derivative zero");
    for(auto& a:data)a={DiffusiveBoundaryType::ZeroFlux,NAN};
    Call(IntegrateDiffusiveBoundaryFluxWithDerivatives(s,u,coefficient,data,0,empty,r));
    Require(r.flux==0,"exact zero-flux boundary");

    DiffusiveSampling interior;
    Call(CreateInteriorDiffusiveSampling(edge,quadLeft,quadRight,three,{},interior));
    u=Samples(interior,Linear);coefficient={1,1,1};
    auto noDerivative=spatial;
    noDerivative.evaluate=[](double v,const DiffusiveFluxPoint&,DiffusiveQuantity& d)->int {d.value=v;return 0;};
    Call(IntegrateDiffusiveFlux(interior,u,coefficient,0,noDerivative,f));
    r.flux=123;
    Require(IntegrateDiffusiveFluxWithDerivatives(interior,u,coefficient,0,noDerivative,r)!=0 && r.flux==123,
            "missing derivative errors without output mutation");
    std::vector<double> w{123};
    Require(CreateLagrangeDerivativeWeights(3,LagrangeDerivativeLocation::Middle,w)!=0&&w[0]==123,
            "odd midpoint rule rejected");
    Require(CreateLagrangeDerivativeWeights(0,LagrangeDerivativeLocation::EndHi,w)!=0,"empty rule rejected");
    Require(CreateLagrangeDerivativeWeights(4,static_cast<LagrangeDerivativeLocation>(99),w)!=0,"bad enum rejected");
    const auto oldCount=interior.SamplePoints().size();
    Require(CreateInteriorDiffusiveSampling(edge,quadLeft,quadRight,three,{3,.9},interior)!=0 &&
            interior.SamplePoints().size()==oldCount,"failed builder preserves old plan");
    Require(CreateInteriorDiffusiveSampling(edge,quadRight,quadLeft,three,{},interior)!=0,"opposite cells rejected");
    Require(CreateBoundaryDiffusiveSampling({{edge[1],edge[0]}},quadLeft,three,{},s)!=0,"inward boundary rejected");
    Require(CreateInteriorDiffusiveSampling(edge,quadLeft,quadRight,three,{4,0},interior)!=0,"zero spacing rejected");
    Require(CreateInteriorDiffusiveSampling(edge,quadLeft,quadRight,three,{4,1.1},interior)!=0,"exterior sampling rejected");
    Require(CreateInteriorDiffusiveSampling(edge,quadLeft,quadRight,{{-1,1},{1,1}},{},interior)!=0,
            "endpoint quadrature rejected");
    double keep=123;
    Require(IntegrateDiffusiveFlux(interior,{},coefficient,0,{},keep)!=0 && keep==123,"wrong sample count");
    Require(IntegrateDiffusiveFlux(interior,u,{1,-1,1},0,{},keep)!=0,"negative coefficient rejected");
    Require(IntegrateDiffusiveFlux(interior,u,coefficient,NAN,{},keep)!=0,"nonfinite time rejected");
    Require(IntegrateDiffusiveFlux(s,u,coefficient,0,{},keep)!=0,"wrong kernel type rejected");
    auto broken=u;broken[0]=NAN;
    Require(IntegrateDiffusiveFlux(interior,broken,coefficient,0,{},keep)!=0,"nonfinite sample rejected");
    DiffusiveSampling blank;
    Require(IntegrateDiffusiveFlux(blank,u,coefficient,0,{},keep)!=0,"uninitialized plan rejected");
    auto overflow=spatial;
    overflow.evaluate=[](double,const DiffusiveFluxPoint&,DiffusiveQuantity& d)->int {d={INFINITY,1};return 0;};
    Require(IntegrateDiffusiveFlux(interior,u,coefficient,0,overflow,keep)!=0,"nonfinite law rejected");
    auto failed=spatial;failed.evaluate=[](double,const DiffusiveFluxPoint&,DiffusiveQuantity&)->int{return 91;};
    Require(IntegrateDiffusiveFlux(interior,u,coefficient,0,failed,keep)==91,"callback error propagated");
    Call(IntegrateDiffusiveFluxWithDerivatives(interior,u,{0,0,0},0,{},r));
    Require(r.flux==0,"zero coefficient flux");
    for(auto a:r.derivativeSamples)Require(a==0,"zero coefficient state partials");
    Require(std::abs(r.derivativeDiffusivity[0])>0,"zero coefficient retains coefficient partial");
}

double Sine(const Point& p) {return std::sin(1.2*p.p[0]+.7*p.p[1]);}
double SineIntegral(const EdgeVertices& edge)
{
    double length;Point normal;Call(GetEdgeGeometry(edge,length,normal));
    const double mid=.6*(edge[0].p[0]+edge[1].p[0])+.35*(edge[0].p[1]+edge[1].p[1]);
    const double half=.6*(edge[1].p[0]-edge[0].p[0])+.35*(edge[1].p[1]-edge[0].p[1]);
    return -length*(1.2*normal.p[0]+.7*normal.p[1])*std::cos(mid)*(half==0 ? 1:std::sin(half)/half);
}
void Convergence()
{
    for(bool quad:{false,true}) for(bool boundary:{false,true}) for(int n:{2,4,6}) {
        double previous=0,minimum=100;
        for(double h:{.8,.4,.2,.1}) {
            auto left=quad?quadLeft:rectangleLeft,right=quad?quadRight:rectangleRight;
            for(auto* cell:{&left,&right})for(auto& p:*cell){p.p[0]=.31+h*p.p[0];p.p[1]=.27+h*p.p[1];}
            const EdgeVertices edge{{left[1],left[2]}};
            DiffusiveSampling s;
            if(boundary)Call(CreateBoundaryDiffusiveSampling(edge,left,five,{n,.9},s));
            else Call(CreateInteriorDiffusiveSampling(edge,left,right,five,{n,.9},s));
            const auto u=Samples(s,Sine);
            double f;
            if(boundary) {
                std::vector<DiffusiveBoundaryValue> data(5);
                for(std::size_t q=0;q<5;++q)data[q]={DiffusiveBoundaryType::PrescribedState,Sine(s.QuadraturePoints()[q])};
                Call(IntegrateDiffusiveBoundaryFlux(s,u,{1,1,1,1,1},data,0,{},f));
            } else Call(IntegrateDiffusiveFlux(s,u,{1,1,1,1,1},0,{},f));
            double length;Point normal;Call(GetEdgeGeometry(edge,length,normal));
            const double error=std::abs(f-SineIntegral(edge))/length;
            if(previous>0)minimum=std::min(minimum,std::log2(previous/error));
            previous=error;
        }
        std::cout << (quad?"quad":"rectangle") << (boundary?" boundary":" interior")
                  << " N=" << n << " minimum rate=" << minimum << '\n';
        Require(minimum>n-.45,"normal-sampling convergence order");
    }
}
}

int main(int argc,char** argv)
{
    return transport_test::Main(argc,argv,[]{
        Coefficients();GeometryAndFluxes();Jacobians();
        Call(PetscPushErrorHandler(PetscReturnErrorHandler,nullptr));
        PhysicsAndFailures();
        Call(PetscPopErrorHandler());
        Convergence();
        std::cout << "Finite-difference comparisons=" << differences
                  << " max_scaled_fd_error=" << maxFdError << '\n';
    });
}
