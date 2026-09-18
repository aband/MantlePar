// Combined spatial-operator tests; no time integrator is introduced.
// flux_mode: 0=accuracy/convergence, 1=assembled Jacobian, 2=properties/limits.
#include "transport_test_support.h"

namespace {
using namespace transport_test;

void FrontProfile(const Grid& g,const std::vector<Reconstruction>& r,const Field& f,
                   const Options& o)
{
    auto csv=CSV(Sibling(o.output,"_profile"));
    csv<<"distance,distance_over_h,x,y,exact,reconstructed,cell_i,cell_j,weight_large,weight_small,weight_constant\n";
    const Point normal=f.Direction();const PetscReal move=Dot(normal,f.velocity)*f.time;
    const Point centre{{.5+move*normal.p[0],.5+move*normal.p[1]}};
    PetscReal lo=1e30,hi=-lo,over=0,under=0;
    for(int k=-1200;k<=1200;++k){const PetscReal d=k/2000.0;
        const Point p{{centre.p[0]+d*normal.p[0],centre.p[1]+d*normal.p[1]}};
        const auto c=Locate(g,p);if(c<0)continue;PetscReal value,w0;Call(r[c].Evaluate(p,value));Call(r[c].ConstantWeight(w0));
        PetscReal wl=0,ws=0;for(std::size_t a=0;a<r[c].CandidateCount();++a){ReconstructionCandidateInfo info;Call(r[c].GetCandidate(a,info));
            (info.family==ReconstructionFamilyKind::Large?wl:ws)+=info.nonlinearWeight;}
        over=std::max(over,value-1);under=std::max(under,-value);
        if(value>.1&&value<.9){lo=std::min(lo,d);hi=std::max(hi,d);}
        csv<<d<<','<<d/g.h<<','<<p.p[0]<<','<<p.p[1]<<','<<f(p).u<<','<<value<<','
           <<c%g.size.i<<','<<c/g.size.i<<','<<wl<<','<<ws<<','<<w0<<'\n';
    }
    Finish(csv);
    PetscReal lower=0,upper=4;
    for(int i=0;i<60;++i){const auto m=(lower+upper)/2;if(std::erf(m)<.8)lower=m;else upper=m;}
    const PetscReal exactWidth=2*(lower+upper)*std::sqrt(f.kappa*f.time);
    auto metrics=CSV(Sibling(o.output,"_front_metrics"));
    metrics<<"h,overshoot,undershoot,transition_width,exact_transition_width,profile_spacing\n";
    metrics<<g.h<<','<<over<<','<<under<<','<<(hi>=lo?hi-lo:0)<<','<<exactWidth<<",0.0005\n";
    Finish(metrics);
}

void Convergence(const Options& o)
{
    const Field f=MakeField(o,true);const auto edgeRule=Rule(5),cellRule=Rule(6);
    auto csv=CSV(o.output);csv<<"n,h,quantity,region,l1,l2,linf,rate_l2,rate_linf\n";
    std::map<std::string,Norm> previous;PetscReal oldH=0,first=0,last=0,lastRate=0;
    const char* names[]={"all","interior","boundary"};
    for(PetscInt level=0,n=o.n0;level<o.levels;++level,n*=2){
        Grid g;MakeGrid(n,o,g);const auto u=Averages(g,f),exactRHS=Averages(g,f,true);
        std::vector<Reconstruction> recon;InitializeReconstructions(g,o,recon);Update(g,u,recon);
        const auto faces=Faces(g,edgeRule,o.samples);
        const auto numerical=Assemble(g,faces,edgeRule,recon,f,o.lf!=0);
        std::array<Norm,3> residualErrors,fluxErrors,valueErrors;
        long double referenceBalance=0,referenceScale=1;
        for(std::size_t c=0;c<g.cells.size();++c){const int region=Interior(g,c,o.order-1)?1:2;
            const PetscReal error=numerical.rhs[c]-exactRHS[c];
            residualErrors[0].Add(error,g.areas[c]);residualErrors[region].Add(error,g.areas[c]);
            referenceBalance+=g.areas[c]*exactRHS[c];referenceScale+=std::abs(g.areas[c]*exactRHS[c]);
            for(std::size_t j=0;j<cellRule.points.size();++j)for(std::size_t i=0;i<cellRule.points.size();++i){
                const Point ref{{cellRule.points[i],cellRule.points[j]}};const Point p=MapCellPoint(ref,g.cells[c]);
                const PetscReal w=cellRule.weights[i]*cellRule.weights[j]*CellJacobian(ref,g.cells[c]);
                PetscReal v;Call(recon[c].Evaluate(p,v));valueErrors[0].Add(v-f(p).u,w);valueErrors[region].Add(v-f(p).u,w);
            }
        }
        for(std::size_t e=0;e<faces.size();++e){const auto& face=faces[e];const auto exact=ExactFlux(face,f,true,true);
            PetscReal length;Point normal;Call(GetEdgeGeometry(face.edge,length,normal));
            const PetscReal error=(numerical.flux[e]-exact)/length;
            fluxErrors[0].Add(error,length);fluxErrors[face.right<0?2:1].Add(error,length);
            if(face.right<0){referenceBalance+=exact;referenceScale+=std::abs(exact);}
        }
        Require(std::abs(referenceBalance)<2e-9*referenceScale,
                "Independent analytical volume and boundary references violate the PDE balance");
        for(int quantity=0;quantity<3;++quantity)for(int region=0;region<3;++region){
            const auto& norm=quantity==0?residualErrors[region]:quantity==1?fluxErrors[region]:valueErrors[region];
            if(!norm.weight)continue;
            const std::string name=quantity==0?"spatial_rhs":quantity==1?"combined_flux":"reconstruction";
            const auto key=name+names[region];const auto old=previous[key];
            csv<<n<<','<<g.h<<','<<name<<','<<names[region]<<','<<norm.L1()<<','<<norm.L2()<<','<<norm.maximum<<','
               <<Rate(old.L2(),norm.L2(),oldH,g.h)<<','<<Rate(old.maximum,norm.maximum,oldH,g.h)<<'\n';previous[key]=norm;
        }
        const PetscReal error=residualErrors[1].L2();if(level==0)first=error;
        lastRate=Rate(last,error,oldH,g.h);last=error;oldH=g.h;
        std::cout<<"N="<<n<<" interior spatial RHS L2="<<error<<" rate="<<lastRate
                 <<" all-cell L2="<<residualErrors[0].L2()<<'\n';
        if(o.field==1&&level+1==o.levels)FrontProfile(g,recon,f,o);
    }
    Finish(csv);
    Require(last<first||last<3e-9,"Combined spatial operator must improve under refinement");
    // Differentiating a reconstructed quantity to a flux, then taking its
    // discrete divergence can lose TWO orders. Randomly perturbed meshes need
    // not have the cancellation seen on uniform grids. Boundary rates are
    // reported separately. Do not impose smooth asymptotic rates on thin fronts.
    if(o.field==0&&last>3e-9)
        Require(lastRate>std::max<PetscInt>(1,std::min<PetscInt>(o.samples,o.order-1)-1)-.8,
                "Combined spatial RHS convergence below its conservative order floor");
}

void Jacobian(const Options& input)
{
    Options o=input;o.n0=8;
    Grid g;MakeGrid(8,o,g);std::vector<Reconstruction> recon;InitializeReconstructions(g,o,recon);
    Field f=MakeField(o,true);f.kind=0;f.kappa=.07;
    std::vector<PetscReal> u(g.cells.size());
    for(std::size_t c=0;c<u.size();++c)
        u[c]=.8+.05*std::sin(.7*(c%g.size.i)+.4*(c/g.size.i))+.01*static_cast<PetscReal>((13*c+7)%11);
    const auto q=Rule(3);
    const auto faces=Faces(g,q,o.samples);
    Update(g,u,recon,true);const auto base=Assemble(g,faces,q,recon,f,o.lf!=0,true,true,true);
    auto csv=CSV(o.output);csv<<"direction,scaled_jacobian_error\n";
    PetscReal worst=0;
    for(int direction=0;direction<9;++direction){std::vector<PetscReal> d(u.size(),0),action(u.size(),0);
        if(direction<5)d[static_cast<std::size_t>(direction)*(u.size()-1)/4]=1;
        else for(std::size_t c=0;c<d.size();++c)d[c]=.3*std::sin((direction-.5)*(c+1));
        for(std::size_t c=0;c<u.size();++c)for(const auto& entry:base.jacobian[c])action[c]+=entry.second*d[entry.first];
        const auto evaluate=[&](PetscReal step){auto values=u;
            for(std::size_t c=0;c<u.size();++c)values[c]+=step*d[c];
            Update(g,values,recon);return Assemble(g,faces,q,recon,f,o.lf!=0).rhs;};
        const PetscReal h=2e-6;
        const auto plus=evaluate(h),minus=evaluate(-h),halfPlus=evaluate(h/2),halfMinus=evaluate(-h/2);
        PetscReal error=0,scale=1;
        for(std::size_t c=0;c<u.size();++c){
            const PetscReal coarse=(plus[c]-minus[c])/(2*h),fine=(halfPlus[c]-halfMinus[c])/h;
            const PetscReal fd=(4*fine-coarse)/3;
            error=std::max(error,std::abs(action[c]-fd));scale=std::max(scale,std::max(std::abs(action[c]),std::abs(fd)));}
        const PetscReal relative=error/scale;worst=std::max(worst,relative);csv<<direction<<','<<relative<<'\n';
        Require(relative<2e-6,"Assembled full reconstruction/advection/diffusion Jacobian disagrees with finite differences");
    }
    Finish(csv);std::cout<<"Worst assembled Jacobian scaled error="<<worst<<'\n';
}

void Properties(const Options& input)
{
    Options o=input;o.order=5;o.constant=0;o.axis=0;
    Grid g;MakeGrid(8,o,g);std::vector<Reconstruction> r;InitializeReconstructions(g,o,r);
    const auto q=Rule(5);
    const auto faces=Faces(g,q,o.samples);
    Field f=MakeField(o,true);f.kind=3;
    Update(g,Averages(g,f),r);
    for(bool global:{false,true}){const auto op=Assemble(g,faces,q,r,f,global);
        for(auto v:op.rhs)Near(v,0,"Combined constant-state preservation",2e-10);}
    // Both the Q4 large and Q2 small candidates reproduce this quadratic;
    // this checks flux signs, cell areas and all boundary contributions against
    // the analytical divergence, independently of convergence-rate fitting.
    f.kind=2;const auto u=Averages(g,f);Update(g,u,r);const auto reference=Averages(g,f,true);
    const auto both=Assemble(g,faces,q,r,f,o.lf!=0),adv=Assemble(g,faces,q,r,f,o.lf!=0,true,false),
               diff=Assemble(g,faces,q,r,f,o.lf!=0,false,true);
    for(std::size_t c=0;c<u.size();++c){Near(both.rhs[c],reference[c],"Quadratic combined operator accuracy",3e-8);
        Near(both.rhs[c],adv.rhs[c]+diff.rhs[c],"Combined conservative assembly",2e-12);}
    auto noDiffusion=f;noDiffusion.kappa=0;
    const auto advLimit=Assemble(g,faces,q,r,noDiffusion,o.lf!=0);
    auto noAdvection=f;noAdvection.velocity={};const auto diffLimit=Assemble(g,faces,q,r,noAdvection,o.lf!=0);
    for(std::size_t c=0;c<u.size();++c){Near(advLimit.rhs[c],adv.rhs[c],"Zero diffusivity limit",2e-12);
        Near(diffLimit.rhs[c],diff.rhs[c],"Zero velocity limit",2e-12);}
}
}

int main(int argc,char** argv)
{
    return transport_test::Main(argc,argv,[]{const auto o=transport_test::ReadOptions();
        if(o.mode==0)Convergence(o);else if(o.mode==1)Jacobian(o);else Properties(o);});
}
