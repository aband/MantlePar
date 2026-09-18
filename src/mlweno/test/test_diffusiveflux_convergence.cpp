// Modes: -flux_mode 0 = reconstruction/diffusive-flux convergence;
//        -flux_mode 1 = exact discontinuity reconstruction diagnostics.
// See transport_flux_tests.md for options, expected orders and CSV definitions.
#include "transport_test_support.h"

namespace {
using namespace transport_test;
const char* regions[]={"all","interior","boundary"};

void SmoothConvergence(const Options& o)
{
    Require(o.kappa>0,"A diffusion convergence test requires kappa>0");
    auto csv=CSV(o.output);
    csv<<"n,h,quantity,region,l1,l2,linf,rate_l2,rate_linf\n";
    std::map<std::string,Norm> previous;
    PetscReal first=0,last=0,oldH=0,rate=0;
    const auto edgeRule=Rule(5),cellRule=Rule(6);const auto f=MakeField(o,false);
    for(PetscInt level=0,n=o.n0;level<o.levels;++level,n*=2){
        Grid g;MakeGrid(n,o,g);const auto u=Averages(g,f);
        std::vector<Reconstruction> r;InitializeReconstructions(g,o,r);Update(g,u,r);
        const auto faces=Faces(g,edgeRule,o.samples);
        std::array<Norm,3> valueErrors,fluxErrors;
        for(std::size_t c=0;c<g.cells.size();++c){const int region=Interior(g,c,o.order-1)?1:2;
            long double mean=0;
            for(std::size_t j=0;j<cellRule.points.size();++j)for(std::size_t i=0;i<cellRule.points.size();++i){
                const Point reference{{cellRule.points[i],cellRule.points[j]}};
                const Point p=MapCellPoint(reference,g.cells[c]);
                const PetscReal w=cellRule.weights[i]*cellRule.weights[j]*CellJacobian(reference,g.cells[c]);
                PetscReal v;Call(r[c].Evaluate(p,v));mean+=w*v;
                valueErrors[0].Add(v-f(p).u,w);valueErrors[region].Add(v-f(p).u,w);
            }
            Near(static_cast<PetscReal>(mean/g.areas[c]),u[c],"Reconstruction cell-average conservation",2e-9);
        }
        for(const auto& face:faces){const auto numerical=EvaluateFace(g,face,edgeRule,r,f,false,false,true);
            const PetscReal exact=ExactFlux(face,f,false,true);PetscReal length;Point normal;
            Call(GetEdgeGeometry(face.edge,length,normal));
            const PetscReal error=(numerical.flux-exact)/length;
            fluxErrors[0].Add(error,length);fluxErrors[face.right<0?2:1].Add(error,length);
        }
        for(int quantity=0;quantity<2;++quantity)for(int region=0;region<3;++region){
            const auto& norm=quantity?fluxErrors[region]:valueErrors[region];
            if(!norm.weight)continue;
            const std::string name=quantity?"diffusive_flux":"reconstruction";
            const auto key=name+regions[region];const auto old=previous[key];
            const PetscReal rl2=Rate(old.L2(),norm.L2(),oldH,g.h),rmax=Rate(old.maximum,norm.maximum,oldH,g.h);
            csv<<n<<','<<g.h<<','<<name<<','<<regions[region]<<','<<norm.L1()<<','<<norm.L2()<<','
               <<norm.maximum<<','<<rl2<<','<<rmax<<'\n';
            previous[key]=norm;
        }
        const auto error=fluxErrors[1].L2();if(level==0)first=error;
        rate=Rate(last,error,oldH,g.h);last=error;oldH=g.h;
        std::cout<<"N="<<n<<" interior edge-average flux L2="<<error<<" rate="<<rate<<'\n';
    }
    Finish(csv);
    Require(last<first||last<2e-10*o.kappa,"Reconstructed diffusive flux must improve under refinement");
    // Exact-sample order is tested separately. Reconstruction costs a spatial
    // derivative: a degree p-1 reconstruction supplies at least p-1 flux order.
    // Erf layers can be underresolved; their measured rates are diagnostic.
    if(o.field==0&&last>2e-10*o.kappa)
        Require(rate>std::min<PetscInt>(o.samples,o.order-1)-.8,
                "Smooth reconstruction/flux convergence below the expected derivative order");
}

struct Jump {
    Point normal{{1,0}};PetscReal offset=.5,slope=0;
    PetscReal Signed(const Point& p)const{return Dot(normal,p)-offset;}
    PetscReal Value(const Point& p,bool positive)const{return (positive?1:0)+slope*Signed(p);}
};
Jump JumpField(const Options& o)
{
    Field f;f.axis=o.axis;f.orientation=o.orientation;Jump j;j.normal=f.Direction();
    const Point centre=o.cut?Point{{.473,.517}}:Point{{.5,.5}};
    j.offset=Dot(j.normal,centre);j.slope=o.field?.2:0;return j;
}
std::pair<PetscReal,PetscReal> Range(const QuadVertices& c,const Jump& j)
{
    PetscReal lo=std::numeric_limits<PetscReal>::infinity(),hi=-lo;
    for(const auto& p:c){const auto v=j.Signed(p);lo=std::min(lo,v);hi=std::max(hi,v);}
    return {lo,hi};
}
int JumpRegion(const Grid& g,const Reconstruction& r,const Jump& jump)
{
    const auto target=Range(g.cells[Id(g.size,r.Target())],jump);const PetscReal tol=1e-12*g.h;
    if(target.first < -tol&&target.second > tol)return 1; // cut cell
    PetscReal lo=1e30,hi=-lo;const auto& patch=r.RequiredCells();
    for(PetscInt j=patch.begin.j;j<patch.end.j;++j)for(PetscInt i=patch.begin.i;i<patch.end.i;++i){
        const auto range=Range(g.cells[Id(g.size,{i,j})],jump);lo=std::min(lo,range.first);hi=std::max(hi,range.second);}
    return lo>=-tol||hi<=tol ? 3:2; // clean stencil or stencil touching both sides
}
void WriteJumpProfile(const Grid& g,const std::vector<Reconstruction>& r,const Jump& jump,
                       const Options& o)
{
    auto csv=CSV(Sibling(o.output,"_profile"));
    csv<<"distance,distance_over_h,x,y,exact,reconstructed,cell_i,cell_j,weight_large,weight_small,weight_constant\n";
    const Point centre0{{.5,.5}};
    const PetscReal shift=jump.offset-Dot(jump.normal,centre0);
    const Point centre{{.5+shift*jump.normal.p[0],.5+shift*jump.normal.p[1]}};
    // Resolve both a domain-scale profile and offsets O(h) around the jump.
    std::vector<PetscReal> distances;
    for(int k=-600;k<=600;++k)if(k)distances.push_back(k/1000.0);
    for(PetscReal factor:{-4.,-2.,-1.,-.5,-.25,-.1,.1,.25,.5,1.,2.,4.})distances.push_back(factor*g.h);
    std::sort(distances.begin(),distances.end());
    distances.erase(std::unique(distances.begin(),distances.end()),distances.end());
    for(const auto d:distances){const Point p{{centre.p[0]+d*jump.normal.p[0],centre.p[1]+d*jump.normal.p[1]}};
        const auto c=Locate(g,p);if(c<0)continue;PetscReal value,w0;Call(r[c].Evaluate(p,value));Call(r[c].ConstantWeight(w0));
        PetscReal wl=0,ws=0;for(std::size_t k=0;k<r[c].CandidateCount();++k){ReconstructionCandidateInfo info;Call(r[c].GetCandidate(k,info));
            (info.family==ReconstructionFamilyKind::Large?wl:ws)+=info.nonlinearWeight;}
        csv<<d<<','<<d/g.h<<','<<p.p[0]<<','<<p.p[1]<<','<<jump.Value(p,d>0)<<','<<value<<','
           <<c%g.size.i<<','<<c/g.size.i<<','<<wl<<','<<ws<<','<<w0<<'\n';
    }
    Finish(csv);
    auto weights=CSV(Sibling(o.output,"_weights"));
    weights<<"i,j,region,family,candidate,smoothness,nonlinear_weight\n";
    for(std::size_t c=0;c<r.size();++c){const int region=JumpRegion(g,r[c],jump);if(region==3)continue;
        for(std::size_t k=0;k<r[c].CandidateCount();++k){ReconstructionCandidateInfo info;Call(r[c].GetCandidate(k,info));
            weights<<c%g.size.i<<','<<c/g.size.i<<','<<region<<','
                   <<(info.family==ReconstructionFamilyKind::Large?"large":"small")<<','<<k<<','
                   <<info.smoothness<<','<<info.nonlinearWeight<<'\n';}
        PetscReal w0;Call(r[c].ConstantWeight(w0));weights<<c%g.size.i<<','<<c/g.size.i<<','<<region<<",constant,0,0,"<<w0<<'\n';
    }
    Finish(weights);
}

void Discontinuity(const Options& o)
{
    const auto jump=JumpField(o);const auto q=Rule(9),meanRule=Rule(6),edgeRule=Rule(3);
    auto csv=CSV(o.output);auto traces=CSV(Sibling(o.output,"_traces"));
    csv<<"n,h,quantity,region,l1,l2,linf,rate_l2,rate_linf,overshoot,undershoot,transition_width,conservation_error,region_area\n";
    traces<<"n,x,y,left_exact,left_trace,right_exact,right_trace\n";
    const char* names[]={"all","cut","near_jump","smooth_stencil"};
    std::array<Norm,4> previous{};Norm previousFlux;
    PetscReal oldH=0,firstL1=0,lastL1=0,firstTrace=0,lastTrace=0;
    bool hasTraces=false;
    for(PetscInt level=0,n=o.n0;level<o.levels;++level,n*=2){
        Grid g;MakeGrid(n,o,g);std::vector<PetscReal> u(g.cells.size());
        std::vector<std::array<Polygon,2>> split(g.cells.size());
        for(std::size_t c=0;c<g.cells.size();++c){const Polygon poly(g.cells[c].begin(),g.cells[c].end());
            split[c][0]=Clip(poly,jump.normal,jump.offset,false);split[c][1]=Clip(poly,jump.normal,jump.offset,true);
            const auto whole=AreaCentroid(poly),positive=AreaCentroid(split[c][1]),negative=AreaCentroid(split[c][0]);
            Near(positive.first+negative.first,g.areas[c],"Clipped-cell area",2e-12);
            u[c]=positive.first/g.areas[c]+jump.slope*jump.Signed(whole.second);
        }
        std::vector<Reconstruction> r;InitializeReconstructions(g,o,r);Update(g,u,r);
        std::array<Norm,4> norms{};PetscReal overshoot=0,undershoot=0,meanError=0;
        for(std::size_t c=0;c<r.size();++c){const int region=JumpRegion(g,r[c],jump);
            for(int side=0;side<2;++side)PolygonQuadrature(split[c][side],q,[&](const Point& p,PetscReal w){
                PetscReal value;Call(r[c].Evaluate(p,value));const PetscReal exact=jump.Value(p,side!=0);
                norms[0].Add(value-exact,w);norms[region].Add(value-exact,w);
                // Remove the known smooth background before measuring ringing.
                const PetscReal step=value-jump.slope*jump.Signed(p);
                overshoot=std::max(overshoot,step-1);undershoot=std::max(undershoot,-step);
            });
            PetscReal mean=0;Call(IntegrateCell(g.cells[c],meanRule,[&](const Point& p){PetscReal v=0;Call(r[c].Evaluate(p,v));return v;},mean));
            meanError=std::max(meanError,std::abs(mean/g.areas[c]-u[c]));
        }
        Require(meanError<2e-8,"Discontinuous reconstruction must preserve cell averages");
        // Measured transition envelope along the interface normal; no assumption
        // of monotonicity and no false TVD/positivity guarantee for ML-WENO.
        const Point mid{{.5,.5}};const PetscReal shift=jump.offset-Dot(jump.normal,mid);
        const Point centre{{.5+shift*jump.normal.p[0],.5+shift*jump.normal.p[1]}};
        PetscReal lo=1e30,hi=-lo;
        for(int k=-400;k<=400;++k){const PetscReal d=k*g.h/100;
            const Point p{{centre.p[0]+d*jump.normal.p[0],centre.p[1]+d*jump.normal.p[1]}};
            const auto c=Locate(g,p);if(c<0)continue;PetscReal v;Call(r[c].Evaluate(p,v));v-=jump.slope*d;
            if(v>.1&&v<.9){lo=std::min(lo,d);hi=std::max(hi,d);}}
        const PetscReal width=hi>=lo?hi-lo:0;
        for(int region=0;region<4;++region){const auto& e=norms[region];if(!e.weight)continue;
            csv<<n<<','<<g.h<<",reconstruction,"<<names[region]<<','<<e.L1()<<','<<e.L2()<<','<<e.maximum<<','
               <<Rate(previous[region].L2(),e.L2(),oldH,g.h)<<','<<Rate(previous[region].maximum,e.maximum,oldH,g.h)<<','
               <<overshoot<<','<<undershoot<<','<<width<<','<<meanError<<','<<static_cast<PetscReal>(e.weight)<<'\n';
        }
        Norm fluxAway;PetscReal traceError=0;
        for(const auto& face:Faces(g,edgeRule,o.samples)){
            // Flux accuracy is meaningful where the entire sampling/reconstruction
            // support stays on one smooth side of the discontinuity.
            if(JumpRegion(g,r[face.left],jump)==3&&
               (face.right<0||JumpRegion(g,r[face.right],jump)==3)){
                const auto& s=face.sampling;std::vector<PetscReal> sampled(s.SamplePoints().size(),0);
                const bool positive=jump.Signed(s.QuadraturePoints()[0])>0;
                bool smooth=true;
                for(std::size_t a=0;a<sampled.size();++a){
                    if((jump.Signed(s.SamplePoints()[a])>0)!=positive){smooth=false;break;}
                    if(s.SampleSides()[a]==DiffusiveSampleSide::Boundary)continue;
                    const auto owner=s.SampleSides()[a]==DiffusiveSampleSide::Left?face.left:static_cast<std::size_t>(face.right);
                    Call(r[owner].Evaluate(s.SamplePoints()[a],sampled[a]));
                }
                if(smooth){PetscReal numerical=0,length=0;Point normal;
                    const std::vector<PetscReal> kappa(edgeRule.points.size(),o.kappa);
                    if(face.right<0){std::vector<DiffusiveBoundaryValue> bc(edgeRule.points.size());
                        for(std::size_t a=0;a<bc.size();++a)bc[a]={DiffusiveBoundaryType::PrescribedState,jump.Value(s.QuadraturePoints()[a],positive)};
                        Call(IntegrateDiffusiveBoundaryFlux(s,sampled,kappa,bc,0,{},numerical));
                    }else Call(IntegrateDiffusiveFlux(s,sampled,kappa,0,{},numerical));
                    Call(GetEdgeGeometry(face.edge,length,normal));
                    fluxAway.Add(numerical/length+o.kappa*jump.slope*Dot(jump.normal,normal),length);
                }
            }
            if(face.right<0)continue;
            if(std::abs(jump.Signed(face.edge[0]))>1e-12*g.h||std::abs(jump.Signed(face.edge[1]))>1e-12*g.h)continue;
            const bool rightPositive=Dot(jump.normal,face.sampling.Normal())>0;
            for(auto s:edgeRule.points){const Point p=MapEdgePoint(s,face.edge);PetscReal l,rr;
                Call(r[face.left].Evaluate(p,l));Call(r[face.right].Evaluate(p,rr));
                hasTraces=true;
                traceError=std::max({traceError,std::abs(l-jump.Value(p,!rightPositive)),std::abs(rr-jump.Value(p,rightPositive))});
                traces<<n<<','<<p.p[0]<<','<<p.p[1]<<','<<jump.Value(p,!rightPositive)<<','<<l<<','
                      <<jump.Value(p,rightPositive)<<','<<rr<<'\n';}
        }
        if(fluxAway.weight){csv<<n<<','<<g.h<<",diffusive_flux,smooth_stencil,"<<fluxAway.L1()<<','<<fluxAway.L2()<<','
            <<fluxAway.maximum<<','<<Rate(previousFlux.L2(),fluxAway.L2(),oldH,g.h)<<','
            <<Rate(previousFlux.maximum,fluxAway.maximum,oldH,g.h)<<",0,0,0,0,"<<static_cast<PetscReal>(fluxAway.weight)<<'\n';}
        previousFlux=fluxAway;
        std::cout<<"N="<<n<<" jump L1="<<norms[0].L1()<<" L2="<<norms[0].L2()
                 <<" overshoot="<<overshoot<<" undershoot="<<undershoot<<" width="<<width<<'\n';
        if(level==0){firstL1=norms[0].L1();firstTrace=traceError;}
        lastL1=norms[0].L1();lastTrace=traceError;previous=norms;oldH=g.h;
        if(level+1==o.levels)WriteJumpProfile(g,r,jump,o);
    }
    Finish(csv);Finish(traces);
    Require(lastL1<firstL1||lastL1<1e-10,"Global jump reconstruction L1 error must decrease");
    if(hasTraces)Require(lastTrace<firstTrace||lastTrace<1e-9,"Aligned one-sided trace errors must decrease");
    // Pointwise error in cut cells, overshoots and transition width are measured
    // diagnostics; a polynomial cannot resolve a true within-cell jump exactly.
}
}
int main(int argc,char** argv)
{
    return transport_test::Main(argc,argv,[]{const auto o=transport_test::ReadOptions();
        if(o.mode==0)SmoothConvergence(o);else {transport_test::Require(o.mode==1,"Use flux_mode 0 or 1");Discontinuity(o);}});
}
