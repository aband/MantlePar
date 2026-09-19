#include "evolve.h"
#include "visualization.h"
#include "evolution_output.h"
#include <petscao.h>
#include <yaml-cpp/yaml.h>
#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <map>
#include <numeric>
#include <sstream>

namespace mantle::couple {
namespace {
template<class F> PetscErrorCode Local(F&& f) {
    PetscFunctionBeginUser;
    try { const auto e=f(); PetscFunctionReturn(e); }
    catch (const std::exception& e) { SETERRQ(PETSC_COMM_SELF,PETSC_ERR_USER,"%s",e.what()); }
}
PetscErrorCode Agree(MPI_Comm comm,PetscErrorCode e) {
    PetscFunctionBeginUser;
    int local=e,global=0; PetscCallMPI(MPI_Allreduce(&local,&global,1,MPI_INT,MPI_MAX,comm));
    PetscCheck(!global,comm,static_cast<PetscErrorCode>(global),"Phase-coupled evolution failed on at least one rank");
    PetscFunctionReturn(PETSC_SUCCESS);
}
PetscReal Dot(Point a,Point b) { return a.p[0]*b.p[0]+a.p[1]*b.p[1]; }
void Require(bool ok,const std::string& message) { if (!ok) throw std::runtime_error(message); }
std::filesystem::path Output(const Configuration& c,const std::string& name) {
    return input::ResolveOutputPath(c.input,name,c.input.mesh.family);
}
}

TransportVelocities PhaseTransportVelocities(Point solid,Point relative,PetscReal phi,PetscReal theta,
    const phase::PhaseState& a,const phase::PhaseState& b)
{
    const auto cl=.5*(a.cl+b.cl),cs=.5*(a.cs+b.cs);
    const auto bulk=phi*cl+(1-phi)*cs;
    // (2.26): assembled Darcy unknown is phi^(-theta)*(v_l-v_s).
    // Use the flux phi^(1+theta)*u_d directly, including at exact zero melt.
    const auto factor=phi==0?0:std::pow(phi,1+theta);
    TransportVelocities out;
    for (int k=0;k<2;++k) {
        const auto flux=factor*relative.p[k];
        out.mixture.p[k]=solid.p[k]+flux;
        // Pure component C=0: no species flux and no division by zero.
        out.effective.p[k]=solid.p[k]+(bulk>0?cl/bulk*flux:0);
    }
    return out;
}

PetscErrorCode LoadPreheatState(MPI_Comm comm,const Configuration& c,const std::string& directory,InitialState& s)
{
    PetscFunctionBeginUser;
    PetscMPIInt rank; PetscCallMPI(MPI_Comm_rank(comm,&rank));
    std::vector<PetscScalar> fields[2];
    PetscReal sourceTime=0;
    PetscCall(Agree(comm,Local([&]() -> PetscErrorCode {
        if (rank) return PETSC_SUCCESS;
        const auto path=std::filesystem::absolute(directory).lexically_normal();
        Require(std::filesystem::weakly_canonical(path)!=std::filesystem::weakly_canonical(Output(c,"setup.txt").parent_path()),"Source and output directories must differ");
        const auto m=YAML::LoadFile((path/"transport_state.json").string());
        Require(m["schema_version"].as<int>()==1 && m["status"].as<std::string>()=="complete","Preheat manifest must be complete schema 1");
        Require(m["representation"].as<std::string>()=="cell averages" && m["ordering"].as<std::string>()=="j*nx+i","Unsupported preheat representation/order");
        Require(m["H"]["units"].as<std::string>()=="h/(cp*dT)" && m["C"]["units"].as<std::string>()=="fraction","Preheat units do not match");
        Require(m["H"]["file"].as<std::string>()=="cellH1.dat" && m["C"]["file"].as<std::string>()=="cellC1.dat","Unsupported preheat filenames");
        const auto mesh=m["mesh"]; const auto& target=c.input.mesh;
        Require(mesh["family"].as<std::string>()==target.family && mesh["nx"].as<PetscInt>()==target.cells[0] && mesh["ny"].as<PetscInt>()==target.cells[1],"Preheat mesh does not match evolution mesh");
        const auto close=[](double a,double b) { return std::isfinite(a) && std::abs(a-b)<=1e-12*std::max({1.,std::abs(a),std::abs(b)}); };
        const double domain[4]={target.x[0],target.x[1],target.y[0],target.y[1]};
        for (int k=0;k<4;++k) Require(close(mesh["domain"][k].as<double>(),domain[k]),"Preheat domain does not match");
        if (target.family!="rectangular") Require(close(mesh["perturbation"].as<double>(),target.perturbation) && mesh["seed"].as<std::uint64_t>()==target.seed,"Preheat mesh perturbation does not match");
        const auto& d=c.phase.derived();
        Require(close(m["scales"]["enthalpy_Jkg"].as<double>(),d.h0) && close(m["scales"]["length_m"].as<double>(),d.l0) && close(m["scales"]["time_s"].as<double>(),d.t0),"Preheat scales do not match; explicit physical conversion is required");
        sourceTime=m["time"].as<PetscReal>(); Require(std::isfinite(sourceTime),"Nonfinite source time");
        for (int k=0;k<2;++k) {
            std::ifstream file(path/(k?"cellC1.dat":"cellH1.dat")); Require(bool(file),"Cannot open preheat field");
            fields[k].resize(s.mesh.CellCount());
            for (auto& v:fields[k]) {
                double value; Require(bool(file>>value),"Incomplete preheat cell-average matrix");
                Require(std::isfinite(value) && value>=0 && (!k || value<=c.phase.parameters().Xe),"Inadmissible preheat cell average"); v=value;
            }
            file>>std::ws; Require(file.eof(),"Extra entries in preheat cell-average matrix");
        }
        std::filesystem::create_directories(Output(c,"setup.txt").parent_path());
        for (const auto* name:{"transport_state.json","visualization.json","porosity_series.json","evolution_series.json"}) {
            std::ofstream marker(Output(c,name)); marker.exceptions(std::ios::badbit|std::ios::failbit);
            marker<<"{\"status\":\"incomplete\"}\n";
        }
        // Preserve the exact imported matrices beside the evolved matrices.
        for (int k=0;k<2;++k) {
            std::ofstream file(Output(c,k?"starting_cellC.dat":"starting_cellH.dat")); file.exceptions(std::ios::badbit|std::ios::failbit);
            file<<std::setprecision(17);
            for (std::size_t n=0;n<fields[k].size();++n) file<<fields[k][n]<<((n+1)%target.cells[0]?' ':'\n');
        }
        std::ofstream provenance(Output(c,"source_state.json")); provenance.exceptions(std::ios::badbit|std::ios::failbit);
        provenance<<std::setprecision(17)<<"{\"directory\":"<<std::quoted(path.string())<<",\"source_time\":"<<sourceTime
            <<",\"source_time_years\":"<<sourceTime*d.t0/(365*24*3600)<<",\"evolution_start\":"<<c.input.time.start<<"}\n";
        return PETSC_SUCCESS;
    })));
    PetscCallMPI(MPI_Bcast(&sourceTime,1,MPIU_REAL,0,comm));
    Vec natural=nullptr; PetscCall(DMDACreateNaturalVector(s.cellDM,&natural));
    const auto scatter=[&]() -> PetscErrorCode {
        PetscFunctionBeginUser;
        for (int k=0;k<2;++k) {
            if (!rank) {
                std::vector<PetscInt> ids(fields[k].size()); std::iota(ids.begin(),ids.end(),0);
                PetscCall(VecSetValues(natural,static_cast<PetscInt>(ids.size()),ids.data(),fields[k].data(),INSERT_VALUES));
            }
            PetscCall(VecAssemblyBegin(natural)); PetscCall(VecAssemblyEnd(natural));
            Vec target=k?s.composition:s.enthalpy;
            PetscCall(DMDANaturalToGlobalBegin(s.cellDM,natural,INSERT_VALUES,target));
            PetscCall(DMDANaturalToGlobalEnd(s.cellDM,natural,INSERT_VALUES,target));
        }
        PetscFunctionReturn(PETSC_SUCCESS);
    };
    const auto error=scatter(),cleanup=VecDestroy(&natural); PetscCall(error); PetscCall(cleanup);
    s.phaseCoupled=true; s.sourceState=directory; s.sourceTime=sourceTime;
    s.time=c.input.time.start; s.acceptedSteps=0;
    PetscCall(PetscPrintf(comm,"Loaded exact %" PetscInt_FMT "x%" PetscInt_FMT " preheat H/C averages at source t=%g; evolution starts at t=%g\n",
        s.mesh.CellDimensions().i,s.mesh.CellDimensions().j,static_cast<double>(sourceTime),static_cast<double>(s.time)));
    PetscFunctionReturn(PETSC_SUCCESS);
}

namespace {
struct Face {
    MeshIndex left{},right{}; PetscInt ids[2]{};
    bool boundary=false,reverse=false;
    PetscInt edgeId=0;
    PetscReal area[2]{},length=0;
    Point normal{}; EdgeVertices vertices;
    BoundaryPoint boundaryPoint;
    DiffusiveSampling diffusion;
};
struct Rates { PetscReal boundaryH=0,boundaryC=0,sourceH=0,maximumRate=0,divergence=0; };
struct Workspace {
    StateReconstruction reconstruction;
    GaussRule1D rule;
    std::vector<Face> faces;
    PetscBool savePorosity=PETSC_FALSE;
    std::ofstream porositySeries;
    EvolutionOutput movie;
    InitialFlowSamples movieFlow;
    Vec rhsH=nullptr,rhsC=nullptr,oldH=nullptr,oldC=nullptr,volume=nullptr,rate=nullptr,divergence=nullptr;
    PetscErrorCode Destroy() {
        PetscErrorCode first=0; const auto record=[&](PetscErrorCode e) { if (!first) first=e; };
        record(reconstruction.Destroy());
        for (auto p:{&rhsH,&rhsC,&oldH,&oldC,&volume,&rate,&divergence}) record(VecDestroy(p));
        return first;
    }
};
std::string PorosityFilename(PetscMPIInt rank) {
    std::ostringstream name;
    name<<"porosity_series_rank_"<<std::setw(6)<<std::setfill('0')<<rank<<".csv";
    return name.str();
}
PetscErrorCode OpenPorositySeries(MPI_Comm comm,const Configuration& c,Workspace& w) {
    PetscFunctionBeginUser;
    PetscCall(PetscOptionsGetBool(nullptr,nullptr,"-couple_porosity_snapshots",&w.savePorosity,nullptr));
    if (!w.savePorosity) PetscFunctionReturn(PETSC_SUCCESS);
    PetscMPIInt rank; PetscCallMPI(MPI_Comm_rank(comm,&rank));
    PetscCall(Agree(comm,Local([&]() -> PetscErrorCode {
        w.porositySeries.exceptions(std::ios::badbit|std::ios::failbit);
        w.porositySeries.open(Output(c,PorosityFilename(rank)));
        w.porositySeries<<std::setprecision(17)<<"step,time,time_years";
        const auto& mesh=w.reconstruction.Mesh(); const auto r=mesh.OwnedCells();
        for (PetscInt j=r.begin.j;j<r.end.j;++j) for (PetscInt i=r.begin.i;i<r.end.i;++i) {
            PetscInt id; PetscCall(mesh.CellId({i,j},id)); w.porositySeries<<",phi_"<<id;
        }
        w.porositySeries<<'\n';
        return PETSC_SUCCESS;
    })));
    PetscFunctionReturn(PETSC_SUCCESS);
}
PetscErrorCode WritePorosityFrame(MPI_Comm comm,const Configuration& c,const InitialState& s,Workspace& w) {
    PetscFunctionBeginUser;
    if (!w.savePorosity) PetscFunctionReturn(PETSC_SUCCESS);
    PetscCall(Agree(comm,Local([&]() -> PetscErrorCode {
        w.porositySeries<<s.acceptedSteps<<','<<s.time<<','<<s.time*c.phase.derived().t0/(365*24*3600);
        const auto& mesh=w.reconstruction.Mesh(); const auto r=mesh.OwnedCells();
        // RHS has already updated and limited this accepted state's reconstruction.
        // Match the static phase plots: phase evaluated at mapped cell centers.
        for (PetscInt j=r.begin.j;j<r.end.j;++j) for (PetscInt i=r.begin.i;i<r.end.i;++i) {
            QuadVertices corners; PetscCall(mesh.GetCellCorners({i,j},corners));
            const auto p=MapCellPoint(Point{{0,0}},corners);
            PetscReal H,C; PetscCall(w.reconstruction.Evaluate({i,j},p,H,C));
            const auto phi=c.phase.evaluate(H,C,c.pressure(p,s.time)).phil;
            PetscCheck(std::isfinite(phi) && phi>=0 && phi<1,PETSC_COMM_SELF,PETSC_ERR_FP,"Invalid snapshot porosity");
            w.porositySeries<<','<<phi;
        }
        w.porositySeries<<'\n'; w.porositySeries.flush();
        return PETSC_SUCCESS;
    })));
    PetscFunctionReturn(PETSC_SUCCESS);
}
PetscErrorCode FinishPorositySeries(MPI_Comm comm,const Configuration& c,const InitialState& s,Workspace& w) {
    PetscFunctionBeginUser;
    if (!w.savePorosity) PetscFunctionReturn(PETSC_SUCCESS);
    PetscCall(Agree(comm,Local([&]() -> PetscErrorCode { w.porositySeries.close(); return PETSC_SUCCESS; })));
    PetscMPIInt rank,ranks; PetscCallMPI(MPI_Comm_rank(comm,&rank)); PetscCallMPI(MPI_Comm_size(comm,&ranks));
    PetscCall(Agree(comm,Local([&]() -> PetscErrorCode {
        if (rank) return PETSC_SUCCESS;
        const auto temporary=Output(c,"porosity_series.json.tmp");
        std::ofstream manifest; manifest.exceptions(std::ios::badbit|std::ios::failbit); manifest.open(temporary);
        manifest<<std::setprecision(17)<<"{\"schema_version\":1,\"status\":\"complete\",\"sampling\":\"cell centers\",\"units\":\"fraction\","
            <<"\"frames\":"<<s.acceptedSteps+1<<",\"cell_count\":"<<s.mesh.CellCount()
            <<",\"time_start\":"<<c.input.time.start<<",\"time_end\":"<<s.time
            <<",\"time_scale_years\":"<<c.phase.derived().t0/(365*24*3600)<<",\"files\":[";
        for (PetscMPIInt r=0;r<ranks;++r) {
            if (r) manifest<<',';
            manifest<<'"'<<PorosityFilename(r)<<'"';
        }
        manifest<<"]}\n"; manifest.close();
        std::filesystem::rename(temporary,Output(c,"porosity_series.json"));
        return PETSC_SUCCESS;
    })));
    PetscFunctionReturn(PETSC_SUCCESS);
}
PetscErrorCode Build(MPI_Comm comm,const Configuration& c,const InitialState& s,Workspace& w) {
    PetscFunctionBeginUser;
    PetscCall(w.reconstruction.Initialize(comm,s));
    PetscCall(CreateGaussRule(c.input.quadrature.edgePoints,w.rule));
    for (auto p:{&w.rhsH,&w.rhsC,&w.oldH,&w.oldC,&w.volume,&w.rate,&w.divergence}) PetscCall(VecDuplicate(s.enthalpy,p));
    PetscCall(VecSet(w.volume,0));
    std::vector<PetscInt> ids,volumeIds; std::vector<PetscScalar> volumes;
    const auto& mesh=w.reconstruction.Mesh();
    PetscCall(Agree(comm,Local([&]() -> PetscErrorCode {
        for (auto edge:mesh.OwnedEdgeIds()) {
            EdgeTopology t; PetscCall(mesh.GetEdgeTopology(edge,t));
            Face f; f.edgeId=edge; f.boundary=t.IsBoundary(); f.reverse=!t.leftCell;
            f.left=f.reverse?*t.rightCell:*t.leftCell;
            if (!f.boundary) f.right=*t.rightCell;
            PetscCall(mesh.GetEdgeVertices(edge,f.vertices));
            if (f.reverse) std::swap(f.vertices[0],f.vertices[1]);
            PetscCall(GetEdgeGeometry(f.vertices,f.length,f.normal));
            QuadVertices left,right; PetscCall(mesh.GetCellCorners(f.left,left));
            PetscCall(mesh.GetCellArea(f.left,f.area[0])); PetscCall(mesh.CellId(f.left,f.ids[0]));
            if (f.boundary) {
                f.ids[1]=f.ids[0]; f.area[1]=f.area[0];
                f.boundaryPoint.cell=f.left; f.boundaryPoint.edgeId=edge; f.boundaryPoint.outwardNormal=f.normal;
                if (t.axis==EdgeAxis::AlongI) {
                    f.boundaryPoint.side=t.vertices[0].j==0?CellSide::Bottom:CellSide::Top; f.boundaryPoint.sideEdge=f.left.i;
                } else { f.boundaryPoint.side=t.vertices[0].i==0?CellSide::Left:CellSide::Right; f.boundaryPoint.sideEdge=f.left.j; }
                PetscCall(CreateBoundaryDiffusiveSampling(f.vertices,left,w.rule,c.sampling,f.diffusion));
            } else {
                PetscCall(mesh.GetCellCorners(f.right,right)); PetscCall(mesh.GetCellArea(f.right,f.area[1]));
                PetscCall(mesh.CellId(f.right,f.ids[1]));
                PetscCall(CreateInteriorDiffusiveSampling(f.vertices,left,right,w.rule,c.sampling,f.diffusion));
            }
            ids.push_back(f.ids[0]); ids.push_back(f.ids[1]); w.faces.push_back(std::move(f));
        }
        const auto r=mesh.OwnedCells();
        for (PetscInt j=r.begin.j;j<r.end.j;++j) for (PetscInt i=r.begin.i;i<r.end.i;++i) {
            PetscInt id; PetscReal area; PetscCall(mesh.CellId({i,j},id)); PetscCall(mesh.GetCellArea({i,j},area));
            volumeIds.push_back(id); volumes.push_back(area);
        }
        return PETSC_SUCCESS;
    })));
    AO ao; PetscCall(DMDAGetAO(s.cellDM,&ao));
    PetscCall(AOApplicationToPetsc(ao,static_cast<PetscInt>(ids.size()),ids.data()));
    PetscCall(AOApplicationToPetsc(ao,static_cast<PetscInt>(volumeIds.size()),volumeIds.data()));
    for (std::size_t k=0;k<w.faces.size();++k) { w.faces[k].ids[0]=ids[2*k]; w.faces[k].ids[1]=ids[2*k+1]; }
    PetscCall(VecSetValues(w.volume,static_cast<PetscInt>(volumes.size()),volumeIds.data(),volumes.data(),INSERT_VALUES));
    PetscCall(VecAssemblyBegin(w.volume)); PetscCall(VecAssemblyEnd(w.volume));
    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode Phase(const Configuration& c,const InitialState& s,const StateReconstruction& r,
    MeshIndex cell,Point p,phase::PhaseState& phase,PetscReal& H,PetscReal& C) {
    PetscFunctionBeginUser;
    PetscCall(r.Evaluate(cell,p,H,C)); phase=c.phase.evaluate(H,C,c.pressure(p,s.time));
    PetscCheck(phase.TDp>=0 && phase.phil<1,PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,
        "Coupled Darcy-Stokes requires nonnegative temperature and porosity < 1");
    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode RHS(MPI_Comm comm,const Configuration& c,InitialState& s,Workspace& w,Rates& totals) {
    PetscFunctionBeginUser;
    PetscCall(w.reconstruction.Update(s)); PetscCall(w.reconstruction.Limit(c));
    PetscCall(SolveReconstructedFlow(comm,c,s,w.reconstruction));
    InitialFlowSamples flow; PetscCall(SampleInitialFlow(comm,c,s,flow));
    for (auto v:{w.rhsH,w.rhsC,w.rate,w.divergence,s.temperature,s.porosity}) PetscCall(VecSet(v,0));
    std::vector<PetscInt> ids; std::vector<PetscScalar> heat,phi,temperature,sourceRates;
    PetscReal local[3]{};
    const auto LD=c.phase.derived().LD;
    const auto cooling=c.phase.parameters().alpha0*c.phase.derived().l0*c.phase.parameters().g/c.phase.parameters().cp;
    PetscCall(Agree(comm,Local([&]() -> PetscErrorCode {
        std::map<PetscInt,std::vector<FlowPointSample>> velocities;
        for (const auto& p:flow.edges) { auto& points=velocities[p.edgeId]; points.resize(w.rule.points.size()); points.at(p.q)=p; }
        for (const auto& f:w.faces) {
            PetscReal fluxH=0,fluxC=0,fluxMixture=0,rate=0;
            for (std::size_t q=0;q<w.rule.points.size();++q) {
                const auto& velocity=velocities.at(f.edgeId).at(f.reverse?w.rule.points.size()-1-q:q);
                const auto p=MapEdgePoint(w.rule.points[q],f.vertices);
                PetscReal hL,cL,hR,cR; phase::PhaseState a,b;
                PetscCall(Phase(c,s,w.reconstruction,f.left,p,a,hL,cL));
                if (f.boundary) { b=a; hR=hL; cR=cL; }
                else PetscCall(Phase(c,s,w.reconstruction,f.right,p,b,hR,cR));
                const auto edgePhi=velocity.flowPorosity;
                const auto v=PhaseTransportVelocities(velocity.stokesVelocity,velocity.darcyVelocity,edgePhi,c.flow.material.theta,a,b);
                const auto vn=Dot(v.mixture,f.normal),ve=Dot(v.effective,f.normal),vs=Dot(velocity.stokesVelocity,f.normal);
                const auto alphaH=c.stabilizer.mode==LaxFriedrichsMode::Global?c.stabilizer.globalSpeed:std::abs(vn);
                const auto alphaC=c.stabilizer.mode==LaxFriedrichsMode::Global?c.stabilizer.globalSpeed:std::abs(ve);
                PetscCheck(alphaH+1e-14>=std::abs(vn) && alphaC+1e-14>=std::abs(ve),PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Global LF speed is below the stage flow speed");
                PetscReal fh=.5*(vn*(a.TDp+b.TDp)-alphaH*(b.TDp-a.TDp))-LD*(1-edgePhi)*vs;
                PetscReal fc=.5*(ve*(cL+cR)-alphaC*(cR-cL));
                if (f.boundary) {
                    auto bp=f.boundaryPoint; bp.position=p; bp.time=s.time;
                    AdvectiveBoundaryValue hb,cb;
                    PetscCall(EvaluateAdvectionBoundary(c.boundary.enthalpy,bp,vn,hb));
                    PetscCall(EvaluateAdvectionBoundary(c.boundary.composition,bp,ve,cb));
                    // H and C can have different characteristic directions.
                    // Respect the C boundary policy when evaluating incoming T.
                    const auto inflowC=cb.type==AdvectiveBoundaryType::PrescribedState?cb.value:cL;
                    auto T=a.TDp;
                    if (hb.type==AdvectiveBoundaryType::PrescribedState) T=c.phase.evaluate(hb.value,inflowC,c.pressure(p,s.time)).TDp;
                    fh=hb.type==AdvectiveBoundaryType::ZeroFlux?0:vn*T-LD*(1-edgePhi)*vs;
                    fc=cb.type==AdvectiveBoundaryType::ZeroFlux?0:ve*(cb.type==AdvectiveBoundaryType::PrescribedState?cb.value:cL);
                }
                const auto weight=.5*f.length*w.rule.weights[q];
                fluxH+=weight*fh; fluxC+=weight*fc; fluxMixture+=weight*vn;
                // Frozen-stage transport bound; include thermal/compositional
                // phase sensitivities and latent transport, then recheck stage 2.
                const auto sensitivity=1+std::max(std::abs(a.dTD_dCD),std::abs(b.dTD_dCD));
                PetscCheck(std::isfinite(sensitivity),PETSC_COMM_SELF,PETSC_ERR_FP,"Singular phase derivative in CFL estimate");
                rate+=weight*(alphaC+alphaH+2*std::abs(vs))*sensitivity;
            }
            const auto nq=w.rule.points.size();
            std::vector<PetscReal> samples(f.diffusion.SamplePoints().size(),0),kappa(nq,c.thermalDiffusivity);
            for (std::size_t q=0;q<samples.size();++q) {
                const auto side=f.diffusion.SampleSides()[q];
                if (side!=DiffusiveSampleSide::Boundary) {
                    phase::PhaseState phase; PetscReal H,C;
                    PetscCall(Phase(c,s,w.reconstruction,side==DiffusiveSampleSide::Left?f.left:f.right,f.diffusion.SamplePoints()[q],phase,H,C));
                    samples[q]=phase.TDp;
                }
            }
            DiffusiveEdgeFluxResult diffusion;
            if (f.boundary) {
                std::vector<DiffusiveBoundaryValue> bc(nq);
                for (std::size_t q=0;q<nq;++q) { auto bp=f.boundaryPoint; bp.position=f.diffusion.QuadraturePoints()[q]; bp.time=s.time; PetscCall(EvaluateThermalBoundary(c.boundary.temperature,bp,bc[q])); }
                PetscCall(IntegrateDiffusiveBoundaryFluxWithDerivatives(f.diffusion,samples,kappa,bc,s.time,{},diffusion));
            } else PetscCall(IntegrateDiffusiveFluxWithDerivatives(f.diffusion,samples,kappa,s.time,{},diffusion));
            fluxH+=diffusion.flux;
            // Absolute derivative sum bounds the normal-line diffusion stencil;
            // dT/dH <= 1. C has no diffusion equation in the low-D_l model.
            for (auto d:diffusion.derivativeSamples) rate+=std::abs(d);
            const PetscScalar h[2]={-fluxH/f.area[0],fluxH/f.area[1]},co[2]={-fluxC/f.area[0],fluxC/f.area[1]},
                rates[2]={rate/f.area[0],rate/f.area[1]},div[2]={fluxMixture/f.area[0],-fluxMixture/f.area[1]};
            const auto count=f.boundary?1:2;
            PetscCall(VecSetValues(w.rhsH,count,f.ids,h,ADD_VALUES)); PetscCall(VecSetValues(w.rhsC,count,f.ids,co,ADD_VALUES));
            PetscCall(VecSetValues(w.rate,count,f.ids,rates,ADD_VALUES)); PetscCall(VecSetValues(w.divergence,count,f.ids,div,ADD_VALUES));
            if (f.boundary) { local[0]+=fluxH; local[1]+=fluxC; }
        }
        // Full cell quadrature for cooling and phase diagnostics (the document's
        // midpoint approximation is not necessary with available Gauss samples).
        const auto& mesh=w.reconstruction.Mesh(); const auto nx=mesh.CellDimensions().i;
        for (const auto& p:flow.cells) {
            const MeshIndex cell{p.cellId%nx,p.cellId/nx}; PetscReal H,C,area; phase::PhaseState a;
            PetscCall(Phase(c,s,w.reconstruction,cell,p.position,a,H,C)); PetscCall(mesh.GetCellArea(cell,area));
            const auto v=PhaseTransportVelocities(p.stokesVelocity,p.darcyVelocity,p.flowPorosity,c.flow.material.theta,a,a);
            const auto source=-cooling*a.TDp*v.mixture.p[1]; // y up, gravity down.
            ids.push_back(p.cellId); heat.push_back(p.weight/area*source);
            temperature.push_back(p.weight/area*a.TDp); phi.push_back(p.weight/area*a.phil);
            sourceRates.push_back(p.weight/area*cooling*std::abs(v.mixture.p[1]));
            local[2]+=p.weight*source;
        }
        return PETSC_SUCCESS;
    })));
    AO ao; PetscCall(DMDAGetAO(s.cellDM,&ao)); PetscCall(AOApplicationToPetsc(ao,static_cast<PetscInt>(ids.size()),ids.data()));
    PetscCall(VecSetValues(w.rhsH,static_cast<PetscInt>(ids.size()),ids.data(),heat.data(),ADD_VALUES));
    PetscCall(VecSetValues(w.rate,static_cast<PetscInt>(ids.size()),ids.data(),sourceRates.data(),ADD_VALUES));
    PetscCall(VecSetValues(s.temperature,static_cast<PetscInt>(ids.size()),ids.data(),temperature.data(),ADD_VALUES));
    PetscCall(VecSetValues(s.porosity,static_cast<PetscInt>(ids.size()),ids.data(),phi.data(),ADD_VALUES));
    for (auto v:{w.rhsH,w.rhsC,w.rate,w.divergence,s.temperature,s.porosity}) { PetscCall(VecAssemblyBegin(v)); PetscCall(VecAssemblyEnd(v)); }
    PetscReal global[3]; PetscCallMPI(MPI_Allreduce(local,global,3,MPIU_REAL,MPI_SUM,comm));
    totals.boundaryH=global[0]; totals.boundaryC=global[1]; totals.sourceH=global[2];
    PetscCall(VecMax(w.rate,nullptr,&totals.maximumRate)); PetscCall(VecNorm(w.divergence,NORM_INFINITY,&totals.divergence));
    PetscCheck(std::isfinite(totals.maximumRate),comm,PETSC_ERR_FP,"Nonfinite coupled stage rate");
    if (w.movie.Enabled()) w.movieFlow=std::move(flow);
    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode Admissible(MPI_Comm comm,const Configuration& c,const InitialState& s,bool& valid) {
    PetscFunctionBeginUser;
    PetscReal hmin,hmax,cmin,cmax;
    PetscCall(VecMin(s.enthalpy,nullptr,&hmin)); PetscCall(VecMax(s.enthalpy,nullptr,&hmax));
    PetscCall(VecMin(s.composition,nullptr,&cmin)); PetscCall(VecMax(s.composition,nullptr,&cmax));
    valid=std::isfinite(hmin)&&std::isfinite(hmax)&&std::isfinite(cmin)&&std::isfinite(cmax)&&hmin>=0&&cmin>=0&&cmax<=c.phase.parameters().Xe;
    PetscFunctionReturn(PETSC_SUCCESS);
}
PetscErrorCode Integral(Vec field,Vec volume,PetscReal& value) {
    PetscFunctionBeginUser;
    PetscScalar sum; PetscCall(VecDot(field,volume,&sum)); value=PetscRealPart(sum);
    PetscFunctionReturn(PETSC_SUCCESS);
}
PetscErrorCode WriteRow(MPI_Comm comm,const Configuration& c,const InitialState& s,Workspace& w,std::ofstream& history,
    const Rates& rates,PetscReal dt,PetscReal courant,PetscReal balanceH,PetscReal balanceC,PetscInt retries) {
    PetscFunctionBeginUser;
    PetscReal H,C,hmin,hmax,cmin,cmax,pmin,pmax;
    PetscCall(Integral(s.enthalpy,w.volume,H)); PetscCall(Integral(s.composition,w.volume,C));
    PetscCall(VecMin(s.enthalpy,nullptr,&hmin)); PetscCall(VecMax(s.enthalpy,nullptr,&hmax));
    PetscCall(VecMin(s.composition,nullptr,&cmin)); PetscCall(VecMax(s.composition,nullptr,&cmax));
    PetscCall(VecMin(s.porosity,nullptr,&pmin)); PetscCall(VecMax(s.porosity,nullptr,&pmax));
    PetscMPIInt rank; PetscCallMPI(MPI_Comm_rank(comm,&rank));
    PetscCall(Agree(comm,Local([&]() -> PetscErrorCode {
        if (!rank) {
            const auto& d=c.phase.derived(); const auto& m=c.input.mesh; const auto area=(m.x[1]-m.x[0])*(m.y[1]-m.y[0]);
            history<<s.acceptedSteps<<','<<s.time<<','<<s.time*d.t0/(365*24*3600)<<','<<dt<<','<<courant
                <<','<<H<<','<<C<<','<<H/area*d.h0<<','<<C/area<<','<<hmin*d.h0<<','<<hmax*d.h0
                <<','<<cmin<<','<<cmax<<','<<pmin<<','<<pmax<<','<<rates.boundaryH<<','<<rates.boundaryC
                <<','<<rates.sourceH<<','<<balanceH<<','<<balanceC<<','<<rates.divergence
                <<','<<s.flowReport.relativeTrueResidualNorm<<','<<retries<<'\n'; history.flush();
        }
        return PETSC_SUCCESS;
    })));
    PetscCall(WritePorosityFrame(comm,c,s,w));
    PetscCall(w.movie.Write(comm,c,s,w.reconstruction,w.movieFlow));
    if (s.acceptedSteps==0 || s.acceptedSteps%10==0 || s.time>=c.input.time.end)
        PetscCall(PetscPrintf(comm,"  Coupled step %" PetscInt_FMT ": t=%.8g dt=%.3g phi=[%.4g,%.4g] C=[%.4g,%.4g] flow residual=%.2e\n",
            s.acceptedSteps,static_cast<double>(s.time),static_cast<double>(dt),static_cast<double>(pmin),static_cast<double>(pmax),
            static_cast<double>(cmin),static_cast<double>(cmax),static_cast<double>(s.flowReport.relativeTrueResidualNorm)));
    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode Run(MPI_Comm comm,const Configuration& c,InitialState& s,Workspace& w) {
    PetscFunctionBeginUser;
    PetscCheck(s.phaseCoupled && !c.flow.dryPorosity,comm,PETSC_ERR_ARG_WRONGSTATE,"Load a phase-coupled starting state first");
    PetscCheck(c.input.time.cfl>0 && c.input.time.cfl<=1,comm,PETSC_ERR_ARG_OUTOFRANGE,"Coupled CFL must be in (0,1]");
    PetscCall(Build(comm,c,s,w));
    PetscCall(OpenPorositySeries(comm,c,w));
    PetscCall(w.movie.Open(comm,c));
    // Backups always describe the latest accepted state, including on errors.
    PetscCall(VecCopy(s.enthalpy,w.oldH)); PetscCall(VecCopy(s.composition,w.oldC));
    Rates current; PetscCall(RHS(comm,c,s,w,current));
    PetscMPIInt rank; PetscCallMPI(MPI_Comm_rank(comm,&rank));
    std::ofstream history;
    PetscCall(Agree(comm,Local([&]() -> PetscErrorCode {
        if (!rank) {
            history.exceptions(std::ios::badbit|std::ios::failbit); history.open(Output(c,"evolution_history.csv"));
            history<<std::setprecision(17)<<"step,time,time_years,dt,courant,H_integral,C_integral,h_mean_J_kg,C_mean,h_min_J_kg,h_max_J_kg,C_min,C_max,phi_min,phi_max,outward_H_flux,outward_C_flux,adiabatic_H_source,H_balance_error,C_balance_error,mixture_divergence_max,flow_relative_residual,retries\n";
        }
        return PETSC_SUCCESS;
    })));
    PetscCall(WriteRow(comm,c,s,w,history,current,0,0,0,0,0));
    const auto& t=c.input.time;
    while (s.time<t.end) {
        const auto remaining=t.end-s.time;
        if (remaining<=32*std::numeric_limits<PetscReal>::epsilon()*std::max(PetscReal(1),std::abs(t.end))) { s.time=t.end; break; }
        PetscCheck(s.acceptedSteps<t.maximumSteps,comm,PETSC_ERR_NOT_CONVERGED,"Maximum coupled steps reached before time.end");
        PetscReal oldH,oldC; PetscCall(Integral(s.enthalpy,w.volume,oldH)); PetscCall(Integral(s.composition,w.volume,oldC));
        const auto start=s.time;
        PetscReal dt=std::min(remaining,t.control=="fixed"?t.initialStep:(s.acceptedSteps==0?t.initialStep:t.maximumStep));
        const auto limit=current.maximumRate>0?t.cfl/current.maximumRate:t.maximumStep;
        PetscCheck(t.control!="fixed" || dt<=limit*(1+1e-12),comm,PETSC_ERR_ARG_OUTOFRANGE,"Fixed time step exceeds the phase-coupled CFL bound; use control: cfl");
        if (t.control!="fixed") dt=std::min(dt,limit);
        // Save the first-stage RHS: a rejected second stage overwrites w.rhs*.
        Vec firstH=nullptr,firstC=nullptr;
        PetscCall(VecDuplicate(w.rhsH,&firstH)); PetscCall(VecDuplicate(w.rhsC,&firstC));
        PetscCall(VecCopy(w.rhsH,firstH)); PetscCall(VecCopy(w.rhsC,firstC));
        const auto firstRates=current;
        Rates second; PetscInt retries=0; PetscReal balanceH=0,balanceC=0,courant=0;
        const auto step=[&]() -> PetscErrorCode {
            PetscFunctionBeginUser;
            for (;;) {
                PetscCheck(dt>=std::min(remaining,t.minimumStep) && retries<40,comm,PETSC_ERR_NOT_CONVERGED,"Coupled step cannot satisfy admissibility/CFL above minimum step");
                PetscCall(VecWAXPY(s.enthalpy,dt,firstH,w.oldH)); PetscCall(VecWAXPY(s.composition,dt,firstC,w.oldC));
                s.time=start+dt;
                bool valid; PetscCall(Admissible(comm,c,s,valid));
                if (valid) {
                    PetscCall(RHS(comm,c,s,w,second));
                    courant=dt*std::max(current.maximumRate,second.maximumRate);
                    valid=courant<=t.cfl*(1+1e-12);
                }
                if (valid) {
                    PetscCall(VecAXPY(s.enthalpy,dt,w.rhsH)); PetscCall(VecAXPY(s.composition,dt,w.rhsC));
                    PetscCall(VecAXPBY(s.enthalpy,.5,.5,w.oldH)); PetscCall(VecAXPBY(s.composition,.5,.5,w.oldC));
                    PetscCall(Admissible(comm,c,s,valid));
                }
                if (valid) break;
                PetscCheck(t.control!="fixed",comm,PETSC_ERR_ARG_OUTOFRANGE,"Fixed coupled step failed stage admissibility/CFL");
                dt*=.5; ++retries;
            }
            PetscReal H,C; PetscCall(Integral(s.enthalpy,w.volume,H)); PetscCall(Integral(s.composition,w.volume,C));
            balanceH=H-oldH+.5*dt*(current.boundaryH+second.boundaryH-current.sourceH-second.sourceH);
            balanceC=C-oldC+.5*dt*(current.boundaryC+second.boundaryC);
            PetscCheck(std::abs(balanceH)<1e-10*std::max(PetscReal(1),std::abs(oldH)) && std::abs(balanceC)<1e-10,
                comm,PETSC_ERR_PLIB,"Coupled finite-volume conservation balance failed");
            PetscFunctionReturn(PETSC_SUCCESS);
        };
        auto error=step(); const auto cleanH=VecDestroy(&firstH),cleanC=VecDestroy(&firstC);
        if (error) { s.time=start; PetscCall(VecCopy(w.oldH,s.enthalpy)); PetscCall(VecCopy(w.oldC,s.composition)); PetscCall(error); }
        PetscCall(cleanH); PetscCall(cleanC);
        // Resolve mechanics for the accepted state, including the final frame.
        // Do not publish an accepted step with stale stage-one flow or phases.
        error=RHS(comm,c,s,w,current);
        if (error) { s.time=start; PetscCall(VecCopy(w.oldH,s.enthalpy)); PetscCall(VecCopy(w.oldC,s.composition)); PetscCall(error); }
        ++s.acceptedSteps;
        PetscCall(VecCopy(s.enthalpy,w.oldH)); PetscCall(VecCopy(s.composition,w.oldC));
        auto reported=current;
        reported.boundaryH=.5*(firstRates.boundaryH+second.boundaryH);
        reported.boundaryC=.5*(firstRates.boundaryC+second.boundaryC);
        reported.sourceH=.5*(firstRates.sourceH+second.sourceH);
        PetscCall(WriteRow(comm,c,s,w,history,reported,dt,courant,balanceH,balanceC,retries));
    }
    PetscCall(FinishPorositySeries(comm,c,s,w));
    PetscCall(w.movie.Finish(comm,c,s));
    PetscFunctionReturn(PETSC_SUCCESS);
}
}
PetscErrorCode AdvancePhaseCoupled(MPI_Comm comm,const Configuration& c,InitialState& s) {
    PetscFunctionBeginUser;
    Workspace w;
    const auto error=Run(comm,c,s,w),cleanup=w.Destroy();
    if (error) s.flowReport.converged=false; // Never export a failed stage as a complete state.
    PetscCall(error); PetscCall(cleanup); PetscFunctionReturn(PETSC_SUCCESS);
}
}
