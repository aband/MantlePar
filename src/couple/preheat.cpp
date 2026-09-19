#include "preheat.h"
#include "state_reconstruction.h"
#include "visualization.h"
#include <petscao.h>
#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <map>

namespace mantle::couple {
namespace {
template<class F> PetscErrorCode Local(F&& f)
{
    PetscFunctionBeginUser;
    try { const auto e=f(); PetscFunctionReturn(e); }
    catch (const std::exception& e) { SETERRQ(PETSC_COMM_SELF,PETSC_ERR_USER,"%s",e.what()); }
}
PetscErrorCode Agree(MPI_Comm comm,PetscErrorCode error)
{
    PetscFunctionBeginUser;
    int local=error,global=0;
    PetscCallMPI(MPI_Allreduce(&local,&global,1,MPI_INT,MPI_MAX,comm));
    PetscCheck(!global,comm,static_cast<PetscErrorCode>(global),"Legacy preheat failed on at least one rank");
    PetscFunctionReturn(PETSC_SUCCESS);
}
PetscReal Dot(Point a,Point b) { return a.p[0]*b.p[0]+a.p[1]*b.p[1]; }
struct Face {
    MeshIndex left{},right{};
    bool boundary=false;
    PetscInt ids[2]{}; // PETSc scalar indices, converted from natural IDs once.
    PetscReal area[2]{},length=0;
    Point normal{};
    EdgeVertices vertices;
    DiffusiveSampling diffusion;
    std::vector<Point> velocity;
    BoundaryPoint boundaryPoint;
};
struct Workspace {
    StateReconstruction reconstruction;
    GaussRule1D rule;
    std::vector<Face> faces;
    Vec rhs=nullptr,candidate=nullptr,volume=nullptr,rate=nullptr;
    PetscErrorCode Destroy() {
        PetscErrorCode first=0;
        const auto record=[&](PetscErrorCode e) { if (!first) first=e; };
        record(reconstruction.Destroy()); record(VecDestroy(&rhs)); record(VecDestroy(&candidate));
        record(VecDestroy(&volume)); record(VecDestroy(&rate)); return first;
    }
};
PetscErrorCode ThermalData(const Configuration& c,const Face& f,PetscReal time,
                           std::vector<DiffusiveBoundaryValue>& data)
{
    PetscFunctionBeginUser;
    data.resize(f.diffusion.QuadraturePoints().size());
    for (std::size_t q=0;q<data.size();++q) {
        auto p=f.boundaryPoint; p.time=time; p.position=f.diffusion.QuadraturePoints()[q];
        PetscCall(EvaluateThermalBoundary(c.boundary.temperature,p,data[q]));
    }
    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode Build(MPI_Comm comm,const Configuration& c,const InitialState& s,Workspace& w)
{
    PetscFunctionBeginUser;
    PetscCall(w.reconstruction.Initialize(comm,s));
    PetscCall(Agree(comm,CreateGaussRule(c.input.quadrature.edgePoints,w.rule)));
    PetscCall(VecDuplicate(s.enthalpy,&w.rhs)); PetscCall(VecDuplicate(s.enthalpy,&w.candidate));
    PetscCall(VecDuplicate(s.enthalpy,&w.volume)); PetscCall(VecDuplicate(s.enthalpy,&w.rate));
    PetscCall(VecSet(w.volume,0)); PetscCall(VecSet(w.rate,0));
    InitialFlowSamples flow; PetscCall(SampleInitialFlow(comm,c,s,flow));
    const auto& mesh=w.reconstruction.Mesh();
    std::vector<PetscInt> ids,volumeIds;
    std::vector<PetscScalar> volumes;
    PetscCall(Agree(comm,Local([&]() -> PetscErrorCode {
        std::map<PetscInt,std::vector<Point>> velocity;
        for (const auto& p:flow.edges) {
            auto& points=velocity[p.edgeId]; points.resize(w.rule.points.size());
            points.at(p.q)=p.stokesVelocity;
        }
        for (const auto edge:mesh.OwnedEdgeIds()) {
            EdgeTopology topology; PetscCall(mesh.GetEdgeTopology(edge,topology));
            Face f; f.boundary=topology.IsBoundary();
            const bool reverse=!topology.leftCell;
            f.left=reverse?*topology.rightCell:*topology.leftCell;
            if (!f.boundary) f.right=*topology.rightCell;
            PetscCall(mesh.GetEdgeVertices(edge,f.vertices));
            if (reverse) std::swap(f.vertices[0],f.vertices[1]);
            PetscCall(GetEdgeGeometry(f.vertices,f.length,f.normal));
            f.velocity=velocity.at(edge);
            if (reverse) std::reverse(f.velocity.begin(),f.velocity.end());
            QuadVertices left,right; PetscCall(mesh.GetCellCorners(f.left,left));
            PetscCall(mesh.GetCellArea(f.left,f.area[0]));
            PetscCall(mesh.CellId(f.left,f.ids[0])); ids.push_back(f.ids[0]);
            if (f.boundary) {
                f.ids[1]=f.ids[0]; f.area[1]=f.area[0];
                f.boundaryPoint.cell=f.left; f.boundaryPoint.edgeId=edge; f.boundaryPoint.outwardNormal=f.normal;
                if (topology.axis==EdgeAxis::AlongI) {
                    f.boundaryPoint.side=topology.vertices[0].j==0?CellSide::Bottom:CellSide::Top;
                    f.boundaryPoint.sideEdge=f.left.i;
                } else {
                    f.boundaryPoint.side=topology.vertices[0].i==0?CellSide::Left:CellSide::Right;
                    f.boundaryPoint.sideEdge=f.left.j;
                }
                PetscCall(CreateBoundaryDiffusiveSampling(f.vertices,left,w.rule,c.sampling,f.diffusion));
            } else {
                PetscCall(mesh.GetCellCorners(f.right,right)); PetscCall(mesh.GetCellArea(f.right,f.area[1]));
                PetscCall(mesh.CellId(f.right,f.ids[1]));
                PetscCall(CreateInteriorDiffusiveSampling(f.vertices,left,right,w.rule,c.sampling,f.diffusion));
            }
            ids.push_back(f.ids[1]); w.faces.push_back(std::move(f));
        }
        const auto r=mesh.OwnedCells();
        for (PetscInt j=r.begin.j;j<r.end.j;++j) for (PetscInt i=r.begin.i;i<r.end.i;++i) {
            PetscInt id; PetscReal area; PetscCall(mesh.CellId({i,j},id)); PetscCall(mesh.GetCellArea({i,j},area));
            volumeIds.push_back(id); volumes.push_back(area);
        }
        return PETSC_SUCCESS;
    })));
    AO ao=nullptr; PetscCall(DMDAGetAO(s.cellDM,&ao));
    PetscCall(AOApplicationToPetsc(ao,static_cast<PetscInt>(ids.size()),ids.data()));
    PetscCall(AOApplicationToPetsc(ao,static_cast<PetscInt>(volumeIds.size()),volumeIds.data()));
    PetscCall(Agree(comm,Local([&]() -> PetscErrorCode {
        PetscCall(VecSetValues(w.volume,static_cast<PetscInt>(volumes.size()),volumeIds.data(),volumes.data(),INSERT_VALUES));
        for (std::size_t k=0;k<w.faces.size();++k) {
            auto& f=w.faces[k]; f.ids[0]=ids[2*k]; f.ids[1]=ids[2*k+1];
            PetscReal speed=0;
            for (std::size_t q=0;q<w.rule.points.size();++q) {
                const auto normalSpeed=std::abs(Dot(f.velocity[q],f.normal));
                const auto alpha=c.stabilizer.mode==LaxFriedrichsMode::Global?c.stabilizer.globalSpeed:normalSpeed;
                PetscCheck(alpha+1e-14>=normalSpeed,PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,
                    "Global LF speed is smaller than the flow normal speed");
                speed+=.5*f.length*w.rule.weights[q]*alpha;
            }
            std::vector<PetscReal> u(f.diffusion.SamplePoints().size(),0),kappa(w.rule.points.size(),c.thermalDiffusivity);
            DiffusiveEdgeFluxResult diffusion;
            if (f.boundary) {
                std::vector<DiffusiveBoundaryValue> bc; PetscCall(ThermalData(c,f,s.time,bc));
                PetscCall(IntegrateDiffusiveBoundaryFluxWithDerivatives(f.diffusion,u,kappa,bc,s.time,{},diffusion));
            } else PetscCall(IntegrateDiffusiveFluxWithDerivatives(f.diffusion,u,kappa,s.time,{},diffusion));
            PetscReal derivative[2]{};
            for (std::size_t q=0;q<u.size();++q) {
                const auto side=f.diffusion.SampleSides()[q];
                if (side!=DiffusiveSampleSide::Boundary)
                    derivative[side==DiffusiveSampleSide::Left?0:1]+=diffusion.derivativeSamples[q];
            }
            const PetscScalar rates[2]={(speed+std::abs(derivative[0]))/f.area[0],
                                         (speed+std::abs(derivative[1]))/f.area[1]};
            PetscCall(VecSetValues(w.rate,f.boundary?1:2,f.ids,rates,ADD_VALUES));
        }
        return PETSC_SUCCESS;
    })));
    PetscCall(VecAssemblyBegin(w.volume)); PetscCall(VecAssemblyEnd(w.volume));
    PetscCall(VecAssemblyBegin(w.rate)); PetscCall(VecAssemblyEnd(w.rate));
    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode RHS(MPI_Comm comm,const Configuration& c,const InitialState& s,Workspace& w,PetscReal& boundaryFlux)
{
    PetscFunctionBeginUser;
    PetscCall(w.reconstruction.Update(s,false)); PetscCall(VecSet(w.rhs,0));
    PetscReal localBoundary=0;
    PetscCall(Agree(comm,Local([&]() -> PetscErrorCode {
        for (const auto& f:w.faces) {
            const auto nq=w.rule.points.size();
            std::vector<PetscReal> left(nq),right(nq);
            std::vector<AdvectiveBoundaryValue> advective(nq);
            for (std::size_t q=0;q<nq;++q) {
                const auto p=MapEdgePoint(w.rule.points[q],f.vertices);
                PetscCall(w.reconstruction.Enthalpy(f.left,p,left[q]));
                if (f.boundary) {
                    auto bp=f.boundaryPoint; bp.position=p; bp.time=s.time;
                    PetscCall(EvaluateAdvectionBoundary(c.boundary.enthalpy,bp,Dot(f.velocity[q],f.normal),advective[q]));
                } else PetscCall(w.reconstruction.Enthalpy(f.right,p,right[q]));
            }
            PetscReal advection=0,diffusion=0;
            if (f.boundary) PetscCall(IntegrateAdvectiveBoundaryFlux(f.vertices,w.rule,f.velocity,left,advective,s.time,{},advection));
            else PetscCall(IntegrateAdvectiveFlux(f.vertices,w.rule,f.velocity,left,right,s.time,{},c.stabilizer,advection));
            // Legacy preheat diffusion uses constant cell-average H samples.
            // C and the diagnostic equilibrium temperature do not enter it.
            std::vector<PetscReal> samples(f.diffusion.SamplePoints().size(),0),kappa(nq,c.thermalDiffusivity);
            for (std::size_t q=0;q<samples.size();++q) {
                const auto side=f.diffusion.SampleSides()[q];
                if (side!=DiffusiveSampleSide::Boundary)
                    samples[q]=w.reconstruction.AverageH(side==DiffusiveSampleSide::Left?f.left:f.right);
            }
            if (f.boundary) {
                std::vector<DiffusiveBoundaryValue> bc; PetscCall(ThermalData(c,f,s.time,bc));
                PetscCall(IntegrateDiffusiveBoundaryFlux(f.diffusion,samples,kappa,bc,s.time,{},diffusion));
            } else PetscCall(IntegrateDiffusiveFlux(f.diffusion,samples,kappa,s.time,{},diffusion));
            const auto flux=advection+diffusion;
            const PetscScalar values[2]={-flux/f.area[0],flux/f.area[1]};
            PetscCall(VecSetValues(w.rhs,f.boundary?1:2,f.ids,values,ADD_VALUES));
            if (f.boundary) localBoundary+=flux;
        }
        return PETSC_SUCCESS;
    })));
    PetscCall(VecAssemblyBegin(w.rhs)); PetscCall(VecAssemblyEnd(w.rhs));
    PetscCallMPI(MPI_Allreduce(&localBoundary,&boundaryFlux,1,MPIU_REAL,MPI_SUM,comm));
    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode Diagnostics(MPI_Comm comm,const Configuration& c,InitialState& s,Workspace& w)
{
    PetscFunctionBeginUser;
    PetscCall(w.reconstruction.Update(s));
    GaussRule1D rule; PetscCall(Agree(comm,CreateGaussRule(c.input.quadrature.cellPointsPerAxis,rule)));
    std::vector<PetscInt> ids; std::vector<PetscScalar> temperature,porosity;
    PetscCall(Agree(comm,Local([&]() -> PetscErrorCode {
        const auto& mesh=w.reconstruction.Mesh(); const auto r=mesh.OwnedCells();
        for (PetscInt j=r.begin.j;j<r.end.j;++j) for (PetscInt i=r.begin.i;i<r.end.i;++i) {
            PetscInt id; QuadVertices corners; PetscReal area,t=0,phi=0;
            PetscCall(mesh.CellId({i,j},id)); PetscCall(mesh.GetCellCorners({i,j},corners)); PetscCall(mesh.GetCellArea({i,j},area));
            for (std::size_t b=0;b<rule.points.size();++b) for (std::size_t a=0;a<rule.points.size();++a) {
                Point ref{{rule.points[a],rule.points[b]}}; const auto p=MapCellPoint(ref,corners);
                PetscReal H,C; PetscCall(w.reconstruction.Evaluate({i,j},p,H,C));
                const auto phase=c.phase.evaluate(H,C,c.pressure(p,s.time));
                const auto weight=rule.weights[a]*rule.weights[b]*CellJacobian(ref,corners)/area;
                t+=weight*phase.TDp; phi+=weight*phase.phil;
            }
            ids.push_back(id); temperature.push_back(t); porosity.push_back(phi);
        }
        return PETSC_SUCCESS;
    })));
    AO ao=nullptr; PetscCall(DMDAGetAO(s.cellDM,&ao)); PetscCall(AOApplicationToPetsc(ao,static_cast<PetscInt>(ids.size()),ids.data()));
    PetscCall(VecSetValues(s.temperature,static_cast<PetscInt>(ids.size()),ids.data(),temperature.data(),INSERT_VALUES));
    PetscCall(VecSetValues(s.porosity,static_cast<PetscInt>(ids.size()),ids.data(),porosity.data(),INSERT_VALUES));
    PetscCall(VecAssemblyBegin(s.temperature)); PetscCall(VecAssemblyEnd(s.temperature));
    PetscCall(VecAssemblyBegin(s.porosity)); PetscCall(VecAssemblyEnd(s.porosity));
    PetscFunctionReturn(PETSC_SUCCESS);
}
PetscErrorCode Run(MPI_Comm comm,const Configuration& c,InitialState& s,Workspace& w,PreheatReport& report)
{
    PetscFunctionBeginUser;
    PetscCheck(c.flow.dryPorosity,comm,PETSC_ERR_SUP,"Legacy preheat requires prescribed zero flow porosity");
    PetscCheck(s.flowReport.converged,comm,PETSC_ERR_ARG_WRONGSTATE,"Initialize and solve flow before preheating");
    PetscCheck(s.time<=c.input.time.end,comm,PETSC_ERR_ARG_OUTOFRANGE,"State time exceeds time.end");
    PetscCheck(c.input.time.cfl<=1,comm,PETSC_ERR_ARG_OUTOFRANGE,"Explicit preheat requires time.step.cfl <= 1");
    PetscCall(Build(comm,c,s,w));
    PetscReal rate=0; PetscCall(VecMax(w.rate,nullptr,&rate));
    const auto& time=c.input.time;
    const auto limit=rate>0?time.cfl/rate:time.maximumStep;
    report=PreheatReport{}; report.initialTime=report.finalTime=s.time; report.stepLimit=limit;
    PetscScalar total=0; PetscCall(VecDot(w.volume,s.enthalpy,&total));
    report.initialIntegral=report.finalIntegral=PetscRealPart(total);
    PreheatStep first; first.time=s.time; first.step=s.acceptedSteps; first.integral=report.initialIntegral;
    PetscCall(VecMin(s.enthalpy,nullptr,&first.minimum)); PetscCall(VecMax(s.enthalpy,nullptr,&first.maximum));
    report.history.push_back(first);
    PetscCall(PetscPrintf(comm,"Legacy preheat: fixed dry flow, C fixed, forward Euler; rate-based step limit %.8g\n",static_cast<double>(limit)));
    std::int64_t steps=0;
    while (s.time<time.end) {
        PetscCheck(steps<time.maximumSteps,comm,PETSC_ERR_NOT_CONVERGED,"maximum_steps reached before time.end");
        const auto remaining=time.end-s.time;
        auto dt=time.control=="fixed"?time.initialStep:std::min(time.maximumStep,limit);
        if (time.control=="cfl" && steps==0) dt=std::min(dt,time.initialStep);
        dt=std::min(dt,remaining);
        const auto roundoff=8*std::numeric_limits<PetscReal>::epsilon()*std::max({std::abs(time.end),std::abs(s.time),dt});
        if (remaining-dt<=roundoff) dt=remaining;
        PetscCheck(dt<=limit*(1+1e-12),comm,PETSC_ERR_ARG_OUTOFRANGE,
            "Requested step %g exceeds advection/diffusion limit %g; reduce step or use control: cfl",static_cast<double>(dt),static_cast<double>(limit));
        PetscCheck(dt>=time.minimumStep || dt==remaining,comm,PETSC_ERR_ARG_OUTOFRANGE,"Stable step is below time.step.minimum");
        PetscCheck(s.time+dt>s.time && std::isfinite(dt),comm,PETSC_ERR_FP,"Time step does not advance finite time");
        PreheatStep row; row.dt=dt; row.courant=dt*rate;
        PetscCall(RHS(comm,c,s,w,row.boundaryFlux));
        PetscCall(VecWAXPY(w.candidate,dt,w.rhs,s.enthalpy));
        PetscReal norm=0; PetscCall(VecNorm(w.candidate,NORM_INFINITY,&norm));
        PetscCheck(!PetscIsInfOrNanReal(norm),comm,PETSC_ERR_FP,"Nonfinite preheat candidate rejected");
        PetscCall(VecDot(w.volume,w.candidate,&total)); row.integral=PetscRealPart(total);
        row.balanceError=row.integral-report.finalIntegral+dt*row.boundaryFlux;
        const auto tolerance=1e-10*(1+std::abs(row.integral)+std::abs(report.finalIntegral)+std::abs(dt*row.boundaryFlux));
        PetscCheck(std::abs(row.balanceError)<=tolerance,comm,PETSC_ERR_PLIB,"Preheat heat balance failed; candidate rejected");
        PetscCall(VecMin(w.candidate,nullptr,&row.minimum)); PetscCall(VecMax(w.candidate,nullptr,&row.maximum));
        PetscCall(VecCopy(w.candidate,s.enthalpy));
        s.time=dt==remaining?time.end:s.time+dt; ++s.acceptedSteps; ++steps;
        row.time=s.time; row.step=s.acceptedSteps;
        report.finalTime=s.time; report.finalIntegral=row.integral; report.boundaryIntegral+=dt*row.boundaryFlux;
        report.history.push_back(row);
        if (steps==1 || steps%100==0 || s.time==time.end)
            PetscCall(PetscPrintf(comm,"  step %" PetscInt_FMT ": t=%g, dt=%g, H=[%g,%g], heat balance=%g\n",
                row.step,static_cast<double>(s.time),static_cast<double>(dt),static_cast<double>(row.minimum),
                static_cast<double>(row.maximum),static_cast<double>(row.balanceError)));
    }
    PetscCall(Diagnostics(comm,c,s,w));
    PetscFunctionReturn(PETSC_SUCCESS);
}
}

PetscErrorCode AdvanceLegacyPreheat(MPI_Comm comm,const Configuration& c,InitialState& s,PreheatReport& report)
{
    PetscFunctionBeginUser;
    Workspace w; const auto error=Run(comm,c,s,w,report), cleanup=w.Destroy();
    PetscCall(error); PetscCall(cleanup); PetscFunctionReturn(PETSC_SUCCESS);
}
PetscErrorCode WritePreheatHistory(MPI_Comm comm,const Configuration& c,const PreheatReport& report)
{
    PetscFunctionBeginUser;
    PetscMPIInt rank; PetscCallMPI(MPI_Comm_rank(comm,&rank));
    PetscCall(Agree(comm,Local([&]() -> PetscErrorCode {
        if (rank) return PETSC_SUCCESS;
        const auto path=input::ResolveOutputPath(c.input,"preheat_history.csv",c.input.mesh.family);
        std::filesystem::create_directories(std::filesystem::path(path).parent_path());
        std::ofstream file; file.exceptions(std::ios::badbit|std::ios::failbit); file.open(path);
        file<<std::setprecision(17)<<"step,time,time_years,dt,dt_s,courant,H_integral,h_mean_J_kg,h_min_J_kg,h_max_J_kg,outward_H_flux,balance_error\n";
        const auto& d=c.phase.derived(); const auto& mesh=c.input.mesh;
        const auto area=(mesh.x[1]-mesh.x[0])*(mesh.y[1]-mesh.y[0]);
        for (const auto& r:report.history)
            file<<r.step<<','<<r.time<<','<<r.time*d.t0/(365*24*3600)<<','<<r.dt<<','<<r.dt*d.t0<<','<<r.courant
                <<','<<r.integral<<','<<r.integral/area*d.h0<<','<<r.minimum*d.h0<<','<<r.maximum*d.h0
                <<','<<r.boundaryFlux<<','<<r.balanceError<<'\n';
        file.close(); return PETSC_SUCCESS;
    })));
    PetscFunctionReturn(PETSC_SUCCESS);
}
}
