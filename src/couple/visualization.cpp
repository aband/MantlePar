#include "visualization.h"
#include "state_reconstruction.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>

namespace mantle::couple {
namespace {
namespace fs = std::filesystem;

// Local file/profile failures must be agreed before another collective.
template<class Function> PetscErrorCode Local(Function&& function)
{
    PetscFunctionBeginUser;
    try { PetscCall(function()); }
    catch (const std::exception& e) { SETERRQ(PETSC_COMM_SELF, PETSC_ERR_USER, "%s", e.what()); }
    PetscFunctionReturn(PETSC_SUCCESS);
}
PetscErrorCode Agree(MPI_Comm comm, PetscErrorCode error)
{
    PetscFunctionBeginUser;
    int local = static_cast<int>(error), global = 0;
    PetscCallMPI(MPI_Allreduce(&local, &global, 1, MPI_INT, MPI_MAX, comm));
    PetscCheck(!global, comm, static_cast<PetscErrorCode>(global),
               "Initialization visualization failed on at least one rank");
    PetscFunctionReturn(PETSC_SUCCESS);
}
std::string Quote(const std::string& value)
{
    std::ostringstream out;
    out << '"';
    for (unsigned char c : value) {
        if (c == '"' || c == '\\') out << '\\' << c;
        else if (c < 0x20) out << "\\u" << std::hex << std::setw(4) << std::setfill('0') << int(c);
        else out << c;
    }
    out << '"';
    return out.str();
}
std::string Filename(const char* kind, PetscMPIInt rank)
{
    std::ostringstream out;
    out << kind << "_rank_" << std::setw(6) << std::setfill('0') << rank << ".csv";
    return out.str();
}
std::ofstream Open(const fs::path& path)
{
    std::ofstream out;
    out.exceptions(std::ios::failbit | std::ios::badbit);
    out.open(path);
    out << std::setprecision(17);
    return out;
}
constexpr const char* columns =
    "entity_id,q,x,y,weight,H,C,T,temperature_K,phi,phi1,phi2,cl,cs,"
    "phase_pressure_Pa,region,has_liquid,has_solid,derivatives_finite,"
    "dT_dH,dT_dC,enthalpy_residual,composition_residual,fraction_residual,"
    "x_m,y_m,h_J_kg,dT_dh_K_kg_J,dT_dC_K,eutectic_temperature_K\n";

void Sample(std::ostream& out, const Configuration& c, PetscReal time,
            PetscInt id, std::size_t q, const Point& point, PetscReal weight,
            std::array<double,3>& errors, const StateReconstruction* reconstruction=nullptr,
            MeshIndex cell={})
{
    PetscReal h,composition;
    if (reconstruction) {
        if (reconstruction->Evaluate(cell,point,h,composition))
            throw std::runtime_error("Could not sample evolved H/C reconstruction");
    } else {
        h=c.initialEnthalpy(point,time); composition=c.initialComposition(point,time);
    }
    const double pressure = c.pressure(point,time);
    const auto phase = c.phase.evaluate(h,composition,pressure);
    const auto& scale = c.phase.derived();
    const double nan = std::numeric_limits<double>::quiet_NaN();
    const bool finite = phase.finite_temperature_derivatives &&
        std::isfinite(phase.dTD_dHD) && std::isfinite(phase.dTD_dCD);
    const std::array<double,3> residuals{{phase.TDp + scale.LD*phase.phil - h,
        phase.phi2 + phase.phil*phase.cl - composition,
        phase.phi1 + phase.phi2 + phase.phil - 1}};
    out << id << ',' << q << ',' << point.p[0] << ',' << point.p[1] << ',' << weight
        << ',' << h << ',' << composition << ',' << phase.TDp << ',' << phase.TDp*scale.dT
        << ',' << phase.phil << ',' << phase.phi1 << ',' << phase.phi2
        << ',' << (phase.has_liquid ? phase.cl : nan) << ',' << (phase.has_solid ? phase.cs : nan)
        << ',' << pressure << ',' << static_cast<int>(phase.region)
        << ',' << phase.has_liquid << ',' << phase.has_solid << ',' << finite
        << ',' << (finite ? phase.dTD_dHD : nan) << ',' << (finite ? phase.dTD_dCD : nan);
    for (std::size_t k=0; k<residuals.size(); ++k) {
        out << ',' << residuals[k];
        errors[k] = std::max(errors[k],std::abs(residuals[k]));
    }
    out << ',' << point.p[0]*scale.l0 << ',' << point.p[1]*scale.l0 << ',' << h*scale.h0
        << ',' << (finite?phase.dTD_dHD*scale.dT/scale.h0:nan)
        << ',' << (finite?phase.dTD_dCD*scale.dT:nan) << ',' << phase.Tep*scale.dT << '\n';
}

void Block(std::ostream& out, const LinearBlockSolverOptions& b)
{
    out << "{\"method\":" << Quote(b.kspType) << ",\"preconditioner\":" << Quote(b.pcType)
        << ",\"rtol\":" << b.relativeTolerance << ",\"atol\":" << b.absoluteTolerance
        << ",\"dtol\":" << b.divergenceTolerance << ",\"max_iterations\":" << b.maximumIterations << '}';
}
void Manifest(std::ostream& out, const Configuration& c, const InitialState& s,
              PetscMPIInt ranks, const std::vector<PetscInt>& counts, const std::array<double,3>& errors)
{
    const auto& solver = c.flow.solver;
    const auto& d = c.phase.derived();
    out << std::boolalpha << "{\n\"schema_version\":2,\"status\":\"complete\",\n"
        << "\"simulation\":" << Quote(c.input.simulation.name) << ",\"source\":" << Quote(c.input.sourceFile)
        << ",\"time\":" << s.time << ",\"ranks\":" << ranks
        << ",\"accepted_steps\":" << s.acceptedSteps << ",\"flow_time\":" << (s.phaseCoupled?s.time:c.input.time.start)
        << ",\"evolution_model\":" << Quote(s.phaseCoupled?"full phase coupling":s.acceptedSteps?"legacy dry preheat":"initialization")
        << ",\"mesh_family\":" << Quote(c.input.mesh.family)
        << ",\"nx\":" << s.mesh.CellDimensions().i << ",\"ny\":" << s.mesh.CellDimensions().j
        << ",\"cell_count\":" << s.mesh.CellCount() << ",\"edge_count\":" << s.mesh.EdgeCount()
        << ",\"cell_points_per_axis\":" << c.input.quadrature.cellPointsPerAxis
        << ",\"edge_points\":" << c.input.quadrature.edgePoints
        << ",\"domain\":[" << c.input.mesh.x[0] << ',' << c.input.mesh.x[1] << ','
        << c.input.mesh.y[0] << ',' << c.input.mesh.y[1] << "],\n"
        << "\"sampling\":" << Quote(s.phaseCoupled?"bounded ML-WENO H/C phase at cell centers; current coupled flow at Gauss points":s.acceptedSteps?
            "phase of evolved ML-WENO H/C at cell centers; fixed initial flow at Gauss points":
            "phase at mapped cell centers; finite-element velocity at physical Gauss points") << ','
        << "\"flow_status\":\"solved\",\"flow_porosity\":" << Quote(c.flow.dryPorosity?"prescribed zero (legacy dry preheat)":"equilibrium phase porosity")
        << ",\"flow_theta\":" << c.flow.material.theta << ",\"pressure_model\":"
        << Quote(c.input.phase.settings.At("pressure").At("model").AsString())
        << ",\"scales\":{\"length_m\":" << d.l0 << ",\"temperature_K\":" << d.dT
        << ",\"velocity_m_s\":" << d.u0 << ",\"pressure_Pa\":" << d.p0
        << ",\"enthalpy_Jkg\":" << d.h0 << ",\"time_s\":" << d.t0 << ",\"latent_heat\":" << d.LD << "},\n"
        << "\"solver\":{\"status\":\"solved; settings are configured defaults, see solve for measured results\","
        << "\"method\":" << Quote(solver.kspType) << ",\"preconditioner\":"
        << Quote(solver.preconditioner == LinearPreconditioner::Schur ? "full Schur field split" : solver.preconditioner == LinearPreconditioner::SparseLU ? "sparse LU (serial AIJ)" : "none")
        << ",\"options_prefix\":" << Quote(solver.optionsPrefix)
        << ",\"rtol\":" << solver.relativeTolerance << ",\"atol\":" << solver.absoluteTolerance
        << ",\"dtol\":" << solver.divergenceTolerance << ",\"max_iterations\":" << solver.maximumIterations
        << ",\"velocity\":"; Block(out,solver.velocity);
    out << ",\"pressure\":"; Block(out,solver.pressure);
    out << ",\"error_if_not_converged\":" << solver.errorIfNotConverged
        << ",\"require_subsolver_convergence\":" << solver.requireSubsolverConvergence
        << ",\"require_true_residual\":" << solver.requireTrueResidual
        << ",\"initial_guess_nonzero\":" << solver.initialGuessNonzero
        << ",\"remove_pressure_nullspace\":" << solver.removePressureNullspace
        << ",\"pressure_nullspace\":" << Quote(c.input.flow.pressureNullspace)
        << ",\"project_rhs\":" << c.input.flow.projectRhs << "},\n";
    const auto& r=s.flowReport;
    out << "\"solve\":{\"converged\":" << r.converged
        << ",\"method\":" << Quote(r.kspType) << ",\"preconditioner\":" << Quote(r.pcType)
        << ",\"iterations\":" << r.iterations << ",\"reason\":" << static_cast<int>(r.reason)
        << ",\"ksp_residual\":" << r.kspResidualNorm << ",\"rhs_norm\":" << r.rhsNorm
        << ",\"true_residual\":" << r.trueResidualNorm << ",\"relative_true_residual\":" << r.relativeTrueResidualNorm
        << ",\"true_residual_threshold\":" << r.trueResidualThreshold
        << ",\"true_residual_satisfied\":" << r.trueResidualSatisfied
        << ",\"subsolvers_converged\":" << r.subsolversConverged
        << ",\"pressure_gauge_removed\":" << r.pressureGaugeRemoved
        << ",\"removed_rhs_component\":" << s.flowSystem.removedRhsComponent << ",\"subsolvers\":[";
    for (std::size_t k=0;k<r.subsolvers.size();++k) {
        const auto& b=r.subsolvers[k]; if (k) out << ',';
        out << "{\"prefix\":" << Quote(b.optionsPrefix) << ",\"solves\":" << b.solves
            << ",\"iterations\":" << b.totalIterations << ",\"failures\":" << b.failures << '}';
    }
    out << "]},\n\"transport\":{\"lf_mode\":" << Quote(c.stabilizer.mode == LaxFriedrichsMode::Local ? "local" : "global")
        << ",\"lf_linearization\":" << Quote(c.stabilizer.linearization == LaxFriedrichsLinearization::Full ? "full" : "frozen speed")
        << ",\"global_speed\":" << c.stabilizer.globalSpeed
        << ",\"thermal_diffusivity\":" << c.thermalDiffusivity
        << ",\"diffusion_samples\":" << c.sampling.numberOfSamples
        << ",\"diffusion_extent\":" << c.sampling.extentFraction << "},\n"
        << "\"maximum_balance_errors\":{\"enthalpy\":" << errors[0]
        << ",\"composition\":" << errors[1] << ",\"fractions\":" << errors[2] << "},\n\"files\":[";
    for (PetscMPIInt rank=0; rank<ranks; ++rank) {
        if (rank) out << ',';
        out << "{\"rank\":" << rank << ",\"cells\":" << Quote(Filename("cell_gauss",rank))
            << ",\"cell_rows\":" << counts[6*rank] << ",\"edges\":" << Quote(Filename("edge_gauss",rank))
            << ",\"edge_rows\":" << counts[6*rank+1] << ",\"mesh\":" << Quote(Filename("mesh",rank))
            << ",\"mesh_rows\":" << counts[6*rank+2]
            << ",\"centers\":" << Quote(Filename("cell_center",rank)) << ",\"center_rows\":" << counts[6*rank+3]
            << ",\"flow_cells\":" << Quote(Filename("flow_cell_gauss",rank)) << ",\"flow_cell_rows\":" << counts[6*rank+4]
            << ",\"flow_edges\":" << Quote(Filename("flow_edge_gauss",rank)) << ",\"flow_edge_rows\":" << counts[6*rank+5]
            << ",\"flow_dofs\":" << Quote(Filename("flow_dofs",rank)) << '}';
    }
    out << "]\n}\n";
}
} // namespace

PetscErrorCode WriteVisualizationImpl(MPI_Comm comm, const Configuration& c, const InitialState& s,
                                    const StateReconstruction* reconstruction)
{
    PetscFunctionBeginUser;
    PetscCall(Agree(comm, s.IsEmpty() || !s.mesh.IsInitialized() || !s.flowReport.converged ? PETSC_ERR_ARG_WRONGSTATE : PETSC_SUCCESS));
    PetscMPIInt rank=0, ranks=0;
    PetscCallMPI(MPI_Comm_rank(comm,&rank)); PetscCallMPI(MPI_Comm_size(comm,&ranks));
    fs::path directory;
    GaussRule1D cellRule, edgeRule;
    // Invalidate any old manifest first: partial reruns must not look complete.
    PetscCall(Agree(comm,Local([&]() -> PetscErrorCode {
        directory=fs::path(input::ResolveOutputPath(c.input,"visualization.json",c.input.mesh.family)).parent_path();
        if (!rank) {
            fs::create_directories(directory);
            auto out=Open(directory/"visualization.json");
            out << "{\"schema_version\":2,\"status\":\"incomplete\"}\n"; out.close();
        }
        auto error=CreateGaussRule(c.input.quadrature.cellPointsPerAxis,cellRule);
        if (error) return error;
        return CreateGaussRule(c.input.quadrature.edgePoints,edgeRule);
    })));
    InitialFlowSamples flow; PetscCall(SampleInitialFlow(comm,c,s,flow));
    std::array<PetscInt,6> localCounts{};
    std::array<double,3> localErrors{}, errors{};
    PetscCall(Agree(comm,Local([&]() -> PetscErrorCode {
        auto centers=Open(directory/Filename("cell_center",rank)); centers << columns;
        auto cells=Open(directory/Filename("cell_gauss",rank)); cells << columns;
        auto edges=Open(directory/Filename("edge_gauss",rank)); edges << columns;
        auto mesh=Open(directory/Filename("mesh",rank)); mesh << "cell_id,x0,y0,x1,y1,x2,y2,x3,y3\n";
        const auto range=s.mesh.OwnedCells();
        for (PetscInt j=range.begin.j; j<range.end.j; ++j) for (PetscInt i=range.begin.i; i<range.end.i; ++i) {
            PetscInt id=0; QuadVertices corners;
            auto error=s.mesh.CellId({i,j},id); if (error) return error;
            error=s.mesh.GetCellCorners({i,j},corners); if (error) return error;
            mesh << id;
            for (const auto& point:corners) mesh << ',' << point.p[0] << ',' << point.p[1];
            mesh << '\n'; ++localCounts[2];
            PetscReal area=0; error=s.mesh.GetCellArea({i,j},area); if (error) return error;
            Sample(centers,c,s.time,id,0,MapCellPoint(Point{{0,0}},corners),area,localErrors,reconstruction,{i,j});
            ++localCounts[3];
            const auto n=cellRule.points.size();
            for (std::size_t b=0; b<n; ++b) for (std::size_t a=0; a<n; ++a) {
                const Point reference{{cellRule.points[a],cellRule.points[b]}};
                Sample(cells,c,s.time,id,b*n+a,MapCellPoint(reference,corners),
                       cellRule.weights[a]*cellRule.weights[b]*CellJacobian(reference,corners),localErrors,reconstruction,{i,j});
                ++localCounts[0];
            }
        }
        // Unique edge ownership includes top/right boundary edges and avoids
        // counting shared edges once per adjacent cell or MPI rank.
        for (const auto id:s.mesh.OwnedEdgeIds()) {
            EdgeVertices vertices; PetscReal length=0; Point normal;
            EdgeTopology topology;
            auto topologyError=s.mesh.GetEdgeTopology(id,topology); if (topologyError) return topologyError;
            const auto cell=topology.leftCell?*topology.leftCell:*topology.rightCell;
            auto error=s.mesh.GetEdgeVertices(id,vertices); if (error) return error;
            error=GetEdgeGeometry(vertices,length,normal); if (error) return error;
            for (std::size_t q=0; q<edgeRule.points.size(); ++q) {
                Sample(edges,c,s.time,id,q,MapEdgePoint(edgeRule.points[q],vertices),
                       0.5*length*edgeRule.weights[q],localErrors,reconstruction,cell);
                ++localCounts[1];
            }
        }
        const auto writeFlow=[&](const char* name,const std::vector<FlowPointSample>& samples) {
            auto out=Open(directory/Filename(name,rank));
            out << "entity_id,q,x,y,weight,edge_id,side,us_x,us_y,ud_x,ud_y,ps,pd,flow_phi,x_m,y_m,vs_x_m_s,vs_y_m_s,q_x_m_s,q_y_m_s\n";
            const auto& scale=c.phase.derived();
            for (const auto& p:samples) {
                const auto factor=scale.u0*std::pow(p.flowPorosity,1+c.flow.material.theta);
                out << p.cellId << ',' << p.q << ',' << p.position.p[0] << ',' << p.position.p[1] << ',' << p.weight
                    << ',' << p.edgeId << ',' << p.side << ',' << p.stokesVelocity.p[0] << ',' << p.stokesVelocity.p[1]
                    << ',' << p.darcyVelocity.p[0] << ',' << p.darcyVelocity.p[1] << ',' << p.stokesPressure
                    << ',' << p.darcyPressure << ',' << p.flowPorosity
                    << ',' << p.position.p[0]*scale.l0 << ',' << p.position.p[1]*scale.l0
                    << ',' << p.stokesVelocity.p[0]*scale.u0 << ',' << p.stokesVelocity.p[1]*scale.u0
                    << ',' << p.darcyVelocity.p[0]*factor << ',' << p.darcyVelocity.p[1]*factor << '\n';
            }
            out.close();
        };
        writeFlow("flow_cell_gauss",flow.cells); writeFlow("flow_edge_gauss",flow.edges);
        localCounts[4]=static_cast<PetscInt>(flow.cells.size()); localCounts[5]=static_cast<PetscInt>(flow.edges.size());
        centers.close(); cells.close(); edges.close(); mesh.close();
        return PETSC_SUCCESS;
    })));
    std::vector<PetscInt> counts(static_cast<std::size_t>(ranks)*6);
    PetscCallMPI(MPI_Allgather(localCounts.data(),6,MPIU_INT,counts.data(),6,MPIU_INT,comm));
    PetscCallMPI(MPI_Allreduce(localErrors.data(),errors.data(),3,MPI_DOUBLE,MPI_MAX,comm));
    PetscCall(Agree(comm,Local([&]() -> PetscErrorCode {
        if (!rank) {
            const auto temporary=directory/"visualization.json.tmp";
            auto out=Open(temporary); Manifest(out,c,s,ranks,counts,errors); out.close();
            fs::rename(temporary,directory/"visualization.json");
        }
        return PETSC_SUCCESS;
    })));
    PetscCall(PetscPrintf(comm,"  Cell-center phase and Gauss-point flow data: %s\n",(directory/"visualization.json").string().c_str()));
    PetscFunctionReturn(PETSC_SUCCESS);
}
PetscErrorCode WriteInitializationVisualization(MPI_Comm comm,const Configuration& c,const InitialState& s)
{
    PetscFunctionBeginUser;
    StateReconstruction reconstruction;
    auto error=(s.acceptedSteps || s.phaseCoupled)?reconstruction.Initialize(comm,s):PETSC_SUCCESS;
    if (!error && s.phaseCoupled) error=reconstruction.Limit(c);
    if (!error) error=WriteVisualizationImpl(comm,c,s,(s.acceptedSteps || s.phaseCoupled)?&reconstruction:nullptr);
    const auto cleanup=reconstruction.Destroy();
    PetscCall(error); PetscCall(cleanup);
    PetscFunctionReturn(PETSC_SUCCESS);
}
} // namespace mantle::couple
