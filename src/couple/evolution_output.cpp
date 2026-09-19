#include "evolution_output.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <iomanip>
#include <limits>
#include <sstream>

namespace mantle::couple {
namespace {
template<class F> PetscErrorCode Local(MPI_Comm comm,F&& action) {
    PetscFunctionBeginUser;
    PetscErrorCode error=PETSC_SUCCESS;
    try { error=action(); }
    catch (const std::exception& e) { PetscCall(PetscPrintf(PETSC_COMM_SELF,"Movie output: %s\n",e.what())); error=PETSC_ERR_FILE_WRITE; }
    int local=error,global=0;
    PetscCallMPI(MPI_Allreduce(&local,&global,1,MPI_INT,MPI_MAX,comm));
    PetscCheck(!global,comm,static_cast<PetscErrorCode>(global),"Movie output failed on at least one rank");
    PetscFunctionReturn(PETSC_SUCCESS);
}
std::filesystem::path Path(const Configuration& c,const std::string& name) {
    return input::ResolveOutputPath(c.input,name,c.input.mesh.family);
}
std::string Name(const char* kind,PetscMPIInt rank,const char* extension) {
    std::ostringstream s; s<<"evolution_"<<kind<<"_rank_"<<std::setw(6)<<std::setfill('0')<<rank<<extension;
    return s.str();
}
void OpenFile(std::ofstream& out,const std::filesystem::path& path,bool binary=false) {
    out.exceptions(std::ios::badbit|std::ios::failbit);
    out.open(path,std::ios::out|std::ios::trunc|(binary?std::ios::binary:std::ios::openmode(0)));
    out<<std::setprecision(17);
}
void Binary(std::ofstream& out,const std::vector<double>& values) {
    static_assert(sizeof(double)==8 && std::numeric_limits<double>::is_iec559,"Movie format requires IEEE float64");
    const std::uint16_t one=1;
    if (*reinterpret_cast<const unsigned char*>(&one)==1) {
        out.write(reinterpret_cast<const char*>(values.data()),static_cast<std::streamsize>(values.size()*sizeof(double)));
    } else {
        for (double value:values) {
            std::array<char,8> bytes; std::memcpy(bytes.data(),&value,8); std::reverse(bytes.begin(),bytes.end());
            out.write(bytes.data(),8);
        }
    }
    out.flush();
}
}
PetscErrorCode EvolutionOutput::Open(MPI_Comm comm,const Configuration& c) {
    PetscFunctionBeginUser;
    PetscCall(PetscOptionsGetBool(nullptr,nullptr,"-couple_movie_snapshots",&enabled_,nullptr));
    if (!enabled_) PetscFunctionReturn(PETSC_SUCCESS);
    PetscMPIInt rank; PetscCallMPI(MPI_Comm_rank(comm,&rank));
    PetscCall(Local(comm,[&]() -> PetscErrorCode {
        OpenFile(centersFile_,Path(c,Name("centers",rank,".bin")),true);
        OpenFile(gaussFile_,Path(c,Name("gauss",rank,".bin")),true);
        if (!rank) {
            OpenFile(timesFile_,Path(c,"evolution_times.csv"));
            timesFile_<<"step,time,time_years\n";
            std::ofstream marker; OpenFile(marker,Path(c,"evolution_series.json"));
            marker<<"{\"status\":\"incomplete\"}\n"; marker.close();
        }
        return PETSC_SUCCESS;
    }));
    PetscFunctionReturn(PETSC_SUCCESS);
}
PetscErrorCode EvolutionOutput::Write(MPI_Comm comm,const Configuration& c,const InitialState& s,
                                     const StateReconstruction& reconstruction,const InitialFlowSamples& flow) {
    PetscFunctionBeginUser;
    if (!enabled_) PetscFunctionReturn(PETSC_SUCCESS);
    PetscCheck(s.acceptedSteps==frames_,comm,PETSC_ERR_ARG_WRONGSTATE,"Movie frames must follow accepted steps");
    PetscMPIInt rank; PetscCallMPI(MPI_Comm_rank(comm,&rank));
    PetscCall(Local(comm,[&]() -> PetscErrorCode {
        const auto& d=c.phase.derived();
        std::ofstream centerPoints,gaussPoints;
        if (!frames_) {
            OpenFile(centerPoints,Path(c,Name("center_points",rank,".csv")));
            OpenFile(gaussPoints,Path(c,Name("gauss_points",rank,".csv")));
            centerPoints<<"cell_id,x_m,y_m\n"; gaussPoints<<"cell_id,q,x_m,y_m\n";
        }
        std::vector<double> centers,velocities;
        const auto& mesh=reconstruction.Mesh(); const auto r=mesh.OwnedCells();
        for (PetscInt j=r.begin.j;j<r.end.j;++j) for (PetscInt i=r.begin.i;i<r.end.i;++i) {
            QuadVertices corners; PetscCall(mesh.GetCellCorners({i,j},corners));
            const auto p=MapCellPoint(Point{{0,0}},corners);
            PetscReal H,C; PetscCall(reconstruction.Evaluate({i,j},p,H,C));
            const auto phase=c.phase.evaluate(H,C,c.pressure(p,s.time));
            const auto nan=std::numeric_limits<double>::quiet_NaN();
            centers.insert(centers.end(),{phase.TDp*d.dT,phase.Tep*d.dT,phase.phil,phase.phi1,phase.phi2,
                static_cast<double>(phase.region),phase.has_liquid?phase.cl:nan,phase.has_solid?phase.cs:nan});
            if (!frames_) {
                PetscInt id; PetscCall(mesh.CellId({i,j},id));
                centerPoints<<id<<','<<p.p[0]*d.l0<<','<<p.p[1]*d.l0<<'\n';
            }
        }
        for (const auto& p:flow.cells) {
            const auto factor=d.u0*std::pow(p.flowPorosity,1+c.flow.material.theta);
            velocities.insert(velocities.end(),{d.u0*p.stokesVelocity.p[0],d.u0*p.stokesVelocity.p[1],
                factor*p.darcyVelocity.p[0],factor*p.darcyVelocity.p[1],p.flowPorosity});
            if (!frames_) gaussPoints<<p.cellId<<','<<p.q<<','<<p.position.p[0]*d.l0<<','<<p.position.p[1]*d.l0<<'\n';
        }
        if (!frames_) {
            centers_=static_cast<PetscInt>(centers.size()/8); gauss_=static_cast<PetscInt>(velocities.size()/5);
            centerPoints.close(); gaussPoints.close();
        }
        PetscCheck(centers.size()==static_cast<std::size_t>(centers_)*8 && velocities.size()==static_cast<std::size_t>(gauss_)*5,
            PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"Movie sampling layout changed");
        Binary(centersFile_,centers); Binary(gaussFile_,velocities);
        if (!rank) {
            timesFile_<<s.acceptedSteps<<','<<s.time<<','<<s.time*d.t0/(365*24*3600)<<'\n'; timesFile_.flush();
        }
        return PETSC_SUCCESS;
    }));
    ++frames_;
    PetscFunctionReturn(PETSC_SUCCESS);
}
PetscErrorCode EvolutionOutput::Finish(MPI_Comm comm,const Configuration& c,const InitialState& s) {
    PetscFunctionBeginUser;
    if (!enabled_) PetscFunctionReturn(PETSC_SUCCESS);
    PetscMPIInt rank,ranks; PetscCallMPI(MPI_Comm_rank(comm,&rank)); PetscCallMPI(MPI_Comm_size(comm,&ranks));
    PetscCall(Local(comm,[&]() -> PetscErrorCode {
        centersFile_.close(); gaussFile_.close(); if (!rank) timesFile_.close(); return PETSC_SUCCESS;
    }));
    PetscInt local[3]={centers_,gauss_,frames_}; std::vector<PetscInt> counts(3*ranks);
    PetscCallMPI(MPI_Allgather(local,3,MPIU_INT,counts.data(),3,MPIU_INT,comm));
    PetscCall(Local(comm,[&]() -> PetscErrorCode {
        if (rank) return PETSC_SUCCESS;
        for (PetscMPIInt r=0;r<ranks;++r)
            PetscCheck(counts[3*r+2]==s.acceptedSteps+1,PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"Missing movie frames");
        std::ofstream out; OpenFile(out,Path(c,"evolution_series.json.tmp"));
        out<<"{\"schema_version\":1,\"status\":\"complete\",\"dtype\":\"<f8\",\"layout\":\"frame,point,field\","
            <<"\"frames\":"<<frames_<<",\"nx\":"<<s.mesh.CellDimensions().i<<",\"ny\":"<<s.mesh.CellDimensions().j
            <<",\"cell_count\":"<<s.mesh.CellCount()<<",\"cell_points_per_axis\":"<<c.input.quadrature.cellPointsPerAxis
            <<",\"times\":\"evolution_times.csv\",\"time_end\":"<<s.time<<",\"eutectic_reference_K\":"<<c.phase.parameters().Te0
            <<",\"center_fields\":[\"temperature_K\",\"eutectic_temperature_K\",\"phi\",\"phi1\",\"phi2\",\"region\",\"cl\",\"cs\"],"
            <<"\"gauss_fields\":[\"vs_x_m_s\",\"vs_y_m_s\",\"q_x_m_s\",\"q_y_m_s\",\"flow_phi\"],\"files\":[";
        for (PetscMPIInt r=0;r<ranks;++r) {
            if (r) out<<',';
            out<<"{\"rank\":"<<r<<",\"centers\":\""<<Name("centers",r,".bin")<<"\",\"gauss\":\""<<Name("gauss",r,".bin")
                <<"\",\"center_points\":\""<<Name("center_points",r,".csv")<<"\",\"gauss_points\":\""<<Name("gauss_points",r,".csv")
                <<"\",\"center_count\":"<<counts[3*r]<<",\"gauss_count\":"<<counts[3*r+1]<<'}';
        }
        out<<"]}\n"; out.close();
        std::filesystem::rename(Path(c,"evolution_series.json.tmp"),Path(c,"evolution_series.json"));
        return PETSC_SUCCESS;
    }));
    PetscFunctionReturn(PETSC_SUCCESS);
}
}
