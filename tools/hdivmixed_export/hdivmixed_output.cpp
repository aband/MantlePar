#include "hdivmixed_output.h"
#include <hdf5.h>
#include <array>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <locale>
#include <sstream>
#include <vector>

#ifndef H5_HAVE_PARALLEL
#error "HDivMixed export requires parallel HDF5 using PETSc's MPI implementation"
#endif

namespace {
constexpr int ModeCount = 3;
constexpr std::array<const char*,ModeCount> modeNames{{
    "shared_linear","shared_constant","conforming_velocity"}};

struct Case {
    std::string name;
    std::array<QuadVertices,2> cells;
    std::array<std::array<int,2>,2> dofs;
    Point start{}, end{};
    bool alongJ = false;
};
struct Samples {
    std::vector<double> xyz, corners, coefficients, pressure;
    std::vector<int> connectivity;
    std::array<std::vector<double>,8> basis, divergence;
    std::array<std::vector<double>,ModeCount> modes, modeDivergence;
};
struct Trace {
    std::vector<double> s, xyz, values, normal, tangent, divergence, normals, expected;
    std::array<double,3> commonNormal{}, commonTangent{};
    double maxNormalJump = 0, maxTangentJump = 0, maxDivergenceJump = 0;
    double maxNormalError = 0, maxTraceError = 0, normalizedJump = 0, normalizedError = 0;
};

Point Mix(const Point& a, const Point& b, PetscReal s)
{
    if (s == 0) return a;
    if (s == 1) return b;
    return {{a.p[0]+s*(b.p[0]-a.p[0]),a.p[1]+s*(b.p[1]-a.p[1])}};
}
Point Map(const QuadVertices& q, PetscReal s, PetscReal t)
{ return Mix(Mix(q[0],q[1],s),Mix(q[3],q[2],s),t); }
double Length(const Point& a, const Point& b)
{ return std::hypot(double(b.p[0]-a.p[0]),double(b.p[1]-a.p[1])); }

Case MakeCase(bool perturbed, bool alongJ, PetscReal amplitude)
{
    // An interior two-cell patch of a logically rectangular grid. Six logical
    // vertices are perturbed; the cells retain exactly the same shared edge.
    const double offsets[6][2] = {{-0.4,0.2},{1,-0.4},{0.4,0.6},
                                  {0.5,0.8},{-0.7,0.4},{0.9,-0.5}};
    std::array<Point,6> points;
    const PetscReal p = perturbed ? amplitude : PetscReal(0);
    for (int j = 0; j < 2; ++j) for (int i = 0; i < 3; ++i) {
        const int k = 3*j+i;
        points[k] = Point{{PetscReal(1.3)*(i+p*offsets[k][0]),
                            PetscReal(0.85)*(j+p*offsets[k][1])}};
    }
    Case result;
    result.name = std::string(perturbed ? "quadrilateral" : "rectangular") + (alongJ ? "_j" : "_i");
    result.alongJ = alongJ;
    result.cells[0] = QuadVertices{{points[0],points[1],points[4],points[3]}};
    result.cells[1] = QuadVertices{{points[1],points[2],points[5],points[4]}};
    if (alongJ) {
        // Rotate and reorder: logical Bottom/Right/Top/Left, CCW corners.
        for (auto& q : result.cells) {
            const auto old = q;
            for (int v = 0; v < 4; ++v) {
                const auto& a = old[(v+3)%4];
                q[v] = Point{{PetscReal(0.85)-a.p[1],a.p[0]}};
            }
        }
        result.dofs = {{{{2,6}},{{0,4}}}}; // Top of A, bottom of B.
        result.start = result.cells[0][3]; result.end = result.cells[0][2];
    } else {
        result.dofs = {{{{1,5}},{{3,7}}}}; // Right of A, left of B.
        result.start = result.cells[0][1]; result.end = result.cells[0][2];
    }
    return result;
}

std::array<PetscReal,8> Coefficients(const QuadVertices& q)
{
    std::array<PetscReal,8> result{};
    for (int e = 0; e < 4; ++e) {
        // Symmetric midpoint gives matching coefficients for reversed edges.
        // HDivMixed already includes orientation; no additional sign is used.
        const PetscReal x = PetscReal(0.5)*q[e].p[0]+PetscReal(0.5)*q[(e+1)%4].p[0];
        const PetscReal y = PetscReal(0.5)*q[e].p[1]+PetscReal(0.5)*q[(e+1)%4].p[1];
        result[e] = PetscReal(0.3)+PetscReal(0.17)*x-PetscReal(0.09)*y;
        result[e+4] = PetscReal(0.8)-PetscReal(0.12)*x+PetscReal(0.21)*y;
    }
    return result;
}

std::array<HDivBasisValue,ModeCount> Modes(const HDivMixed::Values& all,
    const std::array<int,2>& dofs, const std::array<PetscReal,8>& coefficients)
{
    std::array<HDivBasisValue,ModeCount> result{};
    result[0] = all[dofs[0]]; result[1] = all[dofs[1]];
    for (int k = 0; k < 8; ++k) {
        for (int d = 0; d < 2; ++d) result[2].value.p[d] += coefficients[k]*all[k].value.p[d];
        result[2].divergence += coefficients[k]*all[k].divergence;
    }
    return result;
}

PetscErrorCode SampleCell(const Case& patch, int cell, PetscInt n, Samples& result)
{
    PetscFunctionBeginUser;
    const auto& q = patch.cells[cell];
    HDivMixed basis;
    PetscCall(basis.Initialize(q));
    const auto coefficients = Coefficients(q);
    const auto nodes = static_cast<std::size_t>((n+1)*(n+1));
    result.xyz.assign(3*nodes,0); result.corners.assign(12,0);
    result.coefficients.assign(coefficients.begin(),coefficients.end());
    result.pressure.assign(nodes,double(HDivMixed::Pressure()));
    for (auto& field : result.basis) field.assign(3*nodes,0);
    for (auto& field : result.modes) field.assign(3*nodes,0);
    for (auto& field : result.divergence) field.resize(nodes);
    for (auto& field : result.modeDivergence) field.resize(nodes);
    for (int v = 0; v < 4; ++v) for (int d = 0; d < 2; ++d)
        result.corners[3*v+d] = double(q[v].p[d]);
    for (PetscInt j = 0; j <= n; ++j) for (PetscInt i = 0; i <= n; ++i) {
        const auto k = static_cast<std::size_t>(j*(n+1)+i);
        const Point p = Map(q,PetscReal(i)/n,PetscReal(j)/n);
        HDivMixed::Values all;
        PetscCall(basis.EvaluateAll(p,all));
        const auto modes = Modes(all,patch.dofs[cell],coefficients);
        for (int d = 0; d < 2; ++d) {
            result.xyz[3*k+d] = double(p.p[d]);
            for (int b = 0; b < 8; ++b) result.basis[b][3*k+d] = double(all[b].value.p[d]);
            for (int m = 0; m < ModeCount; ++m) result.modes[m][3*k+d] = double(modes[m].value.p[d]);
        }
        for (int b = 0; b < 8; ++b) result.divergence[b][k] = double(all[b].divergence);
        for (int m = 0; m < ModeCount; ++m) result.modeDivergence[m][k] = double(modes[m].divergence);
    }
    result.connectivity.reserve(static_cast<std::size_t>(4*n*n));
    for (PetscInt j = 0; j < n; ++j) for (PetscInt i = 0; i < n; ++i) {
        const int a = static_cast<int>(j*(n+1)+i), width = static_cast<int>(n+1);
        for (int vertex : {a,a+1,a+width+1,a+width}) result.connectivity.push_back(vertex);
    }
    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode CheckAll(MPI_Comm comm, bool ok, const char* operation);

PetscErrorCode SampleTraceCell(const Case& patch, int cell, PetscInt segments,
    std::vector<double>& values, std::vector<double>& divergence, std::array<double,6>& normals)
{
    PetscFunctionBeginUser;
    HDivMixed basis;
    PetscCall(basis.Initialize(patch.cells[cell]));
    const auto coefficients = Coefficients(patch.cells[cell]);
    Point normal;
    PetscCall(basis.GetEdgeNormal(static_cast<CellSide>(patch.dofs[cell][0]),normal));
    for (int d = 0; d < 2; ++d) normals[3*cell+d] = double(normal.p[d]);
    const auto stride = static_cast<std::size_t>(segments+1)*ModeCount;
    for (PetscInt i = 0; i <= segments; ++i) {
        // Exactly the same physical points on A and B; no averaging.
        const Point p = Mix(patch.start,patch.end,PetscReal(i)/segments);
        HDivMixed::Values all;
        PetscCall(basis.EvaluateAll(p,all));
        const auto modes = Modes(all,patch.dofs[cell],coefficients);
        for (int m = 0; m < ModeCount; ++m) {
            const auto k = cell*stride+static_cast<std::size_t>(i)*ModeCount+m;
            for (int d = 0; d < 2; ++d) values[3*k+d] = double(modes[m].value.p[d]);
            divergence[k] = double(modes[m].divergence);
        }
    }
    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode SampleTrace(MPI_Comm comm, PetscMPIInt rank, PetscMPIInt ranks,
                            const Case& patch, PetscInt segments, Trace& trace)
{
    PetscFunctionBeginUser;
    const auto n = static_cast<std::size_t>(segments+1), stride = n*ModeCount;
    const double length = Length(patch.start,patch.end);
    trace.commonTangent = {{double(patch.end.p[0]-patch.start.p[0])/length,
                            double(patch.end.p[1]-patch.start.p[1])/length,0}};
    const auto& t = trace.commonTangent;
    trace.commonNormal = patch.alongJ ? std::array<double,3>{{-t[1],t[0],0}}
                                      : std::array<double,3>{{t[1],-t[0],0}};
    trace.s.resize(n); trace.xyz.assign(3*n,0); trace.expected.resize(stride);
    const auto coefficients = Coefficients(patch.cells[0]);
    for (std::size_t i = 0; i < n; ++i) {
        trace.s[i] = double(i)/double(segments);
        const auto p = Mix(patch.start,patch.end,PetscReal(i)/segments);
        for (int d = 0; d < 2; ++d) trace.xyz[3*i+d] = double(p.p[d]);
        const double linear = patch.alongJ ? 2*trace.s[i]-1 : 1-2*trace.s[i];
        trace.expected[i*ModeCount] = linear;
        trace.expected[i*ModeCount+1] = 1;
        trace.expected[i*ModeCount+2] = double(coefficients[patch.dofs[0][0]])*linear+
                                       double(coefficients[patch.dofs[0][1]]);
    }
    std::vector<double> local(2*stride*3,0), localDivergence(2*stride,0);
    std::array<double,6> localNormals{};
    bool sampled = true;
    for (int cell = 0; cell < 2; ++cell) if (rank == cell%ranks)
        sampled = SampleTraceCell(patch,cell,segments,local,localDivergence,localNormals) == PETSC_SUCCESS && sampled;
    PetscCall(CheckAll(comm,sampled,"sample shared-edge values"));
    trace.values.resize(local.size()); trace.divergence.resize(localDivergence.size());
    trace.normals.resize(6); trace.normal.assign(2*stride,0); trace.tangent.assign(2*stride,0);
    PetscCallMPI(MPI_Allreduce(local.data(),trace.values.data(),static_cast<int>(local.size()),MPI_DOUBLE,MPI_SUM,comm));
    PetscCallMPI(MPI_Allreduce(localDivergence.data(),trace.divergence.data(),static_cast<int>(localDivergence.size()),MPI_DOUBLE,MPI_SUM,comm));
    PetscCallMPI(MPI_Allreduce(localNormals.data(),trace.normals.data(),6,MPI_DOUBLE,MPI_SUM,comm));
    for (std::size_t k = 0; k < 2*stride; ++k) {
        for (int d = 0; d < 2; ++d) {
            trace.normal[k] += trace.values[3*k+d]*trace.commonNormal[d];
            trace.tangent[k] += trace.values[3*k+d]*trace.commonTangent[d];
        }
        PetscCheck(std::isfinite(trace.normal[k]) && std::isfinite(trace.tangent[k]) &&
                     std::isfinite(trace.divergence[k]),comm,PETSC_ERR_FP,"Nonfinite shared-edge sample");
        const double expected = trace.expected[k%stride];
        trace.maxTraceError = std::max(trace.maxTraceError,std::abs(trace.normal[k]-expected));
        trace.normalizedError = std::max(trace.normalizedError,std::abs(trace.normal[k]-expected)/
            std::max({1.,std::abs(trace.normal[k]),std::abs(expected)}));
    }
    for (std::size_t k = 0; k < stride; ++k) {
        const double a = trace.normal[k], b = trace.normal[stride+k];
        trace.maxNormalJump = std::max(trace.maxNormalJump,std::abs(a-b));
        trace.maxTangentJump = std::max(trace.maxTangentJump,std::abs(trace.tangent[k]-trace.tangent[stride+k]));
        trace.maxDivergenceJump = std::max(trace.maxDivergenceJump,std::abs(trace.divergence[k]-trace.divergence[stride+k]));
        trace.normalizedJump = std::max(trace.normalizedJump,std::abs(a-b)/std::max({1.,std::abs(a),std::abs(b)}));
    }
    for (int cell = 0; cell < 2; ++cell) for (int d = 0; d < 2; ++d)
        trace.maxNormalError = std::max(trace.maxNormalError,std::abs(trace.normals[3*cell+d]-trace.commonNormal[d]));
    PetscFunctionReturn(PETSC_SUCCESS);
}

bool Passed(const Trace& trace, double tolerance)
{
    return std::max({trace.normalizedJump,trace.normalizedError,trace.maxNormalError}) <= tolerance;
}

class H5Handle {
public:
    H5Handle(hid_t id, herr_t (*close)(hid_t)) : id_(id), close_(close) {}
    ~H5Handle() { if (id_ >= 0) close_(id_); }
    H5Handle(const H5Handle&) = delete;
    H5Handle& operator=(const H5Handle&) = delete;
    hid_t get() const { return id_; }
    herr_t close() { if (id_ < 0) return 0; const auto result = close_(id_); id_ = -1; return result; }
private:
    hid_t id_;
    herr_t (*close_)(hid_t);
};

PetscErrorCode CheckAll(MPI_Comm comm, bool ok, const char* operation)
{
    PetscFunctionBeginUser;
    const int local = ok ? 1 : 0;
    int global = 0;
    PetscCallMPI(MPI_Allreduce(&local, &global, 1, MPI_INT, MPI_MIN, comm));
    PetscCheck(global, comm, PETSC_ERR_LIB, "HDivMixed export failed: %s", operation);
    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode SameInputs(MPI_Comm comm, PetscMPIInt rank, const HDivPlotOptions& o, const std::string& stem)
{
    PetscFunctionBeginUser;
    std::ostringstream signature;
    signature.imbue(std::locale::classic());
    signature << std::setprecision(std::numeric_limits<PetscReal>::max_digits10)
              << o.subdivisions << ' ' << o.trace_segments << ' ' << o.perturbation << ' ' << o.tolerance << '\n' << stem;
    const auto local = signature.str();
    PetscCall(CheckAll(comm, local.size() < static_cast<std::size_t>(std::numeric_limits<int>::max()), "input signature length"));
    int length = rank == 0 ? static_cast<int>(local.size()) : 0;
    PetscCallMPI(MPI_Bcast(&length, 1, MPI_INT, 0, comm));
    std::string reference(static_cast<std::size_t>(length), '\0');
    if (rank == 0) reference = local;
    PetscCallMPI(MPI_Bcast(reference.data(), length, MPI_CHAR, 0, comm));
    PetscCall(CheckAll(comm, local == reference, "options/output stem differ between MPI ranks"));
    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode Group(MPI_Comm comm, hid_t file, const std::string& path)
{
    PetscFunctionBeginUser;
    H5Handle group(H5Gcreate2(file, path.c_str(), H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT), H5Gclose);
    PetscCall(CheckAll(comm, group.get() >= 0, path.c_str()));
    PetscCall(CheckAll(comm, group.close() >= 0, "close group"));
    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode Write(MPI_Comm comm, hid_t file, const std::string& path,
                      const std::vector<hsize_t>& shape, bool owner, const void* data,
                      bool integer = false)
{
    PetscFunctionBeginUser;
    H5Handle space(H5Screate_simple(static_cast<int>(shape.size()), shape.data(), nullptr), H5Sclose);
    PetscCall(CheckAll(comm, space.get() >= 0, "create dataspace"));
    H5Handle dataset(H5Dcreate2(file, path.c_str(), integer ? H5T_STD_I32LE : H5T_IEEE_F64LE,
                                space.get(), H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT), H5Dclose);
    PetscCall(CheckAll(comm, dataset.get() >= 0, path.c_str()));
    hsize_t count = 1;
    for (auto n : shape) count *= n;
    const hsize_t memorySize = owner ? count : 1;
    H5Handle memory(H5Screate_simple(1, &memorySize, nullptr), H5Sclose);
    PetscCall(CheckAll(comm, memory.get() >= 0, "create memory dataspace"));
    bool selected = true;
    if (!owner) {
        const auto a = H5Sselect_none(space.get()), b = H5Sselect_none(memory.get());
        selected = a >= 0 && b >= 0;
    }
    PetscCall(CheckAll(comm, selected, "select empty-rank dataspace"));
    H5Handle transfer(H5Pcreate(H5P_DATASET_XFER), H5Pclose);
    PetscCall(CheckAll(comm, transfer.get() >= 0, "create transfer properties"));
    PetscCall(CheckAll(comm, H5Pset_dxpl_mpio(transfer.get(), H5FD_MPIO_COLLECTIVE) >= 0, "collective dataset I/O"));
    const double unused = 0;
    PetscCall(CheckAll(comm, H5Dwrite(dataset.get(), integer ? H5T_NATIVE_INT : H5T_NATIVE_DOUBLE,
                                     memory.get(), space.get(), transfer.get(), owner ? data : &unused) >= 0,
                       path.c_str()));
    PetscCall(CheckAll(comm, dataset.close() >= 0, "close dataset"));
    PetscFunctionReturn(PETSC_SUCCESS);
}

std::string BasisName(int k)
{ return std::string("basis_") + (k < 10 ? "0" : "") + std::to_string(k); }
std::string XmlEscape(const std::string& text)
{
    std::string result;
    for (char c : text) {
        switch (c) {
        case '&': result += "&amp;"; break;
        case '<': result += "&lt;"; break;
        case '>': result += "&gt;"; break;
        case '"': result += "&quot;"; break;
        case '\'': result += "&apos;"; break;
        default: result += c;
        }
    }
    return result;
}
void Item(std::ostream& out, const std::string& file, const std::string& path,
           const std::string& shape, bool integer = false)
{
    out << "<DataItem Dimensions=\"" << shape << "\" NumberType=\"" << (integer ? "Int" : "Float")
        << "\" Precision=\"" << (integer ? 4 : 8) << "\" Format=\"HDF\">"
        << XmlEscape(file) << ':' << path << "</DataItem>\n";
}

void WriteXdmf(std::ostream& out, const Case& patch, PetscInt n, const std::string& file)
{
    const auto nodes = std::to_string((n+1)*(n+1)), cells = std::to_string(n*n);
    out << "<?xml version=\"1.0\"?>\n<Xdmf Version=\"2.0\"><Domain>\n"
        << "<Grid Name=\"" << patch.name << "\" GridType=\"Collection\" CollectionType=\"Spatial\">\n";
    for (int c = 0; c < 2; ++c) {
        const std::string root = "/"+patch.name+"/cell_"+std::to_string(c);
        out << "<Grid Name=\"Cell_" << (c == 0 ? 'A' : 'B') << "\" GridType=\"Uniform\">\n"
            << "<Topology TopologyType=\"Quadrilateral\" NumberOfElements=\"" << cells << "\">\n";
        Item(out,file,root+"/connectivity",cells+" 4",true);
        out << "</Topology>\n<Geometry GeometryType=\"XYZ\">\n";
        Item(out,file,root+"/coordinates",nodes+" 3");
        out << "</Geometry>\n";
        const auto attribute = [&](const std::string& name, bool vector) {
            out << "<Attribute Name=\"" << name << "\" AttributeType=\"" << (vector ? "Vector" : "Scalar")
                << "\" Center=\"Node\">\n";
            Item(out,file,root+"/"+name,nodes+(vector ? " 3" : ""));
            out << "</Attribute>\n";
        };
        for (int k = 0; k < 8; ++k) {
            attribute(BasisName(k),true); attribute(BasisName(k)+"_divergence",false);
        }
        for (const char* name : modeNames) {
            attribute(name,true); attribute(std::string(name)+"_divergence",false);
        }
        attribute("pressure_basis",false);
        out << "</Grid>\n";
    }
    out << "</Grid>\n</Domain></Xdmf>\n";
}
} // namespace

PetscErrorCode WriteHDivMixedXdmf(MPI_Comm comm, const HDivPlotOptions& o, const std::string& stem)
{
    PetscFunctionBeginUser;
    PetscMPIInt rank, ranks;
    PetscCallMPI(MPI_Comm_rank(comm,&rank)); PetscCallMPI(MPI_Comm_size(comm,&ranks));
    const bool valid = o.subdivisions >= 2 && o.subdivisions <= 512 &&
        o.trace_segments >= 2 && o.trace_segments <= 100000 &&
        !PetscIsInfOrNanReal(o.perturbation) && o.perturbation >= 0 && o.perturbation < PetscReal(0.25) &&
        !PetscIsInfOrNanReal(o.tolerance) && o.tolerance > 0 && !stem.empty() &&
        !std::filesystem::path(stem).filename().empty();
    PetscCall(CheckAll(comm,valid,"require 2<=hdiv_n<=512, 2<=hdiv_trace_n<=100000, 0<=hdiv_perturb<0.25 and positive tolerance"));
    PetscCall(SameInputs(comm,rank,o,stem));
    const std::array<Case,4> cases{{MakeCase(false,false,o.perturbation),MakeCase(true,false,o.perturbation),
                                   MakeCase(false,true,o.perturbation),MakeCase(true,true,o.perturbation)}};
    std::array<std::array<Samples,2>,4> samples;
    std::array<Trace,4> traces;
    for (std::size_t p = 0; p < cases.size(); ++p) {
        bool sampled = true;
        for (int c = 0; c < 2; ++c) if (rank == c%ranks)
            sampled = SampleCell(cases[p],c,o.subdivisions,samples[p][c]) == PETSC_SUCCESS && sampled;
        PetscCall(CheckAll(comm,sampled,"sample cell vector values and divergence"));
        PetscCall(SampleTrace(comm,rank,ranks,cases[p],o.trace_segments,traces[p]));
    }
    const std::filesystem::path h5Path(stem+".h5");
    bool directories = true;
    if (rank == 0 && !h5Path.parent_path().empty()) {
        std::error_code error;
        std::filesystem::create_directories(h5Path.parent_path(),error);
        directories = !error;
    }
    PetscCall(CheckAll(comm,directories,"create output directory"));
    H5Handle access(H5Pcreate(H5P_FILE_ACCESS),H5Pclose);
    PetscCall(CheckAll(comm,access.get() >= 0,"create file access properties"));
    PetscCall(CheckAll(comm,H5Pset_fapl_mpio(access.get(),comm,MPI_INFO_NULL) >= 0,"enable parallel HDF5"));
    H5Handle file(H5Fcreate(h5Path.string().c_str(),H5F_ACC_TRUNC,H5P_DEFAULT,access.get()),H5Fclose);
    PetscCall(CheckAll(comm,file.get() >= 0,"create HDF5 file"));
    const hsize_t nodes = static_cast<hsize_t>((o.subdivisions+1)*(o.subdivisions+1));
    const hsize_t displayCells = static_cast<hsize_t>(o.subdivisions*o.subdivisions);
    const hsize_t traceNodes = static_cast<hsize_t>(o.trace_segments+1);
    const auto stride = static_cast<std::size_t>(traceNodes)*ModeCount;
    const double metadata[] = {1,double(o.subdivisions),double(o.trace_segments),
        double(o.perturbation),double(o.tolerance),double(ranks),8,ModeCount};
    PetscCall(Write(comm,file.get(),"/metadata",{8},rank == 0,metadata));
    for (std::size_t p = 0; p < cases.size(); ++p) {
        const auto& patch = cases[p];
        const std::string root = "/"+patch.name;
        PetscCall(Group(comm,file.get(),root));
        for (int c = 0; c < 2; ++c) {
            const auto& data = samples[p][c];
            const bool owner = rank == c%ranks;
            const auto group = root+"/cell_"+std::to_string(c);
            PetscCall(Group(comm,file.get(),group));
            PetscCall(Write(comm,file.get(),group+"/coordinates",{nodes,3},owner,data.xyz.data()));
            PetscCall(Write(comm,file.get(),group+"/connectivity",{displayCells,4},owner,data.connectivity.data(),true));
            PetscCall(Write(comm,file.get(),group+"/corners",{4,3},owner,data.corners.data()));
            PetscCall(Write(comm,file.get(),group+"/coefficients",{8},owner,data.coefficients.data()));
            for (int k = 0; k < 8; ++k) {
                PetscCall(Write(comm,file.get(),group+"/"+BasisName(k),{nodes,3},owner,data.basis[k].data()));
                PetscCall(Write(comm,file.get(),group+"/"+BasisName(k)+"_divergence",{nodes},owner,data.divergence[k].data()));
            }
            for (int m = 0; m < ModeCount; ++m) {
                PetscCall(Write(comm,file.get(),group+"/"+modeNames[m],{nodes,3},owner,data.modes[m].data()));
                PetscCall(Write(comm,file.get(),group+"/"+modeNames[m]+"_divergence",{nodes},owner,data.modeDivergence[m].data()));
            }
            PetscCall(Write(comm,file.get(),group+"/pressure_basis",{nodes},owner,data.pressure.data()));
        }
        const int dofs[] = {patch.dofs[0][0],patch.dofs[0][1],patch.dofs[1][0],patch.dofs[1][1]};
        PetscCall(Write(comm,file.get(),root+"/shared_local_dofs",{2,2},rank == 0,dofs,true));
        const auto group = root+"/trace";
        const auto& t = traces[p];
        PetscCall(Group(comm,file.get(),group));
        PetscCall(Write(comm,file.get(),group+"/s",{traceNodes},rank == 0,t.s.data()));
        PetscCall(Write(comm,file.get(),group+"/coordinates",{traceNodes,3},rank == 0,t.xyz.data()));
        PetscCall(Write(comm,file.get(),group+"/normals",{2,3},rank == 0,t.normals.data()));
        PetscCall(Write(comm,file.get(),group+"/common_normal",{3},rank == 0,t.commonNormal.data()));
        PetscCall(Write(comm,file.get(),group+"/common_tangent",{3},rank == 0,t.commonTangent.data()));
        PetscCall(Write(comm,file.get(),group+"/expected_normal",{traceNodes,ModeCount},rank == 0,t.expected.data()));
        for (int c = 0; c < 2; ++c) {
            const std::string suffix = c == 0 ? "_a" : "_b";
            PetscCall(Write(comm,file.get(),group+"/values"+suffix,{traceNodes,ModeCount,3},rank == 0,t.values.data()+3*c*stride));
            PetscCall(Write(comm,file.get(),group+"/normal"+suffix,{traceNodes,ModeCount},rank == 0,t.normal.data()+c*stride));
            PetscCall(Write(comm,file.get(),group+"/tangent"+suffix,{traceNodes,ModeCount},rank == 0,t.tangent.data()+c*stride));
            PetscCall(Write(comm,file.get(),group+"/divergence"+suffix,{traceNodes,ModeCount},rank == 0,t.divergence.data()+c*stride));
        }
        const double metrics[] = {t.maxNormalJump,t.maxTangentJump,t.maxDivergenceJump,t.maxNormalError,
                                   t.maxTraceError,t.normalizedJump,t.normalizedError};
        PetscCall(Write(comm,file.get(),group+"/metrics",{7},rank == 0,metrics));
    }
    PetscCall(CheckAll(comm,file.close() >= 0,"close HDF5 file"));
    PetscCall(CheckAll(comm,access.close() >= 0,"close file access properties"));

    bool textWritten = true;
    if (rank == 0) {
        std::ofstream summary(stem+"_summary.csv"), csv(stem+"_traces.csv");
        summary.imbue(std::locale::classic()); csv.imbue(std::locale::classic());
        summary << std::setprecision(17) << "case,max_normal_jump,max_tangential_jump,max_divergence_jump,"
            "max_normal_direction_error,max_expected_trace_error,normalized_normal_jump,normalized_trace_error,tolerance,passed\n";
        csv << std::setprecision(17) << "case,mode,s,x,y,ux_a,uy_a,ux_b,uy_b,normal_a,normal_b,"
            "normal_jump,expected_normal,tangent_a,tangent_b,tangent_jump,divergence_a,divergence_b\n";
        for (std::size_t p = 0; p < cases.size(); ++p) {
            const auto& t = traces[p];
            std::ofstream xdmf(stem+"_"+cases[p].name+".xdmf");
            WriteXdmf(xdmf,cases[p],o.subdivisions,h5Path.filename().string());
            xdmf.close(); textWritten = textWritten && !xdmf.fail();
            summary << cases[p].name << ',' << t.maxNormalJump << ',' << t.maxTangentJump << ',' << t.maxDivergenceJump
                << ',' << t.maxNormalError << ',' << t.maxTraceError << ',' << t.normalizedJump << ',' << t.normalizedError
                << ',' << double(o.tolerance) << ',' << (Passed(t,double(o.tolerance)) ? 1 : 0) << '\n';
            for (std::size_t i = 0; i < t.s.size(); ++i) for (int m = 0; m < ModeCount; ++m) {
                const auto k = i*ModeCount+m, b = stride+k;
                csv << cases[p].name << ',' << modeNames[m] << ',' << t.s[i] << ',' << t.xyz[3*i] << ',' << t.xyz[3*i+1]
                    << ',' << t.values[3*k] << ',' << t.values[3*k+1] << ',' << t.values[3*b] << ',' << t.values[3*b+1]
                    << ',' << t.normal[k] << ',' << t.normal[b] << ',' << t.normal[k]-t.normal[b] << ',' << t.expected[k]
                    << ',' << t.tangent[k] << ',' << t.tangent[b] << ',' << t.tangent[k]-t.tangent[b]
                    << ',' << t.divergence[k] << ',' << t.divergence[b] << '\n';
            }
        }
        summary.close(); csv.close();
        textWritten = textWritten && !summary.fail() && !csv.fail();
    }
    PetscCall(CheckAll(comm,textWritten,"write XDMF/CSV files"));
    for (std::size_t p = 0; p < cases.size(); ++p)
        PetscCall(PetscPrintf(comm,"%s: max normal jump=%.3e, tangential jump=%.3e (allowed), %s\n",
            cases[p].name.c_str(),traces[p].maxNormalJump,traces[p].maxTangentJump,
            Passed(traces[p],double(o.tolerance)) ? "PASS" : "FAIL"));
    const bool conforming = std::all_of(traces.begin(),traces.end(),
        [&](const Trace& t) { return Passed(t,double(o.tolerance)); });
    PetscCheck(conforming,comm,PETSC_ERR_PLIB,"HDivMixed sampled normal-trace check failed; inspect the HDF5/CSV output");
    PetscFunctionReturn(PETSC_SUCCESS);
}
