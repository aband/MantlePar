#include "phase_output.h"

#include <hdf5.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <locale>
#include <sstream>
#include <stdexcept>
#include <utility>
#include <vector>

#ifndef H5_HAVE_PARALLEL
#error "phase_output requires HDF5 built with parallel MPI support"
#endif

namespace {

using mantle::phase::EutecticModel;
using mantle::phase::MaterialParameters;
using mantle::phase::DerivedParameters;

class H5Handle {
public:
    H5Handle(hid_t value, herr_t (*closer)(hid_t)) : value_(value), closer_(closer) {}
    ~H5Handle() { if (value_ >= 0) closer_(value_); }
    H5Handle(const H5Handle&) = delete;
    H5Handle& operator=(const H5Handle&) = delete;
    hid_t get() const { return value_; }
    herr_t close() {
        if (value_ < 0) return 0;
        const herr_t status = closer_(value_);
        value_ = -1;
        return status;
    }
private:
    hid_t value_;
    herr_t (*closer_)(hid_t);
};

PetscErrorCode CheckAll(MPI_Comm comm, bool ok, const char* operation) {
    int local = ok ? 1 : 0, global = 0;
    PetscFunctionBeginUser;
    PetscCallMPI(MPI_Allreduce(&local, &global, 1, MPI_INT, MPI_MIN, comm));
    PetscCheck(global, comm, PETSC_ERR_LIB, "Phase export failed: %s", operation);
    PetscFunctionReturn(PETSC_SUCCESS);
}

struct Rows { PetscInt first, count; };

Rows OwnedRows(PetscInt total, PetscMPIInt rank, PetscMPIInt size) {
    const PetscInt base = total / size, extra = total % size;
    return {rank * base + std::min<PetscInt>(rank, extra), base + (rank < extra ? 1 : 0)};
}

struct Plot {
    PetscInt nc, nh;
    double cmin, cmax, hmin, hmax, pressure, Tep, Xe;
    bool normalized, boundaries;
};

struct ParameterEntry { const char* name; double MaterialParameters::*member; };
constexpr std::array<ParameterEntry, 13> materialEntries{{
    {"Tm0", &MaterialParameters::Tm0}, {"Te0", &MaterialParameters::Te0},
    {"nu", &MaterialParameters::nu}, {"L", &MaterialParameters::L},
    {"cp", &MaterialParameters::cp}, {"Xe", &MaterialParameters::Xe},
    {"rho", &MaterialParameters::rho}, {"rhor", &MaterialParameters::rhor},
    {"mus", &MaterialParameters::mus}, {"mul", &MaterialParameters::mul},
    {"k0", &MaterialParameters::k0}, {"g", &MaterialParameters::g},
    {"alpha0", &MaterialParameters::alpha0}
}};

constexpr std::array<const char*, 16> realNames{{
    "HD", "CD", "TDp", "Temperature_K", "phi1", "phi2", "phil", "cl", "cs",
    "dTD_dHD", "dTD_dCD", "EnthalpyResidual", "CompositionResidual", "FractionResidual",
    "TemperatureAboveEutectic", "Enthalpy_Jkg"
}};
constexpr std::array<const char*, 5> intNames{{
    "Region", "HasSolid", "HasLiquid", "DerivativeFinite", "WriterRank"
}};

Plot Resolve(const EutecticModel& model, const PhasePlotOptions& o) {
    if (o.composition_cells < 1 || o.enthalpy_cells < 1 ||
        o.composition_cells >= PETSC_MAX_INT || o.enthalpy_cells >= PETSC_MAX_INT) {
        throw std::invalid_argument("Cell counts must be positive and less than PETSC_MAX_INT");
    }
    const auto b = model.boundaries(0.0, o.pressure_pa);
    Plot p{o.composition_cells, o.enthalpy_cells, o.composition_min,
           o.composition_max.value_or(model.parameters().Xe),
           o.enthalpy_min.value_or(b.Tep - 0.2),
           o.enthalpy_max.value_or(b.H_liquidus + 0.2),
           o.pressure_pa, b.Tep, model.parameters().Xe, o.normalized_axes, o.write_boundaries};
    if (!std::isfinite(p.cmin) || !std::isfinite(p.cmax) || p.cmin < 0.0 ||
        p.cmax > p.Xe || p.cmax <= p.cmin) {
        throw std::invalid_argument("Require 0 <= phase_c_min < phase_c_max <= Xe");
    }
    if (!std::isfinite(p.hmin) || !std::isfinite(p.hmax) || p.hmax <= p.hmin ||
        !std::isfinite(p.hmax - p.hmin)) {
        throw std::invalid_argument("Require finite phase_h_min < phase_h_max");
    }
    return p;
}

// Including the exact endpoints avoids evaluating a rounded C slightly above Xe.
double Sample(double a, double b, PetscInt i, PetscInt cells) {
    if (i == 0) return a;
    if (i == cells) return b;
    return std::fma(static_cast<double>(i) / static_cast<double>(cells), b - a, a);
}

std::string Signature(const Plot& p, const EutecticModel& model, const std::string& stem) {
    std::ostringstream s;
    s.imbue(std::locale::classic());
    s << std::setprecision(17) << p.nc << ' ' << p.nh << ' ' << p.cmin << ' ' << p.cmax
      << ' ' << p.hmin << ' ' << p.hmax << ' ' << p.pressure << ' ' << p.normalized
      << ' ' << p.boundaries;
    for (const auto& entry : materialEntries) s << ' ' << model.parameters().*(entry.member);
    s << '\n' << stem;
    return s.str();
}

PetscErrorCode CheckSameInputs(MPI_Comm comm, PetscMPIInt rank, const std::string& local) {
    PetscFunctionBeginUser;
    PetscCall(CheckAll(comm, local.size() <= static_cast<std::size_t>(std::numeric_limits<int>::max()),
                       "input signature is too long"));
    int length = rank == 0 ? static_cast<int>(local.size()) : 0;
    PetscCallMPI(MPI_Bcast(&length, 1, MPI_INT, 0, comm));
    std::string reference;
    bool allocated = true;
    try {
        reference.resize(static_cast<std::size_t>(length));
        if (rank == 0) reference = local;
    } catch (const std::exception&) { allocated = false; }
    PetscCall(CheckAll(comm, allocated, "allocate input signature"));
    PetscCallMPI(MPI_Bcast(reference.data(), length, MPI_CHAR, 0, comm));
    PetscCall(CheckAll(comm, reference == local, "material, plot options, or output path differ between ranks"));
    PetscFunctionReturn(PETSC_SUCCESS);
}

std::size_t Product(PetscInt a, PetscInt b) {
    const auto x = static_cast<std::size_t>(a), y = static_cast<std::size_t>(b);
    if (y && x > std::numeric_limits<std::size_t>::max() / y)
        throw std::overflow_error("Local grid size overflows size_t");
    const auto n = x * y;
    if (n > std::vector<double>().max_size() / 3)
        throw std::overflow_error("Local grid is too large for the coordinate buffer");
    return n;
}

struct Samples {
    std::vector<double> xyz;
    std::array<std::vector<double>, realNames.size()> reals;
    std::array<std::vector<int>, intNames.size()> integers;
    std::vector<int> cellRegion;
};

Samples EvaluateRows(const EutecticModel& model, const Plot& p, Rows nodes, Rows cells,
                     PetscMPIInt rank) {
    Samples data;
    const auto n = Product(nodes.count, p.nc + 1);
    data.xyz.resize(3 * n, 0.0);
    for (auto& values : data.reals) values.resize(n);
    for (auto& values : data.integers) values.resize(n);
    data.cellRegion.resize(Product(cells.count, p.nc));
    const auto& d = model.derived();
    std::size_t k = 0;
    for (PetscInt j = nodes.first; j < nodes.first + nodes.count; ++j) {
        const double H = Sample(p.hmin, p.hmax, j, p.nh);
        for (PetscInt i = 0; i <= p.nc; ++i, ++k) {
            const double C = Sample(p.cmin, p.cmax, i, p.nc);
            const auto s = model.evaluate(H, C, p.pressure);
            data.xyz[3 * k] = p.normalized ? C / p.Xe : C;
            data.xyz[3 * k + 1] = p.normalized ? H - p.Tep : H;
            const std::array<double, realNames.size()> values{{
                H, C, s.TDp, s.TDp * d.dT, s.phi1, s.phi2, s.phil, s.cl, s.cs,
                s.dTD_dHD, std::isfinite(s.dTD_dCD) ? s.dTD_dCD : 0.0,
                s.TDp + d.LD * s.phil - H, s.phi2 + s.phil * s.cl - C,
                s.phi1 + s.phi2 + s.phil - 1.0, s.TDp - s.Tep, H * d.h0
            }};
            for (std::size_t f = 0; f < values.size(); ++f) {
                if (!std::isfinite(values[f]))
                    throw std::overflow_error(std::string("Nonfinite output field: ") + realNames[f]);
                data.reals[f][k] = values[f];
            }
            const std::array<int, intNames.size()> flags{{static_cast<int>(s.region),
                s.has_solid ? 1 : 0, s.has_liquid ? 1 : 0,
                s.finite_temperature_derivatives ? 1 : 0, static_cast<int>(rank)}};
            for (std::size_t f = 0; f < flags.size(); ++f) data.integers[f][k] = flags[f];
        }
    }
    k = 0;
    for (PetscInt j = cells.first; j < cells.first + cells.count; ++j) {
        const double H = std::fma((static_cast<double>(j) + 0.5) / p.nh, p.hmax - p.hmin, p.hmin);
        for (PetscInt i = 0; i < p.nc; ++i, ++k) {
            const double C = std::fma((static_cast<double>(i) + 0.5) / p.nc, p.cmax - p.cmin, p.cmin);
            data.cellRegion[k] = static_cast<int>(model.classify(H, C, p.pressure));
        }
    }
    return data;
}

struct Curves {
    std::vector<double> xyz;
    std::vector<int> connectivity, ids;
};

// Each boundary is an affine segment; clip it to the requested H window.
void AddCurve(Curves& curves, const Plot& p, double c0, double h0, double c1, double h1, int id) {
    double t0 = 0.0, t1 = 1.0;
    if (h0 == h1) {
        if (h0 < p.hmin || h0 > p.hmax) return;
    } else {
        const double a = (p.hmin - h0) / (h1 - h0), b = (p.hmax - h0) / (h1 - h0);
        t0 = std::max(0.0, std::min(a, b));
        t1 = std::min(1.0, std::max(a, b));
        if (t1 <= t0) return;
    }
    const int first = static_cast<int>(curves.xyz.size() / 3);
    for (const double t : {t0, t1}) {
        const double C = std::fma(t, c1 - c0, c0), H = std::fma(t, h1 - h0, h0);
        curves.xyz.push_back(p.normalized ? C / p.Xe : C);
        curves.xyz.push_back(p.normalized ? H - p.Tep : H);
        curves.xyz.push_back(1.0e-6); // A small display offset above the surface.
    }
    curves.connectivity.push_back(first);
    curves.connectivity.push_back(first + 1);
    curves.ids.push_back(id);
}

Curves BoundaryCurves(const EutecticModel& model, const Plot& p) {
    Curves curves;
    if (!p.boundaries) return curves;
    const auto a = model.boundaries(p.cmin, p.pressure), b = model.boundaries(p.cmax, p.pressure);
    AddCurve(curves, p, p.cmin, a.Tep, p.cmax, b.Tep, 1);
    AddCurve(curves, p, p.cmin, a.H_eutectic_end, p.cmax, b.H_eutectic_end, 2);
    AddCurve(curves, p, p.cmin, a.H_liquidus, p.cmax, b.H_liquidus, 3);
    if (p.cmin == 0.0) AddCurve(curves, p, 0.0, a.Tmp, 0.0, a.H_liquidus, 4);
    return curves;
}

PetscErrorCode WriteSlab(MPI_Comm comm, hid_t file, const char* name, int dimensions,
                         const hsize_t* shape, const hsize_t* start, const hsize_t* count,
                         hid_t fileType, hid_t memoryType, const void* values) {
    PetscFunctionBeginUser;
    H5Handle space(H5Screate_simple(dimensions, shape, nullptr), H5Sclose);
    PetscCall(CheckAll(comm, space.get() >= 0, "create file dataspace"));
    H5Handle dataset(H5Dcreate2(file, name, fileType, space.get(),
                               H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT), H5Dclose);
    PetscCall(CheckAll(comm, dataset.get() >= 0, name));
    hsize_t items = 1;
    for (int dim = 0; dim < dimensions; ++dim) items *= count[dim];
    const hsize_t memorySize = items ? items : 1;
    H5Handle memory(H5Screate_simple(1, &memorySize, nullptr), H5Sclose);
    PetscCall(CheckAll(comm, memory.get() >= 0, "create memory dataspace"));
    bool selected;
    if (items) selected = H5Sselect_hyperslab(space.get(), H5S_SELECT_SET, start,
                                             nullptr, count, nullptr) >= 0;
    else {
        // All ranks participate, including those without any node/cell rows.
        const auto a = H5Sselect_none(space.get()), b = H5Sselect_none(memory.get());
        selected = a >= 0 && b >= 0;
    }
    PetscCall(CheckAll(comm, selected, "select hyperslab"));
    H5Handle transfer(H5Pcreate(H5P_DATASET_XFER), H5Pclose);
    PetscCall(CheckAll(comm, transfer.get() >= 0, "create transfer properties"));
    PetscCall(CheckAll(comm, H5Pset_dxpl_mpio(transfer.get(), H5FD_MPIO_COLLECTIVE) >= 0,
                       "enable collective dataset I/O"));
    const double unused = 0.0;
    PetscCall(CheckAll(comm, H5Dwrite(dataset.get(), memoryType, memory.get(), space.get(),
                                     transfer.get(), items ? values : &unused) >= 0, name));
    PetscCall(CheckAll(comm, dataset.close() >= 0, "close dataset"));
    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode MakeGroup(MPI_Comm comm, hid_t file, const char* name) {
    PetscFunctionBeginUser;
    H5Handle group(H5Gcreate2(file, name, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT), H5Gclose);
    PetscCall(CheckAll(comm, group.get() >= 0, name));
    PetscCall(CheckAll(comm, group.close() >= 0, "close metadata group"));
    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode WriteScalar(MPI_Comm comm, PetscMPIInt rank, hid_t file,
                           const std::string& name, double value) {
    const hsize_t shape = 1, start = 0, count = rank == 0 ? 1 : 0;
    PetscFunctionBeginUser;
    PetscCall(WriteSlab(comm, file, name.c_str(), 1, &shape, &start, &count,
                        H5T_IEEE_F64LE, H5T_NATIVE_DOUBLE, &value));
    PetscFunctionReturn(PETSC_SUCCESS);
}

std::string EscapeXml(const std::string& text) {
    std::string result;
    for (const char c : text) {
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

void DataItem(std::ostream& out, const std::string& file, const std::string& path,
              const std::string& dimensions, bool integer = false) {
    out << "<DataItem Dimensions=\"" << dimensions << "\" NumberType=\""
        << (integer ? "Int" : "Float") << "\" Precision=\"" << (integer ? 4 : 8)
        << "\" Format=\"HDF\">" << file << ':' << path << "</DataItem>\n";
}

std::string XdmfText(const Plot& p, const Curves& curves, const std::string& h5Filename) {
    const auto file = EscapeXml(h5Filename);
    const auto nodes = std::to_string(p.nh + 1) + " " + std::to_string(p.nc + 1);
    const auto cells = std::to_string(p.nh) + " " + std::to_string(p.nc);
    std::ostringstream x;
    x.imbue(std::locale::classic());
    x << std::setprecision(17) << "<?xml version=\"1.0\"?>\n<Xdmf Version=\"2.0\"><Domain>\n"
      << "<Grid Name=\"Phase\" GridType=\"Collection\" CollectionType=\"Spatial\">\n"
      << "<Grid Name=\"PhaseDiagram\" GridType=\"Uniform\">\n"
      << "<Information Name=\"XAxis\" Value=\"" << (p.normalized ? "CD/Xe" : "CD") << "\"/>\n"
      << "<Information Name=\"YAxis\" Value=\"" << (p.normalized ? "HD-Tep(P)" : "HD") << "\"/>\n"
      << "<Information Name=\"Pressure_Pa\" Value=\"" << p.pressure << "\"/>\n"
      << "<Topology TopologyType=\"2DSMesh\" Dimensions=\"" << nodes << "\"/>\n"
      << "<Geometry GeometryType=\"XYZ\">\n";
    DataItem(x, file, "/Coordinates", nodes + " 3");
    x << "</Geometry>\n";
    for (const auto name : realNames) {
        x << "<Attribute Name=\"" << name << "\" AttributeType=\"Scalar\" Center=\"Node\">\n";
        DataItem(x, file, std::string("/") + name, nodes);
        x << "</Attribute>\n";
    }
    for (const auto name : intNames) {
        x << "<Attribute Name=\"" << name << "\" AttributeType=\"Scalar\" Center=\"Node\">\n";
        DataItem(x, file, std::string("/") + name, nodes, true);
        x << "</Attribute>\n";
    }
    x << "<Attribute Name=\"RegionCell\" AttributeType=\"Scalar\" Center=\"Cell\">\n";
    DataItem(x, file, "/RegionCell", cells, true);
    x << "</Attribute>\n</Grid>\n";
    if (!curves.ids.empty()) {
        x << "<Grid Name=\"PhaseBoundaries\" GridType=\"Uniform\">\n"
          << "<Topology TopologyType=\"Polyline\" NumberOfElements=\"" << curves.ids.size()
          << "\" NodesPerElement=\"2\">\n";
        DataItem(x, file, "/Boundaries/Connectivity", std::to_string(curves.ids.size()) + " 2", true);
        x << "</Topology>\n<Geometry GeometryType=\"XYZ\">\n";
        DataItem(x, file, "/Boundaries/Coordinates", std::to_string(curves.xyz.size() / 3) + " 3");
        x << "</Geometry>\n<Attribute Name=\"BoundaryId\" AttributeType=\"Scalar\" Center=\"Cell\">\n";
        DataItem(x, file, "/Boundaries/Id", std::to_string(curves.ids.size()), true);
        x << "</Attribute>\n</Grid>\n";
    }
    x << "</Grid>\n</Domain></Xdmf>\n";
    return x.str();
}

} // namespace

PetscErrorCode WritePhaseXdmf(MPI_Comm comm, const EutecticModel& model,
                              const PhasePlotOptions& options, const std::string& stem) {
    PetscMPIInt rank, size;
    PetscFunctionBeginUser;
    PetscCallMPI(MPI_Comm_rank(comm, &rank));
    PetscCallMPI(MPI_Comm_size(comm, &size));
    Plot p{};
    std::string signature;
    bool valid = true;
    try {
        if (stem.empty()) throw std::invalid_argument("Output stem is empty");
        p = Resolve(model, options);
        signature = Signature(p, model, stem);
    } catch (const std::exception& error) {
        std::fprintf(stderr, "[rank %d] %s\n", static_cast<int>(rank), error.what());
        valid = false;
    }
    PetscCall(CheckAll(comm, valid, "validate plot inputs"));
    PetscCall(CheckSameInputs(comm, rank, signature));
    const Rows nodes = OwnedRows(p.nh + 1, rank, size), cells = OwnedRows(p.nh, rank, size);
    Samples data;
    Curves curves;
    std::filesystem::path h5Path;
    std::string xml;
    bool sampled = true;
    try {
        h5Path = stem + ".h5";
        if (h5Path.filename().string().find(':') != std::string::npos)
            throw std::invalid_argument("Output filename cannot contain ':' in an XDMF HDF reference");
        data = EvaluateRows(model, p, nodes, cells, rank);
        curves = BoundaryCurves(model, p);
        if (rank == 0) xml = XdmfText(p, curves, h5Path.filename().string());
    } catch (const std::exception& error) {
        std::fprintf(stderr, "[rank %d] %s\n", static_cast<int>(rank), error.what());
        sampled = false;
    }
    PetscCall(CheckAll(comm, sampled, "sample phase diagram"));
    bool directoryOk = true;
    if (rank == 0 && !h5Path.parent_path().empty()) {
        std::error_code error;
        std::filesystem::create_directories(h5Path.parent_path(), error);
        directoryOk = !error;
    }
    PetscCall(CheckAll(comm, directoryOk, "create output directory"));

    H5Handle access(H5Pcreate(H5P_FILE_ACCESS), H5Pclose);
    PetscCall(CheckAll(comm, access.get() >= 0, "create file access properties"));
    PetscCall(CheckAll(comm, H5Pset_fapl_mpio(access.get(), comm, MPI_INFO_NULL) >= 0,
                       "enable MPI file access"));
    H5Handle file(H5Fcreate(h5Path.string().c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, access.get()), H5Fclose);
    PetscCall(CheckAll(comm, file.get() >= 0, "create HDF5 file"));
    const hsize_t shape[3] = {static_cast<hsize_t>(p.nh + 1), static_cast<hsize_t>(p.nc + 1), 3};
    const hsize_t start[3] = {static_cast<hsize_t>(nodes.first), 0, 0};
    const hsize_t count[3] = {static_cast<hsize_t>(nodes.count), shape[1], 3};
    PetscCall(WriteSlab(comm, file.get(), "/Coordinates", 3, shape, start, count,
                        H5T_IEEE_F64LE, H5T_NATIVE_DOUBLE, data.xyz.data()));
    for (std::size_t f = 0; f < realNames.size(); ++f)
        PetscCall(WriteSlab(comm, file.get(), realNames[f], 2, shape, start, count,
                            H5T_IEEE_F64LE, H5T_NATIVE_DOUBLE, data.reals[f].data()));
    for (std::size_t f = 0; f < intNames.size(); ++f)
        PetscCall(WriteSlab(comm, file.get(), intNames[f], 2, shape, start, count,
                            H5T_STD_I32LE, H5T_NATIVE_INT, data.integers[f].data()));
    const hsize_t cellShape[2] = {static_cast<hsize_t>(p.nh), static_cast<hsize_t>(p.nc)};
    const hsize_t cellStart[2] = {static_cast<hsize_t>(cells.first), 0};
    const hsize_t cellCount[2] = {static_cast<hsize_t>(cells.count), cellShape[1]};
    PetscCall(WriteSlab(comm, file.get(), "/RegionCell", 2, cellShape, cellStart, cellCount,
                        H5T_STD_I32LE, H5T_NATIVE_INT, data.cellRegion.data()));

    PetscCall(MakeGroup(comm, file.get(), "/Material"));
    for (const auto& entry : materialEntries)
        PetscCall(WriteScalar(comm, rank, file.get(), std::string("/Material/") + entry.name,
                              model.parameters().*(entry.member)));
    PetscCall(MakeGroup(comm, file.get(), "/Scales"));
    const auto& d = model.derived();
    const std::array<std::pair<const char*, double>, 14> scales{{
        {"dT", d.dT}, {"h0", d.h0}, {"gamma", d.gamma}, {"LD", d.LD},
        {"TDm0", d.TDm0}, {"TDe0", d.TDe0}, {"invk0", d.invk0},
        {"l0", d.l0}, {"u0", d.u0}, {"p0", d.p0}, {"t0", d.t0},
        {"Pressure_Pa", p.pressure}, {"Tep", p.Tep}, {"NormalizedAxes", p.normalized ? 1.0 : 0.0}
    }};
    for (const auto& entry : scales)
        PetscCall(WriteScalar(comm, rank, file.get(), std::string("/Scales/") + entry.first, entry.second));
    if (!curves.ids.empty()) {
        PetscCall(MakeGroup(comm, file.get(), "/Boundaries"));
        const hsize_t zero[2] = {0, 0};
        const hsize_t xyzShape[2] = {static_cast<hsize_t>(curves.xyz.size() / 3), 3};
        const hsize_t xyzCount[2] = {rank == 0 ? xyzShape[0] : 0, 3};
        PetscCall(WriteSlab(comm, file.get(), "/Boundaries/Coordinates", 2, xyzShape, zero, xyzCount,
                            H5T_IEEE_F64LE, H5T_NATIVE_DOUBLE, curves.xyz.data()));
        const hsize_t lineShape[2] = {static_cast<hsize_t>(curves.ids.size()), 2};
        const hsize_t lineCount[2] = {rank == 0 ? lineShape[0] : 0, 2};
        PetscCall(WriteSlab(comm, file.get(), "/Boundaries/Connectivity", 2, lineShape, zero, lineCount,
                            H5T_STD_I32LE, H5T_NATIVE_INT, curves.connectivity.data()));
        PetscCall(WriteSlab(comm, file.get(), "/Boundaries/Id", 1, lineShape, zero, lineCount,
                            H5T_STD_I32LE, H5T_NATIVE_INT, curves.ids.data()));
    }
    PetscCall(CheckAll(comm, file.close() >= 0, "close HDF5 file"));
    bool xmlOk = true;
    if (rank == 0) {
        std::ofstream out(stem + ".xdmf", std::ios::trunc);
        out << xml;
        out.close();
        xmlOk = !out.fail();
    }
    PetscCall(CheckAll(comm, xmlOk, "write XDMF file"));
    PetscFunctionReturn(PETSC_SUCCESS);
}
