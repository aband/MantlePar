#include "mesh_output.h"
#include "mesh.h"

#include <hdf5.h>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <vector>

#ifndef H5_HAVE_PARALLEL
#error "mesh_output requires HDF5 built with parallel MPI support"
#endif

namespace {

// Scoped ownership for HDF5 handles; explicit file close is checked below.
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

// All ranks take the same error path before the next collective HDF5 call.
PetscErrorCode CheckAll(MPI_Comm comm, bool ok, const char* operation)
{
    PetscMPIInt local = ok ? 1 : 0, global = 0;
    PetscFunctionBeginUser;
    PetscCallMPI(MPI_Allreduce(&local, &global, 1, MPI_INT, MPI_MIN, comm));
    PetscCheck(global, comm, PETSC_ERR_LIB, "Mesh export failed: %s", operation);
    PetscFunctionReturn(PETSC_SUCCESS);
}

std::string EscapeXml(const std::string& text)
{
    std::string escaped;
    for (char c : text) {
        switch (c) {
        case '&': escaped += "&amp;"; break;
        case '<': escaped += "&lt;"; break;
        case '>': escaped += "&gt;"; break;
        case '"': escaped += "&quot;"; break;
        case '\'': escaped += "&apos;"; break;
        default: escaped += c;
        }
    }
    return escaped;
}

std::string XdmfText(PetscInt M, PetscInt N, const std::string& h5Filename)
{
    const std::string file = EscapeXml(h5Filename);
    std::ostringstream xml;
    xml << "<?xml version=\"1.0\"?>\n"
        << "<Xdmf Version=\"2.0\">\n  <Domain>\n"
        << "    <Grid Name=\"Mesh\" GridType=\"Uniform\">\n"
        << "      <Topology TopologyType=\"2DSMesh\" Dimensions=\"" << N << ' ' << M << "\"/>\n"
        << "      <Geometry GeometryType=\"XYZ\">\n"
        << "        <DataItem Dimensions=\"" << N << ' ' << M << " 3\" NumberType=\"Float\" Precision=\"8\" Format=\"HDF\">\n"
        << "          " << file << ":/Coordinates\n"
        << "        </DataItem>\n      </Geometry>\n"
        << "      <Attribute Name=\"VertexRank\" AttributeType=\"Scalar\" Center=\"Node\">\n"
        << "        <DataItem Dimensions=\"" << N << ' ' << M << "\" NumberType=\"Int\" Precision=\"4\" Format=\"HDF\">\n"
        << "          " << file << ":/VertexRank\n"
        << "        </DataItem>\n      </Attribute>\n"
        << "    </Grid>\n  </Domain>\n</Xdmf>\n";
    return xml.str();
}

PetscErrorCode WriteSlab(MPI_Comm comm, hid_t file, const char* name,
                          int dimensions, const hsize_t* shape,
                          const hsize_t* start, const hsize_t* count,
                          hid_t fileType, hid_t memoryType, const void* data)
{
    PetscFunctionBeginUser;
    H5Handle space(H5Screate_simple(dimensions, shape, nullptr), H5Sclose);
    PetscCall(CheckAll(comm, space.get() >= 0, "create file dataspace"));
    H5Handle dataset(H5Dcreate2(file, name, fileType, space.get(),
                                H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT), H5Dclose);
    PetscCall(CheckAll(comm, dataset.get() >= 0, "create dataset"));

    hsize_t items = 1;
    for (int d = 0; d < dimensions; ++d) items *= count[d];
    const hsize_t memorySize = items ? items : 1;
    H5Handle memory(H5Screate_simple(1, &memorySize, nullptr), H5Sclose);
    PetscCall(CheckAll(comm, memory.get() >= 0, "create memory dataspace"));
    bool selected = true;
    if (items) {
        selected = H5Sselect_hyperslab(space.get(), H5S_SELECT_SET,
                                       start, nullptr, count, nullptr) >= 0;
    } else {
        // Empty ranks still participate in the collective write.
        const herr_t a = H5Sselect_none(space.get());
        const herr_t b = H5Sselect_none(memory.get());
        selected = a >= 0 && b >= 0;
    }
    PetscCall(CheckAll(comm, selected, "select owned hyperslab"));
    H5Handle transfer(H5Pcreate(H5P_DATASET_XFER), H5Pclose);
    PetscCall(CheckAll(comm, transfer.get() >= 0, "create transfer property list"));
    PetscCall(CheckAll(comm, H5Pset_dxpl_mpio(transfer.get(), H5FD_MPIO_COLLECTIVE) >= 0,
                       "enable collective dataset I/O"));
    const double unused = 0.0;
    PetscCall(CheckAll(comm, H5Dwrite(dataset.get(), memoryType, memory.get(),
                                     space.get(), transfer.get(), items ? data : &unused) >= 0,
                       "write dataset"));
    PetscCall(CheckAll(comm, dataset.close() >= 0, "close dataset"));
    PetscFunctionReturn(PETSC_SUCCESS);
}

} // namespace

PetscErrorCode WriteMeshXdmf(DM dm, Vec vertices, const std::string& stem)
{
    PetscInt M, N, xs, ys, xm, ym;
    PetscMPIInt rank;
    const PetscScalar*** a = nullptr;
    const MPI_Comm comm = PetscObjectComm(reinterpret_cast<PetscObject>(dm));
    PetscFunctionBeginUser;
    PetscCheck(!stem.empty(), comm, PETSC_ERR_ARG_WRONG, "Output stem must not be empty");
    PetscCall(ValidateMesh(dm, vertices));
    PetscCallMPI(MPI_Comm_rank(comm, &rank));
    PetscCall(DMDAGetInfo(dm, nullptr, &M, &N, nullptr, nullptr, nullptr,
                          nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr));
    PetscCall(DMDAGetCorners(dm, &xs, &ys, nullptr, &xm, &ym, nullptr));

    const std::filesystem::path h5Path(stem + ".h5");
    bool directoryOk = true;
    if (rank == 0 && !h5Path.parent_path().empty()) {
        std::error_code error;
        std::filesystem::create_directories(h5Path.parent_path(), error);
        directoryOk = !error;
    }
    PetscCall(CheckAll(comm, directoryOk, "create output directory"));

    const std::size_t localSize = static_cast<std::size_t>(xm) * static_cast<std::size_t>(ym);
    std::vector<double> coordinates(3 * localSize, 0.0);
    std::vector<int> owners(localSize, rank);
    bool finite = true;
    PetscCall(DMDAVecGetArrayDOFRead(dm, vertices, &a));
    std::size_t k = 0;
    for (PetscInt j = ys; j < ys + ym; ++j) {
        for (PetscInt i = xs; i < xs + xm; ++i, ++k) {
            coordinates[3 * k] = static_cast<double>(PetscRealPart(a[j][i][0]));
            coordinates[3 * k + 1] = static_cast<double>(PetscRealPart(a[j][i][1]));
            finite = finite && std::isfinite(coordinates[3 * k])
                            && std::isfinite(coordinates[3 * k + 1]);
        }
    }
    PetscCall(DMDAVecRestoreArrayDOFRead(dm, vertices, &a));
    PetscCall(CheckAll(comm, finite, "convert coordinates to Float64"));

    H5Handle access(H5Pcreate(H5P_FILE_ACCESS), H5Pclose);
    PetscCall(CheckAll(comm, access.get() >= 0, "create file access property list"));
    PetscCall(CheckAll(comm, H5Pset_fapl_mpio(access.get(), comm, MPI_INFO_NULL) >= 0,
                       "enable MPI file access"));
    H5Handle file(H5Fcreate(h5Path.string().c_str(), H5F_ACC_TRUNC,
                            H5P_DEFAULT, access.get()), H5Fclose);
    PetscCall(CheckAll(comm, file.get() >= 0, "create HDF5 file"));

    // XDMF and HDF5 both use slowest dimension first: [j][i][component].
    const hsize_t shape[3] = {static_cast<hsize_t>(N), static_cast<hsize_t>(M), 3};
    const hsize_t start[3] = {static_cast<hsize_t>(ys), static_cast<hsize_t>(xs), 0};
    const hsize_t count[3] = {static_cast<hsize_t>(ym), static_cast<hsize_t>(xm), 3};
    PetscCall(WriteSlab(comm, file.get(), "/Coordinates", 3, shape, start, count,
                        H5T_IEEE_F64LE, H5T_NATIVE_DOUBLE, coordinates.data()));
    PetscCall(WriteSlab(comm, file.get(), "/VertexRank", 2, shape, start, count,
                        H5T_STD_I32LE, H5T_NATIVE_INT, owners.data()));
    PetscCall(CheckAll(comm, file.close() >= 0, "close HDF5 file"));

    // Every HDF5 writer has finished before rank 0 publishes the small XML file.
    bool xmlOk = true;
    if (rank == 0) {
        std::ofstream out(stem + ".xdmf", std::ios::trunc);
        out << XdmfText(M, N, h5Path.filename().string());
        out.close();
        xmlOk = !out.fail();
    }
    PetscCall(CheckAll(comm, xmlOk, "write XDMF file"));
    PetscFunctionReturn(PETSC_SUCCESS);
}
