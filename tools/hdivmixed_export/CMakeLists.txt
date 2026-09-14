set(HDF5_PREFER_PARALLEL TRUE)
find_package(HDF5 REQUIRED COMPONENTS C)

if(NOT HDF5_IS_PARALLEL)
    message(FATAL_ERROR
        "HDivMixed export requires parallel HDF5 built with PETSc's MPI. "
        "Set HDF5_ROOT to that installation.")
endif()

add_library(mantle_hdivmixed_output
    hdivmixed_output.cpp
    hdivmixed_output.h)
target_include_directories(mantle_hdivmixed_output
    PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}")
target_compile_features(mantle_hdivmixed_output PUBLIC cxx_std_17)
target_link_libraries(mantle_hdivmixed_output
    PUBLIC mantle_mfem MPI::MPI_CXX
    PRIVATE HDF5::HDF5)

add_executable(export_hdivmixed export_hdivmixed.cpp)
target_link_libraries(export_hdivmixed PRIVATE mantle_hdivmixed_output)
