# Add this subdirectory after src/core in the existing root CMakeLists.txt.
# Compiler selection and MPIEXEC_EXECUTABLE stay controlled by the root project.
set(HDF5_PREFER_PARALLEL TRUE)
find_package(HDF5 REQUIRED COMPONENTS C)
if(NOT HDF5_IS_PARALLEL)
    message(FATAL_ERROR
        "Mesh export requires parallel HDF5 built with the same MPI as PETSc. "
        "Set HDF5_ROOT to that HDF5 installation.")
endif()

add_library(mantle_mesh_output mesh_output.cpp)
target_include_directories(mantle_mesh_output PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}")
target_compile_features(mantle_mesh_output PUBLIC cxx_std_17)
target_link_libraries(mantle_mesh_output PUBLIC mantle_core PRIVATE HDF5::HDF5)

add_executable(export_mesh export_mesh.cpp)
target_link_libraries(export_mesh PRIVATE mantle_mesh_output)

if(BUILD_TESTING)
    add_test(NAME mesh_export_serial
        COMMAND export_mesh -mesh_nx 8 -mesh_ny 6 -mesh_type 0
                -output "${CMAKE_CURRENT_BINARY_DIR}/output/rectangular")
    set_tests_properties(mesh_export_serial PROPERTIES TIMEOUT 60)
    if(MPIEXEC_EXECUTABLE)
        add_test(NAME mesh_export_mpi_2x2
            COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 4
                    ${MPIEXEC_PREFLAGS} $<TARGET_FILE:export_mesh> ${MPIEXEC_POSTFLAGS}
                    -mesh_px 2 -mesh_py 2 -mesh_nx 8 -mesh_ny 6 -mesh_type 1
                    -output "${CMAKE_CURRENT_BINARY_DIR}/output/quadrilateral")
        set_tests_properties(mesh_export_mpi_2x2 PROPERTIES PROCESSORS 4 TIMEOUT 60)
    endif()
endif()
