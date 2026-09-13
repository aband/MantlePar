# Add this directory after src/phase in the root CMakeLists.txt.
# Compiler selection and the MPI launcher remain controlled by the root project.
set(HDF5_PREFER_PARALLEL TRUE)
find_package(HDF5 REQUIRED COMPONENTS C)
if(NOT HDF5_IS_PARALLEL)
    message(FATAL_ERROR
        "Phase export requires parallel HDF5 built with the same MPI as PETSc. "
        "Set HDF5_ROOT to that HDF5 installation.")
endif()

add_library(mantle_phase_output phase_output.cpp)
target_include_directories(mantle_phase_output PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}")
target_compile_features(mantle_phase_output PUBLIC cxx_std_17)
target_link_libraries(mantle_phase_output
    PUBLIC mantle_phase PkgConfig::PETSC MPI::MPI_CXX
    PRIVATE HDF5::HDF5
)

add_executable(export_phase export_phase.cpp)
target_link_libraries(export_phase PRIVATE mantle_phase_output)

if(BUILD_TESTING)
    add_test(NAME phase_export_serial
        COMMAND export_phase -phase_nc 8 -phase_nh 12
            -output "${CMAKE_CURRENT_BINARY_DIR}/output/serial")
    add_test(NAME phase_export_parameters
        COMMAND export_phase -phase_nc 8 -phase_nh 12
            -phase_pressure 1e9 -phase_tm0 2150 -phase_te0 1500
            -phase_cp 1300 -phase_l 6e5 -phase_xe 0.3
            -output "${CMAKE_CURRENT_BINARY_DIR}/output/parameters")
    set_tests_properties(phase_export_serial phase_export_parameters
        PROPERTIES TIMEOUT 60 LABELS "phase;output")
    if(MPIEXEC_EXECUTABLE)
        add_test(NAME phase_export_mpi_2
            COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 2
                ${MPIEXEC_PREFLAGS} $<TARGET_FILE:export_phase> ${MPIEXEC_POSTFLAGS}
                -phase_nc 8 -phase_nh 12
                -output "${CMAKE_CURRENT_BINARY_DIR}/output/mpi_2")
        # Two node rows and one cell row on four ranks exercise empty selections.
        add_test(NAME phase_export_mpi_empty_ranks
            COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 4
                ${MPIEXEC_PREFLAGS} $<TARGET_FILE:export_phase> ${MPIEXEC_POSTFLAGS}
                -phase_nc 3 -phase_nh 1
                -output "${CMAKE_CURRENT_BINARY_DIR}/output/mpi_empty_ranks")
        set_tests_properties(phase_export_mpi_2
            PROPERTIES PROCESSORS 2 TIMEOUT 60 LABELS "phase;output;mpi")
        set_tests_properties(phase_export_mpi_empty_ranks
            PROPERTIES PROCESSORS 4 TIMEOUT 60 LABELS "phase;output;mpi")
    endif()
endif()
