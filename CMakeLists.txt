cmake_minimum_required(VERSION 3.20)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Useful for clangd in Vim.
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Remove the lines below if you do not wish to have PETSc determine the compilers
execute_process ( COMMAND pkg-config PETSc --variable=ccompiler COMMAND tr -d '\n' OUTPUT_VARIABLE C_COMPILER)
SET(CMAKE_C_COMPILER ${C_COMPILER})
execute_process ( COMMAND pkg-config PETSc --variable=cxxcompiler COMMAND tr -d '\n' OUTPUT_VARIABLE CXX_COMPILER)
if (CXX_COMPILER) 
  SET(CMAKE_CXX_COMPILER ${CXX_COMPILER})
endif (CXX_COMPILER)
execute_process ( COMMAND pkg-config PETSc --variable=fcompiler COMMAND tr -d '\n' OUTPUT_VARIABLE FORTRAN_COMPILER)
if (FORTRAN_COMPILER)
  SET(CMAKE_Fortran_COMPILER ${FORTRAN_COMPILER})                                                                                                                                                                                      
  enable_language(Fortran)
endif (FORTRAN_COMPILER)

project(MantleSolver LANGUAGES C CXX)

# Creates BUILD_TESTING and enables CTest when it is ON.
include(CTest)

find_package(PkgConfig REQUIRED)

# Use the launcher from the same MPI installation as PETSc's compiler.
get_filename_component(PETSC_MPI_BIN "${C_COMPILER}" DIRECTORY)

set(MPIEXEC_EXECUTABLE "${PETSC_MPI_BIN}/mpiexec"
    CACHE FILEPATH "MPI launcher matching PETSc" FORCE)

find_package(MPI REQUIRED COMPONENTS C CXX)

pkg_search_module(PETSC REQUIRED IMPORTED_TARGET PETSc petsc)
pkg_check_modules(LAPACKE REQUIRED IMPORTED_TARGET lapacke)

# Register subsequent modules as we reconstruct them:
add_subdirectory(src/core)
add_subdirectory(tools/mesh_export)
add_subdirectory(src/phase)
add_subdirectory(tools/phase_export)
add_subdirectory(src/mfem)
add_subdirectory(tools/brmixed_export)
add_subdirectory(tools/hdivmixed_export)
add_subdirectory(src/mlweno)
# add_subdirectory(src/transport_array)

# Shared YAML-driven Stokes/Darcy executable at the repository root.
# Independent of BUILD_TESTING.
option(MANTLE_BUILD_DRIVER "Build the shared steady-flow driver" ON)

if(MANTLE_BUILD_DRIVER)
    foreach(_driver_file IN ITEMS
            driver.cpp
            src/core/input.h
            src/core/input.cpp
            src/core/input.cmake)
        if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/${_driver_file}")
            message(FATAL_ERROR
                "The shared driver requires ${_driver_file}. "
                "Copy the previously generated driver/input-reader files into MantlePar.")
        endif()
    endforeach()
    unset(_driver_file)

    # The current src/core/CMakeLists.txt does not register the input reader.
    # Include it once. If moved into src/core/CMakeLists.txt later, remove this line.
    include("${CMAKE_CURRENT_SOURCE_DIR}/src/core/input.cmake")

    # Discover HDF5 in root scope so its imported target is visible here.
    set(HDF5_PREFER_PARALLEL TRUE)
    find_package(HDF5 REQUIRED COMPONENTS C)
    if(NOT HDF5_IS_PARALLEL)
        message(FATAL_ERROR
            "The shared driver requires parallel HDF5 built with PETSc's MPICH. "
            "Set HDF5_ROOT to that HDF5 installation and reconfigure.")
    endif()

    add_executable(mantle_driver
        "${CMAKE_CURRENT_SOURCE_DIR}/driver.cpp"
    )
    target_compile_features(mantle_driver PRIVATE cxx_std_17)
    target_link_libraries(mantle_driver
        PRIVATE
            mantle_mfem
            MPI::MPI_CXX
            HDF5::HDF5
    )
endif()

