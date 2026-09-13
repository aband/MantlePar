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

# add_subdirectory(src/phase)
# add_subdirectory(src/io)
# add_subdirectory(src/mfem)
# add_subdirectory(src/mlweno_array)
# add_subdirectory(src/transport_array)
