cmake_minimum_required(VERSION 3.20)
project(MantleSolver LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Useful for clangd in Vim.
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Creates BUILD_TESTING and enables CTest when it is ON.
include(CTest)

find_package(PkgConfig REQUIRED)
find_package(MPI REQUIRED COMPONENTS C CXX)

pkg_search_module(PETSC REQUIRED IMPORTED_TARGET PETSc petsc)
pkg_check_modules(LAPACKE REQUIRED IMPORTED_TARGET lapacke)

#add_subdirectory(src/core)

# Register subsequent modules as we reconstruct them:
# add_subdirectory(src/phase)
# add_subdirectory(src/io)
# add_subdirectory(src/mfem)
# add_subdirectory(src/mlweno_array)
# add_subdirectory(src/transport_array)
