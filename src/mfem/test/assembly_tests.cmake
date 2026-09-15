# Include explicitly from src/mfem/test/CMakeLists.txt.
include_guard(GLOBAL)

add_executable(test_assembly "${CMAKE_CURRENT_LIST_DIR}/test_assembly.cpp")
target_link_libraries(test_assembly PRIVATE mantle_mfem)

function(mantle_add_assembly_test name kind suite px py nx ny)
    math(EXPR ranks "${px} * ${py}")
    # nx/ny are DMDA vertex counts, including the physical boundary.
    set(arguments -assembly_mesh_type ${kind} -assembly_case ${suite}
        -expected_ranks ${ranks} -mesh_px ${px} -mesh_py ${py}
        -mesh_nx ${nx} -mesh_ny ${ny})
    if(ranks EQUAL 1)
        add_test(NAME ${name}
            COMMAND $<TARGET_FILE:test_assembly> ${arguments})
        set(mode serial)
    else()
        # Preserve the root's PETSc-compatible MPICH launcher and flags.
        # Never rediscover MPI or substitute another system mpiexec here.
        if(NOT MPIEXEC_EXECUTABLE)
            return()
        endif()
        add_test(NAME ${name}
            COMMAND "${MPIEXEC_EXECUTABLE}" ${MPIEXEC_NUMPROC_FLAG} ${ranks}
                ${MPIEXEC_PREFLAGS} $<TARGET_FILE:test_assembly>
                ${MPIEXEC_POSTFLAGS} ${arguments})
        set(mode mpi)
    endif()
    if(suite EQUAL 0)
        set(category mesh)
    else()
        set(category errors)
    endif()
    set_tests_properties(${name} PROPERTIES
        LABELS "mfem;assembly;${category};${mode}" PROCESSORS ${ranks} TIMEOUT 180)
endfunction()

foreach(kind IN ITEMS 0 1 2)
    if(kind EQUAL 0)
        set(mesh_name rectangular)
    elseif(kind EQUAL 1)
        set(mesh_name quadrilateral)
    else()
        set(mesh_name stretched)
    endif()
    set(prefix "mfem_assembly_${mesh_name}")
    mantle_add_assembly_test(${prefix}_serial  ${kind} 0 1 1 9 7)
    mantle_add_assembly_test(${prefix}_mpi_2x1 ${kind} 0 2 1 9 7)
    mantle_add_assembly_test(${prefix}_mpi_1x2 ${kind} 0 1 2 9 7)
    mantle_add_assembly_test(${prefix}_mpi_2x2 ${kind} 0 2 2 9 7)
endforeach()

# A single cell exercises empty cell/pressure owners; the 2x2 process layout
# additionally has a rank owning no HDiv velocity DOFs.
mantle_add_assembly_test(mfem_assembly_minimal_serial  0 0 1 1 2 2)
mantle_add_assembly_test(mfem_assembly_minimal_mpi_2x1 0 0 2 1 2 2)
mantle_add_assembly_test(mfem_assembly_minimal_mpi_1x2 0 0 1 2 2 2)
mantle_add_assembly_test(mfem_assembly_minimal_mpi_2x2 0 0 2 2 2 2)

# Deliberate failures on one rank, collective error propagation, PETSc stack
# discipline, unchanged live outputs, cleanup and successful recovery.
mantle_add_assembly_test(mfem_assembly_errors_serial  1 1 1 1 6 5)
mantle_add_assembly_test(mfem_assembly_errors_mpi_2x1 1 1 2 1 6 5)
mantle_add_assembly_test(mfem_assembly_errors_mpi_1x2 1 1 1 2 6 5)
mantle_add_assembly_test(mfem_assembly_errors_mpi_2x2 1 1 2 2 6 5)

if(NOT MPIEXEC_EXECUTABLE)
    message(STATUS "Assembly: registering serial tests only (MPI launcher unavailable)")
endif()
