# Include explicitly from src/mfem/test/CMakeLists.txt.
include_guard(GLOBAL)

add_executable(test_local_matrix "${CMAKE_CURRENT_LIST_DIR}/test_local_matrix.cpp")
target_link_libraries(test_local_matrix PRIVATE mantle_mfem)

add_test(NAME mfem_local_matrix_unit_serial
    COMMAND $<TARGET_FILE:test_local_matrix> -expected_ranks 1)
set_tests_properties(mfem_local_matrix_unit_serial PROPERTIES
    LABELS "mfem;local_matrix;serial" PROCESSORS 1 TIMEOUT 180)

function(mantle_add_local_matrix_test name kind px py nx ny)
    math(EXPR ranks "${px} * ${py}")
    # nx/ny count DMDA vertices, not cells.
    set(arguments -local_matrix_mesh_type ${kind} -expected_ranks ${ranks}
        -mesh_px ${px} -mesh_py ${py} -mesh_nx ${nx} -mesh_ny ${ny})
    if(ranks EQUAL 1)
        add_test(NAME ${name}
            COMMAND $<TARGET_FILE:test_local_matrix> ${arguments})
        set(mode serial)
    else()
        # Reuse the root's PETSc-compatible MPICH launcher and flags.
        # Do not search for a different system mpiexec here.
        if(NOT MPIEXEC_EXECUTABLE)
            return()
        endif()
        add_test(NAME ${name}
            COMMAND "${MPIEXEC_EXECUTABLE}" ${MPIEXEC_NUMPROC_FLAG} ${ranks}
                ${MPIEXEC_PREFLAGS} $<TARGET_FILE:test_local_matrix>
                ${MPIEXEC_POSTFLAGS} ${arguments})
        set(mode mpi)
    endif()
    set_tests_properties(${name} PROPERTIES
        LABELS "mfem;local_matrix;mesh;${mode}" PROCESSORS ${ranks} TIMEOUT 180)
endfunction()

foreach(kind IN ITEMS 0 1 2)
    if(kind EQUAL 0)
        set(mesh_name rectangular)
    elseif(kind EQUAL 1)
        set(mesh_name quadrilateral)
    else()
        set(mesh_name stretched)
    endif()
    set(prefix "mfem_local_matrix_${mesh_name}")
    mantle_add_local_matrix_test(${prefix}_serial  ${kind} 1 1 9 7)
    mantle_add_local_matrix_test(${prefix}_mpi_2x1 ${kind} 2 1 9 7)
    mantle_add_local_matrix_test(${prefix}_mpi_1x2 ${kind} 1 2 9 7)
    mantle_add_local_matrix_test(${prefix}_mpi_2x2 ${kind} 2 2 9 7)
endforeach()

# One physical cell: on multiple ranks, some ranks own no cells.
mantle_add_local_matrix_test(mfem_local_matrix_minimal_serial  0 1 1 2 2)
mantle_add_local_matrix_test(mfem_local_matrix_minimal_mpi_2x1 0 2 1 2 2)
mantle_add_local_matrix_test(mfem_local_matrix_minimal_mpi_1x2 0 1 2 2 2)
mantle_add_local_matrix_test(mfem_local_matrix_minimal_mpi_2x2 0 2 2 2 2)

if(NOT MPIEXEC_EXECUTABLE)
    message(STATUS "Local matrix: registering serial tests only (MPI launcher unavailable)")
endif()
