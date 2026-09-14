# Include from src/mfem/test/CMakeLists.txt after mantle_mfem is defined.
include_guard(GLOBAL)

add_executable(test_dof_map "${CMAKE_CURRENT_LIST_DIR}/test_dof_map.cpp")
target_link_libraries(test_dof_map PRIVATE mantle_mfem)

function(mantle_add_dof_map_test name kind px py nx ny small_case)
    math(EXPR ranks "${px} * ${py}")
    # mesh_nx/mesh_ny count vertices, not cells.
    set(arguments -dof_mesh_type ${kind} -expected_ranks ${ranks}
        -mesh_px ${px} -mesh_py ${py} -mesh_nx ${nx} -mesh_ny ${ny})
    if(small_case)
        list(APPEND arguments -dof_small_case 1)
    endif()

    if(ranks EQUAL 1)
        add_test(NAME ${name}
            COMMAND $<TARGET_FILE:test_dof_map> ${arguments})
        set(mode serial)
    else()
        # Reuse the root's PETSc-compatible MPICH launcher and flags.
        if(NOT MPIEXEC_EXECUTABLE)
            return()
        endif()
        add_test(NAME ${name}
            COMMAND "${MPIEXEC_EXECUTABLE}" ${MPIEXEC_NUMPROC_FLAG} ${ranks}
                ${MPIEXEC_PREFLAGS} $<TARGET_FILE:test_dof_map>
                ${MPIEXEC_POSTFLAGS} ${arguments})
        set(mode mpi)
    endif()
    set_tests_properties(${name} PROPERTIES
        LABELS "mfem;dof_map;mesh;${mode}" PROCESSORS ${ranks} TIMEOUT 120)
endfunction()

foreach(kind IN ITEMS 0 1 2)
    if(kind EQUAL 0)
        set(mesh_name rectangular)
    elseif(kind EQUAL 1)
        set(mesh_name quadrilateral)
    else()
        set(mesh_name stretched)
    endif()
    set(prefix "mfem_dof_map_${mesh_name}")
    mantle_add_dof_map_test(${prefix}_serial  ${kind} 1 1 9 7 FALSE)
    mantle_add_dof_map_test(${prefix}_mpi_2x1 ${kind} 2 1 9 7 FALSE)
    mantle_add_dof_map_test(${prefix}_mpi_1x2 ${kind} 1 2 9 7 FALSE)
    mantle_add_dof_map_test(${prefix}_mpi_2x2 ${kind} 2 2 9 7 FALSE)
endforeach()

# A single cell exercises ranks with no owned cells; the 2x2 process grid
# also exercises a rank with no owned HDiv velocity DOFs.
mantle_add_dof_map_test(mfem_dof_map_minimal_serial  0 1 1 2 2 TRUE)
mantle_add_dof_map_test(mfem_dof_map_minimal_mpi_2x1 0 2 1 2 2 TRUE)
mantle_add_dof_map_test(mfem_dof_map_minimal_mpi_1x2 0 1 2 2 2 TRUE)
mantle_add_dof_map_test(mfem_dof_map_minimal_mpi_2x2 0 2 2 2 2 TRUE)

if(NOT MPIEXEC_EXECUTABLE)
    message(STATUS "DofMap: registering serial tests only (MPI launcher unavailable)")
endif()
