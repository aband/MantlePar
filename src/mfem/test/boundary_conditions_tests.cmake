# Include explicitly from src/mfem/test/CMakeLists.txt.
include_guard(GLOBAL)

if(NOT BUILD_TESTING)
    return()
endif()

add_executable(test_boundary_conditions
    "${CMAKE_CURRENT_LIST_DIR}/test_boundary_conditions.cpp")
target_link_libraries(test_boundary_conditions PRIVATE mantle_mfem)

function(mantle_add_boundary_conditions_test name kind suite px py nx ny)
    math(EXPR ranks "${px} * ${py}")
    # nx/ny count DMDA vertices, including physical boundary vertices.
    set(arguments -boundary_mesh_type ${kind} -boundary_case ${suite}
        -expected_ranks ${ranks} -mesh_px ${px} -mesh_py ${py}
        -mesh_nx ${nx} -mesh_ny ${ny})
    if(ranks EQUAL 1)
        add_test(NAME ${name}
            COMMAND $<TARGET_FILE:test_boundary_conditions> ${arguments})
        set(mode serial)
    else()
        # Use the root's PETSc-compatible MPICH launcher and all its flags.
        # Never rediscover MPI or replace it with another system launcher.
        if(NOT MPIEXEC_EXECUTABLE)
            return()
        endif()
        add_test(NAME ${name}
            COMMAND "${MPIEXEC_EXECUTABLE}" ${MPIEXEC_NUMPROC_FLAG} ${ranks}
                ${MPIEXEC_PREFLAGS} $<TARGET_FILE:test_boundary_conditions>
                ${MPIEXEC_POSTFLAGS} ${arguments})
        set(mode mpi)
    endif()
    if(suite EQUAL 0)
        set(category mesh)
    else()
        set(category errors)
    endif()
    set_tests_properties(${name} PROPERTIES
        LABELS "mfem;boundary_conditions;${category};${mode}"
        PROCESSORS ${ranks} TIMEOUT 180)
endfunction()

foreach(kind IN ITEMS 0 1 2 3)
    if(kind EQUAL 0)
        set(mesh_name rectangular)
    elseif(kind EQUAL 1)
        set(mesh_name quadrilateral)
    elseif(kind EQUAL 2)
        set(mesh_name stretched)
    else()
        set(mesh_name skewed)
    endif()
    set(prefix "mfem_boundary_conditions_${mesh_name}")
    mantle_add_boundary_conditions_test(${prefix}_serial  ${kind} 0 1 1 7 5)
    mantle_add_boundary_conditions_test(${prefix}_mpi_2x1 ${kind} 0 2 1 7 5)
    mantle_add_boundary_conditions_test(${prefix}_mpi_1x2 ${kind} 0 1 2 7 5)
    mantle_add_boundary_conditions_test(${prefix}_mpi_2x2 ${kind} 0 2 2 7 5)
endforeach()

# A single cell: ranks can own physical-boundary DOFs but no cells, and one
# rank in a 2x2 layout owns no HDiv DOFs. All still join collective operations.
mantle_add_boundary_conditions_test(mfem_boundary_conditions_minimal_serial  0 0 1 1 2 2)
mantle_add_boundary_conditions_test(mfem_boundary_conditions_minimal_mpi_2x1 0 0 2 1 2 2)
mantle_add_boundary_conditions_test(mfem_boundary_conditions_minimal_mpi_1x2 0 0 1 2 2 2)
mantle_add_boundary_conditions_test(mfem_boundary_conditions_minimal_mpi_2x2 0 0 2 2 2 2)

mantle_add_boundary_conditions_test(mfem_boundary_conditions_errors_serial  1 1 1 1 6 5)
mantle_add_boundary_conditions_test(mfem_boundary_conditions_errors_mpi_2x1 1 1 2 1 6 5)
mantle_add_boundary_conditions_test(mfem_boundary_conditions_errors_mpi_1x2 1 1 1 2 6 5)
mantle_add_boundary_conditions_test(mfem_boundary_conditions_errors_mpi_2x2 1 1 2 2 6 5)

if(NOT MPIEXEC_EXECUTABLE)
    message(STATUS "Boundary conditions: registering serial tests only (MPI launcher unavailable)")
endif()
