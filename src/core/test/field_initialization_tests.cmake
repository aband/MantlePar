# Include from src/core/test/CMakeLists.txt after mantle_core is defined.
include_guard(GLOBAL)

add_executable(test_field_initialization test_field_initialization.cpp)
target_link_libraries(test_field_initialization PRIVATE mantle_core)
target_compile_features(test_field_initialization PRIVATE cxx_std_17)

function(mantle_add_field_initialization_test name kind mesh_px mesh_py field_px field_py)
    math(EXPR ranks "${mesh_px} * ${mesh_py}")
    math(EXPR field_ranks "${field_px} * ${field_py}")
    if(NOT ranks EQUAL field_ranks)
        message(FATAL_ERROR "${name}: mesh and field process grids use different rank counts")
    endif()
    set(test_arguments
        -expected_ranks ${ranks}
        -field_mesh_type ${kind}
        -mesh_px ${mesh_px} -mesh_py ${mesh_py}
        -field_px ${field_px} -field_py ${field_py})
    if(ranks EQUAL 1)
        add_test(NAME ${name}
            COMMAND $<TARGET_FILE:test_field_initialization> ${test_arguments})
        set(mode serial)
    else()
        # Reuse the root's launcher paired with PETSc. Do not find or replace
        # MPI here: a second MPI installation can launch the wrong runtime.
        if(NOT MPIEXEC_EXECUTABLE)
            return()
        endif()
        add_test(NAME ${name}
            COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${ranks}
                ${MPIEXEC_PREFLAGS} $<TARGET_FILE:test_field_initialization>
                ${MPIEXEC_POSTFLAGS} ${test_arguments})
        set(mode mpi)
    endif()
    set_tests_properties(${name} PROPERTIES
        PROCESSORS ${ranks}
        TIMEOUT 120
        LABELS "core;field_initialization;${mode}")
endfunction()

# Each invocation runs accuracy, conservation, assignment, ghost exchange,
# collective failure/rollback, and the tiny empty-mesh-owner regression.
foreach(kind IN ITEMS 0 1 2)
    if(kind EQUAL 0)
        set(mesh_name rectangular)
    elseif(kind EQUAL 1)
        set(mesh_name quadrilateral)
    else()
        set(mesh_name stretched)
    endif()
    set(prefix "core_field_initialization_${mesh_name}")
    mantle_add_field_initialization_test(${prefix}_serial              ${kind} 1 1 1 1)
    mantle_add_field_initialization_test(${prefix}_mpi_2x1_to_2x1      ${kind} 2 1 2 1)
    mantle_add_field_initialization_test(${prefix}_mpi_2x1_to_1x2      ${kind} 2 1 1 2)
    mantle_add_field_initialization_test(${prefix}_mpi_1x2_to_2x1      ${kind} 1 2 2 1)
    mantle_add_field_initialization_test(${prefix}_mpi_2x2_to_2x2      ${kind} 2 2 2 2)
    mantle_add_field_initialization_test(${prefix}_mpi_2x2_to_4x1      ${kind} 2 2 4 1)
endforeach()

if(NOT MPIEXEC_EXECUTABLE)
    message(STATUS "Field initialization: registering serial tests only (MPI launcher unavailable)")
endif()
