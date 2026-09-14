# Include from src/mfem/test/CMakeLists.txt after mantle_mfem is defined.
include_guard(GLOBAL)

add_executable(test_brmixed "${CMAKE_CURRENT_LIST_DIR}/test_brmixed.cpp")
target_link_libraries(test_brmixed PRIVATE mantle_mfem)

add_test(NAME mfem_brmixed_unit_serial
    COMMAND $<TARGET_FILE:test_brmixed> -expected_ranks 1)
set_tests_properties(mfem_brmixed_unit_serial PROPERTIES
    LABELS "mfem;brmixed;serial" TIMEOUT 60)

function(mantle_add_brmixed_mesh_test name kind px py)
    math(EXPR ranks "${px} * ${py}")
    set(arguments -brmixed_mesh_type ${kind} -expected_ranks ${ranks}
        -mesh_px ${px} -mesh_py ${py})
    if(ranks EQUAL 1)
        add_test(NAME ${name}
            COMMAND $<TARGET_FILE:test_brmixed> ${arguments})
        set(mode serial)
    else()
        # Reuse the launcher selected by the root for PETSc's MPI installation.
        if(NOT MPIEXEC_EXECUTABLE)
            return()
        endif()
        add_test(NAME ${name}
            COMMAND "${MPIEXEC_EXECUTABLE}" ${MPIEXEC_NUMPROC_FLAG} ${ranks}
                ${MPIEXEC_PREFLAGS} $<TARGET_FILE:test_brmixed>
                ${MPIEXEC_POSTFLAGS} ${arguments})
        set(mode mpi)
    endif()
    set_tests_properties(${name} PROPERTIES
        LABELS "mfem;brmixed;mesh;${mode}" PROCESSORS ${ranks} TIMEOUT 120)
endfunction()

foreach(kind IN ITEMS 0 1 2)
    if(kind EQUAL 0)
        set(mesh_name rectangular)
    elseif(kind EQUAL 1)
        set(mesh_name quadrilateral)
    else()
        set(mesh_name stretched)
    endif()
    set(prefix "mfem_brmixed_${mesh_name}")
    mantle_add_brmixed_mesh_test(${prefix}_serial  ${kind} 1 1)
    mantle_add_brmixed_mesh_test(${prefix}_mpi_2x1 ${kind} 2 1)
    mantle_add_brmixed_mesh_test(${prefix}_mpi_1x2 ${kind} 1 2)
    mantle_add_brmixed_mesh_test(${prefix}_mpi_2x2 ${kind} 2 2)
endforeach()

if(NOT MPIEXEC_EXECUTABLE)
    message(STATUS "BRMixed: registering serial tests only (MPI launcher unavailable)")
endif()
