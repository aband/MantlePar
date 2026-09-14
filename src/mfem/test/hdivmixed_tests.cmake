# Include from src/mfem/test/CMakeLists.txt after mantle_mfem is defined.
include_guard(GLOBAL)

add_executable(test_hdivmixed "${CMAKE_CURRENT_LIST_DIR}/test_hdivmixed.cpp")
target_link_libraries(test_hdivmixed PRIVATE mantle_mfem)

add_test(NAME mfem_hdivmixed_unit_serial
    COMMAND $<TARGET_FILE:test_hdivmixed> -expected_ranks 1)
set_tests_properties(mfem_hdivmixed_unit_serial PROPERTIES
    LABELS "mfem;hdivmixed;serial" TIMEOUT 60)

function(mantle_add_hdivmixed_mesh_test name kind px py)
    math(EXPR ranks "${px} * ${py}")
    set(arguments -hdivmixed_mesh_type ${kind} -expected_ranks ${ranks}
        -mesh_px ${px} -mesh_py ${py})
    if(ranks EQUAL 1)
        add_test(NAME ${name}
            COMMAND $<TARGET_FILE:test_hdivmixed> ${arguments})
        set(mode serial)
    else()
        # Reuse the root's PETSc-compatible MPI launcher and flags.
        if(NOT MPIEXEC_EXECUTABLE)
            return()
        endif()
        add_test(NAME ${name}
            COMMAND "${MPIEXEC_EXECUTABLE}" ${MPIEXEC_NUMPROC_FLAG} ${ranks}
                ${MPIEXEC_PREFLAGS} $<TARGET_FILE:test_hdivmixed>
                ${MPIEXEC_POSTFLAGS} ${arguments})
        set(mode mpi)
    endif()
    set_tests_properties(${name} PROPERTIES
        LABELS "mfem;hdivmixed;mesh;${mode}" PROCESSORS ${ranks} TIMEOUT 120)
endfunction()

foreach(kind IN ITEMS 0 1 2)
    if(kind EQUAL 0)
        set(mesh_name rectangular)
    elseif(kind EQUAL 1)
        set(mesh_name quadrilateral)
    else()
        set(mesh_name stretched)
    endif()
    set(prefix "mfem_hdivmixed_${mesh_name}")
    mantle_add_hdivmixed_mesh_test(${prefix}_serial  ${kind} 1 1)
    mantle_add_hdivmixed_mesh_test(${prefix}_mpi_2x1 ${kind} 2 1)
    mantle_add_hdivmixed_mesh_test(${prefix}_mpi_1x2 ${kind} 1 2)
    mantle_add_hdivmixed_mesh_test(${prefix}_mpi_2x2 ${kind} 2 2)
endforeach()

if(NOT MPIEXEC_EXECUTABLE)
    message(STATUS "HDivMixed: registering serial tests only (MPI launcher unavailable)")
endif()
