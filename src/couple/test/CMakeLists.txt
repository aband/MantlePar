if(NOT BUILD_TESTING)
    return()
endif()

include("${CMAKE_CURRENT_LIST_DIR}/visualization_tests.cmake")

add_executable(test_couple_initial_flow test_initial_flow.cpp)
target_link_libraries(test_couple_initial_flow PRIVATE mantle_couple)
add_test(NAME couple_initial_flow
    COMMAND test_couple_initial_flow -input "${CMAKE_CURRENT_LIST_DIR}/../example/column/input.yaml")
set_tests_properties(couple_initial_flow PROPERTIES TIMEOUT 120 LABELS "couple;flow;serial")
if(MPIEXEC_EXECUTABLE AND MPIEXEC_NUMPROC_FLAG)
    add_test(NAME couple_initial_flow_mpi_2
        COMMAND "${MPIEXEC_EXECUTABLE}" "${MPIEXEC_NUMPROC_FLAG}" 2 ${MPIEXEC_PREFLAGS}
            $<TARGET_FILE:test_couple_initial_flow> ${MPIEXEC_POSTFLAGS}
            -input "${CMAKE_CURRENT_LIST_DIR}/../example/column/input.yaml")
    set_tests_properties(couple_initial_flow_mpi_2 PROPERTIES PROCESSORS 2 TIMEOUT 120 LABELS "couple;flow;mpi")
endif()
