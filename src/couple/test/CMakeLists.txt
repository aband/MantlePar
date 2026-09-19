if(NOT BUILD_TESTING)
    return()
endif()

include("${CMAKE_CURRENT_LIST_DIR}/visualization_tests.cmake")

if(Python3_Interpreter_FOUND)
    execute_process(COMMAND "${Python3_EXECUTABLE}" -c "import numpy, matplotlib, scipy"
        RESULT_VARIABLE _movie_python OUTPUT_QUIET ERROR_QUIET)
    if(_movie_python EQUAL 0)
        add_test(NAME couple_movie_output
            COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_LIST_DIR}/test_movie_output.py"
                --output-dir "${CMAKE_CURRENT_BINARY_DIR}/movie_output")
        set_tests_properties(couple_movie_output PROPERTIES TIMEOUT 120 LABELS "couple;output;no-solver")
    endif()
endif()

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

add_executable(test_couple_evolution test_evolution.cpp)
target_link_libraries(test_couple_evolution PRIVATE mantle_couple)
add_test(NAME couple_evolution_velocities COMMAND test_couple_evolution)
if(Python3_Interpreter_FOUND)
    execute_process(COMMAND "${Python3_EXECUTABLE}" -c "import numpy, matplotlib, scipy, yaml"
        RESULT_VARIABLE _evolve_python OUTPUT_QUIET ERROR_QUIET)
    if(_evolve_python EQUAL 0)
        add_test(NAME couple_run_script
            COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_LIST_DIR}/test_run_script.py")
        set_tests_properties(couple_run_script PROPERTIES TIMEOUT 30 LABELS "couple;output;no-solver")
        find_program(MANTLE_PDFTOTEXT_EXECUTABLE pdftotext)
        execute_process(COMMAND "${Python3_EXECUTABLE}" -c
            "import sys; sys.path.insert(0, r'${CMAKE_CURRENT_LIST_DIR}/..'); from conditions_report import renderer_python; renderer_python()"
            RESULT_VARIABLE _conditions_python OUTPUT_QUIET ERROR_QUIET)
        if(MANTLE_PDFTOTEXT_EXECUTABLE AND _conditions_python EQUAL 0)
            add_test(NAME couple_conditions_report
                COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_LIST_DIR}/test_conditions_report.py")
            set_tests_properties(couple_conditions_report PROPERTIES TIMEOUT 60 LABELS "couple;output;no-solver")
        endif()
        add_test(NAME couple_evolution
            COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_LIST_DIR}/test_evolution.py"
                --executable $<TARGET_FILE:mantle_couple_evolve>
                --input "${CMAKE_CURRENT_LIST_DIR}/../example/formalevolve/input.yaml"
                --output-dir "${CMAKE_CURRENT_BINARY_DIR}/evolution/serial")
        set_tests_properties(couple_evolution PROPERTIES TIMEOUT 240 LABELS "couple;evolution;serial")
        if(MPIEXEC_EXECUTABLE AND MPIEXEC_NUMPROC_FLAG)
            add_test(NAME couple_evolution_mpi_2
                COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_LIST_DIR}/test_evolution.py"
                    --executable $<TARGET_FILE:mantle_couple_evolve>
                    --input "${CMAKE_CURRENT_LIST_DIR}/../example/formalevolve/input.yaml"
                    --output-dir "${CMAKE_CURRENT_BINARY_DIR}/evolution/mpi"
                    --mpiexec "${MPIEXEC_EXECUTABLE}" "--numproc-flag=${MPIEXEC_NUMPROC_FLAG}")
            set_tests_properties(couple_evolution_mpi_2 PROPERTIES TIMEOUT 240 PROCESSORS 2 LABELS "couple;evolution;mpi")
        endif()
    endif()
endif()
