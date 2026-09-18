include_guard(GLOBAL)

find_package(Python3 QUIET COMPONENTS Interpreter)
set(_couple_python_status 1)
if(Python3_Interpreter_FOUND)
    execute_process(COMMAND "${Python3_EXECUTABLE}" -c "import numpy, matplotlib"
        RESULT_VARIABLE _couple_python_status OUTPUT_QUIET ERROR_QUIET)
endif()

if(_couple_python_status EQUAL 0)
    foreach(_couple_case IN ITEMS column perturbed_quad formalpreheat)
        add_test(NAME couple_visualization_${_couple_case}
            COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_LIST_DIR}/test_visualization.py"
                --executable $<TARGET_FILE:mantle_couple_init>
                --input "${CMAKE_CURRENT_LIST_DIR}/../example/${_couple_case}/input.yaml"
                --output-dir "${CMAKE_CURRENT_BINARY_DIR}/visualization/${_couple_case}")
        set_tests_properties(couple_visualization_${_couple_case} PROPERTIES
            TIMEOUT 120 LABELS "couple;visualization;serial")
        if(MPIEXEC_EXECUTABLE AND MPIEXEC_NUMPROC_FLAG)
            add_test(NAME couple_visualization_${_couple_case}_mpi_2
                COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_LIST_DIR}/test_visualization.py"
                    --executable $<TARGET_FILE:mantle_couple_init>
                    --input "${CMAKE_CURRENT_LIST_DIR}/../example/${_couple_case}/input.yaml"
                    --output-dir "${CMAKE_CURRENT_BINARY_DIR}/visualization/${_couple_case}_mpi"
                    --mpiexec "${MPIEXEC_EXECUTABLE}" "--numproc-flag=${MPIEXEC_NUMPROC_FLAG}"
                    "--mpi-preflags=${MPIEXEC_PREFLAGS}" "--mpi-postflags=${MPIEXEC_POSTFLAGS}")
            set_tests_properties(couple_visualization_${_couple_case}_mpi_2 PROPERTIES
                PROCESSORS 2 TIMEOUT 120 LABELS "couple;visualization;mpi")
        endif()
    endforeach()
    add_test(NAME couple_visualization_phase_limits_and_plots
        COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_LIST_DIR}/test_visualization.py"
            --executable $<TARGET_FILE:mantle_couple_init>
            --input "${CMAKE_CURRENT_LIST_DIR}/../example/column/input.yaml"
            --output-dir "${CMAKE_CURRENT_BINARY_DIR}/visualization/phase_limits"
            --phase-limits --plots)
    set_tests_properties(couple_visualization_phase_limits_and_plots PROPERTIES
        TIMEOUT 180 LABELS "couple;visualization;phase;plot")
else()
    message(STATUS "Couple visualization tests require Python, numpy and matplotlib; C++ export remains available")
endif()
unset(_couple_python_status)
unset(_couple_case)
