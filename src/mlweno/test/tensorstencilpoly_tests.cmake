# Place this file, test_tensorstencilpoly.cpp and plot_tensorstencilpoly.py
# in src/mlweno/test/. The existing plot script is unchanged.
#
# Include from src/mlweno/CMakeLists.txt AFTER defining mantle_mlweno:
# if(BUILD_TESTING)
#     include("${CMAKE_CURRENT_SOURCE_DIR}/test/tensorstencilpoly_tests.cmake")
# endif()
#
# Run: ctest --test-dir build -R '^mlweno_tensorstencilpoly_' --output-on-failure
# Gradient cases only: ctest --test-dir build -R '^mlweno_tensorstencilpoly_gradient' --output-on-failure
# CSV tables and optional plots: build/src/mlweno/tensorstencilpoly_results/
# Requires TensorStencilPoly::SmoothnessWithGradient (both overloads).

include_guard(GLOBAL)
if(NOT BUILD_TESTING)
    return()
endif()

if(NOT TARGET mantle_mlweno)
    message(FATAL_ERROR "Define mantle_mlweno before including tensorstencilpoly_tests.cmake")
endif()

add_executable(test_tensorstencilpoly
    "${CMAKE_CURRENT_LIST_DIR}/test_tensorstencilpoly.cpp"
)
target_compile_features(test_tensorstencilpoly PRIVATE cxx_std_17)
target_link_libraries(test_tensorstencilpoly PRIVATE mantle_mlweno)

set(_tensor_results "${CMAKE_CURRENT_BINARY_DIR}/tensorstencilpoly_results")
file(MAKE_DIRECTORY "${_tensor_results}")
set(_tensor_mesh_names rectangular quadrilateral)
set(_tensor_data_fixtures)

foreach(_tensor_mesh_type RANGE 0 1)
    list(GET _tensor_mesh_names ${_tensor_mesh_type} _tensor_mesh_name)
    foreach(_tensor_degree IN ITEMS 1 2)
        set(_tensor_case "${_tensor_mesh_name}_q${_tensor_degree}")
        set(_tensor_test "mlweno_tensorstencilpoly_${_tensor_case}")
        add_test(NAME ${_tensor_test}
            COMMAND test_tensorstencilpoly
                -tensor_mesh_type ${_tensor_mesh_type}
                -tensor_degree ${_tensor_degree}
                -tensor_n0 16
                -tensor_levels 4
                -tensor_seed 7
                -tensor_perturbation 0.15
                -tensor_rate_tolerance 0.35
                -tensor_output "${_tensor_results}/tensorstencilpoly_${_tensor_case}.csv"
        )
        set_tests_properties(${_tensor_test} PROPERTIES
            LABELS "mlweno;tensorstencilpoly;convergence;serial"
            PROCESSORS 1
            TIMEOUT 300
            FIXTURES_SETUP "tensor_data_${_tensor_case}"
        )
        list(APPEND _tensor_data_fixtures "tensor_data_${_tensor_case}")
    endforeach()
    foreach(_tensor_gradient_mode IN ITEMS gradient gradient_pseudo1d)
        set(_tensor_test "mlweno_tensorstencilpoly_${_tensor_gradient_mode}_${_tensor_mesh_name}")
        add_test(NAME ${_tensor_test}
            COMMAND test_tensorstencilpoly
                -tensor_test ${_tensor_gradient_mode}
                -tensor_mesh_type ${_tensor_mesh_type}
        )
        set_tests_properties(${_tensor_test} PROPERTIES
            LABELS "mlweno;tensorstencilpoly;gradient;serial"
            PROCESSORS 1
            TIMEOUT 60
        )
    endforeach()
endforeach()

add_test(NAME mlweno_tensorstencilpoly_gradient_contract
    COMMAND test_tensorstencilpoly -tensor_test gradient_contract
)
set_tests_properties(mlweno_tensorstencilpoly_gradient_contract PROPERTIES
    LABELS "mlweno;tensorstencilpoly;gradient;serial"
    PROCESSORS 1
    TIMEOUT 60
)

# Numerical verification does not require Python or plotting packages.
# When available, CTest schedules plotting after all four data fixtures pass.
option(MANTLE_TENSOR_CONVERGENCE_PLOTS
       "Generate tensor polynomial convergence plots when matplotlib is available" ON)
if(MANTLE_TENSOR_CONVERGENCE_PLOTS)
    find_package(Python3 QUIET COMPONENTS Interpreter)
    if(Python3_Interpreter_FOUND)
        execute_process(
            COMMAND "${Python3_EXECUTABLE}" -c "import matplotlib"
            RESULT_VARIABLE _tensor_matplotlib_status
            OUTPUT_QUIET ERROR_QUIET
        )
    else()
        set(_tensor_matplotlib_status 1)
    endif()
    if(_tensor_matplotlib_status EQUAL 0)
        add_test(NAME mlweno_tensorstencilpoly_plots
            COMMAND "${Python3_EXECUTABLE}"
                "${CMAKE_CURRENT_LIST_DIR}/plot_tensorstencilpoly.py"
                --input-dir "${_tensor_results}"
                --output-dir "${_tensor_results}"
        )
        set_tests_properties(mlweno_tensorstencilpoly_plots PROPERTIES
            LABELS "mlweno;tensorstencilpoly;plot"
            TIMEOUT 60
            FIXTURES_REQUIRED "${_tensor_data_fixtures}"
        )
    else()
        message(STATUS
            "Tensor convergence: Python/matplotlib unavailable; registering numerical tests only")
    endif()
endif()

unset(_tensor_results)
unset(_tensor_mesh_names)
unset(_tensor_mesh_type)
unset(_tensor_mesh_name)
unset(_tensor_degree)
unset(_tensor_gradient_mode)
unset(_tensor_case)
unset(_tensor_test)
unset(_tensor_data_fixtures)
unset(_tensor_matplotlib_status)
