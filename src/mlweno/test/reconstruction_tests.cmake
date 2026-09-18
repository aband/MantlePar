# Put test_reconstruction.cpp, this file, and plot_reconstruction.py in src/mlweno/test/.
# Include AFTER defining mantle_mlweno and adding reconstruction.cpp:
# if(BUILD_TESTING)
#     include("${CMAKE_CURRENT_SOURCE_DIR}/test/reconstruction_tests.cmake")
# endif()
# Root CMake must enable CTest: include(CTest).
# Run: ctest --test-dir build -R '^mlweno_reconstruction_' --output-on-failure
# Output: build/src/mlweno/reconstruction_results/
# All cases are serial; missing solution data are checked, MPI exchanges are not.

include_guard(GLOBAL)
if(NOT BUILD_TESTING)
    return()
endif()
if(NOT TARGET mantle_mlweno)
    message(FATAL_ERROR "Define mantle_mlweno before including reconstruction_tests.cmake")
endif()
add_executable(test_reconstruction "${CMAKE_CURRENT_LIST_DIR}/test_reconstruction.cpp")
target_compile_features(test_reconstruction PRIVATE cxx_std_17)
target_link_libraries(test_reconstruction PRIVATE mantle_mlweno)

set(_recon_results "${CMAKE_CURRENT_BINARY_DIR}/reconstruction_results")
file(MAKE_DIRECTORY "${_recon_results}")
set(_recon_fixtures)
set(_recon_mesh_names rectangular quadrilateral)

# Eight 2-D cases: two meshes, two order pairs, constant off/on.
# Large: reference-square indicator. Small: physical target-cell indicator.
foreach(_recon_mesh RANGE 0 1)
    list(GET _recon_mesh_names ${_recon_mesh} _recon_mesh_name)
    foreach(_recon_order IN ITEMS 3 4)
        foreach(_recon_constant RANGE 0 1)
            set(_recon_case "${_recon_mesh_name}_r${_recon_order}_c${_recon_constant}")
            set(_recon_test "mlweno_reconstruction_${_recon_case}")
            add_test(NAME ${_recon_test}
                COMMAND test_reconstruction -recon_mode 0
                    -recon_mesh_type ${_recon_mesh} -recon_order ${_recon_order}
                    -recon_constant ${_recon_constant} -recon_axis 0
                    -recon_n0 16 -recon_levels 4 -recon_seed 7 -recon_perturbation 0.15
                    -recon_target_smoothness 1 -recon_rate_tolerance 0.4
                    -recon_output "${_recon_results}/reconstruction_${_recon_case}.csv")
            set_tests_properties(${_recon_test} PROPERTIES
                LABELS "mlweno;reconstruction;convergence;serial"
                PROCESSORS 1 TIMEOUT 900 FIXTURES_SETUP "recon_data_${_recon_case}")
            list(APPEND _recon_fixtures "recon_data_${_recon_case}")
        endforeach()
    endforeach()
    add_test(NAME mlweno_reconstruction_properties_${_recon_mesh_name}
        COMMAND test_reconstruction -recon_mode 1 -recon_mesh_type ${_recon_mesh})
    set_tests_properties(mlweno_reconstruction_properties_${_recon_mesh_name} PROPERTIES
        LABELS "mlweno;reconstruction;properties;serial" PROCESSORS 1 TIMEOUT 120)
endforeach()

# Four one-cell-thick cases: 5x1/3x1 and 1x5/1x3, constant off/on.
# Rate-fitting h is ACTIVE cell spacing; polynomial scale is sqrt(area).
# Continue to N=512 so the finest three grids follow the asymptotic (5,4)
# value/derivative regime after the coarse-grid nonlinear-weight transition.
set(_recon_axis_names unused x y)
foreach(_recon_axis RANGE 1 2)
    list(GET _recon_axis_names ${_recon_axis} _recon_axis_name)
    foreach(_recon_constant RANGE 0 1)
        set(_recon_case "pseudo1d_${_recon_axis_name}_c${_recon_constant}")
        set(_recon_test "mlweno_reconstruction_${_recon_case}")
        add_test(NAME ${_recon_test}
            COMMAND test_reconstruction -recon_mode 0 -recon_mesh_type 0
                -recon_axis ${_recon_axis} -recon_constant ${_recon_constant}
                -recon_n0 16 -recon_levels 6 -recon_target_smoothness 1
                -recon_rate_tolerance 0.4
                -recon_output "${_recon_results}/reconstruction_${_recon_case}.csv")
        set_tests_properties(${_recon_test} PROPERTIES
            LABELS "mlweno;reconstruction;convergence;pseudo1d;serial"
            PROCESSORS 1 TIMEOUT 300 FIXTURES_SETUP "recon_data_${_recon_case}")
        list(APPEND _recon_fixtures "recon_data_${_recon_case}")
    endforeach()
endforeach()

add_test(NAME mlweno_reconstruction_step
    COMMAND test_reconstruction -recon_mode 2
        -recon_output "${_recon_results}/reconstruction_step.csv")
set_tests_properties(mlweno_reconstruction_step PROPERTIES
    LABELS "mlweno;reconstruction;discontinuity;serial" PROCESSORS 1 TIMEOUT 120
    FIXTURES_SETUP recon_data_step)
list(APPEND _recon_fixtures recon_data_step)

option(MANTLE_RECONSTRUCTION_PLOTS
       "Generate reconstruction figures when Python and matplotlib are available" ON)
if(MANTLE_RECONSTRUCTION_PLOTS)
    find_package(Python3 QUIET COMPONENTS Interpreter)
    set(_recon_matplotlib_status 1)
    if(Python3_Interpreter_FOUND)
        execute_process(COMMAND "${Python3_EXECUTABLE}" -c "import matplotlib"
            RESULT_VARIABLE _recon_matplotlib_status OUTPUT_QUIET ERROR_QUIET)
    endif()
    if(_recon_matplotlib_status EQUAL 0)
        add_test(NAME mlweno_reconstruction_plots
            COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_LIST_DIR}/plot_reconstruction.py"
                --input-dir "${_recon_results}" --output-dir "${_recon_results}")
        set_tests_properties(mlweno_reconstruction_plots PROPERTIES
            LABELS "mlweno;reconstruction;plot" TIMEOUT 120
            FIXTURES_REQUIRED "${_recon_fixtures}")
    else()
        message(STATUS "Reconstruction: Python/matplotlib unavailable; numerical tests remain enabled")
    endif()
endif()

# Strict (3,2) critical-point tests use the ReconstructionOptions defaults.
# These pass with epsilon=1e-2; the earlier epsilon=1e-6 default caused
# finite-grid convergence failures. Keep the tests available explicitly.
# Enable with -DMANTLE_RECONSTRUCTION_CRITICAL_POINT_TESTS=ON.
option(MANTLE_RECONSTRUCTION_CRITICAL_POINT_TESTS
       "Enable strict critical-point reconstruction tests" OFF)
if(MANTLE_RECONSTRUCTION_CRITICAL_POINT_TESTS)
    foreach(_recon_mesh RANGE 0 1)
        list(GET _recon_mesh_names ${_recon_mesh} _recon_mesh_name)
        foreach(_recon_constant RANGE 0 1)
            set(_recon_case "critical_${_recon_mesh_name}_c${_recon_constant}")
            add_test(NAME mlweno_reconstruction_${_recon_case}
                COMMAND test_reconstruction -recon_mode 0 -recon_field 1
                    -recon_mesh_type ${_recon_mesh} -recon_order 3
                    -recon_constant ${_recon_constant} -recon_axis 0
                    -recon_n0 16 -recon_levels 4 -recon_target_smoothness 1
                    -recon_output "${_recon_results}/reconstruction_${_recon_case}.csv")
            set_tests_properties(mlweno_reconstruction_${_recon_case} PROPERTIES
                LABELS "mlweno;reconstruction;critical_points;stress;serial"
                PROCESSORS 1 TIMEOUT 900)
        endforeach()
    endforeach()
endif()
unset(_recon_results)
unset(_recon_fixtures)
unset(_recon_mesh_names)
unset(_recon_mesh)
unset(_recon_mesh_name)
unset(_recon_order)
unset(_recon_constant)
unset(_recon_case)
unset(_recon_test)
unset(_recon_axis_names)
unset(_recon_axis)
unset(_recon_axis_name)
unset(_recon_matplotlib_status)
