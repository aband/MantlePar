# Include from src/mlweno/test/CMakeLists.txt after mantle_mlweno is defined.
# See transport_flux_tests.md. All tests require one MPI rank and real PETSc.
include_guard(GLOBAL)
if(NOT BUILD_TESTING)
    return()
endif()
if(NOT TARGET mantle_mlweno)
    message(FATAL_ERROR "Define mantle_mlweno before including diffusiveflux_tests.cmake")
endif()
foreach(_df_file IN ITEMS diffusiveflux.h diffusiveflux.cpp advectiveflux.h)
    if(NOT EXISTS "${CMAKE_CURRENT_LIST_DIR}/../${_df_file}")
        message(FATAL_ERROR "Place ${_df_file} in src/mlweno before enabling the transport tests")
    endif()
endforeach()
add_executable(test_diffusiveflux "${CMAKE_CURRENT_LIST_DIR}/test_diffusiveflux.cpp")
add_executable(test_diffusiveflux_convergence "${CMAKE_CURRENT_LIST_DIR}/test_diffusiveflux_convergence.cpp")
foreach(_df_target IN ITEMS test_diffusiveflux test_diffusiveflux_convergence)
    target_compile_features(${_df_target} PRIVATE cxx_std_17)
    target_link_libraries(${_df_target} PRIVATE mantle_mlweno)
endforeach()
add_test(NAME mlweno_diffusion_kernels COMMAND test_diffusiveflux)
set_tests_properties(mlweno_diffusion_kernels PROPERTIES
    LABELS "mlweno;transport;diffusion;accuracy;jacobian;serial" PROCESSORS 1 TIMEOUT 180)
set(_df_results "${CMAKE_CURRENT_BINARY_DIR}/diffusion_results")
file(MAKE_DIRECTORY "${_df_results}")
set(_df_fixtures)
function(_mantle_diffusion_case name category)
    add_test(NAME mlweno_diffusion_${name} COMMAND test_diffusiveflux_convergence
        ${ARGN} -flux_output "${_df_results}/${name}.csv")
    set_tests_properties(mlweno_diffusion_${name} PROPERTIES
        LABELS "mlweno;transport;diffusion;${category};serial" PROCESSORS 1 TIMEOUT 900
        FIXTURES_SETUP diffusion_${name})
    set(_df_fixtures ${_df_fixtures} diffusion_${name} PARENT_SCOPE)
endfunction()
foreach(_df_mesh RANGE 0 1)
    foreach(_df_order IN ITEMS 3 5)
        foreach(_df_constant RANGE 0 1)
            set(_df_case "m${_df_mesh}_p${_df_order}_c${_df_constant}")
            _mantle_diffusion_case(smooth_${_df_case} convergence
                -flux_mesh ${_df_mesh} -flux_order ${_df_order} -flux_constant ${_df_constant}
                -flux_n0 12 -flux_levels 3 -flux_mode 0)
            _mantle_diffusion_case(jump_cut_${_df_case} discontinuity
                -flux_mesh ${_df_mesh} -flux_order ${_df_order} -flux_constant ${_df_constant}
                -flux_n0 12 -flux_levels 3 -flux_mode 1 -flux_cut 1)
        endforeach()
    endforeach()
endforeach()
foreach(_df_axis RANGE 1 2)
    foreach(_df_constant RANGE 0 1)
        _mantle_diffusion_case(smooth_axis${_df_axis}_c${_df_constant} "convergence;pseudo1d"
            -flux_axis ${_df_axis} -flux_order 5 -flux_constant ${_df_constant}
            -flux_n0 16 -flux_levels 5)
        _mantle_diffusion_case(jump_axis${_df_axis}_c${_df_constant} "discontinuity;pseudo1d"
            -flux_mode 1 -flux_axis ${_df_axis} -flux_order 5 -flux_constant ${_df_constant}
            -flux_n0 16 -flux_levels 4 -flux_cut 1)
    endforeach()
endforeach()
foreach(_df_order IN ITEMS 3 5)
    _mantle_diffusion_case(jump_aligned_p${_df_order} discontinuity
        -flux_mode 1 -flux_cut 0 -flux_order ${_df_order} -flux_constant 1)
endforeach()
_mantle_diffusion_case(jump_oblique_p3 discontinuity -flux_mode 1 -flux_mesh 1
    -flux_orientation 2 -flux_order 3 -flux_constant 0)
_mantle_diffusion_case(jump_oblique_piecewise_p5 discontinuity -flux_mode 1 -flux_mesh 1
    -flux_orientation 2 -flux_order 5 -flux_constant 1 -flux_field 1)
_mantle_diffusion_case(jump_horizontal_quad discontinuity -flux_mode 1 -flux_mesh 1
    -flux_orientation 1 -flux_order 3 -flux_constant 1)

option(MANTLE_TRANSPORT_TEST_PLOTS "Plot transport accuracy and discontinuity CSV files" ON)
if(MANTLE_TRANSPORT_TEST_PLOTS)
    find_package(Python3 QUIET COMPONENTS Interpreter)
    set(_df_python_status 1)
    if(Python3_Interpreter_FOUND)
        execute_process(COMMAND "${Python3_EXECUTABLE}" -c "import matplotlib"
            RESULT_VARIABLE _df_python_status OUTPUT_QUIET ERROR_QUIET)
    endif()
    if(_df_python_status EQUAL 0)
        add_test(NAME mlweno_diffusion_plots COMMAND "${Python3_EXECUTABLE}"
            "${CMAKE_CURRENT_LIST_DIR}/plot_transport_tests.py" --input-dir "${_df_results}")
        set_tests_properties(mlweno_diffusion_plots PROPERTIES LABELS "mlweno;transport;plot"
            TIMEOUT 180 FIXTURES_REQUIRED "${_df_fixtures}")
    else()
        message(STATUS "Transport plots disabled: Python/matplotlib unavailable; numerical tests remain enabled")
    endif()
endif()
