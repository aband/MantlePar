# Include after defining mantle_mlweno with the updated reconstruction and
# tensorstencilpoly sources. The root project should include(CTest).
# Example inside src/mlweno/CMakeLists.txt:
# include("${CMAKE_CURRENT_SOURCE_DIR}/test/reconstruction_jacobian_tests.cmake")
# Run: ctest --test-dir build -R '^mlweno_reconstruction_jacobian_' --output-on-failure
include_guard(GLOBAL)
if(NOT BUILD_TESTING)
    return()
endif()
if(NOT TARGET mantle_mlweno)
    message(FATAL_ERROR "Define mantle_mlweno before including reconstruction_jacobian_tests.cmake")
endif()
add_executable(test_reconstruction_jacobian
    "${CMAKE_CURRENT_LIST_DIR}/test_reconstruction_jacobian.cpp")
target_compile_features(test_reconstruction_jacobian PRIVATE cxx_std_17)
target_link_libraries(test_reconstruction_jacobian PRIVATE mantle_mlweno)
set(_jacobian_names rectangular quadrilateral pseudo1d state scales)
foreach(_jacobian_case RANGE 0 4)
    list(GET _jacobian_names ${_jacobian_case} _jacobian_name)
    add_test(NAME mlweno_reconstruction_jacobian_${_jacobian_name}
        COMMAND test_reconstruction_jacobian -jacobian_case ${_jacobian_case})
    set_tests_properties(mlweno_reconstruction_jacobian_${_jacobian_name} PROPERTIES
        LABELS "mlweno;reconstruction;jacobian;serial" PROCESSORS 1 TIMEOUT 300)
endforeach()
unset(_jacobian_names)
unset(_jacobian_case)
unset(_jacobian_name)
