# Test registration for src/mlweno/test.
if(NOT BUILD_TESTING)
    return()
endif()

include("${CMAKE_CURRENT_LIST_DIR}/reconstruction_tests.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/tensorstencilpoly_tests.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/reconstruction_jacobian_tests.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/diffusiveflux_tests.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/advection_diffusion_tests.cmake")

