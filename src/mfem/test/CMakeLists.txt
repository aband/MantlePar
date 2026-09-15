# Test registration for src/mfem/test.
if(NOT BUILD_TESTING)
    return()
endif()

include("${CMAKE_CURRENT_LIST_DIR}/basis_tests.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/brmixed_tests.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/hdivmixed_tests.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/dof_map_tests.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/local_matrix_tests.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/assembly_tests.cmake")
