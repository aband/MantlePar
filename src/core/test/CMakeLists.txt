add_executable(test_mesh test_mesh.cpp)
target_link_libraries(test_mesh PRIVATE mantle_core)

add_test(NAME core_mesh_serial COMMAND test_mesh)
set_tests_properties(core_mesh_serial PROPERTIES TIMEOUT 60)

if(MPIEXEC_EXECUTABLE)
    foreach(layout IN ITEMS 2x1 1x2 2x2)
        string(REPLACE "x" ";" dims "${layout}")
        list(GET dims 0 px)
        list(GET dims 1 py)
        math(EXPR ranks "${px} * ${py}")
        add_test(NAME core_mesh_mpi_${layout}
            COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${ranks}
                    ${MPIEXEC_PREFLAGS} $<TARGET_FILE:test_mesh>
                    ${MPIEXEC_POSTFLAGS} -mesh_px ${px} -mesh_py ${py}
        )
        set_tests_properties(core_mesh_mpi_${layout}
            PROPERTIES PROCESSORS ${ranks} TIMEOUT 60
        )
    endforeach()
endif()

 include("${CMAKE_CURRENT_LIST_DIR}/integral_tests.cmake")
 include("${CMAKE_CURRENT_LIST_DIR}/utility_tests.cmake")
 include("${CMAKE_CURRENT_LIST_DIR}/field_initialization_tests.cmake")

