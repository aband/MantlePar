# Include from src/core/test/CMakeLists.txt after the existing mesh tests.
# Reuses the PETSc compiler and MPI launcher selected by the root project.
add_executable(test_integral "${CMAKE_CURRENT_LIST_DIR}/test_integral.cpp")
target_compile_features(test_integral PRIVATE cxx_std_17)
target_link_libraries(test_integral PRIVATE mantle_core)

add_test(NAME core_integral_unit COMMAND test_integral)
set_tests_properties(core_integral_unit PROPERTIES TIMEOUT 60 LABELS "core;integral;unit")

set(integral_mesh_names rectangular quadrilateral stretched)
foreach(mesh_type RANGE 0 2)
    list(GET integral_mesh_names ${mesh_type} mesh_name)
    add_test(NAME core_integral_${mesh_name}_serial
        COMMAND test_integral -integral_mesh_type ${mesh_type} -mesh_px 1 -mesh_py 1)
    set_tests_properties(core_integral_${mesh_name}_serial
        PROPERTIES TIMEOUT 60 LABELS "core;integral;mesh")

    if(MPIEXEC_EXECUTABLE)
        foreach(layout IN ITEMS 2x1 1x2 2x2)
            string(REPLACE "x" ";" dims "${layout}")
            list(GET dims 0 px)
            list(GET dims 1 py)
            math(EXPR ranks "${px} * ${py}")
            add_test(NAME core_integral_${mesh_name}_mpi_${layout}
                COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${ranks}
                        ${MPIEXEC_PREFLAGS} $<TARGET_FILE:test_integral> ${MPIEXEC_POSTFLAGS}
                        -integral_mesh_type ${mesh_type} -mesh_px ${px} -mesh_py ${py})
            set_tests_properties(core_integral_${mesh_name}_mpi_${layout}
                PROPERTIES PROCESSORS ${ranks} TIMEOUT 60 LABELS "core;integral;mesh;mpi")
        endforeach()
    endif()
endforeach()
