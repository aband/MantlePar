# Place beside test_mesh_info.cpp and test_math_utils.cpp in src/core/test/.
# Include from that directory's existing CMakeLists.txt.
# Both implementation files are already part of mantle_core.
include_guard(GLOBAL)

add_executable(test_math_utils "${CMAKE_CURRENT_LIST_DIR}/test_math_utils.cpp")
target_compile_features(test_math_utils PRIVATE cxx_std_17)
target_link_libraries(test_math_utils PRIVATE mantle_core)
add_test(NAME core_math_utils COMMAND test_math_utils)
set_tests_properties(core_math_utils
    PROPERTIES TIMEOUT 60 LABELS "core;math_utils;serial")

add_executable(test_mesh_info "${CMAKE_CURRENT_LIST_DIR}/test_mesh_info.cpp")
target_compile_features(test_mesh_info PRIVATE cxx_std_17)
target_link_libraries(test_mesh_info PRIVATE mantle_core)

set(mesh_info_test_names rectangular quadrilateral stretched)
foreach(mesh_info_kind RANGE 0 2)
    list(GET mesh_info_test_names ${mesh_info_kind} mesh_info_name)
    add_test(NAME core_mesh_info_${mesh_info_name}_serial
        COMMAND test_mesh_info -mesh_info_type ${mesh_info_kind} -mesh_px 1 -mesh_py 1)
    set_tests_properties(core_mesh_info_${mesh_info_name}_serial
        PROPERTIES TIMEOUT 60 LABELS "core;mesh_info;serial")

    # Reuse the MPI launcher selected by the root PETSc/MPICH configuration.
    if(MPIEXEC_EXECUTABLE)
        foreach(mesh_info_layout IN ITEMS 2x1 1x2 2x2)
            string(REPLACE "x" ";" mesh_info_dims "${mesh_info_layout}")
            list(GET mesh_info_dims 0 mesh_info_px)
            list(GET mesh_info_dims 1 mesh_info_py)
            math(EXPR mesh_info_ranks "${mesh_info_px} * ${mesh_info_py}")
            add_test(NAME core_mesh_info_${mesh_info_name}_mpi_${mesh_info_layout}
                COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${mesh_info_ranks}
                    ${MPIEXEC_PREFLAGS} $<TARGET_FILE:test_mesh_info> ${MPIEXEC_POSTFLAGS}
                    -mesh_info_type ${mesh_info_kind}
                    -mesh_px ${mesh_info_px} -mesh_py ${mesh_info_py})
            set_tests_properties(core_mesh_info_${mesh_info_name}_mpi_${mesh_info_layout}
                PROPERTIES PROCESSORS ${mesh_info_ranks} TIMEOUT 60
                    LABELS "core;mesh_info;mpi")
        endforeach()
    endif()
endforeach()
