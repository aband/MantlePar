# Included from src/phase/CMakeLists.txt when BUILD_TESTING is ON.
# Pointwise equilibrium tests require neither PETSc initialization nor MPI.
add_executable(test_eutectic test_eutectic.cpp)
target_link_libraries(test_eutectic PRIVATE mantle_phase)

foreach(case_name IN ITEMS
    regions
    boundaries
    conservation
    derivatives
    scales
    parameter_updates
    invalid_inputs
    state
)
    add_test(NAME phase_eutectic_${case_name}
        COMMAND test_eutectic ${case_name}
    )
    set_tests_properties(phase_eutectic_${case_name}
        PROPERTIES TIMEOUT 60 LABELS "phase;eutectic"
    )
endforeach()
