# Included after core, input, phase, mfem and mlweno.
# Evolution also uses yaml-cpp directly to prepare restart input.
find_package(yaml-cpp 0.7 CONFIG REQUIRED)

add_library(mantle_couple
    initialization.cpp
    initial_flow.cpp
    flow_sampling.cpp
    state_reconstruction.cpp
    preheat.cpp
    evolve.cpp
    evolution_output.cpp
    visualization.cpp
)

target_include_directories(mantle_couple
    PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}"
)

target_compile_features(mantle_couple PUBLIC cxx_std_17)

# initialization.h exposes types from these modules. Propagate their includes
# and dependencies to targets that link against mantle_couple.
target_link_libraries(mantle_couple
    PUBLIC
        mantle_input
        mantle_core
        mantle_phase
        mantle_mfem
        mantle_mlweno
)
if(TARGET yaml-cpp::yaml-cpp)
    target_link_libraries(mantle_couple PRIVATE yaml-cpp::yaml-cpp)
else()
    target_link_libraries(mantle_couple PRIVATE yaml-cpp)
endif()

add_executable(mantle_couple_init
    initialize.cpp
)

target_link_libraries(mantle_couple_init
    PRIVATE mantle_couple
)

add_executable(mantle_couple_preheat preheat_driver.cpp)
target_link_libraries(mantle_couple_preheat PRIVATE mantle_couple)

add_executable(mantle_couple_evolve evolve_driver.cpp)
target_link_libraries(mantle_couple_evolve PRIVATE mantle_couple)

if(BUILD_TESTING)
    add_subdirectory(test)
endif()
