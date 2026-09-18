# Included by the repository's root CMakeLists.txt after core, phase, mfem and mlweno.
add_library(mantle_couple
    initialization.cpp
    initial_flow.cpp
    flow_sampling.cpp
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
        mantle_core
        mantle_phase
        mantle_mfem
        mantle_mlweno
)

add_executable(mantle_couple_init
    initialize.cpp
)

target_link_libraries(mantle_couple_init
    PRIVATE mantle_couple
)

if(BUILD_TESTING)
    add_subdirectory(test)
endif()
