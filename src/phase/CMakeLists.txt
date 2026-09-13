# Binary eutectic phase model; depends only on the C++ standard library.
add_library(mantle_phase
    eutectic_rescaled.cpp
    eutectic_rescaled.h
)

target_include_directories(mantle_phase
    PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}"
)

target_compile_features(mantle_phase PUBLIC cxx_std_17)
