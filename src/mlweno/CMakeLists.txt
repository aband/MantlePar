# Included by the repository's root CMakeLists.txt after src/core.
pkg_check_modules(LAPACKE REQUIRED IMPORTED_TARGET lapacke)

add_library(mantle_mlweno
    reconstruction.cpp
    tensorstencilpoly.cpp
    advectiveflux.cpp
    diffusiveflux.cpp
)

target_include_directories(mantle_mlweno
    PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}"
)

target_compile_features(mantle_mlweno PUBLIC cxx_std_17)

# Public headers expose core types; LAPACKE is used only by the implementation.
target_link_libraries(mantle_mlweno
    PUBLIC mantle_core
    PRIVATE PkgConfig::LAPACKE
)

if(BUILD_TESTING)
    add_subdirectory(test)
endif()
