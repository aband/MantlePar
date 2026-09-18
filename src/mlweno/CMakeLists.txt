# Included by the repository's root CMakeLists.txt after src/core.
add_library(mantle_mlweno
	reconstruction.cpp
	tensorstencilpoly.cpp
)

target_include_directories(mantle_mlweno
    PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}"
)

target_compile_features(mantle_mlweno PUBLIC cxx_std_17)

# basis.h exposes core types. Propagate core's includes and dependencies
# (including PETSc) to targets that link against mantle_mfem.
target_link_libraries(mantle_mlweno
    PUBLIC mantle_core
)

if(BUILD_TESTING)
    add_subdirectory(test)
endif()
