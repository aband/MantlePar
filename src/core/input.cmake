# Include AFTER add_library(mantle_core ...) in src/core/CMakeLists.txt.
# Leaves the project's PETSc/MPICH compiler and launcher choices unchanged.
find_package(yaml-cpp 0.7 CONFIG REQUIRED)

if(TARGET yaml-cpp::yaml-cpp)
    set(_mantle_input_yaml_target yaml-cpp::yaml-cpp)
elseif(TARGET yaml-cpp)
    set(_mantle_input_yaml_target yaml-cpp)
else()
    message(FATAL_ERROR "yaml-cpp did not provide a supported CMake target")
endif()

target_sources(mantle_core PRIVATE
    "${CMAKE_CURRENT_LIST_DIR}/input.cpp"
    "${CMAKE_CURRENT_LIST_DIR}/input.h"
)
target_compile_features(mantle_core PUBLIC cxx_std_17)
target_link_libraries(mantle_core PRIVATE ${_mantle_input_yaml_target})
unset(_mantle_input_yaml_target)
