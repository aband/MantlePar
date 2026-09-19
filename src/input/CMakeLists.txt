find_package(yaml-cpp 0.7 CONFIG REQUIRED)

if(TARGET yaml-cpp::yaml-cpp)
    set(_mantle_input_yaml_target yaml-cpp::yaml-cpp)
elseif(TARGET yaml-cpp)
    set(_mantle_input_yaml_target yaml-cpp)
else()
    message(FATAL_ERROR "yaml-cpp did not provide a supported CMake target")
endif()

add_library(mantle_input input.cpp)
target_include_directories(mantle_input PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}")
target_compile_features(mantle_input PUBLIC cxx_std_17)
target_link_libraries(mantle_input
    PUBLIC PkgConfig::PETSC
    PRIVATE ${_mantle_input_yaml_target}
)
unset(_mantle_input_yaml_target)
