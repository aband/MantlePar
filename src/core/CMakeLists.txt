add_library(mantle_core
    mesh.cpp
)

target_include_directories(mantle_core
    PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}"
)

target_link_libraries(mantle_core
    PUBLIC
        PkgConfig::PETSC
        PkgConfig::LAPACKE
)

if(BUILD_TESTING)
    add_subdirectory(test)
endif()
