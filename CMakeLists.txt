cmake_minimum_required(VERSION 3.16)

project(tinytensor
    VERSION 0.1.0
    DESCRIPTION "A lightweight, header-only C++20 tensor library"
    LANGUAGES CXX
)

# Determine if tinytensor is built as a subproject or standalone
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    set(TINYTENSOR_STANDALONE ON)
else()
    set(TINYTENSOR_STANDALONE OFF)
endif()

# Options
option(TINYTENSOR_BUILD_TESTS "Build tinytensor tests" ${TINYTENSOR_STANDALONE})
option(TINYTENSOR_INSTALL "Generate install target" ${TINYTENSOR_STANDALONE})

# Create header-only library target
add_library(tinytensor INTERFACE)
add_library(tinytensor::tinytensor ALIAS tinytensor)

target_include_directories(tinytensor
    INTERFACE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)

target_compile_features(tinytensor INTERFACE cxx_std_20)

# Tests
if(TINYTENSOR_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

# Installation
if(TINYTENSOR_INSTALL)
    include(GNUInstallDirs)
    include(CMakePackageConfigHelpers)

    # Install headers
    install(
        DIRECTORY include/
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    )

    # Install target
    install(
        TARGETS tinytensor
        EXPORT tinytensor-targets
        INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    )

    # Export targets
    install(
        EXPORT tinytensor-targets
        FILE tinytensor-targets.cmake
        NAMESPACE tinytensor::
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/tinytensor
    )

    # Package config
    configure_package_config_file(
        ${CMAKE_CURRENT_SOURCE_DIR}/cmake/tinytensor-config.cmake.in
        ${CMAKE_CURRENT_BINARY_DIR}/tinytensor-config.cmake
        INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/tinytensor
    )

    write_basic_package_version_file(
        ${CMAKE_CURRENT_BINARY_DIR}/tinytensor-config-version.cmake
        VERSION ${PROJECT_VERSION}
        COMPATIBILITY SameMajorVersion
    )

    install(
        FILES
            ${CMAKE_CURRENT_BINARY_DIR}/tinytensor-config.cmake
            ${CMAKE_CURRENT_BINARY_DIR}/tinytensor-config-version.cmake
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/tinytensor
    )
endif()
