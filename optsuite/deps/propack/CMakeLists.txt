cmake_minimum_required(VERSION 3.3)

project(PROPACK VERSION 2.1.1 LANGUAGES C Fortran)

# disable in-source builds
set(CMAKE_DISABLE_SOURCE_CHANGES ON)
set(CMAKE_DISABLE_IN_SOURCE_BUILD ON)

# set C90 standard
set(CMAKE_C_STANDARD 90)
set(CMAKE_C_STANDARD_REQUIRED True)

# define options
option(BUILD_SHARED_LIBS "Enable shared library. Default: ON" ON)

# define compile options
if ("${CMAKE_BUILD_TYPE}" STREQUAL "")
    message(STATUS "CMAKE_BUILD_TYPE is unset, defaulting to Release")
    set(CMAKE_BUILD_TYPE "Release")
endif()

if ("${CMAKE_Fortran_COMPILER_ID}" STREQUAL "GNU")
    add_compile_options(-Wall -m64 -march=native
        -Wno-unused-dummy-argument -Wno-unused-variable
        -Wno-compare-reals -Wno-do-subscript -Wno-function-elimination)
    if ("${CMAKE_BUILD_TYPE}" STREQUAL "Debug")
        add_compile_options(-Wpedantic)
    endif()
    if ("${CMAKE_BUILD_TYPE}" STREQUAL "Release")
        add_compile_options(-Wno-intrinsic-shadow -Wno-conversion -Wno-uninitialized)
    endif()
elseif ("${CMAKE_Fortran_COMPILER_ID}" STREQUAL "Intel")
    if ("${CMAKE_BUILD_TYPE}" STREQUAL "Debug")
        add_compile_options(-warn all)
    endif()
    if ("${CMAKE_BUILD_TYPE}" STREQUAL "Release")
        add_compile_options(-nowarn)
    endif()
else()
    message(WARNING "Unsupport Fortran compiler ${CMAKE_Fortran_COMPILER_ID}")
endif()

# openmp support

# blas/lapack support
find_package(BLAS REQUIRED)
find_package(LAPACK REQUIRED)

# enable tests
enable_testing()

# now move to subdirectories
add_subdirectory(${PROJECT_SOURCE_DIR}/double)
add_subdirectory(${PROJECT_SOURCE_DIR}/single)
add_subdirectory(${PROJECT_SOURCE_DIR}/complex16)
add_subdirectory(${PROJECT_SOURCE_DIR}/complex8)

# export the targets and installing
install(TARGETS dpropack cpropack spropack zpropack
    EXPORT propack-targets
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    )
install(EXPORT propack-targets
    FILE propack-targets.cmake
    NAMESPACE PROPACK::
    DESTINATION lib/cmake/propack/
    )

# package config file
include(CMakePackageConfigHelpers)
configure_package_config_file(${CMAKE_CURRENT_SOURCE_DIR}/PROPACKConfig.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/PROPACKConfig.cmake
    INSTALL_DESTINATION lib/cmake/propack
    )
install(FILES
    ${CMAKE_CURRENT_BINARY_DIR}/PROPACKConfig.cmake
    DESTINATION lib/cmake/propack
    )
