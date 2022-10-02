cmake_minimum_required(VERSION 3.12)

project(optsuite VERSION 0.1.0)

# disable in-source builds
set(CMAKE_DISABLE_SOURCE_CHANGES ON)
set(CMAKE_DISABLE_IN_SOURCE_BUILD ON)

# set C++11 standard
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# set module path
set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

# include optsuite build utils
include(cmake/optsuite_utils.cmake)

# enumerate source/headers
file(GLOB_RECURSE ALL_SOURCE ${PROJECT_SOURCE_DIR}/src/*.cpp)
set(SOURCE ${ALL_SOURCE})
list(FILTER SOURCE EXCLUDE REGEX "${PROJECT_SOURCE_DIR}/src/Interface/.*\.cpp")
set(MATLAB_SOURCE ${PROJECT_SOURCE_DIR}/src/Interface/matlab.cpp)
file(GLOB BASE_HEADER ${PROJECT_SOURCE_DIR}/include/OptSuite/Base/*.h)

# define options
option(BUILD_SHARED_LIBS "Enable shared library. Default: ON" ON)
option(ENABLE_SINGLE
    "Enable the build of a single precision library. Default: OFF" OFF)
option(ENABLE_IF_MATLAB
    "Enable the build of matlab interface. Default: OFF" OFF)
option(EIGEN_USE_BUILTIN
    "Force Eigen to use built-in BLAS/LAPACK implementations. Default: OFF" OFF)
option(ENABLE_PRECOMPILE_HEADERS "Enable the use of precompiled headers. Default: ON" ON)
option(USE_FFTW "Enable the use of FFTW. Default: ON" ON)
option(USE_PROPACK "Enable the use of nested PROPACK. Default: ON" ON)
option(USE_ARPACK "Enable the use of ARPACK. Default: ON" ON)
option(USE_SUITESPARSE "Enable the use of SUITESPARSE. Default: ON" ON)


# config file
configure_file(include/OptSuite/config.h.in config.h)

# global include directories
include_directories("${PROJECT_BINARY_DIR}" "${PROJECT_SOURCE_DIR}/include")

# CMAKE_BUILD_TYPE must be "Debug" or "Release"
if ("${CMAKE_BUILD_TYPE}" STREQUAL "")
    message(STATUS "CMAKE_BUILD_TYPE is unset, defaulting to Debug")
    set(CMAKE_BUILD_TYPE "Debug")
endif()
set(valid_builds Debug Release)
if (NOT "${CMAKE_BUILD_TYPE}" IN_LIST valid_builds)
    message(FATAL_ERROR "CMAKE_BUILD_TYPE must be Debug or Release")
endif()

# for building shared libraries, defines these macros
# see also include/OptSuite/api_macro.h
if (BUILD_SHARED_LIBS)
    add_definitions(-DOPTSUITE_BUILD_DYN_LIB)
endif()

if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
    add_compile_options(-Wall -Wextra -m64)
    if ("${CMAKE_BUILD_TYPE}" STREQUAL "Debug")
        add_compile_options(-Wpedantic)
    else()
        add_compile_options(-Wno-unused-parameter)
        # due to a bug in mingw-w64/gcc, avx or avx2 should be disabled
        if (NOT MINGW)
            add_compile_options(-march=native)
        endif()
    endif()
elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Intel")
    # note: not working for current version because C++17 support is poor
    add_compile_options($<$<COMPILE_LANGUAGE:CXX,C>:-Wall> -inline-forceinline)
elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
    add_compile_options(-Wall -Wextra -march=native)
    if ("${CMAKE_BUILD_TYPE}" STREQUAL "Debug")
        add_compile_options(-Wpedantic)
    else()
        add_compile_options(-Wno-unused-parameter)
    endif()
else()
    message(WARNING "Unsupported CXX compiler ${CMAKE_CXX_COMPILER_ID}")
endif()

# subprojects
add_subdirectory(deps/SRBio deps/SRBio/build)
if (USE_PROPACK)
    add_subdirectory(deps/propack deps/propack/build)
endif()

# define targets
add_library(OptSuite ${SOURCE})
target_link_libraries(OptSuite PUBLIC Eigen3::Eigen)

# define pre-compiled headers
if (${CMAKE_VERSION} VERSION_GREATER_EQUAL "3.16" AND ENABLE_PRECOMPILE_HEADERS)
    #target_precompile_headers(OptSuite PRIVATE ${BASE_HEADER})
    target_precompile_headers(OptSuite PRIVATE
        <Eigen/Eigen>
        <vector>
        <unordered_map>
        <memory>
        <iostream>
        <algorithm>
        <string>
        <utility>
        <complex>
        <limits>
        )
endif()

# define single precision targets
if (ENABLE_SINGLE)
    add_library(OptSuite_f ${SOURCE})
    target_compile_definitions(OptSuite_f PUBLIC OPTSUITE_SCALAR_TOKEN=1)
    target_link_libraries(OptSuite_f PUBLIC Eigen3::Eigen)
endif()

# eigen3 prereq
find_package(Eigen3 3.3 REQUIRED NO_MODULE)

if (EIGEN_USE_BUILTIN)
    add_definitions(-DOPTSUITE_EIGEN_USE_BUILTIN)
endif()

# blas/lapack support
# note: Though blas/lapack may not be required by Eigen,
#       other libraries (e.g. PROPACK, ARPACK) may require them.
find_package(BLAS REQUIRED)
find_package(LAPACKE REQUIRED)

# check whether it is MKL
if (BLAS_LIBRARIES MATCHES "mkl_core")
    # check mkl.h presence
    find_file(MKL_H
        NAMES "mkl.h"
        PATHS ENV MKLROOT
        PATH_SUFFIXES "mkl" "include"
        )

    if (MKL_H)
        message(STATUS "Found mkl.h: ${MKL_H}")
        message(STATUS "Enable MKL support in OptSuite.")
        set(MKL_WORKS TRUE)
        add_definitions(-DOPTSUITE_USE_MKL)
        get_filename_component(MKL_INC ${MKL_H} DIRECTORY)
        include_directories(${MKL_INC})
    else()
        message(FATAL_ERROR "Unable to locate mkl.h. Please check your MKL installation.")
    endif()
else()
    set(MKL_WORKS FALSE)
endif()

target_link_libraries(OptSuite PUBLIC ${LAPACKE_LIBRARIES} ${BLAS_LIBRARIES})
if (ENABLE_SINGLE)
    target_link_libraries(OptSuite_f PUBLIC ${LAPACKE_LIBRARIES} ${BLAS_LIBRARIES})
endif()

# fftw support
if (USE_FFTW)
    if (ENABLE_SINGLE)
        find_package(FFTW COMPONENTS FLOAT_LIB DOUBLE_LIB)
    else()
        find_package(FFTW COMPONENTS DOUBLE_LIB)
    endif()

    if (FFTW_FOUND)
        add_definitions(-DOPTSUITE_USE_FFTW)
        target_link_libraries(OptSuite PUBLIC FFTW::Double)
        if (ENABLE_SINGLE)
            target_link_libraries(OptSuite_f PUBLIC FFTW::Float)
        endif()
    else()
        message(STATUS "FFTW not found")
    endif()
endif()

# matlab support
if (ENABLE_IF_MATLAB)
    find_package(Matlab COMPONENTS MAT_LIBRARY)

    if (Matlab_FOUND)
        message(STATUS "Matlab found at ${Matlab_ROOT_DIR}")

        # compile defs
        set(Matlab_COMPILE_DEFS MATLAB_DEFAULT_RELEASE=R2017b MATLAB_MEX_FILE)

        # add matlab interface library
        add_library(OptSuite_matlab ${MATLAB_SOURCE})

        target_compile_definitions(OptSuite_matlab PRIVATE ${Matlab_COMPILE_DEFS})
        target_include_directories(OptSuite_matlab PRIVATE ${Matlab_INCLUDE_DIRS})

        # MATLAB libraries must be propagated by PUBLIC
        target_link_libraries(OptSuite_matlab PUBLIC Eigen3::Eigen OptSuite
            ${Matlab_MEX_LIBRARY} ${Matlab_MX_LIBRARY} ${Matlab_MAT_LIBRARY})

        # single precision lib
        if (ENABLE_SINGLE)
            add_library(OptSuite_matlab_f ${MATLAB_SOURCE})

            target_compile_definitions(OptSuite_matlab_f PUBLIC OPTSUITE_SCALAR_TOKEN=1)
            target_include_directories(OptSuite_matlab_f PRIVATE ${Matlab_INCLUDE_DIRS})

            # MATLAB libraries must be propagated by PUBLIC
            target_link_libraries(OptSuite_matlab_f PUBLIC Eigen3::Eigen OptSuite_f
                ${Matlab_MEX_LIBRARY} ${Matlab_MX_LIBRARY} ${Matlab_MAT_LIBRARY})
        endif()
    else()
        message(WARNING "-DENABLE_IF_MATLAB=ON is set but Matlab not found. "
            "Try passing -DMatlab_ROOT_DIR to specify Matlab installation location")
    endif()
endif()

# propack support
if (USE_PROPACK)
    add_definitions(-DOPTSUITE_USE_PROPACK)
    target_link_libraries(OptSuite PUBLIC dpropack)
    if (ENABLE_SINGLE)
        target_link_libraries(OptSuite_f PUBLIC spropack)
    endif()
endif()

# suitesparse support

if (USE_SUITESPARSE)
    find_package(SuiteSparse MODULE COMPONENTS CHOLMOD AMD CAMD COLAMD CCOLAMD SUITESPARSECONFIG metis)

    if (SuiteSparse_FOUND)
        include_directories(${SuiteSparse_INCLUDE_DIRS})
        add_definitions(-DOPTSUITE_USE_SUITE_SPARSE)
        target_link_libraries(OptSuite PUBLIC ${SuiteSparse_LIBRARIES})
        if (ENABLE_SINGLE)
            target_link_libraries(OptSuite_f PUBLIC ${SuiteSparse_LIBRARIES})
        endif()
    else()
        message(STATUS "SuiteSparse not found")
    endif()
endif()

# ARPACK support
if (USE_ARPACK)
    find_package(ARPACK)
    if (ARPACK_FOUND)
        add_definitions(-DOPTSUITE_USE_ARPACK)
        target_link_libraries(OptSuite PUBLIC ARPACK::ARPACK)
        if (ENABLE_SINGLE)
            target_link_libraries(OptSuite_f PUBLIC ARPACK::ARPACK)
        endif()
    else()
        message(STATUS "ARPACK not found")
    endif()
endif()

# subdirectories (for example/testing)
add_subdirectory(test)
add_subdirectory(example)

# print config info
show_config_summary()
