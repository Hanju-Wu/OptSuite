function(libsummary title include libraries)
    message("   -- ${title}:")
    foreach(inc ${include})
        message("      -- compile: ${inc}")
    endforeach()

    # libraries may have a form: optimized;xxx;debug;xxx
    foreach(lib ${libraries})
        if ("${lib}" STREQUAL "debug")
            set(PREFIX_KEYWORD "debug")
        elseif ("${lib}" STREQUAL "optimized")
            set(PREFIX_KEYWORD "optimized")
        else()
            if ("${CMAKE_BUILD_TYPE}" STREQUAL "Debug" AND PREFIX_KEYWORD STREQUAL "optimized")
                # type mismatch, skip
            elseif ((NOT ("${CMAKE_BUILD_TYPE}" STREQUAL "Debug")) AND PREFIX_KEYWORD STREQUAL "debug")
                # type mismatch, skip
            else()
                message("      -- link:    ${lib}")
            endif()
            set(PREFIX_KEYWORD "")
        endif()
    endforeach()
endfunction(libsummary)

function(cprsummary title compiler debug_flags minsizerel_flags release_flags relwithdebinfo_flags more_flags)
    message("   -- ${title}:      ${compiler}")
    if("${CMAKE_BUILD_TYPE}" STREQUAL "Debug" OR "${CMAKE_BUILD_TYPE}" STREQUAL "DEBUG")
        message("   -- ${title}FLAGS: ${debug_flags} ${more_flags}")
    endif()
    if("${CMAKE_BUILD_TYPE}" STREQUAL "MinSizeRel" OR "${CMAKE_BUILD_TYPE}" STREQUAL "MINSIZEREL")
        message("   -- ${title}FLAGS: ${minsizerel_flags} ${more_flags}")
    endif()
    if("${CMAKE_BUILD_TYPE}" STREQUAL "Release" OR "${CMAKE_BUILD_TYPE}" STREQUAL "RELEASE")
        message("   -- ${title}FLAGS: ${release_flags} ${more_flags}")
    endif()
    if("${CMAKE_BUILD_TYPE}" STREQUAL "RelWithDebInfo" OR "${CMAKE_BUILD_TYPE}" STREQUAL "RELWITHDEBINFO")
        message("   -- ${title}FLAGS: ${relwithdebinfo_flags} ${more_flags}")
    endif()
endfunction(cprsummary)

function(show_config_summary)
    message("")
    message("-- Configuration summary for OptSuite:")
    message("   -- PREFIX: ${CMAKE_INSTALL_PREFIX}")
    message("   -- BUILD: ${CMAKE_BUILD_TYPE}")
    message("   -- SHARED_LIBS: ${BUILD_SHARED_LIBS}")
    message("   -- PATH: ${CMAKE_PREFIX_PATH}")
    message("   -- ENABLE_SINGLE: ${ENABLE_SINGLE}")
    message("   -- MKL: ${MKL_WORKS}")
    if (MKL_WORKS)
        message("      -- compile: ${MKL_INC}")
    endif()

    # compilers
    cprsummary("CXX" "${CMAKE_CXX_COMPILER}"
        "${CMAKE_CXX_FLAGS_DEBUG}"
        "${CMAKE_CXX_FLAGS_MINSIZEREL}"
        "${CMAKE_CXX_FLAGS_RELEASE}"
        "${CMAKE_CXX_FLAGS_RELWITHDEBINFO}"
        "${CMAKE_CXX_FLAGS}")

    # matlab
    if (Matlab_FOUND)
        message("   -- MATLAB: ON")
        message("      -- DIR: ${Matlab_ROOT_DIR}")
    else()
        message("   -- MATLAB: OFF")
    endif()

    # eigen
    libsummary("Eigen" "${EIGEN3_INCLUDE_DIRS}" "")

    # blas/lapack
    libsummary("BLAS" "" "${BLAS_LIBRARIES}")
    libsummary("LAPACKE" "${LAPACKE_INCLUDE_DIRS}" "${LAPACKE_LIBRARIES}")

    # fftw
    if (FFTW_FOUND)
        if (ENABLE_SINGLE)
            libsummary("FFTW" "${FFTW_INCLUDE_DIRS}" "${FFTW_FLOAT_LIB};${FFTW_DOUBLE_LIB}")
        else()
            libsummary("FFTW" "${FFTW_INCLUDE_DIRS}" "${FFTW_DOUBLE_LIB}")
        endif()
    else()
        message("   -- FFTW: OFF")
    endif()


    # propack
    if (USE_PROPACK)
        message("   -- PROPACK: (bundled)")
    else()
        message("   -- PROPACK: OFF")
    endif()

    # arpack
    if (ARPACK_FOUND)
        get_target_property(arpack_inc ARPACK::ARPACK INTERFACE_INCLUDE_DIRECTORIES)
        get_target_property(arpack_lib ARPACK::ARPACK INTERFACE_LINK_LIBRARIES)
        libsummary("ARPACK" ${arpack_inc} ${arpack_lib})
    else()
        message("   -- ARPACK: OFF")
    endif()

    # SuiteSparse
    if (SuiteSparse_FOUND)
        libsummary("SuiteSparse" "${SuiteSparse_INCLUDE_DIRS}" "${SuiteSparse_LIBRARIES}")
    else()
        message("   -- SuiteSparse: OFF")
    endif()

    message("")
endfunction()

function(reuse_pch_from tgt from_tgt)
    if (${CMAKE_VERSION} VERSION_GREATER_EQUAL "3.16")
        target_precompile_headers(${tgt} REUSE_FROM ${from_tgt})
    endif()
endfunction()

