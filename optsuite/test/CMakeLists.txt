file(GLOB TEST_SOURCE *.cpp)

foreach(f ${TEST_SOURCE})
    get_filename_component(tg ${f} NAME_WE)
    add_executable(t_${tg} ${f})
    target_link_libraries(t_${tg} OptSuite)
endforeach()

if (MKL_WORKS)
    add_executable(t_mkl_mul mkl/mkl_mul.cpp)
    target_link_libraries(t_mkl_mul OptSuite)
endif()

if (ENABLE_IF_MATLAB AND Matlab_FOUND)
    add_subdirectory(matlab)
endif()

if (FFTW_FOUND)
    add_executable(t_fft fftw/fft.cpp)
    target_link_libraries(t_fft OptSuite)
endif()

if (ARPACK_FOUND)
    add_executable(t_eigs arpack/eigs.cpp)
    target_link_libraries(t_eigs OptSuite)
endif()

if (SuiteSparse_FOUND)
    add_executable(t_cholmod suitesparse/cholmod.cpp)
    target_link_libraries(t_cholmod OptSuite)
endif()
