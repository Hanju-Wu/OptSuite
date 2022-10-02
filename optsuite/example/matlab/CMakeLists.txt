file(GLOB EXAMPLE_MATLAB_SOURCE *.cpp)

foreach(f ${EXAMPLE_MATLAB_SOURCE})
    get_filename_component(tg ${f} NAME_WE)
    # matcomp needs MKL, wf needs fftw
    if ("${tg}" STREQUAL "matcomp" OR "${tg}" STREQUAL "wf")
        continue()
    endif()
    add_executable(${tg} ${f})
    target_link_libraries(${tg} OptSuite_matlab OptSuite)
endforeach()

if (MKL_WORKS AND USE_PROPACK)
    add_executable(matcomp matcomp.cpp)
    target_link_libraries(matcomp OptSuite_matlab OptSuite)
endif()

if (FFTW_FOUND)
    add_executable(wf wf.cpp)
    target_link_libraries(wf OptSuite_matlab OptSuite)
    if (ENABLE_SINGLE)
        add_executable(wf_f wf.cpp)
        target_link_libraries(wf_f OptSuite_matlab_f OptSuite_f)
    endif()
endif()

