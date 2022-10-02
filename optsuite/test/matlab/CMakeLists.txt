file(GLOB TEST_MATLAB_SOURCE *.cpp)
if (NOT USE_PROPACK)
    list(FILTER TEST_MATLAB_SOURCE EXCLUDE REGEX "svd\.cpp")
endif()

foreach(f ${TEST_MATLAB_SOURCE})
    get_filename_component(tg ${f} NAME_WE)
    add_executable(${tg} ${f})
    target_compile_definitions(${tg} PRIVATE ${Matlab_COMPILE_DEFS})
    target_link_libraries(${tg} Eigen3::Eigen OptSuite_matlab OptSuite)
endforeach()

