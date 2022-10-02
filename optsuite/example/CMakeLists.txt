file(GLOB EXAMPLE_SOURCE *.cpp)

foreach(f ${EXAMPLE_SOURCE})
    get_filename_component(tg ${f} NAME_WE)
    add_executable(${tg} ${f})
    target_link_libraries(${tg} OptSuite)
endforeach()

if (ENABLE_SINGLE)
    foreach(f ${EXAMPLE_SOURCE})
        get_filename_component(tg ${f} NAME_WE)
        add_executable(${tg}_f ${f})
        target_link_libraries(${tg} OptSuite_f)
    endforeach()
endif()

if (ENABLE_IF_MATLAB AND Matlab_FOUND)
    add_subdirectory(matlab)
endif()
