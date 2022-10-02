file(GLOB COMPLEX_DOUBLE_SOURCE *.F)
list(FILTER COMPLEX_DOUBLE_SOURCE EXCLUDE REGEX "zmgs\.risc\.F")
add_library(zpropack ${COMPLEX_DOUBLE_SOURCE})
target_link_libraries(zpropack ${LAPACK_LIBRARIES} ${BLAS_LIBRARIES})

add_subdirectory(Examples)
add_test(NAME test_complex16
    COMMAND ${PROJECT_SOURCE_DIR}/scripts/run.sh complex16 ${PROJECT_SOURCE_DIR}
    WORKING_DIRECTORY ${PROJECT_BINARY_DIR}/complex16/Examples
    )
