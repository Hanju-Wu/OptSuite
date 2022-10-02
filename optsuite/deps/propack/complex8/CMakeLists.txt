file(GLOB COMPLEX_DOUBLE_SOURCE *.F)
list(FILTER COMPLEX_DOUBLE_SOURCE EXCLUDE REGEX "cmgs\.risc\.F")
add_library(cpropack ${COMPLEX_DOUBLE_SOURCE})
target_link_libraries(cpropack ${LAPACK_LIBRARIES} ${BLAS_LIBRARIES})

add_subdirectory(Examples)
add_test(NAME test_complex8
    COMMAND ${PROJECT_SOURCE_DIR}/scripts/run.sh complex8 ${PROJECT_SOURCE_DIR}
    WORKING_DIRECTORY ${PROJECT_BINARY_DIR}/complex8/Examples
    )
