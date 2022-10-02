file(GLOB DOUBLE_SOURCE *.F)
list(FILTER DOUBLE_SOURCE EXCLUDE REGEX "dmgs\.risc\.F")
add_library(dpropack ${DOUBLE_SOURCE})
target_link_libraries(dpropack ${LAPACK_LIBRARIES} ${BLAS_LIBRARIES})

add_subdirectory(Examples)
add_test(NAME test_double
    COMMAND ${PROJECT_SOURCE_DIR}/scripts/run.sh double ${PROJECT_SOURCE_DIR}
    WORKING_DIRECTORY ${PROJECT_BINARY_DIR}/double/Examples
    )
