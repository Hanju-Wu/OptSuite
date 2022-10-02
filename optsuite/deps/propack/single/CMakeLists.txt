file(GLOB SINGLE_SOURCE *.F)
list(FILTER SINGLE_SOURCE EXCLUDE REGEX "smgs\.risc\.F")
add_library(spropack ${SINGLE_SOURCE})
target_link_libraries(spropack ${LAPACK_LIBRARIES} ${BLAS_LIBRARIES})

add_subdirectory(Examples)
add_test(NAME test_single
    COMMAND ${PROJECT_SOURCE_DIR}/scripts/run.sh single ${PROJECT_SOURCE_DIR}
    WORKING_DIRECTORY ${PROJECT_BINARY_DIR}/single/Examples
    )
