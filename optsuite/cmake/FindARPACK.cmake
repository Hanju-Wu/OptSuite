# - Find the ARPACK library
#
#   Copyright (c) 2021, Haoyang Liu
#
# Usage:
#   find_package(ARPACK [REQUIRED] [QUIET])
#
# It sets the following variables:
#   ARPACK_FOUND                  ... true if ARPACK is found on the system
#   ARPACK_LIBRARIES              ... full paths to all found ARPACK libraries
#
# The following variables will be checked by the function
#   ARPACK_ROOT_DIR               ... if set, the libraries are exclusively searched
#                                     under this path
#

if( NOT ARPACK_ROOT_DIR AND DEFINED ENV{ARPACK_DIR} )
    set( ARPACK_ROOT_DIR $ENV{ARPACK_DIR} )
endif()

if( ARPACK_ROOT_DIR )
    # find headers
    find_file(
        ARPACK_HPP
        NAMES "arpack.hpp"
        PATHS ${ARPACK_ROOT_DIR}
        PATH_SUFFIXES "include" "include/arpack"
        NO_DEFAULT_PATH)

    # find libs
    find_library(
        ARPACK_LIBRARIES
        NAMES "arpack"
        PATHS ${ARPACK_ROOT_DIR}
        PATH_SUFFIXES "lib" "lib64"
        NO_DEFAULT_PATH)

else()
    # find headers
    find_file(
        ARPACK_HPP
        NAMES "arpack.hpp"
        PATH_SUFFIXES "include" "include/arpack"
        )

    # find libs
    find_library(
        ARPACK_LIBRARIES
        NAMES "arpack"
        PATH_SUFFIXES "lib" "lib64")

endif( ARPACK_ROOT_DIR )

#--------------------------------------- print messages
if (NOT ARPACK_HPP)
    message(STATUS "No arpack.hpp found. Please check your ARPACK installation.")
    message(STATUS "Hint: iso-c-binding must be enabled.")
endif()

if (NOT ARPACK_LIBRARIES)
    message(STATUS "No ARPACK libraries found. Please check your ARPACK installation.")
endif()

#--------------------------------------- end messages

#--------------------------------------- libs
if (ARPACK_HPP AND ARPACK_LIBRARIES)
    get_filename_component(incdir ${ARPACK_HPP} DIRECTORY)
    add_library(ARPACK::ARPACK INTERFACE IMPORTED)
    set_target_properties(ARPACK::ARPACK
        PROPERTIES INTERFACE_LINK_LIBRARIES "${ARPACK_LIBRARIES}"
        INTERFACE_INCLUDE_DIRECTORIES "${incdir}")
endif()

#--------------------------------------- end libs

include(FindPackageHandleStandardArgs)

find_package_handle_standard_args(ARPACK
    REQUIRED_VARS ARPACK_LIBRARIES ARPACK_HPP
    )

mark_as_advanced(
    ARPACK_FOUND
    ARPACK_LIBRARIES
    )
