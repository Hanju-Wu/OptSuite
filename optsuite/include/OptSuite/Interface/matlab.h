/*
 * ==========================================================================
 *
 *       Filename:  matlab.h
 *
 *    Description:
 *
 *        Version:  1.0
 *        Created:  11/28/2020 12:28:56 PM
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  Haoyang Liu (@liuhy), liuhaoyang@pku.edu.cn
 *   Organization:  BICMR, PKU
 *
 * ==========================================================================
 */

#ifndef OPTSUITE_INTERFACE_MATLAB_H
#define OPTSUITE_INTERFACE_MATLAB_H

#include "OptSuite/core_n.h"

namespace OptSuite { namespace Interface {
    template<typename T>
    void get_matrix_from_file(const char *, const char *,
                              Eigen::Matrix<T, Dynamic, Dynamic> &);
    template<typename T>
    void get_matrix_from_file(const char *, const char *,
                              Eigen::SparseMatrix<T, ColMajor, SparseIndex> &);
}}

#endif
