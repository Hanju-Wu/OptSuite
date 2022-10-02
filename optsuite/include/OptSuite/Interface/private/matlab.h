/*
 * ===========================================================================
 *
 *       Filename:  matlab.h
 *
 *    Description:  private header for MATLAB interface
 *
 *        Version:  1.0
 *        Created:  03/03/2021 03:20:24 PM
 *       Revision:  none
 *       Compiler:  g++
 *
 *         Author:  Haoyang Liu (), liuhaoyang@pku.edu.cn
 *   Organization:  BICMR, Peking University
 *
 * ===========================================================================
 */

#ifndef OPTSUITE_INTERFACE_PRIVATE_MATLAB_H
#define OPTSUITE_INTERFACE_PRIVATE_MATLAB_H

#include <cstring>
#include <type_traits>
#include "OptSuite/core_n.h"
#include "matrix.h"
#include "mat.h"


namespace OptSuite { namespace Interface {
#if OPTSUITE_SCALAR_TOKEN == 0
    inline bool mx_is_scalar(const mxArray *p){
        return mxIsDouble(p) && !mxIsComplex(p);
    }
    inline bool mx_is_complex_scalar(const mxArray *p){
        return mxIsDouble(p) && mxIsComplex(p);
    }
#elif OPTSUITE_SCALAR_TOKEN == 1
    inline bool mx_is_scalar(const mxArray *p){
        return mxIsSingle(p) && !mxIsComplex(p);
    }
    inline bool mx_is_complex_scalar(const mxArray *p){
        return mxIsSingle(p) && mxIsComplex(p);
    }
#endif
    int read_variable_from_mat(const char *, const char *,
                              MATFile **, mxArray **);
    template<typename T>
    int mxArray_to_dense(const mxArray *, Eigen::Matrix<T, Dynamic, Dynamic> &);

    template<typename T>
    int mxArray_to_sparse(const mxArray *, Eigen::SparseMatrix<T, ColMajor, SparseIndex> &);
}}

#endif

