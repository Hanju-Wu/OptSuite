/*
 * ===========================================================================
 *
 *       Filename:  nuclear_norm.cpp
 *
 *    Description:
 *
 *        Version:  1.0
 *        Created:  01/03/2021 04:13:43 PM
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  Haoyang Liu (), liuhaoyang@pku.edu.cn
 *   Organization:  BICMR, Peking University
 *      Copyright:  Copyright (c) 2021, Haoyang Liu
 *
 * ===========================================================================
 */

#include "OptSuite/Base/func/nuclear_norm.h"
#include "OptSuite/Base/mat_wrapper.h"
#include "OptSuite/Base/factorized_mat.h"

namespace OptSuite { namespace Base {
    Scalar NuclearNorm::operator()(const Ref<const mat_t> x){
        using Eigen::DecompositionOptions;
        svd.compute(x);
        return mu * svd.singularValues().sum();
    }

    Scalar NuclearNorm::operator()(const fmat_t& x){
        Index r = x.rank();
        Mat R_U = qr.compute(x.U().transpose()).matrixQR().topRows(r).triangularView<Eigen::Upper>();
        Mat R_V = qr.compute(x.V().transpose()).matrixQR().topRows(r).triangularView<Eigen::Upper>();

        svd.compute(R_U * R_V.transpose());
        return mu * svd.singularValues().sum();
    }

    Scalar NuclearNorm::operator()(const var_t& x){
        const mat_wrapper_t* x_ptr = dynamic_cast<const mat_wrapper_t*>(&x);
        const fmat_t* x_ptr_f = dynamic_cast<const fmat_t*>(&x);

        OPTSUITE_ASSERT(x_ptr || x_ptr_f);

        if (x_ptr)
            return (*this)(x_ptr->mat());
        else
            return (*this)(*x_ptr_f);
    }
}}
