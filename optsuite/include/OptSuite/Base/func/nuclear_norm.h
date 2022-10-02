/*
 * ===========================================================================
 *
 *       Filename:  nuclear_norm.h
 *
 *    Description:  
 *
 *        Version:  1.0
 *        Created:  01/03/2021 04:12:24 PM
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  Haoyang Liu (), liuhaoyang@pku.edu.cn
 *   Organization:  BICMR, Peking University
 *
 * ===========================================================================
 */

#ifndef OPTSUITE_BASE_FUNC_NUCLEAR_NORM_H
#define OPTSUITE_BASE_FUNC_NUCLEAR_NORM_H

#include "OptSuite/Base/func/base.h"

namespace OptSuite { namespace Base {
    class NuclearNorm : public Func<Scalar> {
        Eigen::JacobiSVD<Mat> svd;
        Eigen::HouseholderQR<Mat> qr;
        public:
            inline NuclearNorm(Scalar mu_ = 1) : mu(mu_) {}
            Scalar operator()(const Ref<const mat_t>) override;
            Scalar operator()(const var_t&) override;
            // signature for factorized_mat
            Scalar operator()(const fmat_t&);

            Scalar mu;
    };
}}

#endif

