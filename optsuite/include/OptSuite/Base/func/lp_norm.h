/*
 * ===========================================================================
 *
 *       Filename:  lp_norm.h
 *
 *    Description:  
 *
 *        Version:  1.0
 *        Created:  01/03/2021 04:06:25 PM
 *       Revision:  04/05/2021 02:58:00 PM
 *       Compiler:  g++
 *
 *         Author:  Haoyang Liu (), liuhaoyang@pku.edu.cn
 *   Organization:  BICMR, Peking University
 *
 * ===========================================================================
 */

#ifndef OPTSUITE_BASE_FUNC_LP_NORM_H
#define OPTSUITE_BASE_FUNC_LP_NORM_H

#include "OptSuite/Base/func/base.h"

namespace OptSuite { namespace Base {
    class L1Norm : public Func<Scalar> {
        mat_t weights_;
        public:
            inline L1Norm(Scalar mu_ = 1) : mu(mu_) {}
            inline L1Norm(const Ref<const mat_t> w, Scalar mu_ = 1) : mu(mu_){
                weights_ = w;
            }
            using Func<Scalar>::operator();
            Scalar operator()(const Ref<const mat_t>) override;

            mat_t& weights();
            const mat_t& weights() const;

            Scalar mu;
    };

    class L2Norm : public Func<Scalar> {
        public:
            inline L2Norm(Scalar mu_ = 1) : mu(mu_) {}
            using Func<Scalar>::operator();
            Scalar operator()(const Ref<const mat_t>) override;

            Scalar mu;
    };

    class L1_2Norm : public Func<Scalar> {
        public:
            inline L1_2Norm(Scalar mu_ = 1) : mu(mu_) {}
            using Func<Scalar>::operator();
            Scalar operator()(const Ref<const mat_t>) override;

            Scalar mu;
    };

    class L2_1Norm : public Func<Scalar> {
        public:
            inline L2_1Norm(Scalar mu_ = 1) : mu(mu_) {}
            using Func<Scalar>::operator();
            Scalar operator()(const Ref<const mat_t>) override;

            Scalar mu;
    };

    class L0Norm : public Func<Scalar> {
        public:
            inline L0Norm(Scalar mu_ = 1) : mu(mu_) {}
            using Func<Scalar>::operator();
            Scalar operator()(const Ref<const mat_t>) override;

            Scalar mu;
    };

    class LInfNorm : public Func<Scalar> {
        public:
            inline LInfNorm(Scalar mu_ = 1) : mu(mu_) {}
            using Func<Scalar>::operator();
            Scalar operator()(const Ref<const mat_t>) override;

            Scalar mu;
    };
}}

#endif

