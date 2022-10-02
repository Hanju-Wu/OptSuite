/*
 * ===========================================================================
 *
 *       Filename:  shrinkage_lp.h
 *
 *    Description:  
 *
 *        Version:  1.1
 *        Created:  04/04/2021 02:12:48 PM
 *       Revision:  none
 *       Compiler:  g++
 *
 *         Author:  Haoyang Liu (), liuhaoyang@pku.edu.cn
 *   Organization:  BICMR, Peking University
 *
 * ===========================================================================
 */

#ifndef OPTSUITE_BASE_FUNC_SHRINKAGE_LP_H
#define OPTSUITE_BASE_FUNC_SHRINKAGE_LP_H

#include "OptSuite/Base/func/base.h"
#include "OptSuite/Utils/optionlist.h"
#include "OptSuite/Base/func/ball_lp.h"

namespace OptSuite { namespace Base {
    class ShrinkageL1 : public Proximal<Scalar> {
        mutable Utils::OptionList options_;
        void register_options();
        Index nnz_, nnz_t_ = -1;
        mat_t d, weights_;
        std::vector<Index> ind;

        void do_truncation(Ref<mat_t>);
        void gen_spmat(const Ref<const mat_t>);
        public:
            inline ShrinkageL1(Scalar mu_ = 1) : mu(mu_) {
                register_options();
            }
            inline ShrinkageL1(const Ref<const Vec> w, Scalar mu_ = 1) : mu(mu_) {
                weights_ = w;
                register_options();
            }
            ~ShrinkageL1() = default;

            using Proximal<Scalar>::operator();
            void operator()(const Ref<const mat_t>, Scalar, Ref<mat_t>) override;

            Utils::OptionList& options();
            const Utils::OptionList& options() const;

            mat_t& weights();
            const mat_t& weights() const;

            Index nnz() const;
            Index nnz_t() const;

            Scalar mu;
    };

    class ShrinkageL2 : public Proximal<Scalar> {
        public:
            inline ShrinkageL2(Scalar mu_ = 1) : mu(mu_) {}
            ~ShrinkageL2() = default;
            
            using Proximal<Scalar>::operator();
            void operator()(const Ref<const mat_t>, Scalar, Ref<mat_t>) override;
            Scalar mu;
    };

    class ShrinkageL2Rowwise : public Proximal<Scalar> {
        using vec_t = Vec;
        vec_t lambda;
        public:
            inline ShrinkageL2Rowwise(Scalar mu_ = 1) : mu(mu_) {}
            ~ShrinkageL2Rowwise() = default;

            using Proximal<Scalar>::operator();
            void operator()(const Ref<const mat_t>, Scalar, Ref<mat_t>) override;

            Scalar mu;
    };

    class ShrinkageL2Colwise : public Proximal<Scalar> {
        using vec_t = Vec;
        vec_t lambda;
        public:
            inline ShrinkageL2Colwise(Scalar mu_ = 1) : mu(mu_) {}
            ~ShrinkageL2Colwise() = default;

            using Proximal<Scalar>::operator();
            void operator()(const Ref<const mat_t>, Scalar, Ref<mat_t>) override;

            Scalar mu;
    };

    class ShrinkageL0 : public Proximal<Scalar> {
        public:
            inline ShrinkageL0(Scalar mu_ = 1) : mu(mu_) {}
            ~ShrinkageL0() = default;

            using Proximal<Scalar>::operator();
            void operator()(const Ref<const mat_t>, Scalar, Ref<mat_t>) override;

            Scalar mu;
    };

    class ShrinkageLInf : public Proximal<Scalar> {
        public:
            inline ShrinkageLInf(Scalar mu_ = 1) : mu(mu_) {}
            ~ShrinkageLInf() = default;

            using Proximal<Scalar>::operator();
            void operator()(const Ref<const mat_t>, Scalar, Ref<mat_t>) override;

            Scalar mu;
    };
}}

#endif

