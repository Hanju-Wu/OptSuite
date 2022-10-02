/*
 * ===========================================================================
 *
 *       Filename:  shrinkage_nuclear.h
 *
 *    Description:  
 *
 *        Version:  1.0
 *        Created:  01/03/2021 03:57:34 PM
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  Haoyang Liu (), liuhaoyang@pku.edu.cn
 *   Organization:  BICMR, Peking University
 *
 * ===========================================================================
 */

#ifdef OPTSUITE_USE_PROPACK

#ifndef OPTSUITE_BASE_FUNC_SHRINKAGE_NUCLEAR_H
#define OPTSUITE_BASE_FUNC_SHRINKAGE_NUCLEAR_H

#include "OptSuite/Base/func/base.h"
#include "OptSuite/Utils/optionlist.h"
#include "OptSuite/LinAlg/lansvd.h"

namespace OptSuite { namespace Base {
    class ShrinkageNuclear : public Proximal<Scalar> {
        using vec_t = Vec;
        using fmat_t = FactorizedMat<Scalar>;
        using smat_t = SpMatWrapper<Scalar>;
        vec_t d;
        Eigen::JacobiSVD<mat_t> svd; // JacobiSVD is using LAPACKE/MKL
        LinAlg::LANSVD<Scalar> lansvd;
        unsigned op = Eigen::DecompositionOptions::ComputeThinU |
                      Eigen::DecompositionOptions::ComputeThinV;

        Index compute_rank() const;
        mutable Utils::OptionList options_;
        Index rank = -1, rank_prev = -1;
        Index rank_same_cnt = -1;
        Index iter = -1;

        void register_options();
        void shrinkage_impl_f(const MatOp<Scalar>&, const fmat_t&, Scalar, fmat_t&);

        public:
            ShrinkageNuclear(Scalar = 1);
            ~ShrinkageNuclear() = default;
            inline bool has_objective_cache() const override { return true; }
            Scalar cached_objective() const override;

            void operator()(const Ref<const mat_t>, Scalar, Ref<mat_t>) override;
            void operator()(const var_t&, Scalar, const var_t&, Scalar, var_t&) override;

            // new functions
            void operator()(const fmat_t&, Scalar, const smat_t&, Scalar, fmat_t&);
            void operator()(const fmat_t&, Scalar, const mat_wrapper_t&, Scalar, fmat_t&);

            Utils::OptionList& options();
            const Utils::OptionList& options() const;

            Index sv() const { return rank; }
            Index svp() const { return rank_prev; }

            Scalar mu;
    };
}}

#endif

#endif // OPTSUITE_USE_PROPACK
