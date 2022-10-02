/*
 * ==========================================================================
 *
 *       Filename:  arpack.h
 *
 *    Description:  wrapper for ARPACK-ng package
 *
 *        Version:  1.0
 *        Created:  03/09/2021 01:50:40 PM
 *       Revision:  none
 *       Compiler:  g++
 *
 *         Author:  Haoyang Liu (@liuhy), liuhaoyang@pku.edu.cn
 *   Organization:  BICMR, PKU
 *
 * ==========================================================================
 */

#ifdef OPTSUITE_USE_ARPACK

#ifndef OPTSUITE_LINALG_ARPACK_H
#define OPTSUITE_LINALG_ARPACK_H

#include <string>
#include <vector>
#include "OptSuite/core_n.h"
#include "OptSuite/Base/mat_op.h"
#include "OptSuite/Utils/optionlist.h"

namespace OptSuite { namespace LinAlg {
    constexpr int ARPACK_Sym = 1;
    constexpr int ARPACK_Nonsym = 0;
    template <typename dtype, int self_adjoint>
    class ARPACK {
        using mat_t = Eigen::Matrix<dtype, Dynamic, Dynamic>;
        using vec_t = Eigen::Matrix<dtype, Dynamic, 1>;
        using spmat_t = Eigen::SparseMatrix<dtype, ColMajor, SparseIndex>;
        using matop_t = Base::MatOp<dtype>;

        // internal class
        class __proxy_mat_op : public matop_t {
            const Ref<const mat_t> m;
            public:
            inline __proxy_mat_op(const Ref<const mat_t>& A) : m(A) {
                this->resize(A.rows(), A.cols());
            }
            inline void apply(const Ref<const mat_t>& x, Ref<mat_t> y) const override {
                y = m * x;
            }
            inline void apply_transpose(const Ref<const mat_t>&, Ref<mat_t>) const override {}
        };

        class __proxy_spmat_op : public matop_t {
            const Ref<const spmat_t> m;
            public:
            inline __proxy_spmat_op(const Ref<const spmat_t>& A) : m(A) {
                this->resize(A.rows(), A.cols());
            }
            inline void apply(const Ref<const mat_t>& x, Ref<mat_t> y) const override {
                y = m.transpose() * x;
            }
            inline void apply_transpose(const Ref<const mat_t>&, Ref<mat_t>) const override {}
        };


        // internal variables
        mutable Utils::OptionList options_;
        void register_options();
        mat_t V_;
        mat_t U_;
        vec_t d_;
        int info_;

        // internal storage
        std::vector<Index> ipntr;
        std::vector<dtype> workd;
        std::vector<dtype> resid;
        std::vector<dtype> workl;

        // private functions
        int compute_self_adjoint_normal(const matop_t&, Index, const std::string&, bool = false);
        Index get_ncv(Index, Index) const;

        public:
        ARPACK();
        ARPACK(const Ref<const mat_t>, Index, const std::string&, bool = false);
        ARPACK(const Ref<const spmat_t>, Index, const std::string&, bool = false);
        ARPACK(const matop_t&, Index, const std::string&, bool = false);

        ARPACK& compute(const Ref<const mat_t>, Index, const std::string&m, bool = false);
        ARPACK& compute(const Ref<const spmat_t>, Index, const std::string&, bool = false);
        ARPACK& compute(const matop_t&, Index, const std::string&, bool = false);

        const mat_t& U() const;
        const vec_t& d() const;
        const int& info() const;

        const Utils::OptionList& options() const;
        Utils::OptionList& options();
    };
}}

#endif

#endif // OPTSUITE_USE_ARPACK

