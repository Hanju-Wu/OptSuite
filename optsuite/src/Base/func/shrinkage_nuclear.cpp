/*
 * ===========================================================================
 *
 *       Filename:  shrinkage_nuclear.cpp
 *
 *    Description:
 *
 *        Version:  1.0
 *        Created:  01/03/2021 03:59:30 PM
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  Haoyang Liu (), liuhaoyang@pku.edu.cn
 *   Organization:  BICMR, Peking University
 *      Copyright:  Copyright (c) 2021, Haoyang Liu
 *
 * ===========================================================================
 */

#ifdef OPTSUITE_USE_PROPACK

#include "OptSuite/Base/func/shrinkage_nuclear.h"
#include "OptSuite/Base/mat_wrapper.h"
#include "OptSuite/Base/spmat_wrapper.h"
#include "OptSuite/Base/factorized_mat.h"
#include "OptSuite/Base/structure.h"

namespace OptSuite { namespace Base {
    ShrinkageNuclear::ShrinkageNuclear(Scalar mu_) : mu(mu_){
        register_options();
    }

    Utils::OptionList& ShrinkageNuclear::options(){
        return options_;
    }

    const Utils::OptionList& ShrinkageNuclear::options() const {
        return options_;
    }

    void ShrinkageNuclear::register_options(){
        std::vector<Utils::RegOption> v;

        v.push_back({"truncation", true, "Truncation switch"});
        v.push_back({"ir", false, "Use lansvd-irl for SVD"});
        v.push_back({"gap", 5_s, "gap"});
        v.push_back({"window", 10_i, "Truncation window"});
        v.push_back({"istart", 4_i, "Index for start"});
        v.push_back({"threshold", 1e-5_s, "threshold"});
        v.push_back({"maxrank", 150_i, "maximum rank"});
        v.push_back({"fix_rank_count", 5_i, "fix rank(X) if rank(X) does not change within this number of calls"});

        // v is constructed, now initialize options_
        this->options_ = Utils::OptionList(v, nullptr);

    }

    Index ShrinkageNuclear::compute_rank() const {
        // import from options
        bool truncation = options_.get_bool("truncation");
        Scalar threshold = options_.get_scalar("threshold");
        Scalar gap = options_.get_scalar("gap");
        Index window = options_.get_integer("window");
        Index istart = options_.get_integer("istart");
        Index maxrank = options_.get_integer("maxrank");

        // find the first index < threshold
        // using binary search algorithm
        Index lo = 0_i, hi = d.size() - 1_i, i;
        while ( lo < hi ){
            i = (hi + lo) / 2_i;
            if (d[i] < threshold)
                hi = i;
            else // d[i] > threshold
                lo = i + 1_i;
        }

        Index rank_t = d[lo] >= threshold ? lo + 1_i : lo;
        Index rank_g = d.size();
        Index iend = d.size() - 1;

        if (truncation){
            // compute the end of trunction candidate (tend)
            for (i = istart; i < d.size() - 1; ++i){
                if (d[i] / d[i+1] > gap)
                    break;
            }
            iend = std::min(rank_t - 1_i, i + 1_i);

            if (istart < iend){
                Scalar max_ratio = 0;
                Scalar ratio_k = 0;
                Index imax_ratio = 0;
                for (Index kk = 0; kk < iend - istart; ++kk){
                    // sd1 = mean(d(kk-window:kk))
                    // sd2 = mean(d(kk+1:kk+window+1);
                    Index s1 = std::max(0_i, kk + istart + 1 - window);
                    Index l1 = std::min(window, kk + istart + 1);
                    Index s2 = kk + istart + 1;
                    Index l2 = std::min(window, iend - kk - istart);
                    Scalar sd1 = d.block(s1, 0, l1, 1).mean();
                    Scalar sd2 = d.block(s2, 0, l2, 1).mean() + OptSuite::eps;

                    ratio_k = sd1 / sd2;
                    if (ratio_k > max_ratio){
                        max_ratio = ratio_k;
                        imax_ratio = kk;
                    }
                }

                if (max_ratio > gap)
                    rank_g = imax_ratio + istart + 1;
            } else {
                rank_g = istart + 1;
            }

        }

        return std::min(maxrank, std::min(rank_t, rank_g));
    }

    void ShrinkageNuclear::operator()(const Ref<const mat_t> x, Scalar t, Ref<mat_t> y){
        const vec_t& sv = svd.compute(x, op).singularValues();
        const mat_t& U = svd.matrixU();
        const mat_t& V = svd.matrixV();

        d.array() = (sv.array() - t * mu).max(0);

        rank = compute_rank();
        if (rank == 0)
            y = mat_t::Zero(x.rows(), x.cols());
        else
            y = U.leftCols(rank) * d.head(rank).asDiagonal() * V.leftCols(rank).transpose();
    }
    void ShrinkageNuclear::operator()(const fmat_t& xp, Scalar tau,
            const mat_wrapper_t& gp, Scalar v, fmat_t& x){
        // construct mat op
        FactorizePMatOp<Scalar> Aop(xp, -tau, gp);

        // call impl
        shrinkage_impl_f(Aop, xp, v, x);
    }

    void ShrinkageNuclear::operator()(const fmat_t& xp, Scalar tau, const smat_t& gp,
            Scalar v, fmat_t& x){

        // construct mat op
        FactorizePSpMatOp<Scalar> Aop(xp, -tau, gp);

        // call impl
        shrinkage_impl_f(Aop, xp, v, x);
    }

    void ShrinkageNuclear::shrinkage_impl_f(const MatOp<Scalar>& Aop,
           const fmat_t& xp, Scalar v, fmat_t& x){
        Index p;
        Index fix_rank_count = options_.get_integer("fix_rank_count");
        bool is_irl = options_.get_bool("ir");
        // check the rank of xp (only when #iteration has changed)
        if (workspace != nullptr && workspace->get<Index>("iter") != iter){
            iter = workspace->get<Index>("iter");
            if (rank_prev == xp.rank())
                ++rank_same_cnt;
            else
                rank_same_cnt = 0;

            rank_prev = rank;
        }

        if (rank_same_cnt > fix_rank_count){ // rank(X) doesn't change in X iters
            p = xp.rank();
            // use small tolerance
            lansvd.options().set_scalar("tol", 1e-4_s);
        } else { // now add some guard vectors
            p = std::min(
                    1_i + Index(xp.rank() * 1.2_s),
                    std::min(xp.rows(), xp.cols())
                );
            // use large tolerance
            lansvd.options().set_scalar("tol", 100 * 1e-4_s);

        }

        // set options for lansvd
        lansvd.options().set_bool("ir", is_irl);
        // call lansvd
        const vec_t& sv = lansvd.compute(Aop, p).d();
        const mat_t& U = lansvd.U();
        const mat_t& V = lansvd.V();

        d.array() = (sv.array() - v * mu).max(0);

        // note: rank can be larger or smaller than xp.rank()
        // thus x.rank() can be adjusted automatically
        rank = compute_rank();

        vec_t sqrt_d = d.head(rank).cwiseSqrt();

        x.set_UV(
                (U.leftCols(rank) * sqrt_d.asDiagonal()).transpose(),
                (V.leftCols(rank) * sqrt_d.asDiagonal()).transpose()
                );

    }

    void ShrinkageNuclear::operator()(const var_t& xp, Scalar tau, const var_t& gp,
            Scalar v, var_t& x){
        using namespace Utils;
        const mat_wrapper_t* xp_ptr = dynamic_cast<const mat_wrapper_t*>(&xp);
        const fmat_t* xp_ptr_f = dynamic_cast<const fmat_t*>(&xp);
        const mat_wrapper_t* gp_ptr = dynamic_cast<const mat_wrapper_t*>(&gp);
        const spmat_wrapper_t* gp_ptr_s = dynamic_cast<const spmat_wrapper_t*>(&gp);
              mat_wrapper_t*  x_ptr = dynamic_cast<mat_wrapper_t*>(&x);
              fmat_t* x_ptr_f = dynamic_cast<fmat_t*>(&x);


        // only the following combination is supported:
        // 1) dense + dense
        // 2) dense + sparse
        // 3) factor + sparse
        // 4) factor + dense
        bool is_dense_x = xp_ptr && x_ptr;
        bool is_factor_x = xp_ptr_f && x_ptr_f;
        bool is_dense_g = gp_ptr != NULL;
        bool is_sparse_g = gp_ptr_s != NULL;

        if (is_dense_x && (is_dense_g || is_sparse_g)) {
            // x_tmp is created on every call of operator()
            mat_wrapper_t x_tmp;
            x_tmp.set_zero_like(*xp_ptr);
            if (is_dense_g){
                x_tmp.mat() = xp_ptr->mat() - tau * gp_ptr->mat();
            } else {
                x_tmp.mat() = xp_ptr->mat();
                x_tmp.mat() -= tau * gp_ptr_s->spmat();
            }
            (*this)(x_tmp.mat(), v, x_ptr->mat());
        }
        else if (is_factor_x && is_dense_g){
            (*this)(*xp_ptr_f, tau, *gp_ptr, v, *x_ptr_f);
        }
        else if (is_factor_x && is_sparse_g){
            (*this)(*xp_ptr_f, tau, *gp_ptr_s, v, *x_ptr_f);
        }
    }

    Scalar ShrinkageNuclear::cached_objective() const {
        if (rank == 0)
            return 0_s;
        if (options_.get_bool("truncation"))
            return mu * d.array().block(0, 0, rank, 1).sum();
        else
            return mu * d.array().sum();
    }

}}

#endif
