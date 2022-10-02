/*
 * ===========================================================================
 *
 *       Filename:  shrinkage_lp.cpp
 *
 *    Description:
 *
 *        Version:  1.0
 *        Created:  01/03/2021 04:11:13 PM
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  Haoyang Liu (), liuhaoyang@pku.edu.cn
 *   Organization:  BICMR, Peking University
 *      Copyright:  Copyright (c) 2021, Haoyang Liu
 *
 * ===========================================================================
 */

#include "OptSuite/Base/func/shrinkage_lp.h"
#include "OptSuite/Base/func/ball_lp.h"
#include "OptSuite/Base/mat_wrapper.h"
#include "OptSuite/Base/spmat_wrapper.h"
#include "OptSuite/Base/structure.h"

namespace OptSuite { namespace Base {
    void ShrinkageL1::operator()(const Ref<const mat_t> x, Scalar t, Ref<mat_t> y){
        if (weights_.rows() == 0) // unweighted
            d.array() = (x.array().abs() - t * mu).max(0);
        else // weighted
            d.array() = (x.array().abs() - t * mu * weights_.array()).max(0);

        y.array() = x.array().sign() * d.array();

        // perform truncation
        if (options_.get_bool("truncation")){
            do_truncation(y);
        } else { // no truncation - report nnz_ only
            ind.clear();
            ind.reserve(d.rows());
            for (Index i = 0; i < d.rows(); ++i){
                if (d(i, 0) > OptSuite::eps) ind.push_back(i);
            }
            nnz_ = static_cast<Index>(ind.size());
            nnz_t_ = nnz_;
        }

        // write nnz_ into workspace
        if (workspace != nullptr){
            workspace->set("nnz", nnz_);
            workspace->set("nnz_p", x.rows());
        }

        // optionally write a sparse x into workspace
        if (workspace != nullptr){
            if (options_.get_bool("spx") && nnz_t_ < 0.05_s * x.rows()){
                gen_spmat(y);
            } else { // erase the object
                workspace->erase("spx");
            }
        }
    }

    void ShrinkageL1::do_truncation(Ref<mat_t> y){
        ind.clear();
        ind.reserve(y.rows());

        // first, only select i such that d(i) > eps (d = y.abs())
        for (Index i = 0; i < d.rows(); ++i){
            if (d(i, 0) > OptSuite::eps) ind.push_back(i);
        }
        nnz_ = ind.size();

        // sort d(ind)
        auto comp = [&](Index a, Index b){ return d(a, 0) < d(b, 0); };
        std::sort(ind.begin(), ind.end(), comp);

        // drop y(j) if y(0) + ... + y(j) < (1 - r)|y|_1
        Scalar l1_norm = d.sum();
        Scalar r = 1 - options_.get_scalar("ratio");

        size_t j;
        Scalar sum = 0;
        for (j = 0; j < ind.size(); ++j){
            sum += d(ind[j], 0);
            if (sum > r * l1_norm)
                break;
            // truncation
            y(ind[j], 0) = 0_s;
        }
        nnz_t_ = nnz_ - static_cast<Index>(j);
    }

    void ShrinkageL1::gen_spmat(const Ref<const mat_t> y){
        // ind contains indices of non-zeros (with or without truncation)
        // create a sparse x by triplets
        std::vector<Triplet<Scalar, Index>> triplets;
        triplets.reserve(nnz_t_);

        for (size_t j = nnz_ - nnz_t_; j < ind.size(); ++j){
            triplets.emplace_back(ind[j], 0, y(ind[j], 0));
        }

        std::shared_ptr<SpMatWrapper<Scalar>> spx = std::make_shared<SpMatWrapper<Scalar>>();
        spx->spmat().resize(y.rows(), 1);
        spx->spmat().setFromTriplets(triplets.begin(), triplets.end());

        workspace->set("spx", spx);
    }

    Utils::OptionList& ShrinkageL1::options(){
        return options_;
    }
    const Utils::OptionList& ShrinkageL1::options() const {
        return options_;
    }

    ShrinkageL1::mat_t& ShrinkageL1::weights(){
        return weights_;
    }

    const ShrinkageL1::mat_t& ShrinkageL1::weights() const {
        return weights_;
    }

    void ShrinkageL1::register_options(){
        std::vector<Utils::RegOption> v;

        v.emplace_back("truncation", false, "Truncation switch");
        v.emplace_back("ratio", 0.9999_s, "Minimum l1-norm ratio after truncation");
        v.emplace_back("spx", false, "Write a sparse X into workspace.");

        // v is constructed, now initialize options_
        this->options_ = Utils::OptionList(v, nullptr);

    }

    Index ShrinkageL1::nnz() const {
        return nnz_;
    }

    Index ShrinkageL1::nnz_t() const {
        return nnz_t_;
    }

    void ShrinkageL2::operator()(const Ref<const mat_t> x, Scalar t, Ref<mat_t> y){
        Scalar lambda = 1 - t * mu / x.norm();
        y.array() = x.array() * std::max(0_s, lambda);
    }

    void ShrinkageL2Rowwise::operator()(const Ref<const mat_t> x, Scalar t, Ref<mat_t> y){
        lambda = 1 - t * mu / x.rowwise().norm().array();
        y = x.array().colwise() * lambda.array().max(0);
    }

    void ShrinkageL2Colwise::operator()(const Ref<const mat_t> x, Scalar t, Ref<mat_t> y){
        lambda = 1 - t * mu / x.colwise().norm().array();
        y = x.array().rowwise() * lambda.transpose().array().max(0);
    }

    void ShrinkageL0::operator()(const Ref<const mat_t> x, Scalar t, Ref<mat_t> y) {
        y.array() = x.array() * (x.array().pow(2) > 2_s * t * mu).cast<Scalar>().array();
    }

    void ShrinkageLInf::operator()(const Ref<const mat_t> x, Scalar t, Ref<mat_t> y) {
        OPTSUITE_ASSERT(x.cols() == 1);
        
        std::vector<SparseIndex> indexes(x.rows());
        std::iota(indexes.begin(), indexes.end(), 0);
        mat_t z;
        z.array()=x.array().abs();
        std::sort(indexes.begin(), indexes.end(), [&z](SparseIndex a, SparseIndex b) { return std::fabs(z.coeff(a, 0)) > std::fabs(z.coeff(b, 0)); });
        bool  root_found = false;
        Scalar prefix_sum = 0;
        Scalar lambda = 0;
        for (SparseIndex i = 0; i < static_cast<SparseIndex>(indexes.size()); i++) {
            prefix_sum += z(indexes[i],0);
            lambda = (prefix_sum - t * mu) / (i + 1);
            if (lambda < z(indexes[i],0) && (i + 1 == static_cast<SparseIndex>(indexes.size()) || z(indexes[i+1],0) <= lambda)) {
                root_found = true;
                break;
            }
        }
        if(root_found==false)
        {
            OPTSUITE_ASSERT(0);
        }
        y = x.array().sign() * x.array().abs().min(lambda);
    }
}}
