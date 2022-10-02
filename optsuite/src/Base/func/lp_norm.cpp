/*
 * ===========================================================================
 *
 *       Filename:  lp_norm.cpp
 *
 *    Description:  
 *
 *        Version:  1.0
 *        Created:  01/03/2021 04:08:37 PM
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  Haoyang Liu (), liuhaoyang@pku.edu.cn
 *   Organization:  BICMR, Peking University
 *      Copyright:  Copyright (c) 2021, Haoyang Liu
 *
 * ===========================================================================
 */

#include "OptSuite/Base/func/lp_norm.h"

using Eigen::Infinity;

namespace OptSuite { namespace Base {
    Scalar L1Norm::operator()(const Ref<const mat_t> x){
        if (weights_.rows() == 0) // unweighted
            return mu * x.lpNorm<1>();
        else // weighted
            return mu * (weights_.array() * x.array()).matrix().lpNorm<1>();
    }

    L1Norm::mat_t& L1Norm::weights(){
        return weights_;
    }

    const L1Norm::mat_t& L1Norm::weights() const {
        return weights_;
    }

    Scalar L2Norm::operator()(const Ref<const mat_t> x){
        return mu * x.norm();
    }

    Scalar L1_2Norm::operator()(const Ref<const mat_t> x){
        return mu * x.rowwise().norm().sum();
    }

    Scalar L2_1Norm::operator()(const Ref<const mat_t> x){
        return mu * x.colwise().norm().sum();
    }

    Scalar L0Norm::operator()(const Ref<const mat_t> x){
        return mu * (x.array().abs() > OptSuite::eps).cast<Scalar>().matrix().lpNorm<1>();
    }

    Scalar LInfNorm::operator()(const Ref<const mat_t> x){
        return mu * x.lpNorm<Infinity>();
    }

}}

