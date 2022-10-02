/*
 * @Author: Wu Hanju 
 * @Date: 2022-09-02 22:31:40 
 * @Last Modified by: Wu Hanju
 * @Last Modified time: 2022-09-06 09:49:26
 */

#include "OptSuite/Base/func/ball_lp.h"

namespace OptSuite { namespace Base {
    template<typename dtype>
    void L1NormBall<dtype>::operator()(const Ref<const mat_t> x, Scalar t, Ref<mat_t> y) {
        OPTSUITE_ASSERT(x.cols() == 1);
        dtype l1_norm = x.template lpNorm<1>();
        if (l1_norm <= mu) {
            y = x;
            return;
        }
        std::vector<SparseIndex> indexes(x.rows());
        std::iota(indexes.begin(), indexes.end(), 0);
        mat_t z;
        z.array()=x.array().abs();
        std::sort(indexes.begin(), indexes.end(), [&z](SparseIndex a, SparseIndex b) { return std::fabs(z.coeff(a, 0)) > std::fabs(z.coeff(b, 0)); });
        bool  root_found = false;
        dtype prefix_sum = 0;
        dtype lambda = 0;
        for (SparseIndex i = 0; i < static_cast<SparseIndex>(indexes.size()); i++) {
            prefix_sum += z(indexes[i],0);
            lambda = (prefix_sum - mu) / (i + 1);
            if (lambda < z(indexes[i],0) && (i + 1 == static_cast<SparseIndex>(indexes.size()) || z(indexes[i+1],0) <= lambda)) {
                root_found = true;
                break;
            }
        }
        if(root_found==false)
        {
            OPTSUITE_ASSERT(0);
        }
        y = x.array().sign() * (x.array().abs() - lambda).max(0_s);
        }

    template class L1NormBall<Scalar>;


    template<typename dtype>
    void L0NormBall<dtype>::operator()(const Ref<const mat_t> x, Scalar t, Ref<mat_t> y) {
        OPTSUITE_ASSERT(x.cols() == 1);
        SparseIndex count = std::floor(mu);
        std::vector<SparseIndex> indexes(x.rows());
        std::iota(indexes.begin(), indexes.end(), 0);
        std::sort(indexes.begin(), indexes.end(), [&x](SparseIndex a, SparseIndex b) { return std::fabs(x.coeff(a, 0)) > std::fabs(x.coeff(b, 0)); });
        y = x;
        for (SparseIndex i = count; i < static_cast<SparseIndex>(indexes.size()); i++) { y.coeffRef(indexes[i],0) = 0; }
    }

    template class L0NormBall<Scalar>;
    
    template<typename dtype>
    void L2NormBall<dtype>::operator()(const Ref<const mat_t> x, Scalar t, Ref<mat_t> y) {
        y = x.array() * (mu / std::max(mu, x.norm()));
    }

    template class L2NormBall<Scalar>;


    template<typename dtype>
    void LInfNormBall<dtype>::operator()(const Ref<const mat_t> x, Scalar t, Ref<mat_t> y) {
        y = x.array().sign() * x.array().abs().min(mu);
    }

    template class LInfNormBall<Scalar>;

}}