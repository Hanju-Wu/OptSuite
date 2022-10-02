/*
 * @Author: Wu Hanju 
 * @Date: 2022-09-03 09:38:24 
 * @Last Modified by: Wu Hanju
 * @Last Modified time: 2022-09-03 10:01:45
 */

#include "OptSuite/Base/func/probability_simplex.h"

namespace OptSuite { namespace Base {
    template<typename dtype>
    void Simplex<dtype>::operator()(const Ref<const mat_t> x, Scalar t, Ref<mat_t> y) {
        OPTSUITE_ASSERT(x.cols() == 1);
        std::vector<SparseIndex> indexes(x.rows());
        std::iota(indexes.begin(), indexes.end(), 0);
        std::sort(indexes.begin(), indexes.end(), [&x](SparseIndex a, SparseIndex b) { return x.coeff(a, 0) > x.coeff(b, 0); });
        bool  root_found = false;
        dtype prefix_sum = 0;
        dtype lambda = 0;
        for (SparseIndex i = 0; i < static_cast<SparseIndex>(indexes.size()); i++) {
            prefix_sum += x(indexes[i],0);
            lambda = (prefix_sum - mu) / (i + 1);
            if (lambda < x(indexes[i],0) && (i + 1 == static_cast<SparseIndex>(indexes.size()) || x(indexes[i+1],0) <= lambda)) {
                root_found = true;
                break;
            }
        }
        if(root_found==false)
        {
            OPTSUITE_ASSERT(0);
        }
        y = (x.array() - lambda).max(0_s);
        }

    template class Simplex<Scalar>;
}}