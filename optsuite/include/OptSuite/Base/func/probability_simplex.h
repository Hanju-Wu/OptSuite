/*
 * @Author: Wu Hanju 
 * @Date: 2022-09-03 09:33:46 
 * @Last Modified by: Wu Hanju
 * @Last Modified time: 2022-09-03 09:51:27
 */
#ifndef OPTSUITE_BASE_FUNC_PROBABILITY_SIMPLEX_H
#define OPTSUITE_BASE_FUNC_PROBABILITY_SIMPLEX_H

#include "OptSuite/Base/func/base.h"
#include <numeric>
#include <vector>

namespace OptSuite { namespace Base {
    template<typename dtype>
    class Simplex : public Proximal<dtype> {
        using typename Proximal<dtype>::mat_t;

        public:
            explicit Simplex(Scalar mu_ = 1) : mu(mu_) {}
            ~Simplex() = default;

            using Proximal<Scalar>::operator();
            void operator()(const Ref<const mat_t>, Scalar, Ref<mat_t>) override;

        private:
            Scalar mu;
    };
}}

#endif