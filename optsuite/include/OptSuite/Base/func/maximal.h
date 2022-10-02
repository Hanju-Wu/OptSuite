/*
 * @Author: Wu Hanju 
 * @Date: 2022-09-03 18:18:15 
 * @Last Modified by: Wu Hanju
 * @Last Modified time: 2022-09-03 18:26:31
 */
#ifndef OPTSUITE_BASE_FUNC_MAXIMAL_H
#define OPTSUITE_BASE_FUNC_MAXIMAL_H

#include "OptSuite/Base/func/base.h"
#include <numeric>
#include <vector>

namespace OptSuite { namespace Base {
    template<typename dtype>
    class Maximal : public Proximal<dtype> {
        using typename Proximal<dtype>::mat_t;

        public:
            explicit Maximal(Scalar mu_ = 1) : mu(mu_) {}
            ~Maximal() = default;

            using Proximal<Scalar>::operator();
            void operator()(const Ref<const mat_t>, Scalar, Ref<mat_t>) override;

            Scalar mu;
    };
}}

#endif