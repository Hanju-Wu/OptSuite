/*
 * @Author: Wu Hanju 
 * @Date: 2022-09-02 22:32:05 
 * @Last Modified by:   Wu Hanju 
 * @Last Modified time: 2022-09-02 22:32:05 
 */

#ifndef OPTSUITE_BASE_FUNC_BALL_LP_H
#define OPTSUITE_BASE_FUNC_BALL_LP_H

#include "OptSuite/Base/func/base.h"
#include <numeric>
#include <vector>

namespace OptSuite { namespace Base {
    template<typename dtype>
    class L1NormBall : public Proximal<dtype> {
        using typename Proximal<dtype>::mat_t;

        public:
            explicit L1NormBall(Scalar mu_ = 1) : mu(mu_) {}
            ~L1NormBall() = default;

            using Proximal<Scalar>::operator();
            void operator()(const Ref<const mat_t>, Scalar, Ref<mat_t>) override;

            Scalar mu;
    };

    template<typename dtype>
    class L0NormBall : public Proximal<dtype> {
        using typename Proximal<dtype>::mat_t;

        public:
            explicit L0NormBall(Scalar mu_ = 1) : mu(mu_) {}
            ~L0NormBall() = default;

            using Proximal<Scalar>::operator();
            void operator()(const Ref<const mat_t>, Scalar, Ref<mat_t>) override;

            Scalar mu;
    };

    template<typename dtype>
    class L2NormBall : public Proximal<dtype> {
        using typename Proximal<dtype>::mat_t;

        public:
            explicit L2NormBall(Scalar mu_ = 1) : mu(mu_) {}
            ~L2NormBall() = default;

            using Proximal<Scalar>::operator();
            void operator()(const Ref<const mat_t>, Scalar, Ref<mat_t>) override;

            Scalar mu;
    };


    template<typename dtype>
    class LInfNormBall : public Proximal<dtype> {
        using typename Proximal<dtype>::mat_t;

        public:
            explicit LInfNormBall(Scalar mu_ = 1) : mu(mu_) {}
            ~LInfNormBall() = default;
            
            using Proximal<Scalar>::operator();
            void operator()(const Ref<const mat_t>, Scalar, Ref<mat_t>) override;

            Scalar mu;
    };

}}

#endif