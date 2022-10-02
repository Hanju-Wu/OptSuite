/*
 * ===========================================================================
 *
 *       Filename:  simple.h
 *
 *    Description:  
 *
 *        Version:  1.0
 *        Created:  01/03/2021 04:01:39 PM
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  Haoyang Liu (), liuhaoyang@pku.edu.cn
 *   Organization:  BICMR, Peking University
 *
 * ===========================================================================
 */

#ifndef OPTSUITE_BASE_FUNC_SIMPLE_H
#define OPTSUITE_BASE_FUNC_SIMPLE_H

#include "OptSuite/Base/func/base.h"

namespace OptSuite { namespace Base {
    template<typename dtype = Scalar>
    class Zero : public Func<dtype> {
        using typename Func<dtype>::mat_t;
        public:
            using Func<dtype>::operator();
            Scalar operator()(const Ref<const mat_t>){
                return 0_s;
            }
    };

    template<typename dtype>
    class IdentityProx : public Proximal<dtype> {
        using typename Proximal<dtype>::mat_t;
        using typename Proximal<dtype>::mat_array_t;
        using typename Proximal<dtype>::mat_wrapper_t;
        using typename Proximal<dtype>::var_t;
        public:
            IdentityProx() = default;
            ~IdentityProx() = default;

            using Proximal<dtype>::operator();
            void operator()(const Ref<const mat_t> x, dtype, Ref<mat_t> y){
                // simple copy
                y = x;
            }

            void operator()(const var_t& xp, Scalar tau, const var_t& gp,
                            Scalar, var_t& x){
                const mat_wrapper_t* xp_ptr = dynamic_cast<const mat_wrapper_t*>(&xp);
                const mat_wrapper_t* gp_ptr = dynamic_cast<const mat_wrapper_t*>(&gp);
                      mat_wrapper_t*  x_ptr = dynamic_cast<mat_wrapper_t*>(&x);

                OPTSUITE_ASSERT(xp_ptr || gp_ptr || x_ptr);
                x_ptr->mat() = xp_ptr->mat() - tau * gp_ptr->mat();
            }
            inline bool is_identity() const override { return true; }
    };

}}
#endif

