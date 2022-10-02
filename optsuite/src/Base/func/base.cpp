/*
 * ===========================================================================
 *
 *       Filename:  base.cpp
 *
 *    Description:
 *
 *        Version:  1.0
 *        Created:  01/03/2021 03:45:08 PM
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  Haoyang Liu (), liuhaoyang@pku.edu.cn
 *   Organization:  BICMR, Peking University
 *      Copyright:  Copyright (c) 2021, Haoyang Liu
 *
 * ===========================================================================
 */

#include "OptSuite/Base/func/base.h"
#include "OptSuite/Base/mat_wrapper.h"
#include "OptSuite/Base/mat_array.h"
#include "OptSuite/Base/spmat_wrapper.h"
#include "OptSuite/Base/variable.h"

namespace OptSuite { namespace Base {
    void Functional::bind(const std::shared_ptr<Structure>& p){
        workspace = p;
    }

    void Functional::unbind(){
        workspace = nullptr;
    }

    template<typename dtype>
    Scalar Func<dtype>::operator()(const Ref<const mat_t>){
        OPTSUITE_ASSERT(0);
        return 0;
    }

    template<typename dtype>
    Scalar Func<dtype>::operator()(const mat_wrapper_t& mw){
        return (*this)(mw.mat());
    }

    template<typename dtype>
    Scalar Func<dtype>::operator()(const mat_array_t& ma){
        Scalar r = 0;
        for (const auto& i : ma)
            r += (*this)(i);
        return r;
    }

    template<typename dtype>
    Scalar Func<dtype>::operator()(const var_t& var){
        const mat_wrapper_t* var_ptr = dynamic_cast<const mat_wrapper_t*>(&var);
        OPTSUITE_ASSERT(var_ptr);
        return (*this)(*var_ptr);
    }

    template<typename dtype>
    void Proximal<dtype>::operator()(const Ref<const mat_t>, Scalar, Ref<mat_t>){
        OPTSUITE_ASSERT(0);
    }

    template<typename dtype>
    void Proximal<dtype>::operator()(const mat_wrapper_t& mw, Scalar v, mat_wrapper_t& mw_out){
        (*this)(mw.mat(), v, mw_out.mat());
    }

    template<typename dtype>
    void Proximal<dtype>::operator()(const mat_array_t& ma, Scalar v, mat_array_t& ma_out){
        for (int i = 0; i < ma.total_blocks(); ++i)
            (*this)(ma[i], v, ma_out[i]);
    }

    template<typename dtype>
    void Proximal<dtype>::operator()(const var_t& xp, Scalar tau, const var_t& gp,
                                     Scalar v, var_t& x){
        const mat_wrapper_t* xp_ptr = dynamic_cast<const mat_wrapper_t*>(&xp);
        const mat_wrapper_t* gp_ptr = dynamic_cast<const mat_wrapper_t*>(&gp);
              mat_wrapper_t*  x_ptr = dynamic_cast<mat_wrapper_t*>(&x);

        OPTSUITE_ASSERT(xp_ptr != NULL && gp_ptr != NULL && x_ptr != NULL);
        // x_tmp is created on every call of operator()
        mat_wrapper_t x_tmp;
        x_tmp.set_zero_like(*xp_ptr);
        x_tmp.mat() = xp_ptr->mat() - tau * gp_ptr->mat();
        (*this)(x_tmp, v, *x_ptr);
    }


    template<typename dtype>
    Scalar FuncGrad<dtype>::eval(const Ref<const mat_t> x){
        mat_t dummy_y;
        return this->operator()(x, dummy_y, false);
    }

    template<typename dtype>
    Scalar FuncGrad<dtype>::eval(const Ref<const mat_t>, Ref<mat_t>, bool, bool){
        OPTSUITE_ASSERT(0);
        return 0;
    }

    template<typename dtype>
    Scalar FuncGrad<dtype>::eval(const mat_wrapper_t& x){
        mat_wrapper_t dummy_y;
        return this->operator()(x, dummy_y, false);
    }

    template<typename dtype>
    Scalar FuncGrad<dtype>::eval(const mat_wrapper_t& x, mat_wrapper_t& y, bool compute_grad, bool cached_grad){
        return this->operator()(x.mat(), y.mat(), compute_grad, cached_grad);
    }



    template<typename dtype>
    Scalar FuncGrad<dtype>::eval(const mat_array_t& ma){
        Scalar r = 0;
        for(const auto& i : ma)
            r += (*this)(i);
        return r;
    }

    template<typename dtype>
    Scalar FuncGrad<dtype>::eval(const mat_array_t& x, mat_array_t& y, bool compute_grad, bool cached_grad){
        Scalar r = 0;
        for(int i = 0; i < x.total_blocks(); ++i)
            r += (*this)(x[i], y[i], compute_grad, cached_grad);
        return r;
    }

    template<typename dtype>
    Scalar FuncGrad<dtype>::eval(const var_t& x){
        mat_wrapper_t dummy_y;
        return (*this)(x, dummy_y, false);
    }

    template<typename dtype>
    Scalar FuncGrad<dtype>::eval(const var_t& x, var_t& y, bool compute_grad, bool cached_grad){
        const mat_wrapper_t* x_ptr = dynamic_cast<const mat_wrapper_t*>(&x);
              mat_wrapper_t* y_ptr = dynamic_cast<mat_wrapper_t*>(&y);

        OPTSUITE_ASSERT(x_ptr != NULL && y_ptr != NULL);
        return (*this)(*x_ptr, *y_ptr, compute_grad, cached_grad);
    }

    // template instantiation
    template class Proximal<Scalar>;
    template class Func<Scalar>;
    template class FuncGrad<Scalar>;
    template class Proximal<ComplexScalar>;
    template class Func<ComplexScalar>;
    template class FuncGrad<ComplexScalar>;

}}
