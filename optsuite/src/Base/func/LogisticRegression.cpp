/*
 * @Author: Wu Hanju 
 * @Date: 2022-09-02 22:32:32 
 * @Last Modified by: Wu Hanju
 * @Last Modified time: 2022-09-03 20:25:59
 */

#include "OptSuite/Base/func/LogisticRegression.h"
#include "OptSuite/Base/structure.h"

namespace OptSuite { namespace Base {
    template<typename dtype>
    LogisticRegression<dtype>::LogisticRegression(const Ref<const mat_t> A_,const Ref<const mat_t> b_)
    {
        OPTSUITE_ASSERT(b_.cols() == 1);
        OPTSUITE_ASSERT(b_.rows() == A_.cols());
        A=A_;
        b=b_;
        mbA = A.array().rowwise() * (- b.transpose().array());
    }

    template<typename dtype>
    Scalar LogisticRegression<dtype>::set_fun(const Ref<const mat_t> x)
    {
        OPTSUITE_ASSERT(x.cols() == 1);
        OPTSUITE_ASSERT(x.rows() == A.rows());
        t = mbA.transpose() * x;
        q = (-t.array().sign() * t.array()).exp() + 1;
        r=(t.array().sign()+1)/2 * t.array() + q.array().log();
        return r.mean();
    }

    template<typename dtype>
    Scalar LogisticRegression<dtype>::set_fun(const Ref<const spmat_t> x)
    {
        OPTSUITE_ASSERT(x.cols() == 1);
        OPTSUITE_ASSERT(x.rows() == A.rows());
        t = mbA.transpose() * x;
        q = (-t.array().sign() * t.array()).exp() + 1;
        r=(t.array().sign()+1)/2 * t.array() + q.array().log();
        return r.mean();
    }

    template<typename dtype>
    Scalar LogisticRegression<dtype>::eval(const Ref<const mat_t> x, Ref<mat_t> y, bool compute_grad, bool cached_grad)
    {
        OPTSUITE_ASSERT(x.cols() == 1);
        OPTSUITE_ASSERT(x.rows() == A.rows());
        if (!cached_grad){
            spmat_t* spx_ptr = nullptr;
                if (workspace) {
                    spx_ptr = workspace->template find<spmat_t>("spx");
                }
                if (spx_ptr) {
                    fun = set_fun(*spx_ptr);
                } 
                else {
                    fun = set_fun(x);
                }
        }
        if (compute_grad) {
            vec_t s;
            s.resize(q.rows(),1);
            for(int i=0;i<s.rows();i++)
            {
                if(t(i,0)>=0)
                {
                    s(i,0)=1/q(i,0);
                }
                else
                {
                    s(i,0)=exp(t(i,0))/q(i,0);
                }
            }
            mat_t u = mbA.array().rowwise() * s.transpose().array();
            y = u.array().rowwise().mean();
        }
        return fun;
    }

    template<typename dtype>
    Index LogisticRegression<dtype>::rows() const {
        return b.rows();
    }

    template<typename dtype>
    Index LogisticRegression<dtype>::cols() const {
        return A.rows();
    }

    template class LogisticRegression<Scalar>;
}}