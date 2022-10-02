/*
 * ===========================================================================
 *
 *       Filename:  axmb_norm_sqr.cpp
 *
 *    Description:
 *
 *        Version:  1.0
 *        Created:  01/03/2021 03:51:18 PM
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  Haoyang Liu (), liuhaoyang@pku.edu.cn
 *   Organization:  BICMR, Peking University
 *      Copyright:  Copyright (c) 2021, Haoyang Liu
 *
 * ===========================================================================
 */

#include "OptSuite/Base/func/axmb_norm_sqr.h"
#include "OptSuite/Base/mat_op.h"
#include "OptSuite/Base/structure.h"

namespace OptSuite { namespace Base {
    template<typename dtype>
    AxmbNormSqr<dtype>::AxmbNormSqr(const Ref<const mat_t> A, const Ref<const mat_t> b){
        OPTSUITE_ASSERT(A.rows() == b.rows());
        this->A = A;
        this->b = b;
        this->Aop = nullptr;
        this->type = AType::Dense;
        r.resize(b.rows(), b.cols());
    }

    template<typename dtype>
    AxmbNormSqr<dtype>::AxmbNormSqr(const Ref<const spmat_t> A, const Ref<const mat_t> b){
        OPTSUITE_ASSERT(A.rows() == b.rows());
        this->spA = A;
        this->b = b;
        this->Aop = nullptr;
        this->type = AType::Sparse;
        r.resize(b.rows(), b.cols());
    }

    template<typename dtype>
    AxmbNormSqr<dtype>::AxmbNormSqr(const MatOp<dtype>& Aop, const Ref<const mat_t> b){
        OPTSUITE_ASSERT(Aop.rows() == b.rows());
        this->Aop = &Aop;
        this->b = b;
        this->type = AType::Operator;
        r.resize(b.rows(), b.cols());
    }

    template<typename dtype>
    Scalar AxmbNormSqr<dtype>::eval(const Ref<const mat_t> x, Ref<mat_t> y,
            bool compute_grad, bool cached_grad){
        if (!cached_grad){
            spmat_t* spx_ptr = nullptr;
            switch (type){
                case AType::Dense:
                    if (workspace) {
                        spx_ptr = workspace->template find<spmat_t>("spx");
                    }
                    if (spx_ptr) {
                        r = A * (*spx_ptr) - b;
                    } else {
                        r = A * x - b;
                    }
                    break;
                case AType::Sparse:
                    if (workspace) {
                        spx_ptr = workspace->template find<spmat_t>("spx");
                    }
                    if (spx_ptr) {
                        // SURE to perform a SSMULT?
                        r = spA * (*spx_ptr) - b;
                    } else {
                        r = spA * x - b;
                    }
                    break;
                case AType::Operator:
                    Aop->apply(x, r);
                    r = r - b;
                    break;
            }
            fun = 0.5 * r.squaredNorm();
        }

        if (compute_grad){
            switch (type){
                case AType::Dense:
                    y = A.transpose() * r;
                    break;
                case AType::Sparse:
                    y = spA.transpose() * r;
                    break;
                case AType::Operator:
                    Aop->apply_transpose(r, y);
                    break;
            }
        }

        return fun;
    }

    template<typename dtype>
    const typename AxmbNormSqr<dtype>::mat_t& AxmbNormSqr<dtype>::get_A() const {
        return A;
    }

    template<typename dtype>
    const typename AxmbNormSqr<dtype>::spmat_t& AxmbNormSqr<dtype>::get_spA() const {
        return spA;
    }

    template<typename dtype>
    const MatOp<dtype>& AxmbNormSqr<dtype>::get_Aop() const {
        return *this->Aop;
    }

    template<typename dtype>
    const typename AxmbNormSqr<dtype>::mat_t& AxmbNormSqr<dtype>::get_b() const {
        return b;
    }

    template<typename dtype>
    Index AxmbNormSqr<dtype>::rows() const {
        return b.rows();
    }

    template<typename dtype>
    Index AxmbNormSqr<dtype>::cols() const {
        switch (type){
            case AType::Dense:
                return A.cols();
            case AType::Sparse:
                return spA.cols();
            case AType::Operator:
                return Aop->cols();
            default:
                return 0_i;
        }
    }

    // instantiate
    template class AxmbNormSqr<Scalar>;
    template class AxmbNormSqr<ComplexScalar>;

}}
