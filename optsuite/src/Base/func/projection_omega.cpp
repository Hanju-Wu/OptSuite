/*
 * ===========================================================================
 *
 *       Filename:  projection_omega.cpp
 *
 *    Description:
 *
 *        Version:  1.0
 *        Created:  01/03/2021 03:54:51 PM
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  Haoyang Liu (), liuhaoyang@pku.edu.cn
 *   Organization:  BICMR, Peking University
 *      Copyright:  Copyright (c) 2021, Haoyang Liu
 *
 * ===========================================================================
 */

#include "OptSuite/Base/func/projection_omega.h"
#include "OptSuite/Base/mat_wrapper.h"
#include "OptSuite/Base/spmat_wrapper.h"
#include "OptSuite/Base/factorized_mat.h"
#include "OptSuite/LinAlg/lansvd.h"

namespace OptSuite { namespace Base {
    template<typename dtype>
    ProjectionOmega<dtype>::ProjectionOmega(const Ref<const spmat_t> ref){
        outerIndexPtr.resize(ref.cols() + 1_i);
        innerIndexPtr.resize(ref.nonZeros());

        std::memcpy(outerIndexPtr.data(), ref.outerIndexPtr(),
                sizeof(SparseIndex) * outerIndexPtr.size());
        std::memcpy(innerIndexPtr.data(), ref.innerIndexPtr(),
                sizeof(SparseIndex) * innerIndexPtr.size());

        b.resize(ref.nonZeros(), 1);
        std::memcpy(b.data(), ref.valuePtr(),
                sizeof(dtype) * innerIndexPtr.size());
    }

    template<typename dtype>
    Scalar ProjectionOmega<dtype>::compute_fg_impl(const Ref<const mat_t> x, MatWrapper<dtype>& y, bool compute_grad, bool cached_grad){
        if (cached_grad && !compute_grad)
            return fun;

        // initialization
        r.resize(b.rows(), 1);
        if (compute_grad && y.cols() == 0)
            y.mat().resize(x.rows(), x.cols());
        bool x_zero = x.rows() == 0;

        // Gradient is dense, better to form r, y in one-pass
        dtype *y_ptr = y.mat().data();
        dtype *r_ptr = r.data(), *r_ptr_cpy = r.data(), *b_ptr = b.data();
        Index ldy = y.mat().outerStride();
        SparseIndex *outer_ptr = outerIndexPtr.data();
        SparseIndex *inner_ptr = innerIndexPtr.data();

        for (size_t i = 0; i < outerIndexPtr.size() - 1; ++i){
            for (Index j = outer_ptr[i]; j < outer_ptr[i+1]; ++j){
                // for (inner_ptr[j], i)
                if (!cached_grad){
                    if (x_zero)
                        *r_ptr++ = 0_s - *b_ptr++;
                    else
                        *r_ptr++ = x(inner_ptr[j], i) - *b_ptr++;
                }
                if (compute_grad){
                    *(y_ptr + inner_ptr[j]) = *r_ptr_cpy++;
                }
            }
            y_ptr += ldy;
        }
        if (!cached_grad)
            fun = 0.5_s * r.squaredNorm();
        return fun;
    }

    template<typename dtype>
    Scalar ProjectionOmega<dtype>::eval(const var_t& x, var_t& y, bool compute_grad, bool cached_grad){
        const MatWrapper<dtype>* x_ptr = dynamic_cast<const MatWrapper<dtype>*>(&x);
        const fmat_t* x_ptr_f = dynamic_cast<const fmat_t*>(&x);
        MatWrapper<dtype>* y_ptr = dynamic_cast<MatWrapper<dtype>*>(&y);
        SpMatWrapper<dtype>* y_ptr_s = dynamic_cast<SpMatWrapper<dtype>*>(&y);
        Index m, n;

        if (x_ptr && y_ptr) // dense x + dense y
            return compute_fg_impl(x_ptr->mat(), *y_ptr, compute_grad, cached_grad);
        if (x_ptr_f && y_ptr){ // factorized x + dense y
            // y is dense anyway, use the full format of x_ptr_f
            if (!cached_grad){
                if (x_ptr_f->rank() > 0)
                    r_mat.noalias() = (x_ptr_f->U().transpose() * x_ptr_f->V()).eval();
                else // resize r_mat to (0, 0) to indicate x is a zero matrix
                    r_mat.resize(0, 0);
            }
            return compute_fg_impl(r_mat, *y_ptr, compute_grad, cached_grad);
        }
        else if ((x_ptr_f || x_ptr) && (!compute_grad || y_ptr_s)) { // dense/factorized x + sparse y
            // compute r = P(x)
            if (x_ptr){ // dense
                m = x_ptr->mat().rows();
                n = x_ptr->mat().cols();
                if (!cached_grad)
                    projection(*x_ptr);
            } else { // factor
                m = x_ptr_f->rows();
                n = x_ptr_f->cols();
                if (!cached_grad)
                    projection(*x_ptr_f);
            }

            if (!cached_grad){
                r -= b;
                fun = 0.5_s * r.squaredNorm();
            }

            // compute gradient
            // note: the gradient is sparse
            if (compute_grad){
                // if y_ptr_s isn't initialized, set zeros as default
                if (y_ptr_s->spmat().nonZeros() != b.rows())
                    y_ptr_s->set_zero_like(m, n, r.rows(),
                            outerIndexPtr.data(), innerIndexPtr.data());

                std::memcpy(y_ptr_s->spmat().valuePtr(), r.data(), r.rows() * sizeof(dtype));
            }
            return fun;
        } else {
            OPTSUITE_ASSERT(0);
            return 0;
        }
    }

    template<typename dtype>
    const typename ProjectionOmega<dtype>::mat_t& ProjectionOmega<dtype>::rvec() const {
        return r;
    }

    template<typename dtype>
    template<typename xtype>
    void ProjectionOmega<dtype>::projection(const xtype& x){
        r.resize(b.rows(), 1);
        SparseIndex *outer_ptr = outerIndexPtr.data();
        SparseIndex *inner_ptr = innerIndexPtr.data();
        Scalar *r_ptr = r.data();

        for (size_t i = 0; i < outerIndexPtr.size() - 1; ++i){
            for (Index j = outer_ptr[i]; j < outer_ptr[i+1]; ++j){
                *r_ptr++ = get_xij(x, inner_ptr[j], i); // for colmajor
            }
        }
    }

    // instantiate
    template class ProjectionOmega<Scalar>;

}}
