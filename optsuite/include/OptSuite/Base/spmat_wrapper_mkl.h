/*
 * ===========================================================================
 *
 *       Filename:  spmat_wrapper_mkl.h
 *
 *    Description:  
 *
 *        Version:  1.0
 *        Created:  02/04/2021 07:03:22 PM
 *       Revision:  none
 *       Compiler:  g++
 *
 *         Author:  Haoyang Liu (), liuhaoyang@pku.edu.cn
 *   Organization:  BICMR, Peking University
 *
 * ===========================================================================
 */

#ifndef OPTSUITE_BASE_SPMAT_WRAPPER_MKL_H
#define OPTSUITE_BASE_SPMAT_WRAPPER_MKL_H

#include "OptSuite/Base/spmat_wrapper.h"
#include "OptSuite/LinAlg/mkl_sparse.h"

namespace OptSuite { namespace Base {
#ifdef OPTSUITE_USE_MKL
    template <typename dtype>
    class SpMatWrapper_mkl : public SpMatWrapper<dtype> {
        using spmat_t = Eigen::SparseMatrix<dtype, ColMajor, SparseIndex>;
        using mat_t = Eigen::Matrix<dtype, Dynamic, Dynamic>;
        using vec_t = Eigen::Matrix<dtype, Dynamic, 1>;
        using rowmat_t = Eigen::Matrix<dtype, Dynamic, Dynamic, RowMajor>;
        using SpMatWrapper<dtype>::data_;
        LinAlg::MKLSparse A_mkl;

        public:
            SpMatWrapper_mkl() = default;
            SpMatWrapper_mkl(const Ref<const spmat_t>A){
                data_ = A;
                A_mkl.reset(data_);
            }

            SpMatWrapper_mkl(const SpMatWrapper_mkl& other){
                data_ = other.data_;
                A_mkl.reset(data_);
            }

            void assign(const Variable<dtype>& other) override {
                const SpMatWrapper_mkl<dtype>* other_ptr = dynamic_cast<const SpMatWrapper_mkl<dtype>*>(&other);
                OPTSUITE_ASSERT(other_ptr);
                data_ = other_ptr->data_;
                A_mkl.reset(data_);
            }

            inline virtual void set_zero_like(const Ref<const spmat_t> other){
                data_ = other;
                data_.setZero();
                A_mkl.reset(data_);
            }

            inline virtual void set_zero_like(const Variable<dtype>& other) override {
                const SpMatWrapper_mkl<dtype>* other_ptr = dynamic_cast<const SpMatWrapper_mkl<dtype>*>(&other);
                OPTSUITE_ASSERT(other_ptr);
                data_ = other_ptr->data_;
                data_.setZero();
                A_mkl.reset(data_);
            }

            inline virtual void set_zero_like(Index rows, Index cols, Index nnz,
                    const SparseIndex* outer, const SparseIndex* inner){
                Scalar *buff = new Scalar[nnz];
                data_ = Map<const spmat_t>(rows, cols, nnz, outer, inner, buff);
                delete []buff;
                A_mkl.reset(data_);
            }

            inline virtual void multiply(const Ref<const mat_t> x, Ref<mat_t> y, MulOp op = MulOp::NonTranspose) const {
                A_mkl.multiply(x, y, op);
            }

            inline virtual void multiplyT(const Ref<const mat_t> x, Ref<mat_t> y, MulOp op = MulOp::NonTranspose) const {
                A_mkl.multiply<MulOp::Transpose>(x, y, op);
            }



    };
#endif // of #ifdef OPTSUITE_USE_MKL
}}

#endif

