/*
 * ==========================================================================
 *
 *       Filename:  spmat_wrapper.h
 *
 *    Description:
 *
 *        Version:  1.0
 *        Created:  12/10/2020 11:14:13 PM
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  Haoyang Liu (@liuhy), liuhaoyang@pku.edu.cn
 *   Organization:  BICMR, PKU
 *
 * ==========================================================================
 */

#ifndef OPTSUITE_BASE_SPMAT_WRAPPER_H
#define OPTSUITE_BASE_SPMAT_WRAPPER_H

#include "OptSuite/core_n.h"
#include "OptSuite/Base/variable.h"
#include "OptSuite/Base/object_wrapper.h"

namespace OptSuite { namespace Base {
    template<typename dtype>
    class SpMatWrapper : public ObjectWrapper<Eigen::SparseMatrix<dtype, ColMajor, SparseIndex>, dtype> {
        protected:
            using BaseClass = ObjectWrapper<Eigen::SparseMatrix<dtype, ColMajor, SparseIndex>, dtype>;
            using typename BaseClass::VarPtr;
            using typename Variable<dtype>::mat_t;
            using typename Variable<dtype>::spmat_t;
            using BaseClass::data_;
        public:
            using BaseClass::BaseClass;
            SpMatWrapper() = default;
            ~SpMatWrapper() = default;
            SpMatWrapper(const SpMatWrapper&) = default;
            SpMatWrapper(SpMatWrapper&&) = default;
            SpMatWrapper& operator=(const SpMatWrapper&) = default;
            SpMatWrapper& operator=(SpMatWrapper&&) = default;

            inline spmat_t& spmat() { return data_; }
            inline const spmat_t& spmat() const { return data_; }
            inline Index rows() const { return data_.rows(); }
            inline Index cols() const { return data_.cols(); }
            inline Index size() const override { return data_.size(); }

            inline dtype accept_op_dot(const Variable<dtype>& lhs) const override {
                return lhs.dot(data_);
            }

            inline dtype dot(const Ref<const mat_t>& rhs) const override {
                return (data_.conjugate().cwiseProduct(rhs)).sum();
            }

            inline dtype dot(const Ref<const spmat_t>& rhs) const override {
                return (data_.conjugate().cwiseProduct(rhs)).sum();
            }

            inline void set_zero_like(const Ref<const spmat_t> other){
                data_ = other;
                data_.setZero();
            }

            inline void set_zero_like(const Variable<dtype>& other) override {
                const SpMatWrapper* other_ptr = dynamic_cast<const SpMatWrapper*>(&other);
                OPTSUITE_ASSERT(other_ptr);
                data_ = other_ptr->data_;
                data_.setZero();
            }

            inline void set_zero_like(Index rows, Index cols, Index nnz,
                    const SparseIndex* outer, const SparseIndex* inner){
                std::vector<dtype> buff(nnz, static_cast<dtype>(0));
                data_ = Map<const spmat_t>(rows, cols, nnz, outer, inner, buff.data());
            }

            std::vector<Index> shape() const override {
                return std::vector<Index> { data_.rows(), data_.cols() };
            }

            Scalar squared_norm() const override {
                return data_.squaredNorm();
            }

            inline void multiply(const Ref<const mat_t> x, Ref<mat_t> y, MulOp op = MulOp::NonTranspose) const {
                if (op == MulOp::NonTranspose)
                    y = data_ * x;
                else
                    y = x.transpose() * data_;
            }

            inline void multiplyT(const Ref<const mat_t> x, Ref<mat_t> y, MulOp op = MulOp::NonTranspose) const {
                if (op == MulOp::NonTranspose)
                    y = data_.transpose() * x;
                else
                    y = x * data_;
            }

            using BaseClass::value_span;
            inline
            const dtype* value_span(Index& size_out) const override {
                size_out = data_.nonZeros();
                return data_.valuePtr();
            }

            using BaseClass::slice;
            inline dtype& slice(const Index& i) override {
                OPTSUITE_ASSERT(i >= 0 && i < data_.nonZeros());
                return data_.valuePtr()[i];
            }

            inline VarPtr clone() const override {
                return VarPtr { new SpMatWrapper<dtype>(*this) };
            }

            inline VarPtr accept_op_add(const Variable<dtype>& lhs) const override {
                return lhs.eval_op_add(data_);
            }

            VarPtr eval_op_add(const Ref<const mat_t>&) const override;

            inline VarPtr eval_op_add(const Ref<const spmat_t>& rhs) const override {
                return VarPtr { new SpMatWrapper<dtype> { data_ + rhs } };
            }

            inline VarPtr accept_op_sub(const Variable<dtype>& lhs) const override {
                return lhs.eval_op_sub(data_);
            }

            VarPtr eval_op_sub(const Ref<const mat_t>& rhs) const override;

            inline VarPtr eval_op_sub(const Ref<const spmat_t>& rhs) const override {
                return VarPtr { new SpMatWrapper<dtype> { data_ - rhs } };
            }

            inline VarPtr accept_op_mul(const Variable<dtype>& lhs) const override {
                return lhs.eval_op_mul(data_);
            }

            VarPtr eval_op_mul(const Ref<const mat_t>& rhs) const override;

            inline VarPtr eval_op_mul(const Ref<const spmat_t>& rhs) const override {
                return VarPtr { new SpMatWrapper<dtype> { data_ * rhs } };
            }

            inline VarPtr eval_op_mul(const dtype& rhs) const override {
                return VarPtr { new SpMatWrapper<dtype> { data_ * rhs } };
            }

            inline VarPtr accept_op_div(const Variable<dtype>& lhs) const override {
                return lhs.eval_op_div(data_);
            }

            inline VarPtr eval_op_div(const dtype& rhs) const override {
                return VarPtr { new SpMatWrapper<dtype> { data_ / rhs } };
            }

            inline VarPtr accept_op_tmul(const Variable<dtype>& lhs) const override {
                return lhs.eval_op_tmul(data_);
            }


            inline VarPtr eval_op_tmul(const Ref<const spmat_t>& rhs) const override {
                return VarPtr { new SpMatWrapper<dtype> { data_.adjoint() * rhs } };
            }

            inline VarPtr accept_op_mult(const Variable<dtype>& lhs) const override {
                return lhs.eval_op_mult(data_);
            }

            inline VarPtr eval_op_mult(const Ref<const spmat_t>& rhs) const override {
                return VarPtr { new SpMatWrapper<dtype> { data_ * rhs.adjoint() } };
            }

            inline VarPtr accept_op_tmult(const Variable<dtype>& lhs) const override {
                return lhs.eval_op_tmult(data_);
            }

            inline VarPtr eval_op_tmult(const Ref<const spmat_t>& rhs) const override {
                return VarPtr { new SpMatWrapper<dtype> { data_.adjoint() * rhs.adjoint() } };
            }

            VarPtr eval_op_tmul(const Ref<const mat_t>& rhs) const override;
            VarPtr eval_op_mult(const Ref<const mat_t>& rhs) const override;
            VarPtr eval_op_tmult(const Ref<const mat_t>& rhs) const override;

            inline VarPtr eval_op_adjoint() const override {
                return VarPtr { new SpMatWrapper<dtype> { data_.adjoint() } };
            }

            inline VarPtr eval_op_neg() const override {
                return VarPtr { new SpMatWrapper<dtype>(-data_) };
            }
    };
}}

#endif
