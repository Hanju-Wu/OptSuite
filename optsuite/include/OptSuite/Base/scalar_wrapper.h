/*
 * ==========================================================================
 *
 *       Filename:  scalar_wrapper.h
 *
 *    Description:
 *
 *        Version:  1.0
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  Dong Xu (@taroxd), taroxd@pku.edu.cn
 *   Organization:  BICMR, PKU
 *
 * ==========================================================================
 */


#ifndef OPTSUITE_BASE_SCALAR_WRAPPER_H
#define OPTSUITE_BASE_SCALAR_WRAPPER_H

#include "OptSuite/core_n.h"
#include "OptSuite/Base/object_wrapper.h"

namespace OptSuite { namespace Base {
    // dtype can only be Scalar / ComplexScalar / bool here
    // ScalarWrapper<bool> is specialized
    template<typename dtype>
    class ScalarWrapper: public ObjectWrapper<dtype, dtype> {
        using BaseClass = ObjectWrapper<dtype, dtype>;
        using typename BaseClass::mat_t;
        using typename BaseClass::spmat_t;
        using BaseClass::BaseClass;
        using BaseClass::data_;
        public:
            using typename BaseClass::VarPtr;
            using typename BaseClass::BoolVarPtr;

            ScalarWrapper() = default;
            ScalarWrapper(const ScalarWrapper<dtype>&) = default;
            ScalarWrapper(ScalarWrapper<dtype>&&) = default;
            ~ScalarWrapper() = default;

            inline VarPtr clone() const override {
                return VarPtr { new ScalarWrapper<dtype>(*this) };
            }

            inline Scalar squared_norm() const override {
                using std::norm;
                return static_cast<Scalar>(norm(data_));
            }

            inline void set_zero_like(const Variable<dtype>& /*other*/) override {
                data_ = dtype{};
            }

            inline VarPtr eval_op_neg() const override {
                return VarPtr { new ScalarWrapper<dtype>(-data_) };
            }

            inline const dtype* value_span(Index& size_out) const override {
                size_out = 1_i;
                return &data_;
            }

            VarPtr accept_op_add(const Variable<dtype>& lhs) const override;
            VarPtr eval_op_add(const dtype& rhs) const override;
            VarPtr accept_op_sub(const Variable<dtype>& lhs) const override;
            VarPtr eval_op_sub(const dtype& rhs) const override;
            VarPtr accept_op_mul(const Variable<dtype>& lhs) const override;
            VarPtr eval_op_mul(const Ref<const mat_t>& rhs) const override;
            VarPtr eval_op_mul(const Ref<const spmat_t>& rhs) const override;
            VarPtr eval_op_mul(const dtype& rhs) const override;
            VarPtr accept_op_div(const Variable<dtype>& lhs) const override;
            VarPtr eval_op_div(const dtype& rhs) const override;

            BoolVarPtr eval_op_gt(const Variable<dtype>& rhs) const override;
            BoolVarPtr eval_op_gt(const dtype& rhs) const override;
            BoolVarPtr eval_op_ge(const Variable<dtype>& rhs) const override;
            BoolVarPtr eval_op_ge(const dtype& rhs) const override;
            BoolVarPtr eval_op_lt(const Variable<dtype>& rhs) const override;
            BoolVarPtr eval_op_lt(const dtype& rhs) const override;
            BoolVarPtr eval_op_le(const Variable<dtype>& rhs) const override;
            BoolVarPtr eval_op_le(const dtype& rhs) const override;
            BoolVarPtr eval_op_eq(const Variable<dtype>& rhs) const override;
            BoolVarPtr eval_op_eq(const dtype& rhs) const override;
            BoolVarPtr eval_op_ne(const Variable<dtype>& rhs) const override;
            BoolVarPtr eval_op_ne(const dtype& rhs) const override;
    };

    template <>
    class ScalarWrapper<bool>: public ObjectWrapper<bool, bool> {
        using dtype = bool;
        using typename Variable<dtype>::VarPtr;
        using BaseClass = ObjectWrapper<dtype, dtype>;
        using BaseClass::BaseClass;
        using BaseClass::data_;
        using typename BaseClass::vec_t;
        public:
            ScalarWrapper() = default;
            ScalarWrapper(const ScalarWrapper<dtype>&) = default;
            ScalarWrapper(ScalarWrapper<dtype>&&) = default;
            ~ScalarWrapper() = default;

            inline VarPtr clone() const override {
                return VarPtr { new ScalarWrapper<bool>(*this) };
            }

            inline Map<const vec_t> vector_view() const override {
                return Map<const vec_t>(&data_, 1);
            }
    };
}}

#endif
