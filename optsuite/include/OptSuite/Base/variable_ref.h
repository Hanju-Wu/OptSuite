/*
 * ==========================================================================
 *
 *       Filename:  variable_ref.h
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

#ifndef OPTSUITE_BASE_VARIABLE_REF_H
#define OPTSUITE_BASE_VARIABLE_REF_H

#include "OptSuite/core_n.h"
#include "OptSuite/Base/base.h"
#include "OptSuite/Base/variable.h"
#include "OptSuite/Base/scalar_wrapper.h"
#include "OptSuite/Base/mat_wrapper.h"
#include "OptSuite/Base/spmat_wrapper.h"

namespace OptSuite { namespace Base {
    template <typename VarType, typename Derived>
    class VariableRefBase : public OptSuiteBase {
        public:
            using var_t = VarType;

            VariableRefBase() = default;
            explicit VariableRefBase(var_t* var_ptr) : ptr_(var_ptr) {}
            explicit VariableRefBase(std::shared_ptr<var_t> var_ptr) : ptr_(std::move(var_ptr)) {}
            explicit VariableRefBase(std::unique_ptr<var_t>&& var_ptr) : ptr_(std::move(var_ptr)) {}

            inline void rebind(var_t* var_ptr) {
                ptr_.reset(var_ptr);
            }

            inline void rebind(std::shared_ptr<var_t> var_ptr) {
                ptr_ = var_ptr;
            }

            inline void rebind(std::unique_ptr<var_t> var_ptr) {
                ptr_ = std::move(var_ptr);
            }

            explicit inline operator bool() const noexcept { return static_cast<bool>(ptr_); }
            inline var_t& operator*() const noexcept { return *ptr_; }
            inline var_t* operator->() const noexcept { return ptr_.get(); }
            inline var_t* get() const noexcept { return ptr_.get(); }
            inline const std::shared_ptr<var_t>& get_shared() const noexcept { return ptr_; }

            inline Derived clone() const {
                if (ptr_) {
                    return Derived { ptr_->clone() };
                }
                else {
                    return Derived();
                }
            }

            inline std::string to_string() const override {
                if (ptr_) {
                    return ptr_->to_string();
                } else {
                    return "VariableRef(nullptr)";
                }
            }
        protected:
            std::shared_ptr<var_t> ptr_;
    };

    template <typename dtype>
    class VariableRef;

// inline VariableRef<dtype> adjoint() const {
//     OPTSUITE_ASSERT(ptr_);
//     return VariableRef<dtype> { ptr_->eval_op_adjoint() };
// }
#define OPTSUITE_DEFINE_UNARY_OP_INTERFACE(opname, interface_name) \
    inline VariableRef<dtype> opname() const { \
        OPTSUITE_ASSERT(ptr_); \
        return VariableRef<dtype> { ptr_-> eval_op_##interface_name () }; \
    }

// inline VariableRef<dtype> operator+(const VariableRef<dtype>& rhs) const {
//     OPTSUITE_ASSERT(ptr_);
//     OPTSUITE_ASSERT(rhs);
//     return VariableRef<dtype> { ptr_->eval_op_add(*rhs) };
// }
// /* VariableRef + Mat/SpMat/Variable/Scalar */
// template <typename RhsType>
// inline VariableRef<dtype> operator+(const RhsType& rhs) const {
//     OPTSUITE_ASSERT(ptr_);
//     return VariableRef<dtype> { ptr_->eval_op_add(rhs) };
// }
#define OPTSUITE_DEFINE_BINARY_OP_INTERFACE_WITH_DTYPE(opname, interface_name, result_dtype) \
    inline VariableRef<result_dtype> opname(const VariableRef<dtype>& rhs) const { \
        OPTSUITE_ASSERT(ptr_); \
        OPTSUITE_ASSERT(rhs); \
        return VariableRef<result_dtype> { ptr_-> eval_op_##interface_name (*rhs) }; \
    } \
    /* VariableRef + Mat/SpMat/Variable/Scalar */ \
    template <typename RhsType> \
    inline VariableRef<result_dtype> opname(const RhsType& rhs) const { \
        OPTSUITE_ASSERT(ptr_); \
        return VariableRef<result_dtype> { ptr_-> eval_op_##interface_name (rhs) }; \
    }

#define OPTSUITE_DEFINE_BINARY_OP_INTERFACE(opname, interface_name) \
    OPTSUITE_DEFINE_BINARY_OP_INTERFACE_WITH_DTYPE(opname, interface_name, dtype)

#define OPTSUITE_DEFINE_COMP_OP_INTERFACE(opname, interface_name) \
    OPTSUITE_DEFINE_BINARY_OP_INTERFACE_WITH_DTYPE(opname, interface_name, bool)

    template <>
    class VariableRef<void> final : public VariableRefBase<Variable<void>, VariableRef<void>> {
        public:
            using BaseClass = VariableRefBase<Variable<void>, VariableRef<void>>;
            using BaseClass::BaseClass;

            VariableRef() = default;
            ~VariableRef() = default;
            VariableRef(const VariableRef<void>&) = default;
            VariableRef(VariableRef<void>&&) = default;
            VariableRef<void>& operator=(const VariableRef<void>&) = default;
            VariableRef<void>& operator=(VariableRef<void>&&) = default;
    };


    // ========== dtype = bool ===========
    template <>
    class VariableRef<bool> final : public VariableRefBase<Variable<bool>, VariableRef<bool>> {
        using dtype = bool;
        public:
            using BaseClass = VariableRefBase<Variable<bool>, VariableRef<bool>>;
            using BaseClass::BaseClass;

            VariableRef() = default;
            ~VariableRef() = default;
            VariableRef(const VariableRef<bool>&) = default;
            VariableRef(VariableRef<bool>&&) = default;
            VariableRef<bool>& operator=(const VariableRef<bool>&) = default;
            VariableRef<bool>& operator=(VariableRef<bool>&&) = default;

            template <typename Derived>
            explicit VariableRef(const MatrixBase<Derived>& x): BaseClass(std::make_shared<MatWrapper<bool>>(x)) {}

            template <typename Derived>
            inline void rebind(const MatrixBase<Derived>& x) {
                ptr_ = std::make_shared<MatWrapper<dtype>>(x);
            }

            inline bool equals(const VariableRef<bool>& rhs) const {
                OPTSUITE_ASSERT(rhs);
                return ((*this) ^ rhs)->none();
            }

            // ResultDtype should be explicitly given
            template <typename ResultDtype>
            VariableRef<ResultDtype> map_to_scalar(const std::function<ResultDtype(bool)>& op) const {
                OPTSUITE_ASSERT(ptr_);
                return VariableRef<ResultDtype> { ptr_->map_to_scalar(op) };
            }

            OPTSUITE_DEFINE_BINARY_OP_INTERFACE(operator&, land)
            OPTSUITE_DEFINE_BINARY_OP_INTERFACE(operator|, lor)
            OPTSUITE_DEFINE_BINARY_OP_INTERFACE(operator^, lxor)
            OPTSUITE_DEFINE_UNARY_OP_INTERFACE(operator~, neg)
    };


    // =========== dtype = Scalar/ComplexScalar =================
    template <typename dtype>
    class VariableRef final : public VariableRefBase<Variable<dtype>, VariableRef<dtype>> {
        public:
            using var_t = Variable<dtype>;
            using mat_t = Eigen::Matrix<dtype, Dynamic, Dynamic>;
            using spmat_t = Eigen::SparseMatrix<dtype, ColMajor, SparseIndex>;
            using BaseClass = VariableRefBase<Variable<dtype>, VariableRef<dtype>>;
            using BaseClass::ptr_;
            using BaseClass::BaseClass;
            using BaseClass::clone;

            // base class overloads
            using BaseClass::operator*;
            using BaseClass::rebind;

            VariableRef() = default;
            ~VariableRef() = default;
            VariableRef(const VariableRef<dtype>&) = default;
            VariableRef(VariableRef<dtype>&&) = default;
            VariableRef<dtype>& operator=(const VariableRef<dtype>&) = default;
            VariableRef<dtype>& operator=(VariableRef<dtype>&&) = default;

            template <typename Derived>
            explicit VariableRef(const MatrixBase<Derived>& x): BaseClass(std::make_shared<MatWrapper<dtype>>(x)) {}

            template <typename Derived>
            explicit VariableRef(const SparseMatrixBase<Derived>& x) : BaseClass(std::make_shared<SpMatWrapper<dtype>>(x)) {}

            explicit VariableRef(const dtype& x) : BaseClass(std::make_shared<ScalarWrapper<dtype>>(x)) {}
            // TODO when implementing SDP: mat_array

            inline std::string classname() const override {
                return "VariableRef";
            }

            template <typename Derived>
            inline void rebind(const MatrixBase<Derived>& x) {
                ptr_ = std::make_shared<MatWrapper<dtype>>(x);
            }

            template <typename Derived>
            inline void rebind(const SparseMatrixBase<Derived>& x) {
                ptr_ = std::make_shared<SpMatWrapper<dtype>>(x);
            }

            inline void rebind(const dtype& x) {
                ptr_ = std::make_shared<ScalarWrapper<dtype>>(x);
            }

            inline dtype dot(const VariableRef<dtype>& other) const {
                OPTSUITE_ASSERT(ptr_);
                OPTSUITE_ASSERT(other);
                return ptr_->dot(*other);
            }

            template <typename PredicateType>
            inline VariableRef<bool> where(const PredicateType& pred) const {
                OPTSUITE_ASSERT(ptr_);
                auto internal_pred = [&pred] (const Scalar& s) { return pred(s); };
                return VariableRef<bool> { ptr_->where(internal_pred) };
            }

            // slice
            inline VariableRef<dtype> operator()(const std::vector<Index>& indices) const {
                OPTSUITE_ASSERT(ptr_);
                return VariableRef<dtype> { ptr_->slice(indices) };
            }

            inline VariableRef<dtype> operator()(const VariableRef<bool>& boolvar) const {
                OPTSUITE_ASSERT(ptr_);
                OPTSUITE_ASSERT(boolvar);
                return VariableRef<dtype> { ptr_->slice(*boolvar) };
            }

            inline dtype& operator()(const Index i) {
                OPTSUITE_ASSERT(ptr_);
                return ptr_->slice(i);
            }

            inline const dtype& operator()(const Index i) const {
                OPTSUITE_ASSERT(ptr_);
                return ptr_->slice(i);
            }

            inline dtype& operator()(const Index i, const Index j) {
                OPTSUITE_ASSERT(ptr_);
                return ptr_->slice(i, j);
            }

            inline const dtype& operator()(const Index i, const Index j) const {
                OPTSUITE_ASSERT(ptr_);
                return ptr_->slice(i, j);
            }

            inline void slice_set(const VariableRef<bool>& boolvar, const VariableRef<dtype>& rhs) const {
                OPTSUITE_ASSERT(ptr_);
                OPTSUITE_ASSERT(boolvar);
                OPTSUITE_ASSERT(rhs);
                return ptr_->slice_set(*boolvar, *rhs);
            }

            inline void slice_set(const VariableRef<bool>& boolvar, const dtype& rhs) const {
                OPTSUITE_ASSERT(ptr_);
                OPTSUITE_ASSERT(boolvar);
                return ptr_->slice_set(*boolvar, rhs);
            }

            inline void slice_set(const std::vector<Index>& indices, const VariableRef<dtype>& rhs) const {
                OPTSUITE_ASSERT(ptr_);
                OPTSUITE_ASSERT(rhs);
                return ptr_->slice_set(indices, *rhs);
            }

            inline void slice_set(const std::vector<Index>& indices, const dtype& rhs) const {
                OPTSUITE_ASSERT(ptr_);
                return ptr_->slice_set(indices, rhs);
            }

            inline spmat_t* try_as_sparse() const noexcept {
                auto spwrap = dynamic_cast<SpMatWrapper<dtype>*>(ptr_.get());
                return spwrap ? &spwrap->spmat() : nullptr;
            }

            inline mat_t* try_as_dense() const noexcept {
                auto matwrap = dynamic_cast<MatWrapper<dtype>*>(ptr_.get());
                return matwrap ? &matwrap->mat() : nullptr;
            }

            template <typename UnaryOp>
            inline VariableRef<dtype> unary_map(const UnaryOp& op) const {
                OPTSUITE_ASSERT(ptr_);
                VariableRef<dtype> ans = clone();
                ans->unary_map_inplace(op);
                return ans;
            }

            template <typename BinaryOp>
            inline VariableRef<dtype> binary_map(const VariableRef<dtype>& rhs, const BinaryOp& op) const {
                OPTSUITE_ASSERT(ptr_);
                OPTSUITE_ASSERT(rhs);
                VariableRef<dtype> ans = clone();
                ans->binary_map_inplace(*rhs, op);
                return ans;
            }

            OPTSUITE_DEFINE_UNARY_OP_INTERFACE(adjoint, adjoint)
            OPTSUITE_DEFINE_UNARY_OP_INTERFACE(operator-, neg)
#undef OPTSUITE_DEFINE_UNARY_OP_INTERFACE

            OPTSUITE_DEFINE_BINARY_OP_INTERFACE(operator+, add)
            OPTSUITE_DEFINE_BINARY_OP_INTERFACE(operator-, sub)
            OPTSUITE_DEFINE_BINARY_OP_INTERFACE(operator*, mul)
            OPTSUITE_DEFINE_BINARY_OP_INTERFACE(operator/, div)
            OPTSUITE_DEFINE_BINARY_OP_INTERFACE(tmul, tmul)
            OPTSUITE_DEFINE_BINARY_OP_INTERFACE(mult, mult)
            OPTSUITE_DEFINE_BINARY_OP_INTERFACE(tmult, tmult)

            OPTSUITE_DEFINE_COMP_OP_INTERFACE(operator<, lt)
            OPTSUITE_DEFINE_COMP_OP_INTERFACE(operator<=, le)
            OPTSUITE_DEFINE_COMP_OP_INTERFACE(operator>, gt)
            OPTSUITE_DEFINE_COMP_OP_INTERFACE(operator>=, ge)
            OPTSUITE_DEFINE_COMP_OP_INTERFACE(operator!=, ne)
            OPTSUITE_DEFINE_COMP_OP_INTERFACE(operator==, eq)
#undef OPTSUITE_DEFINE_BINARY_OP_INTERFACE
#undef OPTSUITE_DEFINE_COMP_OP_INTERFACE
#undef OPTSUITE_DEFINE_BINARY_OP_INTERFACE_WITH_DTYPE

            // template <typename RhsType>
            // inline VariableRef<dtype>& operator+=(const RhsType& rhs) {
            //     *this = *this + rhs;
            //     return *this;
            // }
#define OPTSUITE_DEFINE_ASSIGN_OP_INTERFACE(op, reducer) \
            template <typename RhsType> \
            inline VariableRef<dtype>& op(const RhsType& rhs) { \
                *this = *this reducer rhs; \
                return *this; \
            }

            OPTSUITE_DEFINE_ASSIGN_OP_INTERFACE(operator+=, +)
            OPTSUITE_DEFINE_ASSIGN_OP_INTERFACE(operator-=, -)
            OPTSUITE_DEFINE_ASSIGN_OP_INTERFACE(operator*=, *)
            OPTSUITE_DEFINE_ASSIGN_OP_INTERFACE(operator/=, /)
#undef OPTSUITE_DEFINE_ASSIGN_OP_INTERFACE

    };

    // template <typename LhsType, typename dtype>
    // auto operator+(const LhsType& lhs, const VariableRef<dtype>& rhs)
    // {
    //     OPTSUITE_ASSERT(rhs);
    //     return VariableRef<dtype>(lhs) + rhs;
    // }
#define OPTSUITE_DEFINE_BINARY_OP_INTERFACE(opname) \
    template <typename LhsType, typename dtype> \
    auto opname(const LhsType& lhs, const VariableRef<dtype>& rhs) -> decltype(VariableRef<dtype>(lhs).opname(rhs)) { \
        OPTSUITE_ASSERT(rhs); \
        return VariableRef<dtype>(lhs).opname(rhs); \
    }

    OPTSUITE_DEFINE_BINARY_OP_INTERFACE(operator+)
    OPTSUITE_DEFINE_BINARY_OP_INTERFACE(operator-)
    OPTSUITE_DEFINE_BINARY_OP_INTERFACE(operator*)
    OPTSUITE_DEFINE_BINARY_OP_INTERFACE(operator/)
    OPTSUITE_DEFINE_BINARY_OP_INTERFACE(operator<)
    OPTSUITE_DEFINE_BINARY_OP_INTERFACE(operator<=)
    OPTSUITE_DEFINE_BINARY_OP_INTERFACE(operator>)
    OPTSUITE_DEFINE_BINARY_OP_INTERFACE(operator>=)
    OPTSUITE_DEFINE_BINARY_OP_INTERFACE(operator==)
    OPTSUITE_DEFINE_BINARY_OP_INTERFACE(operator!=)

#undef OPTSUITE_DEFINE_BINARY_OP_INTERFACE
}}

#endif
