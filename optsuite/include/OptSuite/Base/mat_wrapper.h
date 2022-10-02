/*
 * ===========================================================================
 *
 *       Filename:  mat_wrapper.h
 *
 *    Description:
 *
 *        Version:  1.0
 *        Created:  12/11/2020 01:26:04 PM
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  Haoyang Liu (), liuhaoyang@pku.edu.cn
 *   Organization:  BICMR, Peking University
 *
 * ===========================================================================
 */

#ifndef OPTSUITE_BASE_MAT_WRAPPER_H
#define OPTSUITE_BASE_MAT_WRAPPER_H

#include <functional>
#include "OptSuite/core_n.h"
#include "OptSuite/Base/object_wrapper.h"

namespace OptSuite { namespace Base {
    template<typename dtype> class MatWrapper;

    template<>
    class MatWrapper<bool> : public ObjectWrapper<Eigen::Matrix<bool, Dynamic, Dynamic>, bool> {
        using dtype = bool;  // alias for convenience
        using BaseClass = ObjectWrapper<Eigen::Matrix<bool, Dynamic, Dynamic>, bool>;
        using typename BaseClass::VarPtr;
        using BaseClass::BaseClass;
        using BaseClass::data_;
        public:
            inline VarPtr clone() const override {
                return VarPtr { new MatWrapper<bool>(*this) };
            }

            inline Map<const vec_t> vector_view() const override {
                return Map<const vec_t>(data_.data(), data_.size());
            }

            std::unique_ptr<Variable<bool>>
            map_to_scalar(const std::function<bool(bool)>&) const override;

            std::unique_ptr<Variable<Scalar>>
            map_to_scalar(const std::function<Scalar(bool)>&) const override;

            std::unique_ptr<Variable<ComplexScalar>>
            map_to_scalar(const std::function<ComplexScalar(bool)>&) const override;

            // inline VarPtr eval_op_land(const Ref<const BoolMat>& rhs) const override {
            //     return VarPtr { new MatWrapper<bool> { data_ && rhs } };
            // }
            // inline VarPtr accept_op_land(const Variable<bool>& lhs) const override {
            //     return lhs.eval_op_land(data_);
            // }
#define OPTSUITE_BOOLMAT_DEFINE_LOGICAL_OP(opname, reducer) \
            inline VarPtr eval_op_##opname(const Ref<const BoolMat>& rhs) const override { \
                return VarPtr { new MatWrapper<bool> { data_ reducer rhs } }; \
            } \
            inline VarPtr accept_op_##opname(const Variable<bool>& lhs) const override { \
                return lhs.eval_op_##opname(data_); \
            }

            OPTSUITE_BOOLMAT_DEFINE_LOGICAL_OP(land, &&)
            OPTSUITE_BOOLMAT_DEFINE_LOGICAL_OP(lor, ||)
#undef OPTSUITE_BOOLMAT_DEFINE_LOGICAL_OP

            inline VarPtr eval_op_lxor(const Ref<const BoolMat>& rhs) const override {
                return VarPtr { new MatWrapper<bool> {
                    data_.binaryExpr(rhs, std::bit_xor<bool>())
                } };
            }
            inline VarPtr accept_op_lxor(const Variable<bool>& lhs) const override {
                return lhs.eval_op_lxor(data_);
            }

            inline VarPtr eval_op_neg() const override {
                return VarPtr { new MatWrapper<bool> { data_.unaryExpr(std::logical_not<bool>()) } };
            }
    };

    template<typename dtype>
    class MatWrapper : public ObjectWrapper<Eigen::Matrix<dtype, Dynamic, Dynamic>, dtype> {
        using BaseClass = ObjectWrapper<Eigen::Matrix<dtype, Dynamic, Dynamic>, dtype>;
        using BaseClass::data_;
        public:
            using BaseClass::BaseClass;
            using typename BaseClass::VarPtr;
            using typename BaseClass::BoolVarPtr;
            using typename Variable<dtype>::vec_t;
            using typename Variable<dtype>::mat_t;
            using typename Variable<dtype>::spmat_t;
            MatWrapper() = default;
            ~MatWrapper() = default;
            MatWrapper(const MatWrapper&) = default;
            MatWrapper(MatWrapper&&) = default;
            MatWrapper& operator=(const MatWrapper&) = default;
            MatWrapper& operator=(MatWrapper&&) = default;

            inline mat_t& mat() { return data_; }
            inline const mat_t& mat() const { return data_; }
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
                return rhs.cwiseProduct(data_.conjugate()).sum();
            }

            inline void set_zero_like(const Ref<const mat_t> other) {
                data_.resize(other.rows(), other.cols());
                data_.setZero();
            }

            inline void set_zero_like(const Variable<dtype>& other) override {
                const MatWrapper* other_ptr = dynamic_cast<const MatWrapper*>(&other);
                OPTSUITE_ASSERT(other_ptr);

                data_.resize(other_ptr->data_.rows(), other_ptr->data_.cols());
                data_.setZero();
            }

            inline std::vector<Index> shape() const override {
                return std::vector<Index> { data_.rows(), data_.cols() };
            }

            inline Scalar squared_norm() const override {
                return data_.squaredNorm();
            }

            using BaseClass::value_span; // overload

            inline
            const dtype* value_span(Index& size_out) const override {
                size_out = data_.size();
                return data_.data();
            }

            using BaseClass::slice; // overload

            inline
            const dtype& slice(const Index& i, const Index& j) const override {
                return data_(i, j);  // i, j out-of-bound assert done by eigen
            }

            inline
            Scalar squared_norm_diff(const Variable<dtype>& other) const override {
                const MatWrapper* other_ptr = dynamic_cast<const MatWrapper*>(&other);
                OPTSUITE_ASSERT(other_ptr);
                return (data_ - other_ptr->data_).squaredNorm();
            }

            inline bool has_squared_norm_diff() const override { return true; }

            inline VarPtr clone() const override {
                return VarPtr { new MatWrapper<dtype>(*this) };
            }

            inline BoolVarPtr where(const std::function<bool(const dtype&)>& pred) const override {
                return BoolVarPtr { new MatWrapper<bool>(data_.unaryExpr(pred)) };
            }

            inline VarPtr accept_op_add(const Variable<dtype>& lhs) const override {
                return lhs.eval_op_add(data_);
            }

            inline VarPtr eval_op_add(const Ref<const mat_t>& rhs) const override {
                return VarPtr { new MatWrapper<dtype> { data_ + rhs } };
            }

            inline VarPtr eval_op_add(const Ref<const spmat_t>& rhs) const override {
                return VarPtr { new MatWrapper<dtype> { data_ + rhs } };
            }

            inline VarPtr accept_op_sub(const Variable<dtype>& lhs) const override {
                return lhs.eval_op_sub(data_);
            }

            inline VarPtr eval_op_sub(const Ref<const mat_t>& rhs) const override {
                return VarPtr { new MatWrapper<dtype> { data_ - rhs } };
            }

            inline VarPtr eval_op_sub(const Ref<const spmat_t>& rhs) const override {
                return VarPtr { new MatWrapper<dtype> { data_ - rhs } };
            }

            inline VarPtr accept_op_mul(const Variable<dtype>& lhs) const override {
                return lhs.eval_op_mul(data_);
            }

            inline VarPtr eval_op_mul(const Ref<const mat_t>& rhs) const override {
                return VarPtr { new MatWrapper<dtype> { data_ * rhs } };
            }

            inline VarPtr eval_op_mul(const Ref<const spmat_t>& rhs) const override {
                return VarPtr { new MatWrapper<dtype> { data_ * rhs } };
            }

            inline VarPtr eval_op_mul(const dtype& rhs) const override {
                return VarPtr { new MatWrapper<dtype> { data_ * rhs } };
            }

            inline VarPtr accept_op_div(const Variable<dtype>& lhs) const override {
                return lhs.eval_op_div(data_);
            }

            inline VarPtr eval_op_div(const dtype& rhs) const override {
                return VarPtr { new MatWrapper<dtype> { data_ / rhs } };
            }

            inline VarPtr accept_op_tmul(const Variable<dtype>& lhs) const override {
                return lhs.eval_op_tmul(data_);
            }

            inline VarPtr eval_op_tmul(const Ref<const mat_t>& rhs) const override {
                return VarPtr { new MatWrapper<dtype> { data_.adjoint() * rhs } };
            }

            inline VarPtr eval_op_tmul(const Ref<const spmat_t>& rhs) const override {
                return VarPtr { new MatWrapper<dtype> { data_.adjoint() * rhs } };
            }


            inline VarPtr accept_op_mult(const Variable<dtype>& lhs) const override {
                return lhs.eval_op_mult(data_);
            }

            inline VarPtr eval_op_mult(const Ref<const mat_t>& rhs) const override {
                return VarPtr { new MatWrapper<dtype> { data_ * rhs.adjoint() } };
            }

            inline VarPtr eval_op_mult(const Ref<const spmat_t>& rhs) const override {
                return VarPtr { new MatWrapper<dtype> { data_ * rhs.adjoint() } };
            }

            inline VarPtr accept_op_tmult(const Variable<dtype>& lhs) const override {
                return lhs.eval_op_tmult(data_);
            }

            inline VarPtr eval_op_tmult(const Ref<const mat_t>& rhs) const override {
                return VarPtr { new MatWrapper<dtype> { data_.adjoint() * rhs.adjoint() } };
            }

            inline VarPtr eval_op_tmult(const Ref<const spmat_t>& rhs) const override {
                return VarPtr { new MatWrapper<dtype> { data_.adjoint() * rhs.adjoint() } };
            }


            inline VarPtr eval_op_adjoint() const override {
                return VarPtr { new MatWrapper<dtype> { data_.adjoint() } };
            }

            inline VarPtr eval_op_neg() const override {
                return VarPtr { new MatWrapper<dtype>(-data_) };
            }

            // inline BoolVarPtr eval_op_gt(const Variable<dtype>& rhs) const override {
            //     if (std::is_same<dtype, ComplexScalar>::value) {
            //         return BaseClass::eval_op_gt(rhs);  // not implemented
            //     }
            //     const MatWrapper* rhs_ptr = dynamic_cast<const MatWrapper*>(&rhs);
            //     OPTSUITE_ASSERT(rhs_ptr);
            //     return BoolVarPtr { new MatWrapper<bool> { data_.real().array() > rhs_ptr->data_.real().array() } };
            // }
            // inline BoolVarPtr eval_op_gt(const dtype& rhs) const override {
            //     using std::real;
            //     if (std::is_same<dtype, ComplexScalar>::value) {
            //         return BaseClass::eval_op_gt(rhs);  // not implemented
            //     }
            //     return BoolVarPtr { new MatWrapper<bool> { data_.real().array() > real(rhs) } };
            // }
#define OPTSUITE_MAT_DEF_COMP_1(opname, reducer) \
            inline BoolVarPtr eval_op_##opname(const Variable<dtype>& rhs) const override { \
                if (std::is_same<dtype, ComplexScalar>::value) { \
                    return BaseClass::eval_op_##opname(rhs); \
                } \
                const MatWrapper* rhs_ptr = dynamic_cast<const MatWrapper*>(&rhs); \
                OPTSUITE_ASSERT(rhs_ptr); \
                return BoolVarPtr { new MatWrapper<bool> { data_.real().array() reducer rhs_ptr->data_.real().array() } }; \
            } \
            inline BoolVarPtr eval_op_##opname(const dtype& rhs) const override { \
                using std::real; \
                if (std::is_same<dtype, ComplexScalar>::value) { \
                    return BaseClass::eval_op_##opname(rhs); \
                } \
                return BoolVarPtr { new MatWrapper<bool> { data_.real().array() reducer real(rhs) } }; \
            }

            OPTSUITE_MAT_DEF_COMP_1(gt, >)
            OPTSUITE_MAT_DEF_COMP_1(ge, >=)
            OPTSUITE_MAT_DEF_COMP_1(lt, <)
            OPTSUITE_MAT_DEF_COMP_1(le, <=)
#undef OPTSUITE_MAT_DEF_COMP_1


// remove complex-related specialization
#define OPTSUITE_MAT_DEF_COMP_2(opname, reducer) \
            inline BoolVarPtr eval_op_##opname(const Variable<dtype>& rhs) const override { \
                const MatWrapper* rhs_ptr = dynamic_cast<const MatWrapper*>(&rhs); \
                OPTSUITE_ASSERT(rhs_ptr); \
                return BoolVarPtr { new MatWrapper<bool> { data_.array() reducer rhs_ptr->data_.array() } }; \
            } \
            inline BoolVarPtr eval_op_##opname(const dtype& rhs) const override { \
                return BoolVarPtr { new MatWrapper<bool> { data_.array() reducer rhs } }; \
            }

            OPTSUITE_MAT_DEF_COMP_2(eq, ==)
            OPTSUITE_MAT_DEF_COMP_2(ne, !=)
#undef OPTSUITE_MAT_DEF_COMP_2
    };
}}

#endif
