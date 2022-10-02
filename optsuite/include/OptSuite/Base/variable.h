/*
 * ==========================================================================
 *
 *       Filename:  variable.h
 *
 *    Description:
 *
 *        Version:  2.0
 *        Created:  12/10/2020 02:32:37 PM
 *       Revision:  10/13/2021 02:29:00 PM
 *       Compiler:  g++
 *
 *         Author:  Haoyang Liu (@liuhy), liuhaoyang@pku.edu.cn
 *   Organization:  BICMR, PKU
 *
 * ==========================================================================
 */

#ifndef OPTSUITE_BASE_VARIABLE_H
#define OPTSUITE_BASE_VARIABLE_H

#include "OptSuite/core_n.h"
#include "OptSuite/Base/base.h"

namespace OptSuite { namespace Base {
    // Helper template to define Variable<dtype>.
    template <typename dtype, typename Derived>
    class VariableBase : public OptSuiteBase {
        public:
            using VarPtr = std::unique_ptr<Derived>;

            inline virtual VarPtr clone() const { return not_implemented(nullptr); }

            inline std::string classname() const override {
                return "Variable";
            }

            inline virtual void assign(const Derived&) { not_implemented(); }

            inline virtual std::vector<Index> shape() const {
                return std::vector<Index>{};
            }

            // override for better performance
            inline virtual Index size() const {
                Index ans = 1_i;
                for (Index s : shape()) {
                    ans *= s;
                }
                return ans;
            }
        protected:

            /* non virtual */
            inline void not_implemented() const {
                OPTSUITE_ASSERT_MSG(false, "Not implemented");
            }

            template <typename T>
            inline T not_implemented(const T& retval) const {
                OPTSUITE_ASSERT_MSG(false, "Not implemented");
                return retval;
            }
    };

    template <typename dtype>
    class Variable;

// inline virtual VarPtr eval_op_neg() const { return not_implemented(nullptr); }
#define OPTSUITE_DEF_UNARY_OP_INTERFACE(opname) \
    inline virtual VarPtr eval_op_##opname() const { return not_implemented(nullptr); }



// ===================== Variable<void> ===================
    template <>
    class Variable<void> : public VariableBase<void, Variable<void>> {};



// ===================== Variable<bool> ===================
    template <>
    class Variable<bool> : public VariableBase<bool, Variable<bool>> {
        using BaseClass = VariableBase<bool, Variable<bool>>;
        using BaseClass::not_implemented;
        using BaseClass::size;

        public:
            using vec_t = Eigen::Matrix<bool, Dynamic, 1>;
            using typename BaseClass::VarPtr;

            // returns a vec_t map which lasts at least until *this destructs
            inline virtual Map<const vec_t> vector_view() const {
                return not_implemented(Map<const vec_t>(nullptr, 0));
            }

            // return the indices that this(index) is true
            // MATLAB: indices = find(this)
            inline virtual std::vector<Index> indices() const {
                std::vector<Index> ans;
                Map<const vec_t> vec_view = vector_view();
                ans.reserve(vec_view.size());
                for (Index i = 0; i < vec_view.size(); ++i) {
                    if (vec_view(i)) {
                        ans.push_back(i);
                    }
                }
                return ans;
            }

            inline virtual bool all() const {
                return vector_view().all();
            }

            inline virtual bool any() const {
                return vector_view().any();
            }

            inline virtual bool none() const {
                return !any();
            }

            inline virtual std::unique_ptr<Variable<bool>>
            map_to_scalar(const std::function<bool(bool)>&) const {
                return not_implemented(nullptr);
            }

            inline virtual std::unique_ptr<Variable<Scalar>>
            map_to_scalar(const std::function<Scalar(bool)>&) const {
                return not_implemented(nullptr);
            }

            inline virtual std::unique_ptr<Variable<ComplexScalar>>
            map_to_scalar(const std::function<ComplexScalar(bool)>&) const {
                return not_implemented(nullptr);
            }

#define OPTSUITE_DEF_BINARY_OP_INTERFACE(opname) \
            inline virtual VarPtr eval_op_##opname(const Variable<bool>& rhs) const { return rhs.accept_op_##opname(*this); } \
            inline virtual VarPtr eval_op_##opname(const Ref<const BoolMat>& /* rhs */) const { return not_implemented(nullptr); } \
            inline virtual VarPtr accept_op_##opname(const Variable<bool>& /* lhs */) const { return not_implemented(nullptr); }

            OPTSUITE_DEF_BINARY_OP_INTERFACE(land)
            OPTSUITE_DEF_BINARY_OP_INTERFACE(lor)
            OPTSUITE_DEF_BINARY_OP_INTERFACE(lxor)
#undef OPTSUITE_DEF_BINARY_OP_INTERFACE

            OPTSUITE_DEF_UNARY_OP_INTERFACE(neg)
        protected:

    };

// ===================== Variable<dtype>, dtype=Scalar/ComplexScalar ===================
    template <typename dtype>
    class Variable : public VariableBase<dtype, Variable<dtype>> {
        public:
            using mat_t = Eigen::Matrix<dtype, Dynamic, Dynamic>;
            using spmat_t = Eigen::SparseMatrix<dtype, ColMajor, SparseIndex>;
            using vec_t = Eigen::Matrix<dtype, Dynamic, 1>;
            using BaseClass = VariableBase<dtype, Variable<dtype>>;
            using typename BaseClass::VarPtr;
            using BoolVarPtr = std::unique_ptr<Variable<bool>>;
            using BaseClass::not_implemented;

            inline
            virtual dtype dot(const Variable<dtype>& rhs) const {
                return rhs.accept_op_dot(*this);
            }

            inline
            virtual dtype dot(const Ref<const mat_t>& /* rhs */) const {
                return not_implemented(0_s);
            }

            inline
            virtual dtype dot(const Ref<const spmat_t>& /* rhs */) const {
                return not_implemented(0_s);
            }

            inline
            virtual dtype accept_op_dot(const Variable<dtype>& /* lhs */) const {
                return not_implemented(0_s);
            }

            virtual void set_zero_like(const Variable<dtype>&) {
                not_implemented();
            };

            virtual Scalar squared_norm() const {
                return std::real(this->dot(*this));
            }

            virtual Scalar norm() const {
                return std::sqrt(squared_norm());
            }

            inline
            virtual Scalar squared_norm_diff(const Variable<dtype>&) const { return 0_s; }

            inline
            virtual bool has_squared_norm_diff() const { return false; }

            // MATLAB example: where_result = v(v == x)
            inline
            virtual BoolVarPtr where(const std::function<bool(const dtype&)>&) const {
                return not_implemented(nullptr);
            }

            // the interface to automatically support slice(i) and slice_set(i, rhs)
            // Note: the result should be writable when *this is not const
            //   when that is not the case, also overwrite the non-const version.
            // returning -1 in size_out indicates not-implemented.
            //  This makes returning nullptr with size_out = 0 valid.
            inline virtual const dtype* value_span(Index& size_out) const {
                size_out = -1_i;
                return not_implemented(nullptr);
            }

            inline virtual dtype* value_span(Index& size_out) {
                const auto& cthis = *this;
                return const_cast<dtype*>(cthis.value_span(size_out));
            }

            // prefer implmenting value_span over implementing vector_view
            inline virtual Map<const vec_t> vector_view() const {
                Index ptrsize = 0;
                const dtype* dptr = value_span(ptrsize);
                OPTSUITE_ASSERT(ptrsize == 0 || dptr);
                return Map<const vec_t>(dptr, ptrsize);
            }

            inline virtual Map<vec_t> vector_view() {
                Index ptrsize = 0;
                dtype* dptr = value_span(ptrsize);
                OPTSUITE_ASSERT(ptrsize == 0 || dptr);
                return Map<vec_t>(dptr, ptrsize);
            }

            // when overwriting the const version, the return value must be
            //   a non-const reference when *this is not const.
            // otherwise the non-const version of slice must also be overriden

            // v(i). A matrix-like object is viewed as a vector
            virtual const dtype& slice(const Index&) const;
            virtual dtype& slice(const Index&);

            // v(i, j)
            virtual const dtype& slice(const Index&, const Index&) const;
            virtual dtype& slice(const Index&, const Index&);

            // MATLAB example: v(indices)
            //  indices could be converted from Variable<bool>
            virtual vec_t slice(const std::vector<Index>&) const;
            inline
            virtual vec_t slice(const Variable<bool>& boolvar) const {
                return slice(boolvar.indices());
            }

            // MATLAB example: v(indices) = v2
            virtual void slice_set(const std::vector<Index>&, const Variable<dtype>&);
            virtual void slice_set(const std::vector<Index>&, const dtype&);
            virtual void slice_set(const Variable<bool>&, const Variable<dtype>&);
            virtual void slice_set(const Variable<bool>&, const dtype&);

            // not virtual because of template argument!
            // To implement this, override value_span function
            template <typename UnaryOp>
            inline void unary_map_inplace(UnaryOp&& op) {
                auto data_map = vector_view();
                data_map = data_map.unaryExpr(std::forward<UnaryOp>(op));
            }

            // not virtual because of template argument!
            // To implement this, override value_span function
            template <typename BinaryOp>
            inline void binary_map_inplace(const Variable<dtype>& rhs, BinaryOp&& op) {
                auto lhs_map = vector_view();
                auto rhs_map = rhs.vector_view();
                OPTSUITE_ASSERT(lhs_map.size() == rhs_map.size());
                lhs_map = lhs_map.binaryExpr(rhs_map, std::forward<BinaryOp>(op));
            }

            OPTSUITE_DEF_UNARY_OP_INTERFACE(adjoint)
            OPTSUITE_DEF_UNARY_OP_INTERFACE(neg)

            // note: eval_op_* could be overridden so that it does not call accept_op_*
            // inline virtual VarPtr eval_op_add(const Variable<dtype>& rhs) const { return rhs.accept_op_add(*this); }
            // inline virtual VarPtr eval_op_add(const Ref<const mat_t>& /* rhs */) const { return not_implemented(nullptr); }
            // inline virtual VarPtr eval_op_add(const Ref<const spmat_t>& /* rhs */) const { return not_implemented(nullptr); }
            // inline virtual VarPtr eval_op_add(const dtype& /* rhs */) const { return not_implemented(nullptr); }
            // inline virtual VarPtr accept_op_add(const Variable<dtype>& /* lhs */) const { return not_implemented(nullptr); }
#define OPTSUITE_DEF_BINARY_OP_INTERFACE(opname) \
            inline virtual VarPtr eval_op_##opname(const Variable<dtype>& rhs) const { return rhs.accept_op_##opname(*this); } \
            inline virtual VarPtr eval_op_##opname(const Ref<const mat_t>& /* rhs */) const { return not_implemented(nullptr); } \
            inline virtual VarPtr eval_op_##opname(const Ref<const spmat_t>& /* rhs */) const { return not_implemented(nullptr); } \
            inline virtual VarPtr eval_op_##opname(const dtype& /* rhs */) const { return not_implemented(nullptr); } \
            inline virtual VarPtr accept_op_##opname(const Variable<dtype>& /* lhs */) const { return not_implemented(nullptr); }

            OPTSUITE_DEF_BINARY_OP_INTERFACE(add)
            OPTSUITE_DEF_BINARY_OP_INTERFACE(sub)
            OPTSUITE_DEF_BINARY_OP_INTERFACE(mul)
            OPTSUITE_DEF_BINARY_OP_INTERFACE(div)
            OPTSUITE_DEF_BINARY_OP_INTERFACE(tmul) // lhs' * rhs
            OPTSUITE_DEF_BINARY_OP_INTERFACE(mult) // lhs * rhs'
            OPTSUITE_DEF_BINARY_OP_INTERFACE(tmult) // lhs' * rhs'

            // inline virtual BoolVarPtr eval_op_gt(const Variable<dtype>& rhs) const {
            //     return not_implemented(nullptr);
            // }
            // inline virtual BoolVarPtr eval_op_gt(const dtype& rhs) const {
            //     return not_implemented(nullptr);
            // }
#define OPTSUITE_DEF_COMP_OP_INTERFACE(opname) \
            inline virtual BoolVarPtr eval_op_##opname(const Variable<dtype>& /* rhs */) const {  \
                return not_implemented(nullptr); \
            } \
            inline virtual BoolVarPtr eval_op_##opname(const dtype& /* rhs */) const { \
                return not_implemented(nullptr); \
             }

            OPTSUITE_DEF_COMP_OP_INTERFACE(gt)  // >
            OPTSUITE_DEF_COMP_OP_INTERFACE(ge)  // >=
            OPTSUITE_DEF_COMP_OP_INTERFACE(lt)  // <
            OPTSUITE_DEF_COMP_OP_INTERFACE(le)  // <=
            OPTSUITE_DEF_COMP_OP_INTERFACE(eq)  // ==
            OPTSUITE_DEF_COMP_OP_INTERFACE(ne)  // !=

#undef OPTSUITE_DEF_COMP_OP_INTERFACE
#undef OPTSUITE_DEF_BINARY_OP_INTERFACE
#undef OPTSUITE_DEF_UNARY_OP_INTERFACE
    };
}}

#endif
