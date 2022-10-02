/*
 * ===========================================================================
 *
 *       Filename:  base.h
 *
 *    Description:
 *
 *        Version:  1.0
 *        Created:  01/03/2021 03:41:03 PM
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  Haoyang Liu (), liuhaoyang@pku.edu.cn
 *   Organization:  BICMR, Peking University
 *
 * ===========================================================================
 */

#ifndef OPTSUITE_BASE_FUNC_BASE_H
#define OPTSUITE_BASE_FUNC_BASE_H

#include "OptSuite/core_n.h"
#include "OptSuite/Base/fwddecl.h"

namespace OptSuite { namespace Base {

    class Functional {
        public:
            Functional() = default;
            virtual ~Functional() = default;
            void bind(const std::shared_ptr<Structure>&);
            void unbind();

        protected:
            std::shared_ptr<Structure> workspace = nullptr;
    };

    class StopRuleChecker : public Functional {
        public:
            StopRuleChecker() = default;
            virtual ~StopRuleChecker() = default;
            inline virtual bool operator()() const { return false; }
    };

    template<typename dtype>
    class Func : public Functional {
        public:
            using mat_t = Eigen::Matrix<dtype, Dynamic, Dynamic>;
            using mat_wrapper_t = MatWrapper<dtype>;
            using mat_array_t = MatArray_t<dtype>;
            using var_t = Variable<dtype>;
            using fmat_t = FactorizedMat<dtype>;
            Func() = default;
            virtual ~Func() = default;

            virtual Scalar operator()(const Ref<const mat_t>);
            virtual Scalar operator()(const mat_wrapper_t&);
            virtual Scalar operator()(const mat_array_t&);
            virtual Scalar operator()(const var_t&);
    };

    template<typename dtype>
    class Proximal : public Functional {
        public:
            using mat_t = Eigen::Matrix<dtype, Dynamic, Dynamic>;
            using mat_wrapper_t = MatWrapper<dtype>;
            using spmat_wrapper_t = SpMatWrapper<dtype>;
            using mat_array_t = MatArray_t<dtype>;
            using var_t = Variable<dtype>;
            virtual ~Proximal() = default;

            virtual void operator()(const Ref<const mat_t>, Scalar, Ref<mat_t>);
            virtual void operator()(const mat_wrapper_t&, Scalar, mat_wrapper_t&);
            virtual void operator()(const mat_array_t&, Scalar, mat_array_t&);
            virtual void operator()(const var_t&, Scalar, const var_t&, Scalar, var_t&);
            virtual bool is_identity() const { return false; }
            virtual bool has_objective_cache() const { return false; }
            virtual Scalar cached_objective() const { return 0_s; }
    };

    template<typename dtype>
    class Grad : public Functional {
        public:
            using mat_t = Eigen::Matrix<dtype, Dynamic, Dynamic>;
            Grad() = default;
            virtual ~Grad() = default;

            virtual void operator()(const Ref<const mat_t>, Ref<mat_t>) = 0;

        protected:
            unsigned num_eval = 0;
    };

    template<typename dtype>
    class FuncGrad : public Functional {
        public:
            using mat_t = Eigen::Matrix<dtype, Dynamic, Dynamic>;
            using mat_wrapper_t = MatWrapper<dtype>;
            using mat_array_t = MatArray_t<dtype>;
            using var_t = Variable<dtype>;
            FuncGrad() = default;
            virtual ~FuncGrad() = default;

            virtual Scalar eval(const Ref<const mat_t>);
            virtual Scalar eval(const Ref<const mat_t>, Ref<mat_t>, bool, bool);
            virtual Scalar eval(const mat_array_t&);
            virtual Scalar eval(const mat_array_t&, mat_array_t&, bool, bool);
            virtual Scalar eval(const mat_wrapper_t&);
            virtual Scalar eval(const mat_wrapper_t&, mat_wrapper_t&, bool, bool);
            virtual Scalar eval(const var_t&);
            virtual Scalar eval(const var_t&, var_t&, bool, bool);

#define __def_operator_bracket(T1, T2) \
            inline Scalar operator()(const T1 x){ \
                return this->eval(x); \
            } \
            inline Scalar operator()(const T1 x, T2 g, \
                    bool compute_grad = true, bool use_cache = false){ \
                return this->eval(x, g, compute_grad, use_cache); \
            }

            __def_operator_bracket(Ref<const mat_t>, Ref<mat_t>)
            __def_operator_bracket(mat_array_t&, mat_array_t&)
            __def_operator_bracket(mat_wrapper_t&, mat_wrapper_t&)
            __def_operator_bracket(var_t&, var_t&)

#undef __def_operator_bracket
    };
}}

#endif
