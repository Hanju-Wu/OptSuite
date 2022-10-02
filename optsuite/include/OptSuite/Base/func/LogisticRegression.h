/*
 * @Author: Wu Hanju 
 * @Date: 2022-09-02 22:32:25 
 * @Last Modified by: Wu Hanju
 * @Last Modified time: 2022-09-03 20:25:53
 */

#ifndef OPTSUITE_BASE_FUNC_LOGISTICREGRESSION_H
#define OPTSUITE_BASE_FUNC_LOGISTICREGRESSION_H

#include "OptSuite/Base/func/base.h"

namespace OptSuite { namespace Base {
    template<typename dtype = Scalar>
        class LogisticRegression : public FuncGrad<dtype> {
        public:
            using typename FuncGrad<dtype>::mat_t;
            using vec_t = Eigen::Matrix<dtype, Dynamic, 1>;
            using spmat_t = Eigen::SparseMatrix<dtype, ColMajor, SparseIndex>;
            LogisticRegression(const Ref<const mat_t>, const Ref<const mat_t>);
            ~LogisticRegression() = default;

            using FuncGrad<dtype>::eval;
            Scalar eval(const Ref<const mat_t>, Ref<mat_t>, bool, bool) override;

            const mat_t& get_A() const { return A; }
            const vec_t& get_b() const { return b; }
            Scalar get_fun() const { return fun; }

            Scalar set_fun(const Ref<const mat_t>);
            Scalar set_fun(const Ref<const spmat_t>);

            Index cols() const;
            Index rows() const;
            
        protected:
            using Functional::workspace;

        private:
            mat_t A;
            vec_t b;
            mat_t mbA;
            Scalar fun;

            vec_t t,q,r;
        };
}}

#endif // OPTSUITE_BASE_FUNC_LOGISTICREGRESSION_H