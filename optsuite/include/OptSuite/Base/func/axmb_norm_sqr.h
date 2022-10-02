/*
 * ===========================================================================
 *
 *       Filename:  axmb_norm_sqr.h
 *
 *    Description:  header file for axmb_norm_sqr function
 *
 *        Version:  2.0
 *        Created:  01/03/2021 03:49:31 PM
 *       Revision:  02/28/2021 14:11
 *       Compiler:  g++
 *
 *         Author:  Haoyang Liu (), liuhaoyang@pku.edu.cn
 *   Organization:  BICMR, Peking University
 *
 * ===========================================================================
 */

#ifndef OPTSUITE_BASE_FUNC_AXMB_NORM_SQR_H
#define OPTSUITE_BASE_FUNC_AXMB_NORM_SQR_H

#include "OptSuite/Base/func/base.h"

namespace OptSuite { namespace Base {
    template<typename dtype = Scalar>
    class AxmbNormSqr : public FuncGrad<dtype> {
        using typename FuncGrad<dtype>::mat_t;
        using spmat_t = Eigen::SparseMatrix<dtype, ColMajor, SparseIndex>;
        enum class AType {
            Dense,
            Sparse,
            Operator
        };
        public:
            AxmbNormSqr(const Ref<const mat_t>, const Ref<const mat_t>);
            AxmbNormSqr(const Ref<const spmat_t>, const Ref<const mat_t>);
            AxmbNormSqr(const MatOp<dtype>&, const Ref<const mat_t>);
            ~AxmbNormSqr() = default;

            using FuncGrad<dtype>::eval;
            Scalar eval(const Ref<const mat_t>, Ref<mat_t>, bool, bool) override;
            const mat_t& get_A() const;
            const spmat_t& get_spA() const;
            const MatOp<dtype>& get_Aop() const;
            const mat_t& get_b() const;
            Index cols() const;
            Index rows() const;
        protected:
            using Functional::workspace;
        private:
            AType type;
            mat_t A;
            spmat_t spA;
            const MatOp<dtype>* Aop;
            mat_t b;
            mat_t r;
            Scalar fun;
    };

}}

#endif
