/*
 * ===========================================================================
 *
 *       Filename:  projection_omega.h
 *
 *    Description:
 *
 *        Version:  1.0
 *        Created:  01/03/2021 03:52:34 PM
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  Haoyang Liu (), liuhaoyang@pku.edu.cn
 *   Organization:  BICMR, Peking University
 *
 * ===========================================================================
 */

#ifndef OPTSUITE_BASE_FUNC_PROJECTION_OMEGA_H
#define OPTSUITE_BASE_FUNC_PROJECTION_OMEGA_H

#include "OptSuite/Base/func/base.h"

namespace OptSuite { namespace Composite { namespace Model {
    class MatComp;
}}}

namespace OptSuite { namespace Base {
    template<typename dtype = Scalar>
    class ProjectionOmega : public FuncGrad<dtype> {
        friend OptSuite::Composite::Model::MatComp;
        using typename FuncGrad<dtype>::mat_t;
        using spmat_t = Eigen::SparseMatrix<dtype, ColMajor, SparseIndex>;
        using typename FuncGrad<dtype>::var_t;
        using fmat_t = FactorizedMat<dtype>;

        template <typename xtype>
        void projection(const xtype&);

        inline dtype get_xij(const MatWrapper<dtype>& x, Index  i, Index j){
            const mat_t& mx = x.mat();
            return mx(i, j);
        }
        inline dtype get_xij(const fmat_t& x, Index i, Index j){
            return x.U().col(i).dot(x.V().col(j));
        }

        Scalar compute_fg_impl(const Ref<const mat_t>, MatWrapper<dtype>&, bool = true, bool = false);

        std::vector<SparseIndex> outerIndexPtr;
        std::vector<SparseIndex> innerIndexPtr;
        mat_t b;
        mat_t r, r_mat;
        Scalar fun;

        public:
            // construction by referencing a sparse object
            ProjectionOmega(const Ref<const spmat_t>);

            ~ProjectionOmega() = default;

            // include all eval's in the superclass
            using FuncGrad<dtype>::eval;
            Scalar eval(const var_t&, var_t&, bool, bool) override;

            const mat_t& rvec() const;

    };
}}
#endif
