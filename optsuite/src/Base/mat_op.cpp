/*
 * ===========================================================================
 *
 *       Filename:  mat_op.cpp
 *
 *    Description:  source for mat_op
 *
 *        Version:  1.0
 *        Created:  10/14/2021 07:07:00 PM
 *       Revision:  none
 *       Compiler:  g++
 *
 *         Author:  Haoyang Liu (), liuhaoyang@pku.edu.cn
 *   Organization:  BICMR, Peking University
 *
 * ===========================================================================
 */

#include "OptSuite/Base/mat_op.h"
#include "OptSuite/Base/mat_wrapper.h"
#include "OptSuite/Base/spmat_wrapper.h"
#include "OptSuite/Base/factorized_mat.h"

namespace OptSuite {
    namespace Base {
        template<typename dtype>
        FactorizePSpMatOp<dtype>::FactorizePSpMatOp(const FactorizedMat<dtype> &mf, Scalar t,
                                                    const SpMatWrapper<dtype> &ms)
                : MatOp<dtype>(mf.rows(), mf.cols()), mat_F(&mf), mat_S(&ms), tau(t) {}

        template<typename dtype>
        void FactorizePSpMatOp<dtype>::apply(const Ref<const mat_t> &in, Ref<mat_t> out) const {
            OPTSUITE_ASSERT(mat_F && mat_S);
            // (U'V + tauS)X = U'(VX) + tau (SX)
            if (tau == 0_s)
                out.noalias() = mat_F->U().transpose() * (mat_F->V() * in);
            else {
                tmp.resize(mat_S->spmat().rows(), 1);
                mat_S->multiply(in, tmp);
                out.noalias() = mat_F->U().transpose() * (mat_F->V() * in) +
                                tau * tmp;
            }
        }

        template<typename dtype>
        void FactorizePSpMatOp<dtype>::apply_transpose(const Ref<const mat_t> &in, Ref<mat_t> out) const {
            OPTSUITE_ASSERT(mat_F && mat_S);
            // (U'V + tauS)'X = V'(UX) + tau (S'X)
            if (tau == 0_s)
                out.noalias() = mat_F->V().transpose() * (mat_F->U() * in);
            else {
                tmpT.resize(mat_S->spmat().cols(), 1);
                mat_S->multiplyT(in, tmpT);
                out.noalias() = mat_F->V().transpose() * (mat_F->U() * in) +
                                tau * tmpT;
            }
        }

        template<typename dtype>
        FactorizePMatOp<dtype>::FactorizePMatOp(const FactorizedMat<dtype> &mf, Scalar t, const MatWrapper<dtype> &ms)
                : MatOp<dtype>(mf.rows(), mf.cols()), mat_F(&mf), mat_D(&ms), tau(t) {}

        template<typename dtype>
        void FactorizePMatOp<dtype>::apply(const Ref<const mat_t> &in, Ref<mat_t> out) const {
            OPTSUITE_ASSERT(mat_F && mat_D);
            // (U'V + tauS)X = U'(VX) + tau (SX)
            if (tau == 0_s)
                out.noalias() = mat_F->U().transpose() * (mat_F->V() * in);
            else
                out.noalias() = mat_F->U().transpose() * (mat_F->V() * in) +
                                tau * (mat_D->mat() * in);
        }

        template<typename dtype>
        void FactorizePMatOp<dtype>::apply_transpose(const Ref<const mat_t> &in, Ref<mat_t> out) const {
            OPTSUITE_ASSERT(mat_F && mat_D);
            // (U'V + tauS)'X = V'(UX) + tau (S'X)
            if (tau == 0_s)
                out.noalias() = mat_F->V().transpose() * (mat_F->U() * in);
            else
                out.noalias() = mat_F->V().transpose() * (mat_F->U() * in) +
                                tau * (mat_D->mat().transpose() * in);
        }

        template class FactorizePMatOp<Scalar>;
        template class FactorizePSpMatOp<Scalar>;
    }
}
