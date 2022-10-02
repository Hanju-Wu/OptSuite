/*
 * ===========================================================================
 *
 *       Filename:  mat_op.h
 *
 *    Description:  
 *
 *        Version:  1.0
 *        Created:  12/13/2020 10:38:44 AM
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  Haoyang Liu (), liuhaoyang@pku.edu.cn
 *   Organization:  BICMR, Peking University
 *
 * ===========================================================================
 */

#ifndef OPTSUITE_BASE_MAT_OP_H
#define OPTSUITE_BASE_MAT_OP_H

#include "OptSuite/core_n.h"

namespace OptSuite { namespace Base {
    template<typename T> class SpMatWrapper;
    template<typename T> class MatWrapper;
    template<typename T> class FactorizedMat;

    template<typename dtype>
    class MatOp {
        using mat_t = Eigen::Matrix<dtype, Dynamic, Dynamic>;
        Index rows_ = 0_i;
        Index cols_ = 0_i;
        public:
            MatOp() = default;
            MatOp(Index m, Index n) : rows_(m), cols_(n) {}

            inline void resize(Index m, Index n){
                rows_ = m;
                cols_ = n;
            }

            inline Index rows() const { return rows_; }
            inline Index cols() const { return cols_; }

            // virtual functions
            virtual void apply(const Ref<const mat_t>&, Ref<mat_t>) const = 0;
            virtual void apply_transpose(const Ref<const mat_t>&, Ref<mat_t>) const = 0;
    };

    template<typename dtype>
    class FactorizePSpMatOp : public MatOp<dtype> {
        using mat_t = Eigen::Matrix<dtype, Dynamic, Dynamic>;
        const FactorizedMat<dtype>* mat_F;
        const SpMatWrapper<dtype>* mat_S;
        Scalar tau;
        mutable mat_t tmp, tmpT;

        public:
            FactorizePSpMatOp() : mat_F(nullptr), mat_S(nullptr), tau(0_s) {}
            FactorizePSpMatOp(Index m, Index n) : MatOp<dtype>(m, n) {}
            FactorizePSpMatOp(const FactorizedMat<dtype>& mf, Scalar t, const SpMatWrapper<dtype>& ms);

            void apply(const Ref<const mat_t>&, Ref<mat_t>) const override;
            void apply_transpose(const Ref<const mat_t>&, Ref<mat_t>) const override;
    };

    template<typename dtype>
    class FactorizePMatOp : public MatOp<dtype> {
        using mat_t = Eigen::Matrix<dtype, Dynamic, Dynamic>;
        const FactorizedMat<dtype>* mat_F;
        const MatWrapper<dtype>* mat_D;
        Scalar tau;

        public:
            FactorizePMatOp() : mat_F(nullptr), mat_D(nullptr), tau(0_s) {}
            FactorizePMatOp(Index m, Index n) : MatOp<dtype>(m, n) {}
            FactorizePMatOp(const FactorizedMat<dtype>& mf, Scalar t, const MatWrapper<dtype>& ms);

            void apply(const Ref<const mat_t>&, Ref<mat_t>) const override;
            void apply_transpose(const Ref<const mat_t>&, Ref<mat_t>) const override;
    };
}}

#endif

