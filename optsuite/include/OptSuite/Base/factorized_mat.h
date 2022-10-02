/*
 * ===========================================================================
 *
 *       Filename:  factorized_mat.h
 *
 *    Description:
 *
 *        Version:  1.0
 *        Created:  12/13/2020 10:11:22 AM
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  Haoyang Liu (), liuhaoyang@pku.edu.cn
 *   Organization:  BICMR, Peking University
 *
 * ===========================================================================
 */

#ifndef OPTSUITE_BASE_FACTORIZED_MAT_H
#define OPTSUITE_BASE_FACTORIZED_MAT_H

#include <vector>
#include "OptSuite/core_n.h"
#include "OptSuite/Base/fwddecl.h"
#include "OptSuite/Base/variable.h"

namespace OptSuite { namespace Base {
    template<typename dtype>
    class FactorizedMat : public Variable<dtype> {
        using typename Variable<dtype>::mat_t;
        using typename Variable<dtype>::spmat_t;
        mat_t U_; // of k x m
        mat_t V_; // of k x n

        mutable mat_t tmpV;

        public:
            FactorizedMat() = default;
            FactorizedMat(Index m, Index n, Index k){
                U_.resize(k, m);
                V_.resize(k, n);
            }
            FactorizedMat(const Ref<const mat_t> UU, const Ref<const mat_t> VV){
                OPTSUITE_ASSERT(UU.rows() == VV.rows());
                U_ = UU;
                V_ = VV;
            }

            void assign(const Variable<dtype>& other) override {
                const auto* other_ptr = dynamic_cast<const FactorizedMat*>(&other);
                OPTSUITE_ASSERT(other_ptr);
                *this = *other_ptr;
            }

            inline
            void resize(Index m, Index n, Index k){
                U_.resize(k, m);
                V_.resize(k, n);
            }

            inline
            void set_UV(const Ref<const mat_t> UU, const Ref<const mat_t> VV){
                OPTSUITE_ASSERT(UU.rows() == VV.rows());
                U_ = UU;
                V_ = VV;
            }

            inline Index rows() const { return U_.cols(); }
            inline Index cols() const { return V_.cols(); }
            inline Index rank() const { return U_.rows(); }

            inline const mat_t& U() const { return U_; }
            inline const mat_t& V() const { return V_; }
            inline mat_t mat() const { return U_.adjoint() * V_; }

            dtype dot(const Variable<dtype>&) const override;
            dtype dot(const FactorizedMat&) const;
            dtype dot(const Ref<const spmat_t>&) const;
            dtype dot(const Ref<const mat_t>&) const;

            inline void set_zero_like(const Variable<dtype>& other) override {
                const auto* other_ptr = dynamic_cast<const FactorizedMat*>(&other);
                OPTSUITE_ASSERT(other_ptr);
                set_zero_like(*other_ptr);
            }

            inline void set_zero_like(const FactorizedMat& other){
                this->resize(other.rows(), other.cols(), other.rank());
            }

            inline std::string to_string() const override { return "FactorizedMat"; }
    };

}}

#endif
