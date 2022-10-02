/*
 * ===========================================================================
 *
 *       Filename:  mkl_sparse.cpp
 *
 *    Description:  
 *
 *        Version:  1.0
 *        Created:  02/04/2021 03:13:03 PM
 *       Revision:  none
 *       Compiler:  g++
 *
 *         Author:  Haoyang Liu (), liuhaoyang@pku.edu.cn
 *   Organization:  BICMR, Peking University
 *      Copyright:  Copyright (c) 2021, Haoyang Liu
 *
 * ===========================================================================
 */

#include <iostream>
#include "OptSuite/LinAlg/mkl_sparse.h"

namespace OptSuite { namespace LinAlg {
#ifdef OPTSUITE_USE_MKL
    sparse_status_t MKLSparse::reset(const SpMat& A) {
        destroy();
        // const_cast because MKL interface does not have const.
        // A will not be modified.
        m_A = const_cast<SpMat*>(&A);
        m_is_empty = (A.nonZeros() == 0);
        if (!m_is_empty) {
            /* A with csc <=> A' with csr */
#if OPTSUITE_SCALAR_TYPE == 0
            sparse_status_t status = mkl_sparse_d_create_csr(
#else
            sparse_status_t status = mkl_sparse_s_create_csr(
#endif
                    &m_At_mkl, SPARSE_INDEX_BASE_ZERO,
                    mkl_cols(), mkl_rows(),
                    m_A->outerIndexPtr(),
                    m_A->outerIndexPtr() + 1,
                    m_A->innerIndexPtr(),
                    m_A->valuePtr());

#ifndef NDEBUG
            if (status != SPARSE_STATUS_SUCCESS) {
                OPTSUITE_ASSERT(false && "mkl_sparse_create_csr failed.");
            }
#endif
            return status;
        }
        return SPARSE_STATUS_SUCCESS;
    }

    void MKLSparse::destroy() {
        if (m_At_mkl) {
            mkl_sparse_destroy(m_At_mkl);
            m_At_mkl = nullptr;
        }
    }

    // A * B + beta * X -> X: A.multiply(B, X, NonTranspose, beta)
    // A' * B + beta * X -> X: A.multiply<Transpose>(B, X, NonTranspose, beta)
    // A * B' + beta * X' -> X': A.multiply(B, X, Transpose, beta)
    // A' * B' + beta * X' -> X': A.multiply<Transpose>(B, X, Transpose, beta)
    template <MulOp op>
    sparse_status_t MKLSparse::multiply(const Ref<const Mat> B, Ref<Mat> X,
            MulOp opX, Scalar beta) const {

        auto m = mkl_rows();
        auto k = mkl_cols();
        auto rB = static_cast<MKL_INT>(B.rows());
        auto cB = static_cast<MKL_INT>(B.cols());
        auto rX = static_cast<MKL_INT>(X.rows());
        auto cX = static_cast<MKL_INT>(X.cols());
        const auto ldb = static_cast<MKL_INT>(B.outerStride());
        const auto ldx = static_cast<MKL_INT>(X.outerStride());

        if (op == MulOp::Transpose) {
            std::swap(m, k);
        }

        if (opX == MulOp::Transpose) {
            std::swap(rB, cB);
            std::swap(rX, cX);
        }

        OPTSUITE_ASSERT(k == rB && "Matrix dimension must agree.");
        OPTSUITE_ASSERT(m == rX && cB == cX && "Output matrix dimension must agree.");

        if (m_is_empty) {
            if (beta == 0_s) {
                X.setZero();
            } else {
                X *= beta;
            }
            return SPARSE_STATUS_SUCCESS;
        }

        matrix_descr descr;
        descr.type = SPARSE_MATRIX_TYPE_GENERAL;

        // A' is stored in MKL, calculate A*B if operation == SPARSE_OPERATION_TRANSPOSE
        // A'*B if operation == SPARSE_OPERATION_NON_TRANSPOSE */
        constexpr sparse_operation_t mkl_op =
            op == MulOp::NonTranspose ?
            SPARSE_OPERATION_TRANSPOSE :
            SPARSE_OPERATION_NON_TRANSPOSE;

        sparse_status_t status;
        if (cB == 1) {
#if OPTSUITE_SCALAR_TOKEN == 0
            status = mkl_sparse_d_mv(mkl_op, 1_s, m_At_mkl, descr, B.data(), beta, X.data());
#else
            status = mkl_sparse_s_mv(mkl_op, 1_s, m_At_mkl, descr, B.data(), beta, X.data());
#endif
        } else {
            sparse_layout_t layout = opX == MulOp::Transpose ?
                SPARSE_LAYOUT_ROW_MAJOR :
                SPARSE_LAYOUT_COLUMN_MAJOR;
#if OPTSUITE_SCALAR_TOKEN == 0
            status = mkl_sparse_d_mm(mkl_op, 1_s, m_At_mkl, descr,
                    layout, B.data(), cB, ldb, beta, X.data(), ldx);
#else
            status = mkl_sparse_s_mm(mkl_op, 1_s, m_At_mkl, descr,
                    layout, B.data(), cB, ldb, beta, X.data(), ldx);
#endif
        }

#ifndef NDEBUG
        if (status != SPARSE_STATUS_SUCCESS) {
            if (cB == 1) OPTSUITE_ASSERT(false && "mkl_sparse_?_mv failed");
            else OPTSUITE_ASSERT(false && "mkl_sparse_?_mm failed");
        }
#endif
        return status;
    }
    void MKLSparse::move_impl(MKLSparse&& other) {
        this->m_A = std::move(other.m_A);
        this->m_At_mkl = std::move(other.m_At_mkl);
        this->m_is_empty = std::move(other.m_is_empty);
        other.m_At_mkl = nullptr;
    }

    template sparse_status_t MKLSparse::multiply<>(const Ref<const Mat>, Ref<Mat>, MulOp, Scalar) const;
    template sparse_status_t MKLSparse::multiply<MulOp::Transpose>(const Ref<const Mat>, Ref<Mat>, MulOp, Scalar) const;
#endif // of OPTSUITE_USE_MKL
}}
