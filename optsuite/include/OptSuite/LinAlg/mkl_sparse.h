#ifndef OPTSUITE_LINALG_MKL_SPARSE_H
#define OPTSUITE_LINALG_MKL_SPARSE_H

#include "OptSuite/core_n.h"

namespace OptSuite { namespace LinAlg {

#ifdef OPTSUITE_USE_MKL
    class MKLSparse {
    public:

        MKLSparse(): m_At_mkl(nullptr), m_is_empty(true) {}

        MKLSparse(const SpMat& A) {
            reset(A);
        }
        ~MKLSparse() {
            destroy();
        }
        MKLSparse(const MKLSparse&) = delete;
        MKLSparse(MKLSparse&& other) {
            move_impl(std::move(other));
        }
        MKLSparse& operator=(const MKLSparse&) = delete;
        MKLSparse& operator=(MKLSparse&& other) {
            destroy();
            move_impl(std::move(other));
            return *this;
        }

        Size rows() const { return m_A->rows(); }
        Size cols() const { return m_A->cols(); }

        MKL_INT mkl_rows() const { return static_cast<MKL_INT>(rows()); }
        MKL_INT mkl_cols() const { return static_cast<MKL_INT>(cols()); }

        template <MulOp op = MulOp::NonTranspose>
        sparse_status_t multiply(const Ref<const Mat> x, Ref<Mat> y,
                MulOp opX = MulOp::NonTranspose, Scalar beta = 0_s) const;

        sparse_status_t reset(const SpMat&);
        const SpMat& to_eigen() const { return *m_A; }
        void destroy();

    private:
        SpMat* m_A;
        sparse_matrix_t m_At_mkl;

        void move_impl(MKLSparse&&);
        bool m_is_empty;

    };
#endif
}}
#endif
