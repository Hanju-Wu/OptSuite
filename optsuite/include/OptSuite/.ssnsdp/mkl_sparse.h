#ifndef SSNSDP_MKL_SPARSE_H
#define SSNSDP_MKL_SPARSE_H

#include "core.h"
#include "mkl_wrapper.h"
#include "log.h"

namespace ssnsdp {
#ifdef SSNSDP_MKL_SPARSE_ENABLED
    static_assert((ScalarIsFloat || ScalarIsDouble) && SparseIndexIsInt);
    constexpr bool UseMKLSparse = true;
#else
    constexpr bool UseMKLSparse = false;
#endif

#ifdef SSNSDP_MKL_SPARSE_ENABLED
    class MKLSparse {
    public:

        MKLSparse(): m_At_mkl(nullptr), m_is_empty(true) {}

        MKLSparse(const SparseMatrix& A) {
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
        }

        Size rows() const { return m_A.rows(); }
        Size cols() const { return m_A.cols(); }

        MKL_INT mkl_rows() const { return static_cast<MKL_INT>(rows()); }
        MKL_INT mkl_cols() const { return static_cast<MKL_INT>(cols()); }

        const SparseMatrix& to_eigen() const { return m_A; }

        void reset(const SparseMatrix& A) {
            destroy();
            m_A = A;
            m_is_empty = (A.nonZeros() == 0);
            if (!m_is_empty) {
                // const_cast because MKL interface does not have const.
                // A will not be modified.
                SparseMatrix& B = A.const_cast_derived();
                /* A with csc <=> A' with csr */

                sparse_status_t status = mkl_sparse_create_csr(
                    &m_At_mkl, SPARSE_INDEX_BASE_ZERO,
                    mkl_cols(), mkl_rows(),
                    m_A.outerIndexPtr(),
                    m_A.outerIndexPtr() + 1,
                    m_A.innerIndexPtr(),
                    m_A.valuePtr());

#ifdef SSNSDP_INTERNAL_DEBUG
                if (status != SPARSE_STATUS_SUCCESS) {
                    log_error_format("MKL Sparse failed, status code = %d\n", status);
                    SSNSDP_ASSERT(false);
                }
#endif
            }
        }

        void destroy() {
            if (m_At_mkl) {
                mkl_sparse_destroy(m_At_mkl);
                m_At_mkl = nullptr;
            }
        }

        // A * B + beta * X -> X: A.multiply_impl(B, X, beta)
        // A' * B + beta * X -> X: A.multiply_impl<Transpose>(B, X, beta)
        template <MKLOperator op = MKLOperator::NonTranspose,
            typename DerivedRhs,
            typename DerivedOut>
        void multiply_impl(
            const MatrixBase<DerivedRhs>& B_,
            MatrixBase<DerivedOut> const& X_,
            Scalar beta = Zero) const
        {
            // row vector is not accepted
            constexpr bool B_is_vector = (DerivedRhs::ColsAtCompileTime == 1);

            const DerivedRhs& B = B_.derived();

            DerivedOut& X = X_.const_cast_derived();

            auto m = mkl_rows();
            const auto n = static_cast<MKL_INT>(B.cols());
            auto k = mkl_cols();

            if constexpr (op == MKLOperator::Transpose) {
                std::swap(m, k);
            }

            SSNSDP_ASSERT(k == B.rows());

            if (beta != 0) {
                SSNSDP_ASSERT(m == X.rows() && n == X.cols());
            }

            if (m_is_empty) {
                if (beta == Zero) {
                    X.setZero();
                } else {
                    X *= beta;
                }
                return;
            }

            matrix_descr descr;
            descr.type = SPARSE_MATRIX_TYPE_GENERAL;

            // A' is stored in MKL, calculate A*B if operation == SPARSE_OPERATION_TRANSPOSE
            // A'*B if operation == SPARSE_OPERATION_NON_TRANSPOSE */
            constexpr sparse_operation_t mkl_op =
                op == MKLOperator::NonTranspose ?
                SPARSE_OPERATION_TRANSPOSE :
                SPARSE_OPERATION_NON_TRANSPOSE;

            [[maybe_unused]] sparse_status_t status;
            if (B_is_vector || n == 1) {
                status = mkl_sparse_mv(mkl_op, One, m_At_mkl, descr, B.data(), beta, X.data());
            } else {
                status = mkl_sparse_mm(mkl_op, One, m_At_mkl, descr,
                    SPARSE_LAYOUT_COLUMN_MAJOR, B.data(), n, k, beta, X.data(), m);
            }

#ifdef SSNSDP_INTERNAL_DEBUG
            if (status != SPARSE_STATUS_SUCCESS) {
                log_error_format("MKL Sparse failed, status code = %d\n", status);
                SSNSDP_ASSERT(false);
            }
#endif
        }

        template <MKLOperator op = MKLOperator::NonTranspose, typename DerivedRhs>
        Matrix<Scalar, Dynamic, DerivedRhs::ColsAtCompileTime>
        multiply(const MatrixBase<DerivedRhs>& B_) const {
            const DerivedRhs& B = B_.derived();

            const Size m = (op == MKLOperator::NonTranspose ? rows() : cols());
            Matrix<Scalar, Dynamic, DerivedRhs::ColsAtCompileTime> X(m, B.cols());
            // evaluate B, so that mkl can get raw pointers.
            multiply_impl<op>(B.eval(), X);
            return X;
        }

    private:
        SparseMatrix m_A;
        sparse_matrix_t m_At_mkl;

        bool m_is_empty;

        void move_impl(MKLSparse&& other) {
            this.m_A = std::move(other.m_A);
            this.m_At_mkl = std::move(other.m_At_mkl);
            this.m_is_empty = std::move(other.m_is_empty);
            other.m_At_mkl = nullptr;
        }
    };
#endif // #ifdef SSNSDP_MKL_SPARSE_ENABLED
}
#endif
