#ifndef SSNSDP_MKL_DENSE_H
#define SSNSDP_MKL_DENSE_H

#include "core.h"
#include "mkl_wrapper.h"

namespace ssnsdp {
    constexpr static bool UseMKLDense = (ScalarIsFloat || ScalarIsDouble) && UseMKL;

    // C := alpha*op(A)*op(B) + beta*C,
    template <MKLOperator opA = MKLOperator::NonTranspose,
              MKLOperator opB = MKLOperator::NonTranspose,
              typename DerivedA, typename DerivedB, typename DerivedC>
    SSNSDP_STRONG_INLINE
    void gemm_impl(
        const MatrixBase<DerivedA>& A_,
        const MatrixBase<DerivedB>& B_,
        MatrixBase<DerivedC> const& C_,
        const Scalar alpha = One,
        const Scalar beta = Zero)
    {
        const DerivedA& A = A_.derived();
        const DerivedB& B = B_.derived();
        DerivedC& C = C_.const_cast_derived();

        static_assert(opA != MKLOperator::ConjugateTranspose &&
                      opB != MKLOperator::ConjugateTranspose);

        MKL_INT m, n, k;
        constexpr bool IsRowMajor = DerivedC::IsRowMajor;
        constexpr CBLAS_LAYOUT layout = IsRowMajor ? CblasRowMajor : CblasColMajor;
        constexpr CBLAS_TRANSPOSE transa =
            DerivedA::IsRowMajor == IsRowMajor ? MKLToCblas<opA> : MKLToCblasTransposed<opA>;
        constexpr CBLAS_TRANSPOSE transb =
            DerivedB::IsRowMajor == IsRowMajor ? MKLToCblas<opB> : MKLToCblasTransposed<opB>;

        if constexpr (opA == MKLOperator::NonTranspose) {
            m = static_cast<MKL_INT>(A.rows());
            k = static_cast<MKL_INT>(A.cols());
        } else {
            m = static_cast<MKL_INT>(A.cols());
            k = static_cast<MKL_INT>(A.rows());
        }

        if constexpr (opB == MKLOperator::NonTranspose) {
            n = static_cast<MKL_INT>(B.cols());
            SSNSDP_ASSERT(k == static_cast<MKL_INT>(B.rows()));
        } else {
            n = static_cast<MKL_INT>(B.rows());
            SSNSDP_ASSERT(k == static_cast<MKL_INT>(B.cols()));
        }

        SSNSDP_ASSERT(m == static_cast<MKL_INT>(C.rows()));
        SSNSDP_ASSERT(n == static_cast<MKL_INT>(C.cols()));

        const auto lda = static_cast<MKL_INT>(A.outerStride());
        const auto ldb = static_cast<MKL_INT>(B.outerStride());
        const auto ldc = static_cast<MKL_INT>(C.outerStride());

        cblas_gemm(layout, transa, transb, m, n, k,
            alpha, A.data(), lda, B.data(), ldb, beta, C.data(), ldc);
    }

    template <typename DerivedA, typename DerivedB, typename DerivedC>
    SSNSDP_STRONG_INLINE
    void gemm_nt_impl(const MatrixBase<DerivedA>& A_,
        const MatrixBase<DerivedB>& B_,
        MatrixBase<DerivedC> const& C_,
        const Scalar alpha = One,
        const Scalar beta = Zero)
    {
        gemm_impl<MKLOperator::NonTranspose, MKLOperator::Transpose>(
            A_.derived(), B_.derived(), C_.const_cast_derived(), alpha, beta);
    }

    template <typename DerivedA, typename DerivedB, typename DerivedC>
    SSNSDP_STRONG_INLINE
    void gemm_tn_impl(const MatrixBase<DerivedA>& A_,
        const MatrixBase<DerivedB>& B_,
        MatrixBase<DerivedC> const& C_,
        const Scalar alpha = One,
        const Scalar beta = Zero)
    {
        gemm_impl<MKLOperator::Transpose, MKLOperator::NonTranspose>(
            A_.derived(), B_.derived(), C_.const_cast_derived(), alpha, beta);
    }

    template <typename DerivedA, typename DerivedB, typename DerivedC>
    SSNSDP_STRONG_INLINE
    void gemm_tt_impl(const MatrixBase<DerivedA>& A_,
        const MatrixBase<DerivedB>& B_,
        MatrixBase<DerivedC> const& C_,
        const Scalar alpha = One,
        const Scalar beta = Zero)
    {
        gemm_impl<MKLOperator::Transpose, MKLOperator::Transpose>(
            A_.derived(), B_.derived(), C_.const_cast_derived(), alpha, beta);
    }
}
#endif
