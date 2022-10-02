#ifndef SSNSDP_MKL_WRAPPER_H
#define SSNSDP_MKL_WRAPPER_H

#include "core.h"

namespace ssnsdp {

#ifdef SSNSDP_USE_MKL

    constexpr static auto CblasStorageOrder =
        MatrixX::IsRowMajor ? CblasRowMajor : CblasColMajor;

    enum class MKLOperator {
        NonTranspose = CblasNoTrans,
        Transpose = CblasTrans,
        ConjugateTranspose = CblasConjTrans
    };

    template <MKLOperator op>
    constexpr CBLAS_TRANSPOSE MKLToCblasTransposed = (op == MKLOperator::NonTranspose) ?
        CblasTrans : CblasNoTrans;

    template <MKLOperator op>
    constexpr CBLAS_TRANSPOSE MKLToCblas = static_cast<CBLAS_TRANSPOSE>(op);

    template <typename... Args>
    SSNSDP_STRONG_INLINE
    auto cblas_gemm(Args&&... args) {
        if constexpr (ScalarIsDouble) {
            return cblas_dgemm(std::forward<Args>(args)...);
        } else if constexpr (ScalarIsFloat) {
            return cblas_sgemm(std::forward<Args>(args)...);
        } else {
            SSNSDP_UNREACHABLE();
        }
    }

    template <typename... Args>
    SSNSDP_STRONG_INLINE
    auto cblas_zgemm(Args&&... args) {
        if constexpr (ScalarIsDouble) {
            return cblas_zgemm(std::forward<Args>(args)...);
        } else if constexpr (ScalarIsFloat) {
            return cblas_cgemm(std::forward<Args>(args)...);
        } else {
            SSNSDP_UNREACHABLE();
        }
    }

    template <typename... Args>
    SSNSDP_STRONG_INLINE
    auto cblas_symm(Args&&... args) {
        if constexpr (ScalarIsDouble) {
            return cblas_dsymm(std::forward<Args>(args)...);
        } else if constexpr (ScalarIsFloat) {
            return cblas_ssymm(std::forward<Args>(args)...);
        } else {
            SSNSDP_UNREACHABLE();
        }
    }

    template <typename... Args>
    SSNSDP_STRONG_INLINE
    auto cblas_hemm(Args&&... args) {
        if constexpr (ScalarIsDouble) {
            return cblas_zhemm(std::forward<Args>(args)...);
        } else if constexpr (ScalarIsFloat) {
            return cblas_chemm(std::forward<Args>(args)...);
        } else {
            SSNSDP_UNREACHABLE();
        }
    }

    template <typename... Args>
    SSNSDP_STRONG_INLINE
    auto LAPACKE_syev(Args&&... args) {
        if constexpr (ScalarIsDouble) {
            return LAPACKE_dsyev(std::forward<Args>(args)...);
        } else if constexpr (ScalarIsFloat) {
            return LAPACKE_ssyev(std::forward<Args>(args)...);
        } else {
            SSNSDP_UNREACHABLE();
        }
    }

    template <typename... Args>
    SSNSDP_STRONG_INLINE
    auto LAPACKE_syevd(Args&&... args) {
        if constexpr (ScalarIsDouble) {
            return LAPACKE_dsyevd(std::forward<Args>(args)...);
        } else if constexpr (ScalarIsFloat) {
            return LAPACKE_ssyevd(std::forward<Args>(args)...);
        } else {
            SSNSDP_UNREACHABLE();
        }
    }

    template <typename... Args>
    SSNSDP_STRONG_INLINE
    auto LAPACKE_heev(Args&&... args) {
        if constexpr (ScalarIsDouble) {
            return LAPACKE_zheev(std::forward<Args>(args)...);
        } else if constexpr (ScalarIsFloat) {
            return LAPACKE_cheev(std::forward<Args>(args)...);
        } else {
            SSNSDP_UNREACHABLE();
        }
    }

    SSNSDP_STRONG_INLINE
    auto as_lapacke_complex(ComplexScalar* p) {
        if constexpr (ScalarIsDouble) {
            return reinterpret_cast<lapack_complex_double*>(p);
        } else if constexpr (ScalarIsFloat) {
            return reinterpret_cast<lapack_complex_float*>(p);
        } else {
            SSNSDP_UNREACHABLE();
        }
    }

    template <typename... Args>
    SSNSDP_STRONG_INLINE
    auto mkl_sparse_mm(Args&&... args) {
        if constexpr (ScalarIsDouble) {
            return mkl_sparse_d_mm(std::forward<Args>(args)...);
        } else if constexpr (ScalarIsFloat) {
            return mkl_sparse_s_mm(std::forward<Args>(args)...);
        } else {
            SSNSDP_UNREACHABLE();
        }
    }

    template <typename... Args>
    SSNSDP_STRONG_INLINE
    auto mkl_sparse_mv(Args&&... args) {
        if constexpr (ScalarIsDouble) {
            return mkl_sparse_d_mv(std::forward<Args>(args)...);
        } else if constexpr (ScalarIsFloat) {
            return mkl_sparse_s_mv(std::forward<Args>(args)...);
        } else {
            SSNSDP_UNREACHABLE();
        }
    }

    template <typename... Args>
    SSNSDP_STRONG_INLINE
    auto mkl_sparse_create_csr(Args&&... args) {
        if constexpr (ScalarIsDouble) {
            return mkl_sparse_d_create_csr(std::forward<Args>(args)...);
        } else if constexpr (ScalarIsFloat) {
            return mkl_sparse_s_create_csr(std::forward<Args>(args)...);
        } else {
            SSNSDP_UNREACHABLE();
        }
    }

    template <typename... Args>
    SSNSDP_STRONG_INLINE
    auto mkl_sparse_create_coo(Args&&... args) {
        if constexpr (ScalarIsDouble) {
            return mkl_sparse_d_create_coo(std::forward<Args>(args)...);
        } else if constexpr (ScalarIsFloat) {
            return mkl_sparse_s_create_coo(std::forward<Args>(args)...);
        } else {
            SSNSDP_UNREACHABLE();
        }
    }
#endif // #ifdef SSNSDP_USE_MKL
}

#endif
