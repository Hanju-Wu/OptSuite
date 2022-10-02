#ifndef SSNSDP_MAPPING_H
#define SSNSDP_MAPPING_H

#include "core.h"
#include "svec.h"
#include "smat.h"
#include "cell_array.h"
#include "buffer.h"
#include "mkl_sparse.h"

namespace ssnsdp {

// AXfun series of functions only use upper part of X

    template <typename DerivedA, typename DerivedB, typename DerivedOut>
    SSNSDP_STRONG_INLINE
    void AXfun_accumulate_sdp_internal(
        const SparseMatrixBase<DerivedA>& At_,
        const EigenBase<DerivedB>& X_,
        MatrixBase<DerivedOut> const& AX_)
    {
        EIGEN_STATIC_ASSERT_VECTOR_ONLY(DerivedOut);
        const DerivedA& At = At_.derived();
        const DerivedB& X = X_.derived();
        DerivedOut& AX = AX_.const_cast_derived();
        auto vec_x = allocate_vector(At.rows());
        svec_dense_impl(X, vec_x);
        AX.noalias() += At.adjoint() * vec_x;
        deactivate_buffer();
    }

    template <typename DerivedA, typename DerivedB, typename DerivedOut>
    SSNSDP_STRONG_INLINE
    void AXfun_accumulate_linear_internal(
        const SparseMatrixBase<DerivedA>& At_,
        const MatrixBase<DerivedB>& X_,
        MatrixBase<DerivedOut> const& AX_)
    {
        EIGEN_STATIC_ASSERT_VECTOR_ONLY(DerivedOut);
        const DerivedA& At = At_.derived();
        const DerivedB& X = X_.derived();
        DerivedOut& AX = AX_.const_cast_derived();
        AX.noalias() += At.adjoint() * X;
    }

    template <typename DerivedA, typename DerivedB, typename DerivedOut>
    SSNSDP_STRONG_INLINE
    void AXfun_accumulate_linear_internal(
        const SparseMatrixBase<DerivedA>& At_,
        const SparseMatrixBase<DerivedB>& X_,
        MatrixBase<DerivedOut> const& AX_)
    {
        EIGEN_STATIC_ASSERT_VECTOR_ONLY(DerivedOut);
        const DerivedA& At = At_.derived();
        const DerivedB& X = X_.derived();
        DerivedOut& AX = AX_.const_cast_derived();
        AX.noalias() += VectorX(At.adjoint() * X);
    }

#ifdef SSNSDP_MKL_SPARSE_ENABLED
    template <typename DerivedB, typename DerivedOut>
    SSNSDP_STRONG_INLINE
    void AXfun_accumulate_sdp_internal(
        const MKLSparse& At,
        const EigenBase<DerivedB>& X_,
        MatrixBase<DerivedOut> const& AX_)
    {
        EIGEN_STATIC_ASSERT_VECTOR_ONLY(DerivedOut);
        const DerivedB& X = X_.derived();
        DerivedOut& AX = AX_.const_cast_derived();
        auto vec_x = allocate_vector(At.rows());
        svec_dense_impl(X, vec_x);
        At.multiply_impl<MKLOperator::Transpose>(vec_x, AX, One);
        deactivate_buffer();
    }

    template <typename DerivedB, typename DerivedOut>
    SSNSDP_STRONG_INLINE
    void AXfun_accumulate_linear_internal(
        const MKLSparse& At,
        const MatrixBase<DerivedB>& X_,
        MatrixBase<DerivedOut> const& AX_)
    {
        EIGEN_STATIC_ASSERT_VECTOR_ONLY(DerivedOut);
        const DerivedB& X = X_.derived();
        DerivedOut& AX = AX_.const_cast_derived();
        SSNSDP_ASSUME(blk.is_linear());
        At.multiply_impl<MKLOperator::Transpose>(X, AX, One);
    }

    template <typename DerivedB, typename DerivedOut>
    SSNSDP_STRONG_INLINE
    void AXfun_accumulate_linear_internal(
        const MKLSparse& At,
        const SparseMatrixBase<DerivedB>& X,
        MatrixBase<DerivedOut> const& AX)
    {
        AXfun_accumulate_linear_internal(At.to_eigen(), X, AX);
    }
#endif

    template <typename SparseType, typename DerivedB, typename DerivedOut>
    SSNSDP_STRONG_INLINE
    void AXfun_accumulate(
        const BlockSpec& blk,
        const SparseType& At,
        const EigenBase<DerivedB>& X_,
        MatrixBase<DerivedOut> const& AX_)
    {
        EIGEN_STATIC_ASSERT_VECTOR_ONLY(DerivedOut);
        const DerivedB& X = X_.derived();
        DerivedOut& AX = AX_.const_cast_derived();
        if (blk.is_sdp()) {
            AXfun_accumulate_sdp_internal(At, X, AX);
        } else {
            AXfun_accumulate_linear_internal(At, X, AX);
        }
    }

    template <typename AtType, typename XType, typename OutType>
    inline void AXfun_impl(
        const CellArray<BlockSpec>& blk,
        const CellArray<AtType>& At,
        const CellArray<XType>& X,
        MatrixBase<OutType> const& AX_)
    {
        EIGEN_STATIC_ASSERT_VECTOR_ONLY(OutType);
        OutType& AX = AX_.const_cast_derived();
        SSNSDP_ASSERT(AX.size() == At[0].cols());
        AX.setZero();
        for (Index p = 0; p < blk.size(); ++p) {
            AXfun_accumulate(blk[p], At[p], X[p], AX);
        }
    }

    template <typename AtType, typename XType>
    inline VectorX AXfun(
        const CellArray<BlockSpec>& blk,
        const CellArray<AtType>& At,
        const CellArray<XType>& X)
    {
        const Size m = At[0].cols();
        VectorX AX = VectorX::Zero(m);
        AXfun_impl(blk, At, X, AX);
        return AX;
    }

    template <bool IsUpper, typename DerivedA, typename DerivedB, typename DerivedOut>
    SSNSDP_STRONG_INLINE
    void Atyfun_internal_impl(
        const BlockSpec& blk,
        const SparseMatrixBase<DerivedA>& At,
        const MatrixBase<DerivedB>& y,
        MatrixBase<DerivedOut> const& X_)
    {
        EIGEN_STATIC_ASSERT_VECTOR_ONLY(DerivedB);
        DerivedOut& X = X_.const_cast_derived();
        if (blk.is_sdp()) {
            auto Aty = allocate_vector(At.rows());
            Aty.noalias() = At.derived() * y.derived();
            if constexpr (IsUpper) {
                smat_upper_impl(blk, Aty, X);
            } else {
                smat_impl(blk, Aty, X);
            }
            deactivate_buffer();
        } else {
            SSNSDP_ASSUME(blk.is_linear());
            X.noalias() = At.derived() * y.derived();
        }
    }

#ifdef SSNSDP_MKL_SPARSE_ENABLED
    template <bool IsUpper, typename DerivedB, typename DerivedOut>
    SSNSDP_STRONG_INLINE
    void Atyfun_internal_impl(
        const BlockSpec& blk,
        const MKLSparse& At,
        const MatrixBase<DerivedB>& y,
        MatrixBase<DerivedOut> const& X_)
    {
        EIGEN_STATIC_ASSERT_VECTOR_ONLY(DerivedB);
        DerivedOut& X = X_.const_cast_derived();
        if (blk.is_sdp()) {
            auto Aty = allocate_vector(At.rows());
            At.multiply_impl(y.derived(), Aty);
            if constexpr (IsUpper) {
                smat_upper_impl(blk, Aty, X);
            } else {
                smat_impl(blk, Aty, X);
            }
            deactivate_buffer();
        } else {
            SSNSDP_ASSUME(blk.is_linear());
            At.multiply_impl(y.derived(), X);
        }
    }
#endif

    template <typename SparseType, typename DerivedB, typename DerivedOut>
    SSNSDP_STRONG_INLINE
    void Atyfun_impl(
        const BlockSpec& blk,
        const SparseType& At,
        const MatrixBase<DerivedB>& y,
        MatrixBase<DerivedOut> const& X_)
    {
        Atyfun_internal_impl<false>(blk, At, y, X_);
    }

    template <typename SparseType, typename DerivedB, typename DerivedOut>
    SSNSDP_STRONG_INLINE
    void Atyfun_upper_impl(
        const BlockSpec& blk,
        const SparseType& At,
        const MatrixBase<DerivedB>& y,
        MatrixBase<DerivedOut> const& X_)
    {
        Atyfun_internal_impl<true>(blk, At, y, X_);
    }

    template <typename SparseType, typename DerivedB>
    SSNSDP_STRONG_INLINE
    MatrixX Atyfun(
        const BlockSpec& blk,
        const SparseType& At,
        const MatrixBase<DerivedB>& y)
    {
        const Size cols = blk.is_sdp() ? blk.n : 1;
        MatrixX X(blk.n, cols);
        Atyfun_impl(blk, At, y, X);
        return X;
    }

    template <typename Derived, typename SparseType>
    inline void Atyfun_impl(
        const CellArray<BlockSpec>& blk,
        const CellArray<SparseType>& At,
        const MatrixBase<Derived>& y,
        CellArray<MatrixX>& Q)
    {
        EIGEN_STATIC_ASSERT_VECTOR_ONLY(Derived);
        const Size numblk = blk.size();
        SSNSDP_ASSERT(Q.size() == numblk);
        for (Index p = 0; p < numblk; ++p) {
            Atyfun_impl(blk[p], At[p], y, Q[p]);
        }
    }

    template <typename Derived, typename SparseType>
    inline CellArray<MatrixX> Atyfun(
        const CellArray<BlockSpec>& blk,
        const CellArray<SparseType>& At,
        const MatrixBase<Derived>& y)
    {
        const Size numblk = blk.size();
        CellArray<MatrixX> Q(numblk);
        for (Index p = 0; p < numblk; ++p) {
            Q[p].resize(blk[p].n, blk[p].n);
            Atyfun_impl(blk[p], At[p], y, Q[p]);
        }
        return Q;
    }
}

#endif
