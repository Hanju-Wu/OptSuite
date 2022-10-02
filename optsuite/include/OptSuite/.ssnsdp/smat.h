
#ifndef SSNSDP_SMAT_H
#define SSNSDP_SMAT_H

#include "core.h"

namespace ssnsdp {
    template <typename Derived, typename DerivedOut>
    SSNSDP_STRONG_INLINE
    void smat_upper_impl(const MatrixBase<Derived>& v_, const Size n, MatrixBase<DerivedOut> const& X_) {
        EIGEN_STATIC_ASSERT_VECTOR_ONLY(Derived);
        DerivedOut& X = X_.const_cast_derived();
        const Derived& v = v_.derived();
        SSNSDP_ASSERT(n > 0);
        SSNSDP_ASSERT(v.size() == n * (n + 1) / 2);
        SSNSDP_ASSERT(X.rows() == n && X.cols() == n);

        X(0, 0) = v(0);
        Index index = 1;
        for (Index j = 1; j < n; ++j) {
            // let compiler do the vectorization
            for (Index i = 0; i < j; ++i) {
                X(i, j) = v(index) * Sqrt1_2;
                ++index;
            }
            X(j, j) = v(index);
            ++index;
        }
    }

    template <typename Derived, typename DerivedOut>
    SSNSDP_STRONG_INLINE
    void smat_impl(const MatrixBase<Derived>& v_, const Size n, MatrixBase<DerivedOut> const& X_) {
        EIGEN_STATIC_ASSERT_VECTOR_ONLY(Derived);
        DerivedOut& X = X_.const_cast_derived();
        const Derived& v = v_.derived();
        SSNSDP_ASSERT(n > 0);
        SSNSDP_ASSERT(v.size() == n * (n + 1) / 2);
        SSNSDP_ASSERT(X.rows() == n && X.cols() == n);

        X(0, 0) = v(0);
        Index index = 1;
        for (Index j = 1; j < n; ++j) {
            X.col(j).head(j) = v.segment(index, j) * Sqrt1_2;
            X.row(j).head(j) = X.col(j).adjoint().head(j);
            index += j;
            X(j, j) = v(index);
            ++index;
        }
    }

    template <typename Derived>
    SSNSDP_STRONG_INLINE
    MatrixX smat(const MatrixBase<Derived>& v, const Size n) {
        MatrixX X(n, n);
        smat_impl(v.derived(), n, X);
        return X;
    }

    // smat for sparse matrix. Probably not needed.
    // template <typename Derived>
    // SparseMatrix smat(const SparseMatrixBase<Derived>& v_, const Size n) {
    //     EIGEN_STATIC_ASSERT_VECTOR_ONLY(Derived);

    //     const Derived &v = v_.derived();
    //     SSNSDP_ASSERT(v.size() == n * (n + 1) / 2);

    //     SparseMatrix X(n, n);
    //     Index jstart = 0;
    //     Index j_accu = 0; // 1 + 2 + ... + (j+1)
    //     TripletList tlist;
    //     tlist.reserve(v.nonZeros());
    //     for (typename Derived::InnerIterator it(v); it; ++it) {
    //         Index j = jstart;
    //         Index r = it.index();
    //         Index i;
    //         while (true) {
    //             SSNSDP_ASSUME(j < n);
    //             i = r - j_accu;
    //             if (i > j) {
    //                 j_accu += j + 1;
    //             }
    //             else {
    //                 break;
    //             }
    //             ++j;
    //         }
    //         jstart = j;

    //         if (i < j) {
    //             tlist.emplace_back(i, j, Sqrt1_2 * it.value());
    //             tlist.emplace_back(j, i, Sqrt1_2 * it.value());
    //         }
    //         else {
    //             SSNSDP_ASSUME(i == j);
    //             tlist.emplace_back(i, j, it.value());
    //         }
    //     }

    //     X.setFromTriplets(tlist.begin(), tlist.end());
    //     return X;
    // }
    template <typename Derived, typename DerivedOut>
    SSNSDP_STRONG_INLINE
    void smat_upper_impl(const BlockSpec& blk, const MatrixBase<Derived>& v, MatrixBase<DerivedOut> const& X) {
        smat_upper_impl(v.derived(), blk.n, X.const_cast_derived());
    }

    template <typename Derived, typename DerivedOut>
    SSNSDP_STRONG_INLINE
    void smat_impl(const BlockSpec& blk, const MatrixBase<Derived>& v, MatrixBase<DerivedOut> const& X) {
        smat_impl(v.derived(), blk.n, X.const_cast_derived());
    }

    template <typename Derived>
    SSNSDP_STRONG_INLINE
    MatrixX smat(const BlockSpec& blk, const MatrixBase<Derived>& v) {
        SSNSDP_ASSERT(blk.is_sdp());
        return smat(v.derived(), blk.n);
    }

    // This function template is for debugging and testing purpose.
    // For normal use, choose smat(blk, v) or smat(v, n)
    template <typename Derived>
    SSNSDP_STRONG_INLINE
    auto smat(const EigenBase<Derived>& v_) {
        const Derived& v = v_.derived();
        Size n = static_cast<int>(sqrt(2 * v.size()));
        return smat(v, n);
    }
}

#endif
