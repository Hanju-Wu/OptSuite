// svec(Matrix X)
// Input: X: sparse or dense matrix. Only upper part of X is used
// Output: A vector `v' with the same sparsity type as X

// svec_dense(MatrixX X)
// same as svec except that output type is always dense.

#ifndef SSNSDP_SVEC_H
#define SSNSDP_SVEC_H

#include "core.h"

namespace ssnsdp {
    template <typename Derived, typename DerivedOut>
    void svec_dense_impl(const MatrixBase<Derived>& X_, MatrixBase<DerivedOut> const& v_) {
        EIGEN_STATIC_ASSERT_VECTOR_ONLY(DerivedOut);
        const Derived& X = X_.derived();
        DerivedOut& v = v_.const_cast_derived();
        const Size n = X.rows();
        SSNSDP_ASSERT(n > 0);
        SSNSDP_ASSERT_MSG(n == X.cols(), "Matrix should be square");
        SSNSDP_ASSERT_MSG(v.size() == n * (n + 1) / 2, "Wrong vector size");

        v(0) = X(0, 0);
        Index index = 1;
        for (Index j = 1; j < n; ++j) {
            // let compiler do the vectorization
            for (Index i = 0; i < j; ++i) {
                v(index) = X(i, j) * Sqrt2;
                ++index;
            }
            v(index) = X(j, j);
            ++index;
        }
    }

    // svec for sparse matrix with dense return type
    template <typename Derived, typename DerivedOut>
    VectorX svec_dense_impl(const SparseMatrixBase<Derived>& X_, MatrixBase<DerivedOut> const& v_) {
        EIGEN_STATIC_ASSERT_VECTOR_ONLY(DerivedOut);
        const Derived& X = X_.derived();
        DerivedOut& v = v_.const_cast_derived();
        const Size n = X.rows();
        SSNSDP_ASSERT_MSG(n == X.cols(), "Matrix should be square");
        v.setZero();
        for (Index j = 0; j < X.outerSize(); ++j) {
            const Index j_accu = j * (j + 1) / 2;
            for (typename Derived::InnerIterator it(X, j); it; ++it) {
                const Index i = it.row();
                SSNSDP_ASSERT_MSG(j == it.col(), "CSC storage is assumed");
                if (i < j) {
                    v(i + j_accu) = it.value() * Sqrt2;
                } else if (i == j) {
                    v(i + j_accu) = it.value();
                }
            }
        }
        return v;
    }

    template <typename Derived>
    SSNSDP_STRONG_INLINE
    VectorX svec_dense(const EigenBase<Derived>& X_) {
        const Derived& X = X_.derived();
        const Size n = X.rows();
        VectorX v(n * (n + 1) / 2);
        svec_dense_impl(X, v);
        return v;
    }

    // svec for sparse matrix when nnz of the result vector is known
    template <typename Derived>
    SparseVector svec_sparse(const SparseMatrixBase<Derived>& X_, const Size reserve_size) {
        const Derived& X = X_.derived();
        const Size n = X.rows();
        SSNSDP_ASSERT_MSG(n == X.cols(), "Matrix should be square");
        SparseVector v(n * (n + 1) / 2);

        v.reserve(reserve_size);
        for (Index j = 0; j < X.outerSize(); ++j) {
            const Index j_accu = j * (j + 1) / 2;
            for (typename Derived::InnerIterator it(X, j); it; ++it) {
                const Index i = it.row();
                SSNSDP_ASSERT_MSG(j == it.col(), "CSC storage is assumed");
                if (i < j) {
                    v.insert(i + j_accu) = it.value() * Sqrt2;
                } else if (i == j) {
                    v.insert(i + j_accu) = it.value();
                }
            }
        }
        return v;
    }

    // svec for sparse matrix.
    // Should only be needed during initialization.
    template <typename Derived>
    SparseVector svec_sparse(const SparseMatrixBase<Derived>& X_) {
        const Derived& X = X_.derived();
        const Size nnz = X.nonZeros();
        const Size n = X.rows();
        // If nnz > n, at most n entries on diagonal
        // In that (worst) case, (n + nnz) / 2 entries are required
        return svec_sparse(X, nnz <= n ? n : (nnz + n) / 2);
    }

    template <typename Derived>
    SSNSDP_STRONG_INLINE
    VectorX svec(const MatrixBase<Derived>& X_) {
        return svec_dense(X_.derived());
    }

    template <typename Derived>
    SSNSDP_STRONG_INLINE
    SparseVector svec(const SparseMatrixBase<Derived>& X_) {
        return svec_sparse(X_.derived());
    }
}
#endif
