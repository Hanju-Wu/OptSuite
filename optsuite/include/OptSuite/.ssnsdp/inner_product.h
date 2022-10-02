#ifndef SSNSDP_INNER_PRODUCT_H
#define SSNSDP_INNER_PRODUCT_H

#include "core.h"
#include "cell_array.h"

namespace ssnsdp {
    SSNSDP_STRONG_INLINE
    Scalar inner_product(const Scalar a, const Scalar b) {
        return a * b;
    }

    SSNSDP_STRONG_INLINE
    ComplexScalar inner_product(const ComplexScalar a, const ComplexScalar b) {
        return conj(a) * b;
    }

    template <typename DerivedA, typename DerivedB>
    SSNSDP_STRONG_INLINE
    auto inner_product(const EigenBase<DerivedA>& a_, const EigenBase<DerivedB>& b_) {
        const DerivedA& a = a_.derived();
        const DerivedB& b = b_.derived();

        EIGEN_STATIC_ASSERT_SAME_MATRIX_SIZE(DerivedA, DerivedB);

        // note: this test is not general. Simply test whether it is ComplexScalar
        constexpr bool is_complex = std::is_same_v<typename DerivedA::Scalar, ComplexScalar> &&
            std::is_same_v<typename DerivedB::Scalar, ComplexScalar>;
        // static assert ensures same size, so there is no need to test DerivedB
        constexpr bool is_vector = DerivedA::IsVectorAtCompileTime;

        if constexpr (is_vector) {
            // Eigen built-in implementation
            return a.dot(b);
        }

        SSNSDP_ASSERT(a.rows() == b.rows() && a.cols() == b.cols());
        // matrix case
        if constexpr (is_complex) {
            return a.conjugate().cwiseProduct(b).sum();
        } else {
            return a.cwiseProduct(b).sum();
        }
    }

    template <typename T1, typename T2>
    inline auto inner_product(const CellArray<T1>& a, const CellArray<T2>& b) {
        const Size n = a.size();
        SSNSDP_ASSERT(n == b.size());
        decltype(inner_product(a[0], b[0])) result = Zero;
        for (Index p = 0; p < n; ++p) {
            result += inner_product(a[p], b[p]);
        }
        return result;
    }
}

#endif
