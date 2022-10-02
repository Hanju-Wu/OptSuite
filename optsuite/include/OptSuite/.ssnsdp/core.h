#ifndef SSNSDP_CORE_H
#define SSNSDP_CORE_H

#include "core_macro.h"

#include <cstddef>
#include <cmath>
#include <cassert>
#include <vector>
#include <limits>
#include <complex>
#include <type_traits>
#include <memory>
#include <utility>

// define SSNSDP_USE_MKL to enable the use of MKL
#ifdef SSNSDP_USE_MKL
#ifndef EIGEN_USE_MKL_ALL
#define EIGEN_USE_MKL_ALL
#endif
#include <mkl.h>
#endif

// define SSNSDP_USE_SUITE_SPARSE to enable the use of suitesparse
#ifdef SSNSDP_USE_SUITE_SPARSE

#include <cholmod.h>

// auto link in MSVC
#if SSNSDP_MSVC && !defined(SSNSDP_DISABLE_SUITE_SPARSE_AUTO_LINK)
#ifndef _DEBUG
#pragma comment(lib, "metis.lib")
#pragma comment(lib, "suitesparseconfig.lib")
#pragma comment(lib, "libcholmod.lib")
#pragma comment(lib, "libamd.lib")
#pragma comment(lib, "libcamd.lib")
#pragma comment(lib, "libcolamd.lib")
#pragma comment(lib, "libccolamd.lib")
#else
#pragma comment(lib, "metisd.lib")
#pragma comment(lib, "suitesparseconfigd.lib")
#pragma comment(lib, "libcholmodd.lib")
#pragma comment(lib, "libamdd.lib")
#pragma comment(lib, "libcamdd.lib")
#pragma comment(lib, "libcolamdd.lib")
#pragma comment(lib, "libccolamdd.lib")
#endif
#endif

#endif // ifdef SSNSDP_USE_SUITE_SPARSE

#include <Eigen/Eigen>

namespace ssnsdp {
    using Index = SSNSDP_DEFAULT_INDEX_TYPE;
    using Size = SSNSDP_DEFAULT_INDEX_TYPE;
    using Scalar = SSNSDP_SCALAR_TYPE;
    using SparseIndex = SSNSDP_SPARSE_INDEX_TYPE;
    using RealScalar = Scalar;
    using ComplexScalar = std::complex<Scalar>;

    using std::abs;
    using std::max;
    using std::min;
    using std::sqrt;
    using std::pow;
    using std::exp;
    using std::conj;

    using Eigen::Dynamic;
    using Eigen::Success;
    using Eigen::NoConvergence;
    using Eigen::InvalidInput;
    using Eigen::NumericalIssue;
    using Eigen::EigenvaluesOnly;
    using Eigen::ComputeEigenvectors;
    using Eigen::Matrix;
    using Eigen::UpLoType;
    using Eigen::Upper;
    using Eigen::Lower;
    using Eigen::ColMajor;
    using Eigen::RowMajor;
    using Eigen::Aligned128;

    using Triplet = Eigen::Triplet<Scalar, Index>;
    using TripletList = std::vector<Triplet>;
    using SparseMatrix = Eigen::SparseMatrix<Scalar, ColMajor, SparseIndex>;
    using SparseVector = Eigen::SparseVector<Scalar>;
    using VectorX = Eigen::Matrix<Scalar, Dynamic, 1>;
    // using RowVectorX = Eigen::Matrix<Scalar, 1, Dynamic>;
    using MatrixX = Eigen::Matrix<Scalar, Dynamic, Dynamic>;
    using ComplexVectorX = Eigen::Matrix<std::complex<Scalar>, Dynamic, 1>;
    using ComplexMatrixX = Eigen::Matrix<std::complex<Scalar>, Dynamic, Dynamic>;
    using DiagonalMatrixX = Eigen::DiagonalMatrix<Scalar, Dynamic>;
    using RowMajorMatrixX = Eigen::Matrix<Scalar, Dynamic, Dynamic, RowMajor>;

    using Eigen::MatrixBase;
    using Eigen::SparseMatrixBase;
    using Eigen::TriangularBase;
    using Eigen::EigenBase;
    using Eigen::Ref;
    using Eigen::Map;
    using Eigen::ComputationInfo;

    constexpr auto Alignment = Eigen::internal::traits<VectorX>::Alignment;

    template <typename T>
    constexpr Scalar as_scalar(T x) {
        return static_cast<Scalar>(x);
    }

    template <typename T>
    constexpr Index as_int(T x) {
        return static_cast<Index>(x);
    }

    // convert to Scalar type
    constexpr Scalar operator""_s(const long double x) {
        return static_cast<Scalar>(x);
    }

    // convert to Scalar type
    constexpr Scalar operator""_s(const unsigned long long int x) {
        return static_cast<Scalar>(x);
    }

    constexpr Index operator""_i(const unsigned long long int x) {
        return static_cast<Index>(x);
    }

    // 0
    constexpr Scalar Zero = 0_s;
    // 1
    constexpr Scalar One = 1_s;

    static_assert(std::numeric_limits<Scalar>::has_infinity);
    // inf
    constexpr Scalar Inf = std::numeric_limits<Scalar>::infinity();
    // (sqrt(5) + 1) / 2
    constexpr Scalar Golden1_618 = 1.6180339887498948482045_s;
    // sqrt(2)
    constexpr Scalar Sqrt2 = 1.414213562373095048801688_s;
    // 1/sqrt(2)
    constexpr Scalar Sqrt1_2 = 0.70710678118654752440084436210_s;

    enum class Verbosity {
        // Suppress any output
        Quiet,
        // Output necessary information
        Info,
        // Output more detail during iteration
        Detail,
        // Output information for developer debugging
        Debug,
        // Output information that verbosely indicates the program state
        Verbose,
        // This option is only for testing program about logging.
        // Virtually the same as Verbose
        Everything
    };

    enum class BlockType {
        SDP = 's',
        LINEAR = 'l'
    };

    struct BlockSpec {
        Size n;
        BlockType type;

        bool is_linear() const { return type == BlockType::LINEAR; }
        bool is_sdp() const { return type == BlockType::SDP; }
    };

    // Compile-time verbosity. Intended only for developers.
    constexpr Verbosity CurrentVerbosity = SSNSDP_VERBOSITY;

#if SSNSDP_USE_C_STYLE_IO
    constexpr bool UseCStyleIO = true;
#else
    constexpr bool UseCStyleIO = false;
#endif

#ifdef SSNSDP_USE_MKL
    constexpr bool UseMKL = true;
#else
    constexpr bool UseMKL = false;
#endif

#ifdef SSNSDP_USE_SUITE_SPARSE
    constexpr bool UseSuiteSparse = true;
#else
    constexpr bool UseSuiteSparse = false;
#endif

    constexpr bool ScalarIsFloat = std::is_same_v<Scalar, float>;
    constexpr bool ScalarIsDouble = std::is_same_v<Scalar, double>;
    constexpr bool ScalarIsLongDouble = std::is_same_v<Scalar, long double>;

    constexpr bool SparseIndexIsInt = std::is_same_v<SparseIndex, int>;

    template <typename Derived>
    constexpr bool is_eigen_type_f(const EigenBase<Derived>*) {
        return true;
    }
    constexpr bool is_eigen_type_f(...) {
        return false;
    }

    // is_eigen_type<T>: true if T is derived from EigenBase<T>, false otherwise
    template <typename T>
    constexpr bool is_eigen_type =
        is_eigen_type_f(static_cast<std::add_pointer_t<T>>(nullptr));

    // is_ssnsdp_scalar_type<T>: true if T == Scalar or ComplexScalar
    template <typename T>
    constexpr bool is_ssnsdp_scalar_type =
        std::is_same_v<T, Scalar> || std::is_same_v<T, ComplexScalar>;

    // is_scalar_type<T>: true if T is integer or floating-point number
    template <typename T>
    constexpr bool is_scalar_type = std::is_arithmetic_v<T>;

    // is_value_type<T>: true if T is either eigen type or scalar type
    template <typename T>
    constexpr bool is_value_type = is_eigen_type<T> || is_scalar_type<T>;

    static_assert(is_eigen_type<MatrixX>);
    static_assert(!is_eigen_type<Scalar>);
}

#endif
