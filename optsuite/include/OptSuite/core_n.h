/*
 * ==========================================================================
 *
 *       Filename:  core_n.h
 *
 *    Description:
 *
 *        Version:  1.0
 *        Created:  10/31/2020 05:17:25 PM
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  Haoyang Liu (@liuhy), liuhaoyang@pku.edu.cn
 *   Organization:  BICMR, PKU
 *
 * ==========================================================================
 */
#ifndef OPTSUITE_CORE_H
#define OPTSUITE_CORE_H

#include <complex>
#include <utility>
#include <memory>
#include <limits>
#include "core_macro_n.h"

#ifdef OPTSUITE_USE_MKL
#if !defined(OPTSUITE_EIGEN_USE_BUILTIN) && !defined(EIGEN_USE_MKL_ALL)
#define EIGEN_USE_MKL_ALL
#endif
#include <mkl.h>
#endif

#ifndef OPTSUITE_EIGEN_USE_BUILTIN
#ifndef EIGEN_USE_BLAS
#define EIGEN_USE_BLAS
#endif

#ifndef EIGEN_USE_LAPACKE
#define EIGEN_USE_LAPACKE
#endif
#endif // of OPTSUITE_EIGEN_USE_BUILTIN

#include <Eigen/Eigen>

namespace OptSuite {
    using Scalar = OPTSUITE_SCALAR_TYPE;
    using ComplexScalar = std::complex<Scalar>;
    using Index = OPTSUITE_DEFAULT_INDEX_TYPE;
    using SparseIndex = OPTSUITE_SPARSE_INDEX_TYPE;
    using Size = OPTSUITE_DEFAULT_INDEX_TYPE;
    using std::shared_ptr;

    using Eigen::Dynamic;
    using Eigen::Ref;
    using Eigen::ColMajor;
    using Eigen::RowMajor;
    using Eigen::EigenBase;
    using Eigen::Map;
    using Eigen::Stride;
    using Eigen::MatrixBase;
    using Eigen::SparseMatrixBase;
    using Eigen::Triplet;
    using Eigen::Infinity;

    using Mat = Eigen::Matrix<Scalar, Dynamic, Dynamic>;
    using RMat = Eigen::Matrix<Scalar, Dynamic, Dynamic, RowMajor>;
    using IMat = Eigen::Matrix<Index, Dynamic, Dynamic>;
    using Vec = Eigen::Matrix<Scalar, Dynamic, 1>;
    using RowVec = Eigen::Matrix<Scalar, 1, Dynamic>;
    using SpMat = Eigen::SparseMatrix<Scalar, ColMajor, SparseIndex>;
    using SpVec = Eigen::SparseVector<Scalar>;
    using CMat = Eigen::Matrix<ComplexScalar, Dynamic, Dynamic>;
    using CVec = Eigen::Matrix<ComplexScalar, Dynamic, 1>;
    using CSpMat = Eigen::SparseMatrix<ComplexScalar, ColMajor, SparseIndex>;
    using DiagMat = Eigen::DiagonalMatrix<Scalar, Dynamic>;
    using CDiagMat = Eigen::DiagonalMatrix<ComplexScalar, Dynamic>;
    using PermutationMatrix = Eigen::PermutationMatrix<Dynamic, Dynamic, SparseIndex>;
    using BoolMat = Eigen::Matrix<bool, Dynamic, Dynamic>;


    using Eigen::UnitLower;
    using Eigen::Lower;
    using Eigen::StrictlyLower;
    using Eigen::UnitUpper;
    using Eigen::Upper;
    using Eigen::StrictlyUpper;

    enum class Verbosity {
        // Suppress any output
        Quiet,
        // Output errors
        Error,
        // Output warnings,
        Warning,
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

    enum class MulOp {
        // Non-transpose
        NonTranspose,
        Transpose,
        ConjugateTranspose,
    };

    // convert to Scalar type
    constexpr Scalar operator"" _s(const long double x) {
        return static_cast<Scalar>(x);
    }

    // convert to Scalar type
    constexpr Scalar operator"" _s(const unsigned long long int x) {
        return static_cast<Scalar>(x);
    }

    constexpr Index operator"" _i(const unsigned long long int x) {
        return static_cast<Index>(x);
    }

    // imaginary part for complex number
    constexpr std::complex<Scalar> operator"" _ii(const long double x){
        return std::complex<Scalar>{0.0_s, static_cast<Scalar>(x)};
    }
    constexpr std::complex<Scalar> operator"" _ii(const unsigned long long x){
        return std::complex<Scalar>{0.0_s, static_cast<Scalar>(x)};
    }


    // define machine accuracy
    constexpr Scalar eps = std::numeric_limits<Scalar>::epsilon();
    constexpr Scalar inf = std::numeric_limits<Scalar>::infinity();
}

#endif
