#ifndef SSNSDP_EIG_H
#define SSNSDP_EIG_H

#include "core.h"
#include <type_traits>
#include "mkl_wrapper.h"

// SelfAdjointEigenSolver
// An interface subset of Eigen::SelfAdjointEigenSolver<MatrixX>, implemented by MKL
// available lapacke solvers are ?syev, ?syevd

// ComplexSelfAdjointEigenSolver
// An interface subset of Eigen::SelfAdjointEigenSolver<ComplexMatrixX>, implemented by MKL
// available lapacke solvers are ?heev

namespace ssnsdp {

#ifdef SSNSDP_MKL_EIG_ENABLED
    // Generic Implementation. MatrixScalar must be Scalar or ComplexScalar
    template <typename MatrixScalar>
    class SelfAdjointEigenSolverTemplate {

    public:
        constexpr static bool IsReal = std::is_same_v<MatrixScalar, Scalar>;
        constexpr static bool IsComplex = std::is_same_v<MatrixScalar, ComplexScalar>;
        static_assert(IsReal || IsComplex);

        using MatrixType = Matrix<MatrixScalar, Dynamic, Dynamic>;
        using BuiltInSolver = Eigen::SelfAdjointEigenSolver<MatrixType>;

        enum class LapackeMethod {
            syev,
            syevd,
            heev
        };

        SelfAdjointEigenSolverTemplate() = default;

        template <typename InputType>
        explicit SelfAdjointEigenSolverTemplate(const MatrixBase<InputType>& matrix,
            int options = ComputeEigenvectors)
        {
            compute(matrix.derived(), options);
        }

        const MatrixType& eigenvectors() const {
            SSNSDP_ASSERT_MSG(m_is_inited,
                "SelfAdjointEigenSolver is not initialized.");
            SSNSDP_ASSERT_MSG(m_eigenvectors_ok,
                "The eigenvectors have not been computed together with the eigenvalues.");
            return m_eivec;
        }

        const VectorX& eigenvalues() const {
            SSNSDP_ASSERT_MSG(m_is_inited, "SelfAdjointEigenSolver is not initialized.");
            return m_eivalues;
        }

        ComputationInfo info() const {
            SSNSDP_ASSERT_MSG(m_is_inited, "SelfAdjointEigenSolver is not initialized.");
            return m_info;
        }

        template <typename InputType>
        void compute(const MatrixBase<InputType>& matrix, const int options) {
            if constexpr (ScalarIsDouble || ScalarIsFloat) {
                if constexpr (IsComplex) {
                    compute_heev(matrix.derived(), options);
                } else {
                    compute_syevd(matrix.derived(), options);
                }
            } else {
                compute_builtin(matrix.derived(), options);
            }
        }

        // calculate eigenvalues without creating a class and temporary object.
        // eivalues: n x 1 buffer, matrix: n x n matrix, overwritten by eigenvectors
        template <int options = ComputeEigenvectors, typename InAndOut, typename EivaluesType>
        static ComputationInfo compute_impl(MatrixBase<InAndOut> const& matrix_,
            MatrixBase<EivaluesType> const& eivalues_)
        {
            EivaluesType& eivalues = eivalues_.const_cast_derived();
            InAndOut& matrix = matrix_.const_cast_derived();
            if constexpr (ScalarIsDouble || ScalarIsFloat) {
                if constexpr (IsComplex) {
                    return compute_lapacke_impl<LapackeMethod::heev, options>(matrix, eivalues);
                } else {
                    return compute_lapacke_impl<LapackeMethod::syevd, options>(matrix, eivalues);
                }
            } else {
                BuiltInSolver solver(matrix.derived(), options);
                eivalues = solver.eigenvalues();
                constexpr bool computeEigenvectors = (options & ComputeEigenvectors) == ComputeEigenvectors;
                if constexpr (computeEigenvectors) {
                    matrix = solver.eigenvectors();
                }
                return solver.info();
            }
        }

        template <typename InputType>
        void compute_builtin(const EigenBase<InputType>& matrix, const int options) {
            BuiltInSolver solver(matrix.derived(), options);
            m_info = solver.info();
            m_eivalues = solver.eigenvalues();
            const bool computeEigenvectors = (options & ComputeEigenvectors) == ComputeEigenvectors;
            if (computeEigenvectors) {
                m_eivec = solver.eigenvectors();
            }
            m_is_inited = true;
            m_eigenvectors_ok = computeEigenvectors;
        }

        template <typename InputType>
        void compute_syev(const MatrixBase<InputType>& matrix, const int options) {
            compute_lapacke<LapackeMethod::syev>(matrix, options);
        }

        template <typename InputType>
        void compute_syevd(const MatrixBase<InputType>& matrix, const int options) {
            compute_lapacke<LapackeMethod::syevd>(matrix, options);
        }

        template <typename InputType>
        void compute_heev(const MatrixBase<InputType>& matrix, const int options) {
            compute_lapacke<LapackeMethod::heev>(matrix, options);
        }

    private:
        Matrix<MatrixScalar, Dynamic, Dynamic> m_eivec;
        VectorX m_eivalues;

        // only for debug assertion
        SSNSDP_CLASS_MEMBER_MAYBE_UNUSED
        bool m_is_inited = false;

        SSNSDP_CLASS_MEMBER_MAYBE_UNUSED
        bool m_eigenvectors_ok = false;

        ComputationInfo m_info = Success;

        template <LapackeMethod lapacke_method, int options,
            typename InAndOut, typename EivaluesType>
        static ComputationInfo compute_lapacke_impl(
            MatrixBase<InAndOut> const& a_matrix,
            MatrixBase<EivaluesType> const& eivalues_)
        {
            SSNSDP_ASSUME(ScalarIsDouble || ScalarIsFloat);
            InAndOut& eivec = a_matrix.const_cast_derived();
            SSNSDP_ASSERT(eivec.cols() == eivec.rows());
            EivaluesType& eivalues = eivalues_.const_cast_derived();
            constexpr bool computeEigenvectors = (options & ComputeEigenvectors) == ComputeEigenvectors;
            const auto n = static_cast<lapack_int>(eivec.cols());
            EIGEN_STATIC_ASSERT_VECTOR_ONLY(EivaluesType);
            SSNSDP_ASSERT(eivalues.size() == n);
            const auto lda = static_cast<lapack_int>(eivec.outerStride());

            constexpr char uplo = 'L';
            constexpr char jobz = computeEigenvectors ? 'V' : 'N';

            lapack_int info;

            if constexpr (lapacke_method == LapackeMethod::syevd) {
                SSNSDP_ASSUME(IsReal);
                info = LAPACKE_syevd(LAPACK_COL_MAJOR,
                    jobz, uplo, n, eivec.data(), lda, eivalues.data());
            } else if constexpr (lapacke_method == LapackeMethod::syev) {
                SSNSDP_ASSUME(IsReal);
                info = LAPACKE_syev(LAPACK_COL_MAJOR,
                    jobz, uplo, n, eivec.data(), lda, eivalues.data());
            } else if constexpr (lapacke_method == LapackeMethod::heev) {
                SSNSDP_ASSUME(IsComplex);
                info = LAPACKE_heev(LAPACK_COL_MAJOR,
                    jobz, uplo, n,
                    as_lapacke_complex(eivec.data()),
                    lda, eivalues.data());
            } else {
                info = InvalidInput;
                SSNSDP_UNREACHABLE();
            }

            return (info == 0) ? Success : NoConvergence;
        }

        template <LapackeMethod lapacke_method, typename InputType>
        void compute_lapacke(const MatrixBase<InputType>& a_matrix, const int options) {
            SSNSDP_ASSUME(ScalarIsDouble || ScalarIsFloat);
            const InputType& matrix = a_matrix.derived();
            SSNSDP_ASSERT(matrix.cols() == matrix.rows());
            const bool computeEigenvectors = (options & ComputeEigenvectors) == ComputeEigenvectors;
            const auto n = static_cast<lapack_int>(matrix.cols());
            m_eivalues.resize(n);
            m_eivec = matrix;
            if (computeEigenvectors) {
                m_info = compute_lapacke_impl<lapacke_method, ComputeEigenvectors>(m_eivec, m_eivalues);
            } else {
                m_info = compute_lapacke_impl<lapacke_method, EigenvaluesOnly>(m_eivec, m_eivalues);
            }
            m_is_inited = true;
            m_eigenvectors_ok = computeEigenvectors;
        }
    };

    using SelfAdjointEigenSolver = SelfAdjointEigenSolverTemplate<Scalar>;
    // using ComplexSelfAdjointEigenSolver = SelfAdjointEigenSolverTemplate<ComplexScalar>;
#else
    using SelfAdjointEigenSolver = Eigen::SelfAdjointEigenSolver<MatrixX>;
    // using ComplexSelfAdjointEigenSolver = Eigen::SelfAdjointEigenSolver<ComplexMatrixX>;
#endif
}

#endif
