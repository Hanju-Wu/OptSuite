/*
 * ==========================================================================
 *
 *       Filename:  cholmod_wrapper.h
 *
 *    Description:
 *
 *        Version:  1.0
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  Dong Xu (@taroxd), taroxd@pku.edu.cn
 *   Organization:  BICMR, PKU
 *
 * ==========================================================================
 */

#ifndef OPTSUITE_CHOLMOD_WRAPPER_H
#define OPTSUITE_CHOLMOD_WRAPPER_H

#ifdef OPTSUITE_USE_SUITE_SPARSE

#include "OptSuite/core_n.h"

#include <cholmod.h>

// auto link in MSVC
#if OPTSUITE_MSVC && !defined(OPTSUITE_DISABLE_SUITE_SPARSE_AUTO_LINK)
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

namespace OptSuite { namespace LinAlg {

class CholmodWrapper {
    class CholmodSparseDeleter {
        public:
            CholmodSparseDeleter(std::shared_ptr<cholmod_common> common) : common_(std::move(common)) {
                OPTSUITE_ASSERT(common_);
            }
            ~CholmodSparseDeleter() = default;
            inline void operator()(cholmod_sparse* data) {
                if (data) {
                    cholmod_free_sparse(&data, common_.get());
                }
            }
        private:
            std::shared_ptr<cholmod_common> common_;
    };

    class CholmodFactorDeleter {
        public:
            CholmodFactorDeleter(std::shared_ptr<cholmod_common> common) :
                common_(std::move(common))
            {
                OPTSUITE_ASSERT(common_);
            }
            ~CholmodFactorDeleter() = default;

            inline void operator()(cholmod_factor* data) {
                if (data) {
                    cholmod_free_factor(&data, common_.get());
                }
            }
        private:
            std::shared_ptr<cholmod_common> common_;
    };

    class CholmodDenseDeleter {
        public:
            CholmodDenseDeleter(std::shared_ptr<cholmod_common> common) : common_(std::move(common)) {
                OPTSUITE_ASSERT(common_);
            }
            ~CholmodDenseDeleter() = default;
            inline void operator()(cholmod_dense* data) {
                if (data) {
                    cholmod_free_dense(&data, common_.get());
                }
            }
        private:
            std::shared_ptr<cholmod_common> common_;
    };

    public:
        using VectorXi = Eigen::Matrix<SparseIndex, Dynamic, 1>;
        using SparsePtr = std::unique_ptr<cholmod_sparse, CholmodSparseDeleter>;
        using FactorPtr = std::unique_ptr<cholmod_factor, CholmodFactorDeleter>;
        using DensePtr = std::unique_ptr<cholmod_dense, CholmodDenseDeleter>;
        enum class AnalyzeType {
            Normal,    // orders A
            Symmetric, // orders A
            Row,       // orders A*A'
            Col        // orders A'*A
        };
        CholmodWrapper();
        ~CholmodWrapper() = default;

        CholmodWrapper(const CholmodWrapper&) = delete;
        CholmodWrapper(CholmodWrapper&&) = delete;

        CholmodWrapper& operator=(const CholmodWrapper&) = default;
        CholmodWrapper& operator=(CholmodWrapper&&) = default;

        inline SparsePtr create_sparse_ptr(cholmod_sparse* sparse) const {
            return SparsePtr { sparse, CholmodSparseDeleter { common } };
        }

        inline FactorPtr create_factor_ptr(cholmod_factor* factor) const
        {
            return FactorPtr { factor, CholmodFactorDeleter { common } };
        }

        inline DensePtr create_dense_ptr(cholmod_dense* dense) const {
            return DensePtr { dense, CholmodDenseDeleter { common } };
        }

        // real number only. Complex may be supported later
        inline PermutationMatrix analyze(const Ref<const SpMat>& spmat, AnalyzeType type = AnalyzeType::Normal) const {
            FactorPtr dummy = create_factor_ptr(nullptr);
            return analyze(spmat, type, dummy);
        };
        PermutationMatrix analyze(const Ref<const SpMat>& spmat, AnalyzeType type, FactorPtr& internal_result) const;

        // beta: factorize spmat when beta is nullptr
        //   otherwise factorize spmat*spmat' + (*beta)*I
        // p: Output parameter. The column at which it failed.
	    //   A value of n means the factorization was successful.
        // q: Output parameter. factorize with permutation.
        //
        // Reference:
        //   Matlab API  [LD, p, q] = ldlchol(spmat, beta)
        inline SpMat ldlchol(const Ref<const SpMat>& spmat) const {
            return ldlchol_impl(spmat, nullptr, nullptr, nullptr);
        };
        inline SpMat ldlchol(const Ref<const SpMat>& spmat, const Scalar* beta) const {
            return ldlchol_impl(spmat, beta, nullptr, nullptr);
        };
        inline SpMat ldlchol(const Ref<const SpMat>& spmat, const Scalar* beta, Index& p) const {
            return ldlchol_impl(spmat, beta, &p, nullptr);
        };
        inline SpMat ldlchol(const Ref<const SpMat>& spmat, const Scalar* beta, Index& p, PermutationMatrix& q) const {
            return ldlchol_impl(spmat, beta, &p, &q);
        };

        // almost the same as ldlchol, other than that the return value is not an Eigen Matrix.
        //  use the returned FactorPtr in ldlsolve to prevent memory allocation overhead.
        // Also note that using `eigen_to_cholmod_factor' on what `ldlchol' returns can also obtain a valid FactorPtr.
        inline FactorPtr ldlchol_raw(const Ref<const SpMat>& spmat) const {
            return ldlchol_raw_with_analyze_impl(spmat, nullptr, nullptr, nullptr);
        };
        inline FactorPtr ldlchol_raw(const Ref<const SpMat>& spmat, const Scalar* beta) const {
            return ldlchol_raw_with_analyze_impl(spmat, beta, nullptr, nullptr);
        };
        inline FactorPtr ldlchol_raw(const Ref<const SpMat>& spmat, const Scalar* beta, Index& p) const {
            return ldlchol_raw_with_analyze_impl(spmat, beta, &p, nullptr);
        };
        inline FactorPtr ldlchol_raw(const Ref<const SpMat>& spmat, const Scalar* beta, Index& p, PermutationMatrix& q) const {
            return ldlchol_raw_with_analyze_impl(spmat, beta, &p, &q);
        };

        // The version reusing previous FactorPtr result. q can be obtained from the first call.
        inline void ldlchol_raw(const Ref<const SpMat>& spmat, FactorPtr& L) const {
            return ldlchol_raw_impl(spmat, nullptr, L, nullptr, nullptr);
        };
        inline void ldlchol_raw(const Ref<const SpMat>& spmat, const Scalar* beta, FactorPtr& L) const {
            return ldlchol_raw_impl(spmat, beta, L, nullptr, nullptr);
        };
        inline void ldlchol_raw(const Ref<const SpMat>& spmat, const Scalar* beta, FactorPtr& L, Index& p) const {
            return ldlchol_raw_impl(spmat, beta, L, &p, nullptr);
        };

        Mat ldlsolve(const FactorPtr& factor, const Ref<const Mat>& B) const;

        // When LD is used multiple times, it is recommended to call eigen_to_cholmod_factor first.
        inline Mat ldlsolve(const Ref<const SpMat>& LD, const Ref<const Mat>& B) const {
            return ldlsolve(eigen_to_cholmod_factor(LD), B);
        }

        FactorPtr eigen_to_cholmod_factor(const Ref<const SpMat>& spmat) const;

        // avoid using this function when it is convenient to inline manually
        // inlining the implementation shown in the function body manually would probably yield better performance,
        //    and will not introduce temporary storage
        static inline void ldlsplit(const SpMat& LD, SpMat& L, DiagMat& D) {
            D = LD.diagonal().asDiagonal();
            L = LD.triangularView<Eigen::UnitLower>();
        };
    private:
        std::shared_ptr<cholmod_common> common;
        cholmod_sparse eigen_to_cholmod(const Ref<const SpMat>& spmat) const;
        cholmod_dense eigen_to_cholmod(const Ref<const Mat>& mat) const;
        Map<const SpMat> cholmod_to_eigen(const cholmod_sparse& spmat) const;
        Map<const Mat, 0, Stride<Dynamic, 1>> cholmod_to_eigen(const cholmod_dense& mat) const;
        SpMat ldlchol_impl(const Ref<const SpMat>& spmat, const Scalar* beta_in, Index* p, PermutationMatrix* q) const;
        void ldlchol_raw_impl(const Ref<const SpMat>& spmat, const Scalar* beta_in, FactorPtr& L, Index* p, PermutationMatrix* q) const;
        FactorPtr ldlchol_raw_with_analyze_impl(const Ref<const SpMat>& spmat, const Scalar* beta_in, Index* p, PermutationMatrix* q) const;
};

}}

#endif // #ifdef OPTSUITE_USE_SUITE_SPARSE
#endif
