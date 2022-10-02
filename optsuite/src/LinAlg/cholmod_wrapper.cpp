/*
 * ==========================================================================
 *
 *       Filename:  cholmod_wrapper.cpp
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

#ifdef OPTSUITE_USE_SUITE_SPARSE

#include "OptSuite/LinAlg/cholmod_wrapper.h"
#include <cstring>
#include <type_traits>
#include "OptSuite/Utils/logger.h"

namespace {
    void free_cholmod_common(cholmod_common* common) {
        if (!common) { return; }
        cholmod_finish(common);
        OPTSUITE_ASSERT(common->malloc_count == 0);
        OPTSUITE_ASSERT(common->memory_inuse == 0);
        delete common;
    }
}

namespace OptSuite { namespace LinAlg {
    constexpr static int ctrue = 1;
    constexpr static int cfalse = 0;

    static_assert(std::is_same<int, SparseIndex>::value, "cholmod expects SparseIndex to be int");

    CholmodWrapper::CholmodWrapper() : common { new cholmod_common, free_cholmod_common } {
        // Eigen's built in cholmodsupport does not expose permutation matrix
        // therefore re-implement it

        cholmod_start(common.get());
        common->final_asis = cfalse;
        common->final_super = cfalse;
        common->final_ll = ctrue;
        common->final_pack = ctrue;
        common->final_monotonic = ctrue;
        common->final_resymbol = cfalse;
    }

    // CholmodWrapper::~CholmodWrapper() {
    // }

    PermutationMatrix CholmodWrapper::analyze(const Ref<const SpMat>& spmat,
        AnalyzeType type, FactorPtr& internal_result) const {
        // reference: https://github.com/DrTimothyAldenDavis/SuiteSparse/blob/master/CHOLMOD/MATLAB/analyze.c
        cholmod_sparse X = eigen_to_cholmod(spmat);

        X.xtype = CHOLMOD_PATTERN;
        bool transpose = false;

        if (type == AnalyzeType::Row) {
            X.stype = 0;
            transpose = false;
        } else if (type == AnalyzeType::Col) {
            X.stype = 0;
            transpose = true;
        } else if (type == AnalyzeType::Symmetric) {
            X.stype = -1;
            transpose = false;
        } else if (type == AnalyzeType::Normal) {
            X.stype = -1;
            transpose = false;
        } else {
            OPTSUITE_UNREACHABLE();
        }

        if (X.stype && (X.nrow != X.ncol)) {
            OPTSUITE_ASSERT_MSG(false, "analyze: input must be square");
        }

        common->supernodal = CHOLMOD_SIMPLICIAL;

        // should not be put in if statement
        // the pointer held by C may be referenced by X
        SparsePtr C = create_sparse_ptr(nullptr);

        if (transpose) {
            /* C = X', and then order C*C' */
            C.reset(cholmod_transpose(&X, 0, common.get()));
            OPTSUITE_ASSERT_MSG(C, "analyze failed");
            X = *C;
        }

        internal_result.reset(cholmod_analyze(&X, common.get()));

        auto perm_buf = static_cast<const SparseIndex*>(internal_result->Perm);
        return PermutationMatrix { Map<const VectorXi>(perm_buf, X.nrow) };
    }


    void
    CholmodWrapper::ldlchol_raw_impl(const Ref<const SpMat>& spmat, const Scalar* beta_in, FactorPtr& L, Index* p, PermutationMatrix* q) const {
        // L is the main return value
        // reference: https://github.com/DrTimothyAldenDavis/SuiteSparse/blob/master/CHOLMOD/MATLAB/ldlchol.c

        common->final_asis = cfalse;
        common->final_super = cfalse;
        common->final_ll = cfalse;
        common->final_pack = ctrue;
        common->final_monotonic = ctrue;
        common->final_resymbol = ctrue;
        common->quick_return_if_not_posdef = (p == nullptr);

        common->nmethods = 1;
        common->method[0].ordering = CHOLMOD_NATURAL;
        common->postorder = cfalse;

        std::array<Scalar, 2> beta;

        cholmod_sparse A = eigen_to_cholmod(spmat);

        if (beta_in) {
            A.stype = 0;
            beta[0] = *beta_in;
            beta[1] = 0;
        } else {
            A.stype = -1;  // use lower part of A
            beta[0] = 0;
            beta[1] = 0;
        }

        if (!L) {
            L.reset(cholmod_analyze(&A, common.get()));
        }
        OPTSUITE_ASSERT(L);

        int orig_print_level = common->print;
        if (p) {
            // suppress CHOLMOD error: not positive definite.
            //  as this is checkable when p is not nullptr.
            common->print = 0;
        }
        cholmod_factorize_p(&A, beta.data(), nullptr, 0, L.get(), common.get());
        common->print = orig_print_level;

        OPTSUITE_ASSERT_MSG(p || common->status == CHOLMOD_OK, "matrix is not positive definite");

        if (p) {
            // not converted to matlab convention
            *p = static_cast<Index>(L->minor);
        }

        if (q) {
            auto perm_buf = static_cast<const SparseIndex*>(L->Perm);
            *q = PermutationMatrix { Map<const VectorXi>(perm_buf, A.nrow) };
        }
    }

    CholmodWrapper::FactorPtr CholmodWrapper::ldlchol_raw_with_analyze_impl(const Ref<const SpMat>& spmat, const Scalar* beta_in, Index* p, PermutationMatrix* q) const {
        FactorPtr L = create_factor_ptr(nullptr);
        ldlchol_raw_impl(spmat, beta_in, L, p, q);
        return L;
    }

    SpMat CholmodWrapper::ldlchol_impl(const Ref<const SpMat>& spmat, const Scalar* beta_in, Index* p, PermutationMatrix* q) const {
        // reference: https://github.com/DrTimothyAldenDavis/SuiteSparse/blob/master/CHOLMOD/MATLAB/ldlchol.c
        // necessary check done in ldlchol_raw_impl
        FactorPtr L = ldlchol_raw_with_analyze_impl(spmat, beta_in, p, q);
        SparsePtr Lsparse = create_sparse_ptr(cholmod_factor_to_sparse(L.get(), common.get()));
        OPTSUITE_ASSERT(Lsparse);
        return cholmod_to_eigen(*Lsparse);
    }

    Mat CholmodWrapper::ldlsolve(const FactorPtr& factor, const Ref<const Mat>& B_eigen) const {
        cholmod_dense B = eigen_to_cholmod(B_eigen);
        DensePtr X = create_dense_ptr(cholmod_solve(CHOLMOD_LDLt, factor.get(), &B, common.get()));
        OPTSUITE_ASSERT(X);
        double rcond = cholmod_rcond(factor.get(), common.get());
        if (rcond == 0.0) {
            Utils::Global::logger_e.log<OptSuite::Verbosity::Warning>("Matrix is indefinite or singular to working precision");
        } else if (rcond < eps) {
            Utils::Global::logger_e.log_format<OptSuite::Verbosity::Warning>(
                "Matrix is close to singular or badly scaled.\n"
                "  Results may be inaccurate. RCOND = %4.3e.\n",
                rcond);
        }
        return cholmod_to_eigen(*X);
    }

    cholmod_sparse CholmodWrapper::eigen_to_cholmod(const Ref<const SpMat>& spmat) const {
        // --- safe version following cholmod API ---
        // Returns SparsePtr if this version is to be used.
        // --- --- ---
        // X = cholmod_allocate_sparse(
        //     m, m, nnz, 1, 1, 1, CHOLMOD_REAL, &c);
        //
        // std::memcpy(X->p, spmat.outerIndexPtr(), (m + 1) * sizeof(SparseIndex));
        // X->i = cholmod_alloc_copy(spmat.innerIndexPtr(), nnz, &c);
        // X->x = cholmod_alloc_copy(spmat.valuePtr(), nnz, &c);
        //
        // X->itype = CHOLMOD_INT;
        // ---
        // implementation from CholmodSupport.h : viewAsCholmod
        // assume to be ok
        const int m = spmat.rows();
        const int n = spmat.cols();
        const int nnz = spmat.nonZeros();

        cholmod_sparse out;

        out.nzmax = nnz;
        out.nrow = m;
        out.ncol = n;
        out.p = const_cast<SparseIndex*>(spmat.outerIndexPtr());
        out.i = const_cast<SparseIndex*>(spmat.innerIndexPtr());
        out.x = const_cast<Scalar*>(spmat.valuePtr());
        out.z = nullptr;
        out.sorted = ctrue;
        if (spmat.isCompressed()) {
            out.packed = ctrue;
            out.nz = nullptr;
        } else {
            out.packed = cfalse;
            out.nz = const_cast<SparseIndex*>(spmat.innerNonZeroPtr());
        }
        if (OPTSUITE_SCALAR_TOKEN == 0) {
            out.dtype = CHOLMOD_DOUBLE;
        } else {
            out.dtype = CHOLMOD_SINGLE;
        }

        out.stype = -1;
        out.itype = CHOLMOD_INT;
        out.xtype = CHOLMOD_REAL;

        return out;
    }

    cholmod_dense CholmodWrapper::eigen_to_cholmod(const Ref<const Mat>& mat) const {
        // implementation from CholmodSupport.h : viewAsCholmod
        cholmod_dense out;
        out.nrow = mat.rows();
        out.ncol = mat.cols();
        out.nzmax = out.nrow * out.ncol;
        out.d = mat.outerStride();
        out.x = const_cast<Scalar*>(mat.data());
        out.z = nullptr;
        out.xtype = CHOLMOD_REAL;
        if (OPTSUITE_SCALAR_TOKEN == 0) {
            out.dtype = CHOLMOD_DOUBLE;
        } else {
            out.dtype = CHOLMOD_SINGLE;
        }
        return out;
    }

    CholmodWrapper::FactorPtr CholmodWrapper::eigen_to_cholmod_factor(const Ref<const SpMat>& spmat) const {
        OPTSUITE_ASSERT(spmat.rows() == spmat.cols());
        const Index n = spmat.rows();
        const Index nnz = spmat.nonZeros();
        FactorPtr L = create_factor_ptr(cholmod_allocate_factor(n, common.get()));
        OPTSUITE_ASSERT(L);
        L->ordering = CHOLMOD_NATURAL;
        // L->p = const_cast<SparseIndex*>(spmat.outerIndexPtr());
        // L->i = const_cast<SparseIndex*>(spmat.innerIndexPtr());
        // L->x = const_cast<Scalar*>(spmat.valuePtr());
        // L->nz = const_cast<SparseIndex*>(spmat.innerNonZeroPtr()); when not compressed

        // use the malloc-and-copy way, which makes the FactorPtr easier to manage.
        // In this way, L lives even if spmat is destroyed. Thus L can be exposed to end-users.
        L->p = cholmod_malloc(n + 1, sizeof(SparseIndex), common.get());
        L->i = cholmod_malloc(nnz, sizeof(SparseIndex), common.get());
        L->x = cholmod_malloc(nnz, sizeof(Scalar), common.get());
        L->nz = cholmod_malloc(n, sizeof(SparseIndex), common.get());
        L->prev = cholmod_malloc(n + 2, sizeof(SparseIndex), common.get());
        L->next = cholmod_malloc(n + 2, sizeof(SparseIndex), common.get());
        OPTSUITE_ASSERT(L->p && L->i && L->x && L->nz && L->prev && L->next);

        auto Lp = static_cast<SparseIndex*>(L->p);
        auto Li = static_cast<SparseIndex*>(L->i);
        auto Lx = static_cast<Scalar*>(L->x);
        auto Lnz = static_cast<SparseIndex*>(L->nz);
        auto Lprev = static_cast<SparseIndex*>(L->prev);
        auto Lnext = static_cast<SparseIndex*>(L->next);

        std::copy(spmat.outerIndexPtr(), spmat.outerIndexPtr() + (n + 1), Lp);
        std::copy(spmat.innerIndexPtr(), spmat.innerIndexPtr() + nnz, Li);
        std::copy(spmat.valuePtr(), spmat.valuePtr() + nnz, Lx);
        if (spmat.isCompressed()) {
            for (Index j = 0; j < n; ++j) {
                Lnz[j] = Lp[j + 1] - Lp[j];
            }
        } else {
            std::copy(spmat.innerNonZeroPtr(), spmat.innerNonZeroPtr() + n, Lnz);
        }

        L->z = nullptr;
        if (OPTSUITE_SCALAR_TOKEN == 0) {
            L->dtype = CHOLMOD_DOUBLE;
        } else {
            L->dtype = CHOLMOD_SINGLE;
        }
        L->itype = CHOLMOD_INT;
        L->xtype = CHOLMOD_REAL;

        const Index head = n + 1;
        const Index tail = n;
        Lnext[head] = 0;
        Lprev[head] = -1;
        Lnext[tail] = -1;
        Lprev[tail] = n - 1;
        for (Index j = 0; j < n; ++j) {
            Lnext[j] = j + 1;
            Lprev[j] = j - 1;
        }
        Lprev[0] = head;
        L->nzmax = nnz;
        return L;
    }


    Map<const SpMat> CholmodWrapper::cholmod_to_eigen(const cholmod_sparse& spmat) const {
        return Map<const SpMat> {
            static_cast<Index>(spmat.nrow),
            static_cast<Index>(spmat.ncol),
            static_cast<Index>(spmat.nzmax),
            static_cast<const SparseIndex*>(spmat.p),
            static_cast<const SparseIndex*>(spmat.i),
            static_cast<const Scalar*>(spmat.x)
        };
    }

    Map<const Mat, 0, Stride<Dynamic, 1>> CholmodWrapper::cholmod_to_eigen(const cholmod_dense& mat) const {
        return Map<const Mat, 0, Stride<Dynamic, 1>> {
            static_cast<const Scalar*>(mat.x),
            static_cast<Index>(mat.nrow),
            static_cast<Index>(mat.ncol),
            Stride<Dynamic, 1>(static_cast<Index>(mat.d), 1)
        };
    }
}}

#endif // ifdef OPTSUITE_USE_SUITE_SPARSE
