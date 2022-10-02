#ifndef SSNSDP_CHOL_SOLVER_H
#define SSNSDP_CHOL_SOLVER_H

#include "core.h"
#include <cstring>
#include <variant>

namespace ssnsdp {

    class CholSolver {
    public:

        static constexpr bool SuiteSparseEnabled =
            UseSuiteSparse && (ScalarIsFloat || ScalarIsDouble) && SparseIndexIsInt;

        using PermutationMatrix = Eigen::PermutationMatrix<Dynamic, Dynamic, SparseIndex>;
        using VectorXi = Matrix<SparseIndex, Dynamic, 1>;

        CholSolver() = default;

        explicit CholSolver(const SparseMatrix& AAt) {
            compute(AAt);
        }

        template <typename Derived>
        void init_with_L(MatrixBase<Derived>&& L) {
            init_with_L_internal<MatrixX>(static_cast<Derived&&>(L));
        }

        template <typename Derived>
        void init_with_L(SparseMatrixBase<Derived>&& L) {
            init_with_L_internal<SparseMatrix>(static_cast<Derived&&>(L));
        }

        template <typename Derived, typename PermType>
        void init_with_L(MatrixBase<Derived>&& L, PermType&& perm) {
            init_with_L_internal<MatrixX>(static_cast<Derived&&>(L),
                std::forward<PermType>(perm));
        }

        template <typename Derived, typename PermType>
        void init_with_L(SparseMatrixBase<Derived>&& L, PermType&& perm) {
            init_with_L_internal<SparseMatrix>(static_cast<Derived&&>(L),
                std::forward<PermType>(perm));
        }

        void compute(const SparseMatrix& AAt)
        {
            const Size m = AAt.cols();
            SSNSDP_ASSERT_MSG(m == AAt.rows(), "Matrix must be square");

            if (AAt.nonZeros() * 5 >= m * m) {
                const LLTX llt = MatrixX(AAt).llt();
                m_info = llt.info();
                if (m_info == Success) {
                    m_L = MatrixX(AAt).llt().matrixL();
                    m_status_internal = SolvedWithNoPerm;
                }
            } else {
                if constexpr (SuiteSparseEnabled) {
                    // Eigen's built in cholmodsupport does not expose permutation matrix
                    // therefore re-implement it
#ifdef SSNSDP_USE_SUITE_SPARSE
                    constexpr int ctrue = 1;
                    constexpr int cfalse = 0;
                    const Size nnz = AAt.nonZeros();
                    cholmod_common c;
                    cholmod_start(&c);
                    /* start CHOLMOD */
                    c.final_asis = cfalse;
                    c.final_super = cfalse;
                    c.final_ll = ctrue;
                    c.final_pack = ctrue;
                    c.final_monotonic = ctrue;
                    c.final_resymbol = cfalse;

                    // --- safe version following cholmod API ---
                    // auto X = cholmod_allocate_sparse(
                    //     m, m, nnz, 1, 1, 1, CHOLMOD_REAL, &c);
                    //
                    // std::memcpy(X->p, AAt.outerIndexPtr(), (m + 1) * sizeof(SparseIndex));
                    // X->i = cholmod_alloc_copy(AAt.innerIndexPtr(), nnz, &c);
                    // X->x = cholmod_alloc_copy(AAt.valuePtr(), nnz, &c);
                    //
                    // X->itype = CHOLMOD_INT;
                    // ---
                    // implementation from CholmodSupport.h : viewAsCholmod
                    // assume to be ok

                    cholmod_sparse X;
                    X.nzmax = nnz;
                    X.nrow = m;
                    X.ncol = m;
                    X.p = const_cast<SparseIndex*>(AAt.outerIndexPtr());
                    X.i = const_cast<SparseIndex*>(AAt.innerIndexPtr());
                    X.x = const_cast<Scalar*>(AAt.valuePtr());
                    X.z = nullptr;
                    X.sorted = ctrue;
                    if (AAt.isCompressed()) {
                        X.packed = ctrue;
                        X.nz = nullptr;
                    } else {
                        X.packed = cfalse;
                        X.nz = const_cast<SparseIndex*>(AAt.innerNonZeroPtr());
                    }
                    if constexpr (ScalarIsDouble) {
                        X.dtype = CHOLMOD_DOUBLE;
                    } else {
                        X.dtype = CHOLMOD_SINGLE;
                    }
                    X.stype = -1;
                    X.itype = CHOLMOD_INT;
                    X.xtype = CHOLMOD_REAL;

                    auto L = cholmod_analyze(&X, &c);
                    cholmod_factorize(&X, L, &c);

                    // convert L->minor to Index
                    if (as_int(L->minor) == m) {
                        m_info = Success;
                    } else {
                        // not positive definite
                        m_info = NumericalIssue;
                    }

                    if (m_info == Success) {
                        auto Lsparse = cholmod_factor_to_sparse(L, &c);
                        cholmod_drop(0, Lsparse, &c);

                        Map<const SparseMatrix> Lmap(m, m,
                            as_int(Lsparse->nzmax),
                            static_cast<const SparseIndex*>(Lsparse->p),
                            static_cast<const SparseIndex*>(Lsparse->i),
                            static_cast<const Scalar*>(Lsparse->x));

                        if (Lmap.nonZeros() > m * m / 3) {
                            m_L.emplace<MatrixX>(Lmap);
                        } else {
                            m_L.emplace<SparseMatrix>(Lmap);
                        }

                        const auto perm_buf = static_cast<const SparseIndex*>(L->Perm);
                        m_status_internal = SolvedWithNoPerm;
                        // do not set perm in the trivial case
                        for (Index i = 0; i < m; ++i) {
                            if (perm_buf[i] != i) {
                                m_status_internal = SolvedWithPerm;
                                m_perm = PermutationMatrix(Map<const VectorXi>(perm_buf, m));
                                m_perminv = m_perm.inverse();
                                break;
                            }
                        }

                        cholmod_free_sparse(&Lsparse, &c);
                    }

                    cholmod_free_factor(&L, &c);
                    // cholmod_free_sparse(&X, &c);
#else
                    SSNSDP_UNREACHABLE();
#endif
                } else {
                    // SuiteSparse is not enabled. Resort to Eigen's built-in solver
                    const LLTSparse llt(AAt);
                    m_info = llt.info();
                    m_status_internal = SolvedWithPerm;
                    if (m_info == Success) {
                        SparseMatrix Lmap(llt.matrixL());

                        if (Lmap.nonZeros() > m * m / 3) {
                            m_L.emplace<MatrixX>(Lmap);
                        } else {
                            m_L.emplace<SparseMatrix>(std::move(Lmap));
                        }

                        m_perm = llt.permutationP();
                        m_perminv = llt.permutationPinv();
                    }
                }
            }
        }

        CholSolver(const CholSolver&) = default;
        CholSolver(CholSolver&&) = default;
        CholSolver& operator=(const CholSolver&) = default;
        CholSolver& operator=(CholSolver&&) = default;
        ~CholSolver() = default;

        ComputationInfo info() const { return m_info; }

        template <typename Derived>
        VectorX fwsolve(const MatrixBase<Derived>& v_) const {
            EIGEN_STATIC_ASSERT_VECTOR_ONLY(Derived);
            assert_success();
            const Derived& v = v_.derived();
            const VectorX& v_permed = has_perm() ? (m_perminv * v).eval() : v;
            return std::visit([&v_permed](auto&& L) -> VectorX {
                return L.template triangularView<Lower>().solve(v_permed);
            }, m_L);
        }

        template <typename Derived>
        void fwsolve_in_place(MatrixBase<Derived> const& v_) const {
            EIGEN_STATIC_ASSERT_VECTOR_ONLY(Derived);
            assert_success();
            Derived& v = v_.const_cast_derived();

            if (has_perm()) {
                v = m_perminv * v;
            }

            std::visit([&v](auto&& L) {
                L.template triangularView<Lower>().solveInPlace(v);
            }, m_L);
        }

        template <typename Derived>
        VectorX bwsolve(const MatrixBase<Derived>& v) const {
            EIGEN_STATIC_ASSERT_VECTOR_ONLY(Derived);
            assert_success();
            VectorX result = std::visit([&v](auto&& L) -> VectorX {
                return L.adjoint().template triangularView<Upper>().solve(v.derived());
            }, m_L);

            if (has_perm()) {
                result = m_perm * result;
            }

            return result;
        }

        template <typename Derived>
        void bwsolve_in_place(MatrixBase<Derived> const& v_) const {
            EIGEN_STATIC_ASSERT_VECTOR_ONLY(Derived);
            assert_success();
            Derived& v = v_.const_cast_derived();

            std::visit([&v](auto&& L) {
                L.adjoint().template triangularView<Upper>().solveInPlace(v);
            }, m_L);

            if (has_perm()) {
                v = m_perm * v;
            }
        }

        bool is_sparse() const {
            return std::holds_alternative<SparseMatrix>(m_L);
        }

        bool has_perm() const {
            return m_status_internal == SolvedWithPerm;
        }
    private:
        using LLTX = Eigen::LLT<MatrixX>;
        using LLTSparse = Eigen::SimplicialLLT<SparseMatrix>;
        ComputationInfo m_info;
        enum InternalStatus {
            NotSolved,
            SolvedWithNoPerm,
            SolvedWithPerm
        } m_status_internal = NotSolved;


        PermutationMatrix m_perm;
        PermutationMatrix m_perminv;
        std::variant<MatrixX, SparseMatrix> m_L;

        void assert_success() const {
            SSNSDP_ASSERT_MSG(m_status_internal != NotSolved, "Not Initialized");
            SSNSDP_ASSERT_MSG(m_info == Success, "Solver not successful");
        }

        template <typename LType, typename InputType>
        void init_with_L_internal(InputType&& L) {
            m_L.emplace<LType>(std::forward<InputType>(L));
            m_info = Success;
            m_status_internal = SolvedWithNoPerm;
        }

        template <typename LType, typename InputType, typename PermType>
        void init_with_L_internal(InputType&& L, PermType&& perm) {
            m_L.emplace<LType>(std::forward<InputType>(L));
            m_perm = std::forward<PermType>(perm);
            m_perminv = m_perm.inverse();
            m_info = Success;
            m_status_internal = SolvedWithPerm;
        }
    };
}

#endif
