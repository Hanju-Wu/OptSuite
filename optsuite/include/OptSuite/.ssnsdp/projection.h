#ifndef SSNSDP_PROJECTION_H
#define SSNSDP_PROJECTION_H

#include "core.h"

#include <algorithm>
#include <functional>
#include <utility>

#include "eig.h"
#include "cell_array.h"
#include "buffer.h"

namespace ssnsdp {
// On projection, set
// X(abs(X) < SSNSDP_PROJECTION_TRUNCATE_X) = 0
// undefine this macro to set SSNSDP_PROJECTION_TRUNCATE_X to zero
#define SSNSDP_PROJECTION_TRUNCATE_X 1e-14

    struct Project2Info {
        VectorX dd;
        // posidx is 1:k in SDP block, not stored in linear block
        Size poslen;
        BlockType blktype = BlockType::SDP;

        // P1
        Map<MatrixX, Alignment> positive_part() {
            const Size& n = dd.size();
            return Map<MatrixX, Alignment>(m_V.data(), n, poslen);
        }

        Map<const MatrixX, Alignment> positive_part() const {
            const Size& n = dd.size();
            return Map<const MatrixX, Alignment>(m_V.data(), n, poslen);
        }

        // P2
        Map<MatrixX, Alignment> negative_part() {
            const Size& n = dd.size();
            Scalar* data = aligned_offset(m_V.data(), n * poslen);
            return Map<MatrixX, Alignment>(data, n, n - poslen);
        }

        Map<const MatrixX, Alignment> negative_part() const {
            const Size& n = dd.size();
            const Scalar* data = aligned_offset(m_V.data(), n * poslen);
            return Map<const MatrixX, Alignment>(data, n, n - poslen);
        }

        Map<MatrixX, Alignment> Dsch12() {
            Size rows;
            Size cols;
            Dsch12_get_info(rows, cols);
            return Map<MatrixX, Alignment>(m_Dsch12.data(), rows, cols);
        }

        Map<const MatrixX, Alignment> Dsch12() const {
            Size rows;
            Size cols;
            Dsch12_get_info(rows, cols);
            return Map<const MatrixX, Alignment>(m_Dsch12.data(), rows, cols);
        }

        // call after blktype is set
        void allocate(const Size n) {
            dd.resize(n);
            if (blktype == BlockType::SDP) {
                m_V.resize(n * n + BufferMaxPadding<Scalar>);
                m_Dsch12.resize(n * n);
            } else {
                SSNSDP_ASSUME(blktype == BlockType::LINEAR);
                // m_V is not used in linear block
                m_Dsch12.resize(n);
            }
        }

        template <typename Derived>
        static void project2_impl(
            const BlockSpec& blk,
            const MatrixBase<Derived>& X_,
            MatrixX& Xp, Project2Info& par)
        {
            const Derived& Y = X_.derived();
            constexpr Scalar tol = 1e-15_s;
            constexpr Scalar addtol = 1e-6_s;

            par.blktype = blk.type;
            const Size n = blk.n;
            par.allocate(n);

            if (blk.is_sdp()) {
                auto V_tmp = allocate_matrix<0>(n, n);
                auto dd_tmp = allocate_vector<1>(n);
#ifdef SSNSDP_PROJECTION_TRUNCATE_X
                constexpr Scalar truc_X = as_scalar(SSNSDP_PROJECTION_TRUNCATE_X);
                static_assert(truc_X > 0,
                    "SSNSDP_PROJECTION_TRUNCATE_X should be either undefined or greater than 0");
                auto X = allocate_matrix<2>(n, n);
                // X(abs(X) < 1e-14) = 0
                X = (Y.array().abs() < truc_X).select(Zero, Y);
#else
                const Derived& X = Y;
#endif
                // [V, D] = eig(X)
                // d = diag(D)
                // and then sort in descending order
                V_tmp = X;
#ifdef SSNSDP_MKL_EIG_ENABLED
                par.dd.resize(n);
                SelfAdjointEigenSolver::compute_impl(V_tmp, dd_tmp);
                // no aliasing here!
                par.dd = dd_tmp.reverse();
                const auto& eigvec = V_tmp;
#else
                SelfAdjointEigenSolver eigensolver(V_tmp);
                par.dd = eigensolver.eigenvalues().reverse();
                const auto& eigvec = eigensolver.eigenvectors();
#endif

                const VectorX& d = par.dd;

                // posidx = find(d > tol)
                par.poslen = std::lower_bound(d.data(), d.data() + n, tol, std::greater<Scalar>()) - d.data();
                const Size& r = par.poslen;
                const Size s = n - r;

                auto P1 = par.positive_part();
                auto P2 = par.negative_part();
                P1 = eigvec.rightCols(r).rowwise().reverse();
                P2 = eigvec.leftCols(s).rowwise().reverse();

                if (r == 0) {
                    Xp.setZero(n, n);
                    // par.Dsch12 = [];
                } else if (r == n) {
                    Xp = X;
                    // par.Dsch12 = [];
                } else {
                    // dp = abs(d(posidx));
                    // dn = abs(d(negidx));
                    auto& d_abs = dd_tmp;
                    d_abs = d.cwiseAbs();
                    const Ref<const VectorX, Alignment>& dp = d_abs.head(r);
                    const Ref<const VectorX>& dn = d_abs.tail(s);

                    if (r <= s) {
                        Map<MatrixX, Alignment> p1dp(V_tmp.data(), n, r);
                        p1dp = P1 * dp.asDiagonal();
                        Xp.noalias() = p1dp * P1.adjoint();
                    } else {
                        Map<MatrixX, Alignment> p2dn(V_tmp.data(), n, s);
                        p2dn = P2 * dn.asDiagonal();
                        Xp.noalias() = X + p2dn * P2.adjoint();
                    }

                    // Dsch12 = (dp*ones(1,s))./(dp*ones(1,s) + ones(r,1)*dn');
                    auto Dsch12 = par.Dsch12();
                    Dsch12 = dp.replicate(1, s);
                    Dsch12 = Dsch12
                        .cwiseQuotient(Dsch12 + dn.adjoint().replicate(r, 1))
                        .cwiseMax(addtol);
                }

                deactivate_buffer<0>();
                deactivate_buffer<1>();
#ifdef SSNSDP_PROJECTION_TRUNCATE_X
                deactivate_buffer<2>();
#endif
            } else {
                SSNSDP_ASSUME(blk.is_linear());

                // only an alias
                const Derived& X = Y;
                SSNSDP_ASSERT(X.cols() == 1);
                par.dd = X;
                auto Dsch12 = par.Dsch12();
                Dsch12 = (X.array() > tol).template cast<Scalar>();
                Xp = Dsch12.cast<bool>().select(X, Zero);
            }
        }
    private:
        VectorX m_V;
        VectorX m_Dsch12;

        void Dsch12_get_info(Size& rows, Size& cols) const {
            const Size n = dd.size();
            if (blktype == BlockType::SDP) {
                rows = poslen;
                cols = n - rows;
            } else {
                SSNSDP_ASSUME(blktype == BlockType::LINEAR);
                rows = n;
                cols = 1;
            }
        }
    };

    inline void project2_impl(
        const CellArray<BlockSpec>& blk,
        const CellArray<MatrixX>& X,
        CellArray<MatrixX>& Xp, CellArray<Project2Info>& par)
    {
        const Size numblk = blk.size();

        for (Index p = 0; p < numblk; ++p) {
            Project2Info::project2_impl(blk[p], X[p], Xp[p], par[p]);
        }
    }

    inline std::pair<CellArray<MatrixX>, CellArray<Project2Info>> project2(
        const CellArray<BlockSpec>& blk,
        const CellArray<MatrixX>& X)
    {
        const Size numblk = blk.size();
        CellArray<MatrixX> Xp(numblk);
        CellArray<Project2Info> par(numblk);
        project2_impl(blk, X, Xp, par);
        return std::make_pair(std::move(Xp), std::move(par));
    }

    // blkprojSDP(blk, X, out MatrixX, out rrank)
    template <typename Derived>
    void blkprojSDP_impl(const BlockSpec& blk, const MatrixBase<Derived>& X_, MatrixX& Xp, Size& rrank) {
        const Derived& X = X_.derived();

        if (blk.is_linear()) {
            Xp = X.cwiseMax(Zero);
            return;
        }

        const Size n = X.rows();
        constexpr Scalar smtol = Zero;
        SSNSDP_ASSERT_MSG(n == X.cols(), "X must be square");
        SSNSDP_ASSUME(blk.is_sdp());

#ifdef SSNSDP_MKL_EIG_ENABLED
        auto V = allocate_matrix<0>(n, n);
        auto d = allocate_vector<1>(n);
        V = X;
        SelfAdjointEigenSolver::compute_impl(V, d);
#else
        SelfAdjointEigenSolver eigensolver(X);
        const VectorX& d = eigensolver.eigenvalues();
        const MatrixX& V = eigensolver.eigenvectors();
#endif

        const auto& [negidx, posidx] = std::equal_range(d.data(), d.data() + n, smtol);
        const Size neglen = negidx - d.data();
        const Size poslen = d.data() + n - posidx;

        if (poslen == 0) {
            rrank = 0;
            Xp.setZero(n, n);
        } else if (neglen == 0) {
            rrank = 0;
            Xp = X;
        } else if (poslen <= neglen) {
            rrank = poslen;
            const Ref<const VectorX>& dp = d.tail(poslen);
            const Ref<const MatrixX>& Vtmp = V.rightCols(poslen);

            auto Vdp = allocate_matrix<2>(n, poslen);
            Vdp = Vtmp * dp.asDiagonal();
            Xp.noalias() = Vdp * Vtmp.adjoint();
            deactivate_buffer<2>();
        } else {
           rrank = neglen;
           const Ref<const VectorX>& dn = d.head(neglen);
           const Ref<const MatrixX, Alignment>& Vtmp = V.leftCols(neglen);
           auto Vdn = allocate_matrix<2>(n, neglen);
           Vdn = Vtmp * dn.asDiagonal();
           Xp.noalias() = X - Vdn * Vtmp.adjoint();
           deactivate_buffer<2>();
        }

#ifdef SSNSDP_MKL_EIG_ENABLED
        deactivate_buffer<0>();
        deactivate_buffer<1>();
#endif
    }

    void blkprojSDP_impl(
        const CellArray<BlockSpec>& blk,
        const CellArray<MatrixX>& X,
        CellArray<MatrixX>& Xp,
        CellArray<Size>& rrank)
    {
        const Size numblk = blk.size();
        SSNSDP_ASSERT(Xp.size() == numblk);
        SSNSDP_ASSERT(rrank.size() == numblk);
        for (Index p = 0; p < numblk; ++p) {
            blkprojSDP_impl(blk[p], X[p], Xp[p], rrank[p]);
        }
    }
}

#endif
