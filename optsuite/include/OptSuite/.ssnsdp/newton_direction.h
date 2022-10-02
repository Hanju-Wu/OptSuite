#ifndef SSNSDP_NEWTON_DIRECTION_H
#define SSNSDP_NEWTON_DIRECTION_H

#include "core.h"
#include "projection.h"
#include "chol_solver.h"
#include "cell_array.h"
#include "buffer.h"
#include "mkl_sparse.h"
#include "mkl_dense.h"

namespace ssnsdp {
    class NewtonDirectionSolver {
    public:
        Size maxit = 0;
        Scalar tol = Zero;
        Size stagnate_check = 20;
        Size miniter = 1;
        Scalar epsilon = One;
        Scalar sig = One;

        // observer pointers
        const CellArray<BlockSpec>* blk_p;
        const CellArray<SparseMatrix>* At_p;
        const VectorX* b_p;
        const CholSolver* AL_p;

        const CellArray<Project2Info>* par_p;
#ifdef SSNSDP_MKL_SPARSE_ENABLED
        const CellArray<MKLSparse>* At_mkl_p;
#endif

        // workspace: an aligned buffer with at least trans_mdim * 8 + BufferMaxPadding<Scalar> * 7
        Scalar* workspace;

        // Hotspot of the whole program. Optimized by hand.
        void solve_impl(std::vector<Scalar>& m_resnrm)
        {
            // ComputationInfo m_info;
            const CellArray<BlockSpec>& blk = *blk_p;
            const CellArray<SparseMatrix>& At = *At_p;
            const VectorX& b = *b_p;
            const CholSolver& AL = *AL_p;
            const CellArray<Project2Info>& par = *par_p;
#ifdef SSNSDP_MKL_SPARSE_ENABLED
            const CellArray<MKLSparse>& At_mkl = *At_mkl_p;
#endif
            const Size N = b.size();
            if (maxit <= 0) {
                if (N <= 2500) {
                    maxit = 50;
                } else {
                    maxit = as_int(sqrt(as_scalar(N)));
                }
            }
            if (tol <= 0) {
                tol = 1e-6_s * b.norm();
            }

            Map<VectorX, Alignment> x(workspace, N);
            x.setZero();

            Map<VectorX, Alignment> r(aligned_offset(x), N);
            r = b;
            Scalar err = r.norm();
            m_resnrm.clear();
            m_resnrm.push_back(err);
            Scalar minres = err;
            // no precondition
            Map<VectorX, Alignment> q(aligned_offset(r), N);
            q = r;
            Map<VectorX, Alignment> Aq(aligned_offset(q), N);
            Scalar tau_old = q.norm();
            Scalar rho_old = r.dot(q);
            Scalar theta_old = Zero;
            Map<VectorX, Alignment> d(aligned_offset(Aq), N);
            d.setZero();
            Map<VectorX, Alignment> res(aligned_offset(d), N);
            res = r;
            Map<VectorX, Alignment> Ad(aligned_offset(res), N);
            Ad.setZero();
            Map<VectorX, Alignment> y(aligned_offset(Ad), N);

            constexpr Scalar tiny = 1e-30_s;

            Index iter = 0;
            for (; iter < maxit; ++iter) {
                puts_verbose("Calculate sig * A(PxP) Dsch (PtxPt) At");
                {
                    Aq.setZero();
                    const auto& yorg = q;
                    y = AL.bwsolve(q);

                    for (Size p = 0; p < blk.size(); ++p) {
                        const BlockSpec& pblk = blk[p];
                        const SparseMatrix& Atp = At[p];
#ifdef SSNSDP_MKL_SPARSE_ENABLED
                        const MKLSparse& At_mklp = At_mkl[p];
#endif
                        const Size n = pblk.n;
                        const Size n2 = n * n + BufferMaxPadding<Scalar>;
                        const Project2Info& parp = par[p];
                        Map<const MatrixX, Alignment> Dsch12p = parp.Dsch12();

                        if (pblk.is_sdp()) {
                            // buffer 0 used in Atyfun and AXfun
                            auto Aty = allocate_matrix<3>(n, n);
#ifdef SSNSDP_MKL_SPARSE_ENABLED
                            Atyfun_upper_impl(pblk, At_mklp, y, Aty);
#else
                            Atyfun_upper_impl(pblk, Atp, y, Aty);
#endif
                            const Size rr = parp.poslen;
                            const Size n_rr = n - rr;

                            Map<const MatrixX, Alignment> P1p = parp.positive_part();
                            Map<const MatrixX, Alignment> P2p = parp.negative_part();

                            if (rr > 0 && rr < n) {
                                const auto tmp0_buffer = allocate_scalars<1>(n2);
                                const auto tmp1_buffer = allocate_scalars<2>(n2);
                                // tmp3 share buffer with tmp11 and tmp21
                                Map<MatrixX, Alignment> tmp3(tmp1_buffer, n, n);
                                // tmp4 share buffer with tmp0 and tmp1
                                Map<MatrixX, Alignment> tmp4(tmp0_buffer, n, n);
                                if (rr * 2 <= n) {
                                    Map<RowMajorMatrixX, Alignment> tmp0(tmp0_buffer, rr, n);
                                    Map<MatrixX, Alignment> tmp1(aligned_offset(tmp0), rr, n);
                                    Map<MatrixX, Alignment> tmp11(tmp1_buffer, rr, rr);
                                    Map<MatrixX, Alignment> tmp21(aligned_offset(tmp11), rr, n_rr);
                                    // no extra allocation when tmp0 is row-majored
                                    tmp0.noalias() = P1p.adjoint() * Aty.selfadjointView<Upper>();
                                    if constexpr (UseMKLDense) {
#ifdef SSNSDP_USE_MKL
                                        gemm_impl(tmp0, P1p, tmp11, 0.5_s);
                                        gemm_impl(tmp0, P2p, tmp21);
#else
                                        SSNSDP_UNREACHABLE();
#endif
                                    } else {
                                        tmp11.noalias() = 0.5_s * (tmp0 * P1p);
                                        tmp21.noalias() = tmp0 * P2p;
                                    }
                                    tmp21 = Dsch12p.cwiseProduct(tmp21);
                                    
                                    if constexpr (UseMKLDense) {
#ifdef SSNSDP_USE_MKL
                                        gemm_nt_impl(tmp11, P1p, tmp1);
                                        gemm_nt_impl(tmp21, P2p, tmp1, One, One);
                                        gemm_impl(P1p, tmp1, tmp3);
#else
                                        SSNSDP_UNREACHABLE();
#endif
                                    } else {
                                        tmp1.noalias() = tmp11 * P1p.adjoint() + tmp21 * P2p.adjoint();
                                        // Eigen seems to have some overhead here ...
                                        tmp3.noalias() = P1p * tmp1;
                                    }
                                    
                                    tmp4.triangularView<Upper>() = tmp3 + tmp3.adjoint();
#ifdef SSNSDP_MKL_SPARSE_ENABLED
                                    AXfun_accumulate(pblk, At_mklp, tmp4, Aq);
#else
                                    AXfun_accumulate(pblk, Atp, tmp4, Aq);
#endif
                                } else {
                                    Map<RowMajorMatrixX, Alignment> tmp0(tmp0_buffer, n_rr, n);
                                    Map<MatrixX, Alignment> tmp1(aligned_offset(tmp0), n_rr, n);
                                    Map<MatrixX, Alignment> tmp11(tmp1_buffer, n_rr, n_rr);
                                    Map<MatrixX, Alignment> tmp21(aligned_offset(tmp11), n_rr, rr);
                                    tmp0.noalias() = P2p.adjoint() * Aty.selfadjointView<Upper>();
                                    if constexpr (UseMKLDense) {
#ifdef SSNSDP_USE_MKL
                                        gemm_impl(tmp0, P2p, tmp11, 0.5_s);
                                        gemm_impl(tmp0, P1p, tmp21);
#else
                                        SSNSDP_UNREACHABLE();
#endif
                                    } else {
                                        tmp11.noalias() = 0.5_s * (tmp0 * P2p);
                                        tmp21.noalias() = tmp0 * P1p;
                                    }
                                    tmp21 = (MatrixX::Ones(n_rr, rr) - Dsch12p.adjoint())
                                        .cwiseProduct(tmp21);
                                    if constexpr (UseMKLDense) {
#ifdef SSNSDP_USE_MKL
                                        gemm_nt_impl(tmp11, P2p, tmp1);
                                        gemm_nt_impl(tmp21, P1p, tmp1, One, One);
                                        gemm_impl(P2p, tmp1, tmp3);
#else
                                        SSNSDP_UNREACHABLE();
#endif
                                    } else {
                                        tmp1.noalias() = tmp11 * P2p.adjoint() + tmp21 * P1p.adjoint();
                                        tmp3.noalias() = P2p * tmp1;
                                    }
                                    // AXfun only cares about upper part, no selfadjointView is required
                                    tmp4.triangularView<Upper>() = Aty - tmp3 - tmp3.adjoint();
#ifdef SSNSDP_MKL_SPARSE_ENABLED
                                    AXfun_accumulate(pblk, At_mklp, tmp4, Aq);
#else
                                    AXfun_accumulate(pblk, Atp, tmp4, Aq);
#endif
                                }
                            } else if (rr == n) {
                                // AXfun only cares about upper part, no selfadjointView is required
#ifdef SSNSDP_MKL_SPARSE_ENABLED
                                AXfun_accumulate(pblk, At_mklp, Aty, Aq);
#else
                                AXfun_accumulate(pblk, Atp, Aty, Aq);
#endif
                            }
                            deactivate_buffer<1>();
                            deactivate_buffer<2>();
                            deactivate_buffer<3>();
                        } else {
                            SSNSDP_ASSUME(pblk.is_linear());
                            auto Aty = allocate_matrix<3>(n, 1);
#ifdef SSNSDP_MKL_SPARSE_ENABLED
                            At_mklp.multiply_impl(y, Aty);
#else
                            Aty.noalias() = Atp * y;
#endif
                            Aty = Dsch12p.cwiseProduct(Aty);
#ifdef SSNSDP_MKL_SPARSE_ENABLED
                            At_mklp.multiply_impl<MKLOperator::Transpose>(Aty, Aq, One);
#else
                            Aq.noalias() += Atp.adjoint() * Aty;
#endif
                            deactivate_buffer<3>();
                        }
                    }

                    Aq *= sig;
                    AL.fwsolve_in_place(Aq);
                    Aq += epsilon * yorg;
                }

                const Scalar sigma = q.dot(Aq);

                if (abs(sigma) < tiny) {
                    // m_info = Success;
                    break;
                }

                Scalar alpha = rho_old / sigma;
                r -= alpha * Aq;

                // u = precondfun(blk, At, par, L, r)
                const auto& u = r;
                const Scalar theta = u.norm() / tau_old;
                const Scalar c = One / sqrt(One + theta * theta);
                const Scalar tau = tau_old * theta * c;
                const Scalar gam = c * c * theta_old * theta_old;
                const Scalar eta = c * c * alpha;
                d = gam * d + eta * q;
                x += d;

                Ad = gam * Ad + eta * Aq;

                res -= Ad;

                err = res.norm();
                m_resnrm.push_back(err);

                if (err < minres) {
                    minres = err;
                }

                if ((err < tol && iter > miniter && b.dot(x) > 0) || abs(rho_old) < tiny) {
                    // m_info = Success;
                    break;
                }

                const Scalar rho = r.dot(u);
                const Scalar beta = rho / rho_old;
                q = u + beta * q;
                rho_old = rho;
                tau_old = tau;
                theta_old = theta;
            } // iteration

            // if (iter == maxit) {
            //     m_info = NoConvergence;
            // }
            //
            // return m_info;
        }

        Map<const VectorX, Alignment> solution() const {
            return Map<const VectorX, Alignment>(workspace, b_p->size());
        }
    };
}

#endif
