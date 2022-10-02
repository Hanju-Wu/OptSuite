#ifndef SSNSDP_GRAD_PHI_H
#define SSNSDP_GRAD_PHI_H

#include "core.h"
#include "projection.h"
#include "cell_array.h"
#include "buffer.h"
#include "mkl_dense.h"
#include "log.h"

namespace ssnsdp {

    SSNSDP_WEAK_INLINE
    void grad_phi_impl(
        const CellArray<BlockSpec>& blk,
        const CellArray<Project2Info>& par,
        const CellArray<MatrixX>& H,
        CellArray<MatrixX>& Y)
    {
        const Size blknum = blk.size();
        for (Index p = 0; p < blknum; ++p) {
            const BlockSpec& pblk = blk[p];
            const Size n = pblk.n;
            const Size n2 = n * n + BufferMaxPadding<Scalar>;
            const Project2Info& parp = par[p];
            const auto& Dsch12p = parp.Dsch12();
            const MatrixX& Hp = H[p];
            MatrixX& Yp = Y[p];

            if (pblk.is_sdp()) {
                const Size rr = parp.poslen;
                const Size n_rr = n - rr;

                const auto& P1p = parp.positive_part();
                const auto& P2p = parp.negative_part();
                if (rr > 0 && rr < n) {
                    const auto tmp0_buffer = allocate_scalars<1>(n2);
                    const auto tmp1_buffer = allocate_scalars<2>(n2);
                    // tmp3 share buffer with tmp11 and tmp21
                    Map<MatrixX, Alignment> tmp3(tmp1_buffer, n, n);
                    if (rr * 2 <= n) {
                        Map<MatrixX, Alignment> tmp0(tmp0_buffer, rr, n);
                        Map<MatrixX, Alignment> tmp1(aligned_offset(tmp0), rr, n);
                        Map<MatrixX, Alignment> tmp11(tmp1_buffer, rr, rr);
                        Map<MatrixX, Alignment> tmp21(aligned_offset(tmp11), rr, n_rr);
                        if constexpr (UseMKLDense) {
#ifdef SSNSDP_USE_MKL
                            gemm_tn_impl(P1p, Hp, tmp0);
                            gemm_impl(tmp0, P1p, tmp11, 0.5_s);
                            gemm_impl(tmp0, P2p, tmp21);
#else
                            SSNSDP_UNREACHABLE();
#endif
                        } else {
                            tmp0.noalias() = P1p.adjoint() * Hp;
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
                            tmp3.noalias() = P1p * tmp1;
                        }
                        Yp = tmp3 + tmp3.adjoint();
                    } else {
                        Map<MatrixX, Alignment> tmp0(tmp0_buffer, n_rr, n);
                        Map<MatrixX, Alignment> tmp1(aligned_offset(tmp0), n_rr, n);
                        Map<MatrixX, Alignment> tmp11(tmp1_buffer, n_rr, n_rr);
                        Map<MatrixX, Alignment> tmp21(aligned_offset(tmp11), n_rr, rr);
                        if constexpr (UseMKLDense) {
#ifdef SSNSDP_USE_MKL
                            gemm_tn_impl(P2p, Hp, tmp0);
                            gemm_impl(tmp0, P2p, tmp11, 0.5_s);
                            gemm_impl(tmp0, P1p, tmp21);
#else
                            SSNSDP_UNREACHABLE();
#endif
                        } else {
                            tmp0.noalias() = P2p.adjoint() * Hp;
                            tmp11.noalias() = 0.5_s * (tmp0 * P2p);
                            tmp21.noalias() = tmp0 * P1p;
                        }
                        tmp21 = (MatrixX::Ones(n_rr, rr) - Dsch12p.adjoint()).cwiseProduct(tmp21);
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
                        Yp = Hp - tmp3 - tmp3.adjoint();
                    }

                    deactivate_buffer<1>();
                    deactivate_buffer<2>();
                } else if (rr == n) {
                    Yp = Hp;
                } else {
                    SSNSDP_ASSUME(rr == 0);
                    Yp.setZero(n, n);
                }
            } else {
                SSNSDP_ASSUME(pblk.is_linear());
                Yp = Dsch12p.cwiseProduct(Hp);
            }
        }
    }
}
#endif
