#ifndef SSNSDP_MODEL_H
#define SSNSDP_MODEL_H

#include "core.h"

#include <utility>
#include <algorithm>
#include <tuple>
#include "mapping.h"
#include "projection.h"
#include "log.h"
#include "geomean.h"
#include "chol_solver.h"
#include "newton_direction.h"
#include "grad_phi.h"
#include "inner_product.h"
#include "cell_array.h"
#include "mkl_sparse.h"

/* (P)max <C, X> s.t. A(X) = b, X \in K,
 * (D)min b^Ty s.t.C = At(y) + S,
 *   where K is a positive semidefinite cone.
 */

namespace ssnsdp {
    class Model {
    public:
        Verbosity verbosity = Verbosity::Detail;
        bool scale_data = true;
        Scalar pmu = 1_s;
        Index maxits = 20000;
        Scalar tol = 1e-6_s;
        Scalar tolscale = 1_s;

        Scalar ADMM_switch_tol = 1e-5_s;
        Index ADMM_print_itr = 20;
        Index ADMM_imaxit = 20000;
        Scalar ADMM_rho = Golden1_618;

        Index NEWT_maxit = 1000;
        Scalar NEWT_sigPow = 0.5_s;
        Scalar NEWT_res_fac = 0.98_s;
        Scalar NEWT_eta1 = 1e-6_s;
        Scalar NEWT_eta2 = 0.9_s;
        Scalar NEWT_gamma1 = 0.5_s;
        Scalar NEWT_gamma2 = 0.95_s;
        Scalar NEWT_gamma3 = 5_s;
        Scalar NEWT_lambda = 1_s;
        Scalar NEWT_sigma = 0.01_s;
        Index NEWT_print_itr = 1;

        Index CG_maxit = 300;
        Scalar CG_tol = 1e-2_s;
        Scalar CG_tolmin = 1e-10_s;
        bool CG_adapt = true;

        bool mu_adp_mu = true;

        Index mu_NEWT_adpmu_cri = 1; // 1 or 2
        Scalar mu_NEWT_mu_min = 1e-6_s;
        Scalar mu_NEWT_mu_max = 1e6_s;
        Index mu_NEWT_mu_update_itr = 10;
        Scalar mu_NEWT_mu_delta = 5e-1_s;
        Scalar mu_NEWT_mu_fact = 5.0_s / 3.0_s;

        Scalar mu_ADMM_ratio = 1_s; // 1 or 0
        Scalar mu_ADMM_mu_max = 1e6_s;
        Scalar mu_ADMM_mu_min = 1e-4_s;
        Scalar mu_ADMM_mu_delta = 1.2_s;
        Scalar mu_ADMM_mu_fact = 1.8_s;

        void solve();

        const CellArray<MatrixX>& X() const { return m_X.var; }
        const VectorX& y() const { return m_y.var; }
        const CellArray<MatrixX>& S() const { return m_S.var; }

        const Scalar& hists_maxinf() const { return trans_hists_maxinf; }
        const std::vector<Scalar>& hists_pobj() const { return trans_hists_pobj; }
        const std::vector<Scalar>& hists_dobj() const { return trans_hists_dobj; }
        const std::vector<Scalar>& hists_gaporg() const { return trans_hists_gaporg; }
        const std::vector<Scalar>& hists_gap() const { return trans_hists_gap; }
        const std::vector<Scalar>& hists_pinf() const { return trans_hists_pinf; }
        const std::vector<Scalar>& hists_pinforg() const { return trans_hists_pinforg; }
        const std::vector<Scalar>& hists_dinf() const { return trans_hists_dinf; }
        const std::vector<Scalar>& hists_dinforg() const { return trans_hists_dinforg; }
        const std::vector<Scalar>& hists_pvd() const { return trans_hists_pvd; }
        const std::vector<Scalar>& hists_dvp() const { return trans_hists_dvp; }
        const std::vector<Scalar>& hists_pvdorg() const { return trans_hists_pvdorg; }
        const std::vector<Scalar>& hists_dvporg() const { return trans_hists_dvporg; }
        const std::vector<Size>& hists_cgiter() const { return trans_hists_cgiter; }
        const std::vector<bool>& hists_is_ssn() const { return trans_hists_isNEWT; }
        const Scalar& result_pobj() const { return trans_rec.pobj; }
        const Scalar& result_dobj() const { return trans_rec.dobj; }
        const Scalar& result_gap() const { return trans_rec.gap; }
        const Index& result_iter() const { return out_iter; }
        const Scalar& result_pinf() const { return trans_rec.pinf; }
        const Scalar& result_dinf() const { return trans_rec.dinf; }
        const std::string& result_status() const { return out_status; }

        CellArray<BlockSpec>& blk() { return trans_blkorg; }
        CellArray<SparseMatrix>& At() { return trans_Atorg; }
        CellArray<SparseMatrix>& C() { return trans_Corg; }
        VectorX& b() { return trans_borg; }

        const CellArray<BlockSpec>& blk() const { return trans_blkorg; }
        const CellArray<SparseMatrix>& At() const { return trans_Atorg; }
        const CellArray<SparseMatrix>& C() const { return trans_Corg; }
        const VectorX& b() const { return trans_borg; }

        Model() = default;
        ~Model() = default;
        Model(const Model&) = default;
        Model(Model&&) = default;
        Model& operator=(const Model&) = default;
        Model& operator=(Model&&) = default;

    private:

        CellArray<BlockSpec> m_blk;
        CellArray<SparseMatrix> m_At;
        CellArray<SparseMatrix> m_C;
        VectorX m_b;

        enum class IterType {
            NEWT,
            ADMM
        };

#ifdef SSNSDP_MKL_SPARSE_ENABLED
        CellArray<MKLSparse> trans_At_mkl;
#endif
        VectorX trans_borg;
        CellArray<SparseMatrix> trans_Corg;
        CellArray<SparseMatrix> trans_Atorg;
        CellArray<BlockSpec> trans_blkorg;
        Scalar trans_normborg;
        Scalar trans_normCorg;

        Size trans_mdim;
        Size trans_nblock;
        Size trans_nmax = 0;

        DiagonalMatrixX trans_scale_DA;
        Scalar trans_scale_bscale;
        Scalar trans_scale_Cscale;
        Scalar trans_scale_objscale;

        CholSolver trans_Lchol;

        VectorX trans_bmAX;
        VectorX trans_AC;
        Scalar trans_normb;
        Scalar trans_normC;
        Scalar trans_pmu;

        CellArray<Size> trans_ADMM_rankS;
        // Size trans_ADMM_recompeig = 0;
        Size trans_ADMM_swt = 0;
        Size trans_ADMM_maxiter;
        Size trans_ADMM_subiter = 0;
        Size trans_ADMM_iter = 0;
        // Scalar trans_ADMM_normAty;
        // Scalar trans_ADMM_normS;
        // Scalar trans_ADMM_normZ1Z2;
        // Scalar trans_ADMM_normX;
        Size trans_muADMM_prim_win = 0;
        Size trans_muADMM_dual_win = 0;

        Scalar trans_NEWT_lambda = Zero;
        Scalar trans_NEWT_sigPow = Zero;
        Size trans_NEWT_iter = 0;
        Size trans_NEWT_swt = 0;
        Size trans_NEWT_subiter = 0;
        Size trans_NEWT_maxiter = 20;
        Scalar trans_NEWT_lastres = Inf;
        Scalar trans_NEWT_nrmd;
        Scalar trans_NEWT_FZd;
        Scalar trans_NEWT_FZdorg;
        Scalar trans_NEWT_FZFZnew;
        Scalar trans_NEWT_thetaFZdorg;
        Scalar trans_NEWT_thetaFZFZnew;
        Scalar trans_NEWT_rhs;
        Scalar trans_NEWT_ratio;
        Size trans_NEWT_CG_maxit;

        IterType trans_last_iter = IterType::ADMM;

        // Size trans_rescale = 1;
        bool trans_ischange = true;
        Size trans_cgiter = 0;
        std::vector<Scalar> trans_cgres;

        bool doNEWT = false;
        Size trans_NEWT_totaliter = 0;
        Scalar trans_NEWT_sig;
        Size trans_ADMM_totaliter = 0;
        Scalar retol = One;

        Scalar trans_hists_maxinf = Inf;
        std::vector<Scalar> trans_hists_pobj;
        std::vector<Scalar> trans_hists_dobj;
        std::vector<Scalar> trans_hists_gaporg;
        std::vector<Scalar> trans_hists_gap;
        std::vector<Scalar> trans_hists_pinf;
        std::vector<Scalar> trans_hists_pinforg;
        std::vector<Scalar> trans_hists_dinf;
        std::vector<Scalar> trans_hists_dinforg;
        std::vector<Scalar> trans_hists_pvd;
        std::vector<Scalar> trans_hists_dvp;
        std::vector<Scalar> trans_hists_pvdorg;
        std::vector<Scalar> trans_hists_dvporg;
        std::vector<Size> trans_hists_cgiter;
        std::vector<bool> trans_hists_isNEWT;

        Index out_iter;

        Scalar trans_AtymCSnrm;
        CellArray<MatrixX> trans_ATymCS;

        bool cstop = false;
        const char* str_head;
        const char* str_head2;
        std::string out_status;

        struct XStruct {
            CellArray<MatrixX> var;
            VectorX Avar;
            VectorX Avarorg;
        } m_X;

        struct ZStruct {
            CellArray<MatrixX> var;
            VectorX Avar;
        } m_Z;

        CellArray<MatrixX> m_W;

        struct yStruct {
            VectorX var;
            CellArray<MatrixX> Avar;
        } m_y;

        ZStruct m_S;

        struct FZStruct {
            CellArray<MatrixX> var;
            VectorX Avar;
            VectorX Avarorg;
            Scalar res;
            CellArray<Project2Info> par;
        } m_FZ;

        struct RecoveredVar {
            Scalar pinf;
            Scalar dinf;
            Scalar gap;
            Scalar pobj;
            Scalar dobj;
            Scalar K1;
            Scalar K1dual;
            Scalar C1;
        } trans_rec;

        // AX = Amap(X)
        inline VectorX Amap(const CellArray<MatrixX>& X) const {
            VectorX AX = AXfun(m_blk, m_At, X);
            trans_Lchol.fwsolve_in_place(AX);
            return AX;
        }

        template <typename DerivedOut>
        SSNSDP_STRONG_INLINE
        void Amap_impl(const CellArray<MatrixX>& X, MatrixBase<DerivedOut> const& AX_) const {
            EIGEN_STATIC_ASSERT_VECTOR_ONLY(DerivedOut);
            DerivedOut& AX = AX_.const_cast_derived();
            AXfun_impl(m_blk, m_At, X, AX);
            trans_Lchol.fwsolve_in_place(AX);
        }

        // [AX, AXorg] = Amap_pair(X)
        inline std::pair<VectorX, VectorX> Amap_pair(const CellArray<MatrixX>& X) const {
            VectorX AXorg = AXfun(m_blk, m_At, X);
            VectorX AX = trans_Lchol.fwsolve(AXorg);
            return std::make_pair(std::move(AX), std::move(AXorg));
        }

        template <typename DerivedA, typename DerivedB>
        SSNSDP_STRONG_INLINE
        void Amap_pair_impl(const CellArray<MatrixX>& X,
            MatrixBase<DerivedA> const& AX_,
            MatrixBase<DerivedB> const& AXorg_) const
        {
            EIGEN_STATIC_ASSERT_VECTOR_ONLY(DerivedA);
            EIGEN_STATIC_ASSERT_VECTOR_ONLY(DerivedB);
            DerivedA& AX = AX_.const_cast_derived();
            DerivedB& AXorg = AXorg_.const_cast_derived();
            AXfun_impl(m_blk, m_At, X, AXorg);
            AX = AXorg;
            trans_Lchol.fwsolve_in_place(AX);
        }

#ifdef SSNSDP_MKL_SPARSE_ENABLED
        template <typename Derived>
        SSNSDP_STRONG_INLINE
        void ATmap_impl(const MatrixBase<Derived>& y, CellArray<MatrixX>& out) const {
            EIGEN_STATIC_ASSERT_VECTOR_ONLY(Derived);
            Atyfun_impl(m_blk, trans_At_mkl, trans_Lchol.bwsolve(y), out);
        }
#else
        template <typename Derived>
        SSNSDP_STRONG_INLINE
        void ATmap_impl(const MatrixBase<Derived>& y, CellArray<MatrixX>& out) const {
            EIGEN_STATIC_ASSERT_VECTOR_ONLY(Derived);
            Atyfun_impl(m_blk, m_At, trans_Lchol.bwsolve(y), out);
        }
#endif
        // Ay = ATmap(y)
        template <typename Derived>
        CellArray<MatrixX> ATmap(const MatrixBase<Derived>& y) const {
            CellArray<MatrixX> result = cell_empty();
            ATmap_impl(y, result);
            return result;
        }

        CellArray<MatrixX> cell_zeros() const;
        CellArray<MatrixX> cell_empty() const;

        void comp_res_fp_impl(const ZStruct& Z, FZStruct& FZ, XStruct& X) const;

        void record_optimal_NEWT();

        void recover_var_chk_impl(CellArray<MatrixX>& X, VectorX& y, CellArray<MatrixX>& S, RecoveredVar& rec) const;
    };

    SSNSDP_WEAK_INLINE
    void Model::solve() {
        // for temporary variables
        CellArray<MatrixX> tmp_mats;
        CellArray<MatrixX> tmp_mats_2;
        CellArray<MatrixX> tmp_newt_direction;
        CellArray<MatrixX> tmp_Dsch12;
        CellArray<MatrixX> tmp_grad_phi;
        VectorX tmp_vec;
        VectorX tmp_vec_2;
        ZStruct tmp_z_struct;
        FZStruct tmp_fz_struct;
        XStruct tmp_x_struct;
        RecoveredVar tmp_rec;

        puts_verbose("Initialization");
        {
            trans_pmu = pmu;
            m_b = b();
            m_C = C();
            m_At = At();
            m_blk = blk();
            trans_normborg = max(One, m_b.norm());
            trans_normCorg = max(One, m_C.norm());
            trans_mdim = m_b.size();
            trans_nblock = m_blk.size();

            for (Index k = 0; k < trans_nblock; ++k) {
                const auto& kblk = m_blk[k];
                if (trans_nmax < kblk.n) {
                    trans_nmax = kblk.n;
                }
            }

            allocate_initialize(trans_nmax * trans_nmax + BufferMaxPadding<Scalar>);
        }

        puts_verbose("Preprocessing SDP");
        {
            if (scale_data) {
                VectorX& normA = tmp_vec;
                normA.setZero(trans_mdim);
                // normA = sum(At .* At)';
                for (Size k = 0; k < trans_nblock; ++k) {
                    normA.noalias() +=
                        m_At[k].cwiseProduct(m_At[k]).adjoint() * VectorX::Ones(m_At[k].rows());
                }
                // normA = 1 ./ max(1, sqrt(normA));
                normA = normA.cwiseSqrt().cwiseMax(One).cwiseInverse();
                // DA = spdiags(DA, 0, trans_mdim, trans_mdim);
                trans_scale_DA = normA.asDiagonal();
            } else {
                // DA = speye(trans_mdim);
                trans_scale_DA = VectorX::Ones(trans_mdim).asDiagonal();
                trans_scale_bscale = One;
                trans_scale_Cscale = One;
                trans_scale_objscale = One;
            }

            m_b = trans_scale_DA * m_b;

#ifdef SSNSDP_MKL_SPARSE_ENABLED
            trans_At_mkl.resize(trans_nblock);
#endif

            SparseMatrix AAt(VectorX::Constant(trans_mdim, 1e-13_s).asDiagonal());
            for (Index k = 0; k < trans_nblock; ++k) {
                m_At[k] = m_At[k] * trans_scale_DA;
                AAt += m_At[k].adjoint() * m_At[k];
#ifdef SSNSDP_MKL_SPARSE_ENABLED
                trans_At_mkl[k].reset(m_At[k]);
#endif
            }

            trans_Lchol = CholSolver(AAt);
            trans_Lchol.fwsolve_in_place(m_b);

            if (scale_data) {
                trans_scale_bscale = max(One, m_b.norm());
                trans_scale_Cscale = max(One, m_C.norm());
                trans_scale_objscale = trans_scale_bscale * trans_scale_Cscale;
                m_b /= trans_scale_bscale;
                m_C /= trans_scale_Cscale;
            }
        }

        puts_verbose("More initialization");
        {
            m_X.var = cell_zeros();
            m_X.Avar.setZero(trans_mdim);
            m_X.Avarorg.setZero(trans_mdim);
            m_Z.var = cell_zeros();
            m_Z.Avar.setZero(trans_mdim);
            m_W = m_Z.var;
            m_y.var.setZero(trans_mdim);
            m_y.Avar = m_X.var;
            m_S.var = cell_zeros();
            m_S.Avar.setZero(trans_mdim);

            trans_bmAX = m_b - m_X.Avar;

            trans_AC = Amap(m_C.cast<MatrixX>());
            trans_normb = max(One, m_b.norm());
            trans_normC = max(One, m_C.norm());

            m_FZ.var = cell_empty();
            m_FZ.par.resize(trans_nblock);
            comp_res_fp_impl(m_Z, m_FZ, m_X);

            // initial_ADMM_trans
            trans_ADMM_rankS = CellArray<Size>(trans_nblock);
            trans_ADMM_maxiter = ADMM_imaxit;

            // initial_NEWT_trans
            trans_NEWT_lambda = NEWT_lambda;
            trans_NEWT_sigPow = NEWT_sigPow;

            tmp_mats = cell_empty();
            tmp_mats_2 = cell_empty();
            tmp_grad_phi = cell_empty();
            tmp_newt_direction = cell_empty();
            tmp_fz_struct.par.resize(trans_nblock);
            tmp_fz_struct.var = cell_empty();
            tmp_x_struct.var = cell_empty();
            tmp_x_struct.Avar.setZero(trans_mdim);
            tmp_x_struct.Avarorg.setZero(trans_mdim);
            tmp_z_struct.var = cell_empty();
        }

        puts_verbose("Set up print format");
        if (verbosity >= Verbosity::Detail) {
            str_head2 =
                " iter    "
                "mu      "
                "pobj      "
                "dobj      "
                "gap      "
                "pinf    "
                "dinf    "
                "gaporg   "
                "pinforg  "
                "dinforg   "
                "res       "
                "sig    "
                "CG_tol   "
                "CG_iter";
            str_head =
                " iter    "
                "mu      "
                "pobj      "
                "dobj      "
                "gap      "
                "pinf    "
                "dinf    "
                "gaporg   "
                "pinforg  "
                "dinforg";
            puts_info(str_head);
        }

        puts_verbose("Main iteration");
        Index iter = 1;
        for (; iter <= maxits; ++iter) {
            puts_verbose("Check condition for NEWT");
            {
                if (doNEWT) {
                    doNEWT = (trans_NEWT_iter != trans_NEWT_maxiter && trans_NEWT_swt <= 10);
                } else if (trans_ADMM_iter == trans_ADMM_maxiter || trans_ADMM_swt > 5) {
                    doNEWT = true;
                } else if (iter <= 200) {
                    doNEWT = false;
                } else {
                    const Map<const VectorX> pinf_v(trans_hists_pinforg.data(), trans_hists_pinforg.size());
                    const Map<const VectorX> dinf_v(trans_hists_dinforg.data(), trans_hists_dinforg.size());

                    SSNSDP_ASSUME(pinf_v.size() >= 30);

                    const Scalar pinf_tail_mean = pinf_v.tail<6>().mean();
                    const Scalar dinf_tail_mean = dinf_v.tail<6>().mean();

                    const Scalar pinf_recent_mean = pinf_v.segment<6>(pinf_v.size() - 26).mean();
                    const Scalar dinf_recent_mean = dinf_v.segment<6>(dinf_v.size() - 26).mean();

                    const Scalar pcri = pinf_tail_mean / pinf_recent_mean;
                    const Scalar dcri = dinf_tail_mean / dinf_recent_mean;

                    SSNSDP_ASSUME(iter > 200);

                    // (a || b) && ( (c && d) || (e && f) ) )
                    if ((tolscale >= 100_s || (iter > 500 && trans_hists_maxinf > tol * 3_s)) &&
                        ((pcri > 1.1_s && dcri > 0.95_s) || (pcri > 1.2_s && dcri > 0.93_s)))
                    {
                        doNEWT = true;
                    } else if (iter > 1500 && pinf_tail_mean > 1e-3_s) {
                        doNEWT = true;
                    } else if (iter > 2000 && pinf_tail_mean > 1e-4_s) {
                        doNEWT = true;
                    } else if (tolscale >= 100_s && iter > 1000) {
                        if (dcri > 0.9_s && pcri > 0.9_s) {
                            doNEWT = true;
                        }
                    }
                }
            }

            if (doNEWT) {
                if (trans_last_iter == IterType::ADMM) {
                    for (Index k = 0; k < trans_nblock; ++k) {
                        m_Z.var[k] = m_X.var[k] - trans_pmu * m_S.var[k];
                    }
                    Amap_impl(m_Z.var, m_Z.Avar);

                    comp_res_fp_impl(m_Z, m_FZ, m_X);

                    puts_verbose("Switching to NEWT. Setting params");

                    trans_last_iter = IterType::NEWT;
                    trans_NEWT_iter = 0;
                    trans_NEWT_swt = 0;

                    ++trans_NEWT_subiter;
                    trans_NEWT_maxiter = 400;
                }

                ++trans_NEWT_iter;
                ++trans_NEWT_totaliter;
                trans_NEWT_sig = trans_NEWT_lambda * pow(m_FZ.res, trans_NEWT_sigPow);

                if (CG_adapt) {
                    CG_tol = max(0.1_s * min(m_FZ.res, One), CG_tolmin) * 0.001_s;
                }

                CellArray<MatrixX>& d = tmp_newt_direction;
                puts_verbose("Generate direction");
                {
                    const Scalar& sig = trans_NEWT_sig;

                    // iHW.Dsch12 = (iHW.Dsch12*sig)./(sig+1-iHW.Dsch12);
                    for (Index k = 0; k < trans_nblock; ++k) {
                        auto Dsch12 = m_FZ.par[k].Dsch12();
                        // save Dsch12 for restoration
                        Map<MatrixX> Dsch12_orig(tmp_mats[k].data(), Dsch12.rows(), Dsch12.cols());
                        Dsch12_orig = Dsch12;
                        Dsch12 = (Dsch12 * sig).cwiseQuotient(
                            MatrixX::Constant(Dsch12.rows(), Dsch12.cols(), sig + One) - Dsch12);
                    }

                    Scalar iHW_sig = -One;
                    Scalar iHW_epsilon = sig / (One + 2_s * sig);
                    grad_phi_impl(m_blk, m_FZ.par, m_FZ.var, tmp_grad_phi);
                    for (Index k = 0; k < trans_nblock; ++k) {
                        tmp_grad_phi[k] = tmp_grad_phi[k] * iHW_sig + m_FZ.var[k] * iHW_epsilon;
                    }
                    VectorX& aaa = tmp_vec;
                    Amap_impl(tmp_grad_phi, aaa);
                    aaa = -aaa;

                    NewtonDirectionSolver direction_solver;
                    direction_solver.epsilon = sig * sig / (One + 2_s * sig);
                    direction_solver.sig = One;
                    direction_solver.tol = CG_tol;
                    direction_solver.maxit = CG_maxit;
                    direction_solver.blk_p = &m_blk;
                    direction_solver.At_p = &m_At;
                    direction_solver.b_p = &aaa;
                    direction_solver.AL_p = &trans_Lchol;
                    direction_solver.par_p = &m_FZ.par;
#ifdef SSNSDP_MKL_SPARSE_ENABLED
                    direction_solver.At_mkl_p = &trans_At_mkl;
#endif
                    direction_solver.workspace =
                        allocate_scalars<4>(trans_mdim * 8 + BufferMaxPadding<Scalar> * 7);
                    direction_solver.solve_impl(trans_cgres);

                    trans_cgiter = trans_cgres.size();
                    // d = ATmap(psqmr.solution()) - m_FZ.var;
                    ATmap_impl(direction_solver.solution(), tmp_mats_2);

                    for (Index k = 0; k < trans_nblock; ++k) {
                        d[k] = tmp_mats_2[k] - m_FZ.var[k];
                    }

                    iHW_sig = One;
                    iHW_epsilon = sig;

                    grad_phi_impl(m_blk, m_FZ.par, d, tmp_grad_phi);
                    // d = (d * iHW_epsilon + iHW_sig * tmp_grad_phi) / sig / (One + sig);
                    for (Index k = 0; k < trans_nblock; ++k) {
                        d[k] = (d[k] * iHW_epsilon + iHW_sig * tmp_grad_phi[k]) / sig / (One + sig);
                    }

                    // Restore Dsch12
                    for (Index k = 0; k < trans_nblock; ++k) {
                        auto Dsch12 = m_FZ.par[k].Dsch12();
                        Map<MatrixX> Dsch12_orig(tmp_mats[k].data(), Dsch12.rows(), Dsch12.cols());
                        Dsch12 = Dsch12_orig;
                    }

                    deactivate_buffer<4>();
                }

                puts_verbose("Newton update");
                {
                    auto& Znew = tmp_z_struct;
                    // Znew.var = m_Z.var + d;
                    for (Index k = 0; k < trans_nblock; ++k) {
                        Znew.var[k] = m_Z.var[k] + d[k];
                    }
                    Znew.Avar = Amap(Znew.var);

                    auto& FZnew = tmp_fz_struct;
                    auto& Xnew = tmp_x_struct;
                    comp_res_fp_impl(Znew, FZnew, Xnew);

                    const Scalar dnorm2 = d.squared_norm();
                    trans_NEWT_nrmd = sqrt(dnorm2);
                    trans_NEWT_FZd = -inner_product(FZnew.var, d);
                    trans_NEWT_FZdorg = -inner_product(m_FZ.var, d);
                    trans_NEWT_thetaFZdorg = trans_NEWT_FZdorg / m_FZ.res / trans_NEWT_nrmd;
                    trans_NEWT_FZFZnew = inner_product(m_FZ.var, FZnew.var);
                    trans_NEWT_thetaFZFZnew = trans_NEWT_FZFZnew / FZnew.res / m_FZ.res;

                    if (m_FZ.res < 1e-3_s) {
                        trans_NEWT_rhs = FZnew.res * dnorm2;
                    } else {
                        trans_NEWT_rhs = dnorm2;
                    }

                    trans_NEWT_ratio = trans_NEWT_FZd / trans_NEWT_rhs;

                    if (trans_NEWT_iter == 1 && FZnew.res >= m_FZ.res) {
                        trans_NEWT_lambda *= 10_s;
                        record_optimal_NEWT();
                        continue;
                    }

                    if (FZnew.res < 5e1_s * m_FZ.res) {
                        std::swap(m_Z, Znew);
                        std::swap(m_FZ, FZnew);
                        std::swap(m_X, Xnew);

                        if (m_FZ.res < 1e-4_s) {
                            trans_NEWT_sigPow = 0.4_s;
                        } else {
                            trans_NEWT_sigPow = 0.5_s;
                        }

                        trans_NEWT_CG_maxit = 300;
                    }
                }

                record_optimal_NEWT();

                if (max(trans_hists_pinforg.back(),
                    trans_hists_dinforg.back() * tolscale) < tol * retol)
                {
                    if (tolscale < 100_s) {
                        auto& Xnew = tmp_mats;
                        auto& ynew = tmp_vec;
                        auto& Snew = tmp_mats_2;
                        auto& rec = tmp_rec;
                        recover_var_chk_impl(Xnew, ynew, Snew, rec);
                        const Scalar maxinf =
                            max({rec.pinf, rec.dinf * tolscale, rec.K1, rec.K1dual, rec.C1});
                        if (maxinf < tol * retol) {
                            cstop = true;
                            m_X.var = std::move(Xnew);
                            m_y.var = std::move(ynew);
                            m_S.var = std::move(Snew);
                            trans_rec = std::move(rec);
                        } else {
                            cstop = false;
                        }
                    } else {
                        cstop = true;
                        recover_var_chk_impl(m_X.var, m_y.var, m_S.var, trans_rec);
                    }
                }

                if (verbosity >= Verbosity::Detail && (cstop || iter == 1 || iter == maxits || iter % NEWT_print_itr == 0)) {
                    if (iter % (20 * NEWT_print_itr) == 0 && iter != maxits && !cstop) {
                        log_format("\n%s\n", str_head2);
                    }

                    log_format(
                        "%5d  %2.1e  %+2.1e  %+2.1e  %2.1e  %2.1e  %2.1e"
                        "  %2.1e  %2.1e  %2.1e  %2.1e  %2.4e  %2.4e  %5d"
                        "  %2.4e  %2.4e  %2.4e\n",
                        iter, trans_pmu, trans_hists_pobj.back(), trans_hists_dobj.back(),
                        trans_hists_gap.back(), trans_hists_pinf.back(), trans_hists_dinf.back(),
                        trans_hists_gaporg.back(), trans_hists_pinforg.back(), trans_hists_dinforg.back(),
                        m_FZ.res, trans_NEWT_sig, trans_cgres.back(), trans_cgiter, trans_NEWT_ratio,
                        trans_NEWT_thetaFZdorg, trans_NEWT_thetaFZFZnew);
                }

                if (trans_NEWT_ratio >= NEWT_eta2) {
                    trans_NEWT_lambda = max(NEWT_gamma1 * trans_NEWT_lambda, 1e-16_s);
                } else if (trans_NEWT_ratio >= NEWT_eta1) {
                    trans_NEWT_lambda *= NEWT_gamma2;
                } else {
                    trans_NEWT_lambda *= NEWT_gamma3;
                }
            } else {
                if (trans_last_iter == IterType::NEWT) {
                    trans_NEWT_lastres = m_FZ.res;
                    // m_S.var = (m_X.var - m_Z.var) / trans_pmu;
                    // m_Z.var -= m_FZ.var;
                    for (Index k = 0; k < trans_nblock; ++k) {
                        m_S.var[k] = (m_X.var[k] - m_Z.var[k]) / trans_pmu;
                        m_Z.var[k] -= m_FZ.var[k];
                    }
                    m_S.Avar = (m_X.Avar - m_Z.Avar) / trans_pmu;
                    m_Z.Avar -= m_FZ.Avar;
                    puts_verbose("Switching to ADMM. Setting params");

                    trans_last_iter = IterType::ADMM;
                    trans_ADMM_swt = 0;
                    trans_ADMM_iter = 0;
                    trans_ADMM_maxiter = 200;
                    ++trans_ADMM_subiter;
                    trans_cgiter = 0;
                }
                ++trans_ADMM_iter;
                ++trans_ADMM_totaliter;

                puts_verbose("Step 1: compute y");
                {
                    m_y.var = (m_b - m_X.Avar) / trans_pmu + trans_AC - m_S.Avar;
                    ATmap_impl(m_y.var, m_y.Avar);
                }

                puts_verbose("Step 2: compute X and S");
                {
                    // m_W = m_C - (m_y.Avar + m_X.var / trans_pmu);
                    for (Index k = 0; k < trans_nblock; ++k) {
                        m_W[k] = m_C[k] - (m_y.Avar[k] + m_X.var[k] / trans_pmu);
                    }

                    // [S.var, trans_ADMM_rankS] = blkprojSDP(W, trans_ADMM_rankS);
                    blkprojSDP_impl(m_blk, m_W, m_S.var, trans_ADMM_rankS);
                    // By the implementation of blkprojSDP,
                    // trans_ADMM_recompeig += 0;

                    const Scalar step = ADMM_rho * trans_pmu;
                    Amap_impl(m_S.var, m_S.Avar);

                    trans_AtymCSnrm = Zero;

                    // CellArray<MatrixX>& CS = tmp_mats;
                    // CS = m_C - m_S.var;
                    // trans_ATymCS = m_y.Avar - CS;
                    // m_X.var += step * trans_ATymCS;
                    // trans_AtymCSnrm += trans_ATymCS.norm_squared();
                    trans_ATymCS.resize(trans_nblock);
                    for (Index k = 0; k < trans_nblock; ++k) {
                        trans_ATymCS[k] = m_y.Avar[k] - (m_C[k] - m_S.var[k]);
                        m_X.var[k] += step * trans_ATymCS[k];
                        trans_AtymCSnrm += trans_ATymCS[k].squaredNorm();
                    }

                    Amap_pair_impl(m_X.var, m_X.Avar, m_X.Avarorg);
                }

                puts_verbose("Check Optimality");
                {
                    const Scalar bTy = m_b.dot(m_y.var);
                    const Scalar trCX = inner_product(m_C, m_X.var);
                    const Scalar dobj = trans_scale_objscale * bTy;
                    const Scalar pobj = trans_scale_objscale * trCX;
                    const Scalar gap = abs(trCX - bTy) / max(One, abs(trCX));
                    const Scalar gaporg = abs(pobj - dobj) / max(One, abs(pobj));

                    trans_bmAX = m_b - m_X.Avar;
                    const VectorX& bmAX = trans_bmAX;
                    // rpnrmorg = norm(X.Avarorg./diag(info.scale.DA)*info.scale.bscale-info.borg);
                    const Scalar bmAXnrmorg =
                        (m_X.Avarorg.cwiseQuotient(trans_scale_DA.diagonal()) * trans_scale_bscale -
                            trans_borg).norm();
                    const Scalar pinforg = bmAXnrmorg / trans_normborg;
                    const Scalar pinf = bmAX.norm() / trans_normb;
                    const Scalar dinf = sqrt(trans_AtymCSnrm) / trans_normC;
                    // const Scalar feasratio = pinforg / dinf;

                    trans_hists_pobj.push_back(pobj);
                    trans_hists_dobj.push_back(dobj);
                    trans_hists_gaporg.push_back(gaporg);
                    trans_hists_gap.push_back(gap);
                    trans_hists_pinf.push_back(pinf);
                    trans_hists_pinforg.push_back(pinforg);
                    trans_hists_dinf.push_back(dinf);
                    trans_hists_dinforg.push_back(dinf);
                    trans_hists_pvd.push_back(pinf / dinf);
                    trans_hists_dvp.push_back(dinf / pinf);
                    trans_hists_pvdorg.push_back(pinforg / dinf);
                    trans_hists_dvporg.push_back(dinf / pinforg);
                    trans_hists_cgiter.push_back(trans_cgiter);
                    trans_hists_isNEWT.push_back(false);
                    trans_hists_maxinf = max(pinf, dinf);
                    // trans_ADMM_normAty = y.Avar.norm();
                    // trans_ADMM_normS = S.var.norm();
                    // trans_ADMM_normZ1Z2 = max(trans_ADMM_normAty, trans_ADMM_normS);
                    // trans_ADMM_normX = X.var.norm();

                    if (max(pinforg, dinf * tolscale) < tol * retol) {
                        if (tolscale < 100_s) {
                            auto& Xnew = tmp_mats;
                            auto& ynew = tmp_vec;
                            auto& Snew = tmp_mats_2;
                            auto& rec = tmp_rec;
                            recover_var_chk_impl(Xnew, ynew, Snew, rec);
                            if (max({rec.pinf, rec.dinf * tolscale, rec.K1, rec.K1dual, rec.C1}) < tol * retol) {
                                cstop = true;
                                // Temp variables are no longer used. Use std::move instead of std::swap.
                                m_X.var = std::move(Xnew);
                                m_y.var = std::move(ynew);
                                m_S.var = std::move(Snew);
                                trans_rec = std::move(rec);
                            } else {
                                cstop = false;
                            }
                        } else {
                            cstop = true;
                            recover_var_chk_impl(m_X.var, m_y.var, m_S.var, trans_rec);
                        }
                    }
                    // cstop = max(pinforg, dinf * tolscale) < tol * retol || iter == maxits;

                    if (verbosity >= Verbosity::Detail && (cstop || iter == 1 || iter % ADMM_print_itr == 0)) {
                        if (iter % (20 * ADMM_print_itr) == 0 && !cstop) {
                            log_format("\n %s\n", str_head);
                        }

                        log_format("%5d  %2.1e  %+2.1e  %+2.1e  %2.1e  %2.1e  %2.1e  %2.1e  %2.1e  %2.1e \n",
                            iter, trans_pmu, pobj, dobj,
                            gap, pinf, dinf,
                            gaporg, pinforg, dinf);
                    }
                }
            } // if (doNEWT)

            if (cstop) {
                out_status = str_format("max(prim_infeas, dual_infeas) < %3.2e", tol);
                break;
            }

            if (iter == maxits) {
                out_status = "reach the maximum iteration";
                recover_var_chk_impl(m_X.var, m_y.var, m_S.var, trans_rec);
                break;
            }

            puts_verbose("Set param mu");
            if (!doNEWT) {
                const Scalar& dinf = trans_hists_dinforg.back();
                const Scalar& pinf = trans_hists_pinforg.back();

                bool switch_dual = true;

                if (dinf > tol && iter < 500) {
                    switch_dual = true;
                } else if (pinf < 12_s * tol || dinf < 12_s * tol) {
                    switch_dual = false;
                }
                if (tolscale < 100_s) {
                    if (iter < 150) {
                        mu_ADMM_ratio = One;
                        mu_ADMM_mu_fact = 1.8_s;
                    } else if (switch_dual) {
                        mu_ADMM_ratio = 1e4_s;
                        mu_ADMM_mu_fact = 5_s;
                    } else {
                        mu_ADMM_ratio = tolscale;
                        mu_ADMM_mu_fact = 1.8_s;
                        if (trans_ischange) {
                            trans_pmu /= 80_s;
                            trans_ischange = false;
                        }
                    }
                } else {
                    if (iter < 150) {
                        mu_ADMM_ratio = One;
                    } else if (pinf > tol) {
                        mu_ADMM_ratio = 1e-4_s;
                    } else {
                        mu_ADMM_ratio = tolscale;
                        trans_ischange = false;
                    }
                }
            } // if (!doNEWT)

            if (tolscale >= 100) {
                if (trans_NEWT_totaliter == 500) {
                    retol = 3_s;
                    tolscale /= 2_s;
                    if (verbosity >= Verbosity::Detail) {
                        puts_info("change the tol");
                    }
                } else if (trans_NEWT_totaliter == 750) {
                    retol = 6_s;
                    tolscale /= 2_s;
                    if (verbosity >= Verbosity::Detail) {
                        puts_info("change the tol");
                    }
                }
            }

            puts_verbose("mu update");
            if (mu_adp_mu) {
                if (doNEWT) {
                    const Scalar pmup = trans_pmu;
                    puts_verbose("mu update NEWT");
                    if (iter % mu_NEWT_mu_update_itr == 0) {
                        const Index sitr = iter - mu_NEWT_mu_update_itr;
                        // const Map<const VectorX> pvd(trans_hists_pvd.data() + sitr, mu_NEWT_mu_update_itr);
                        const Map<const VectorX> dvp(trans_hists_dvp.data() + sitr, mu_NEWT_mu_update_itr);

                        // const Scalar avg_pvd = geomean(pvd);
                        const Scalar avg_dvp = geomean(dvp);

                        if (avg_dvp > mu_NEWT_mu_delta) {
                            trans_pmu *= mu_NEWT_mu_fact;
                        } else {
                            trans_pmu /= mu_NEWT_mu_fact;
                        }

                        trans_pmu = min(mu_NEWT_mu_max, max(mu_NEWT_mu_min, trans_pmu));
                    }
                    if (trans_pmu != pmup) {
                        if (verbosity >= Verbosity::Detail) {
                            log_format("  -- mu updated: %f\n", trans_pmu);
                        }

                        puts_verbose("comp_res_fp_updmu");
                        {
                            // the two tmp variables are both used twice
                            tmp_vec = trans_pmu * trans_AC;
                            VectorX& mud = tmp_vec_2;
                            mud = (One + trans_pmu / pmup) * m_X.Avar -
                                (trans_pmu / pmup) * m_Z.Avar - tmp_vec - m_b;
                            // m_Z.var = m_X.var - trans_pmu * m_C - ATmap(mud);
                            ATmap_impl(mud, tmp_mats);
                            for (Index k = 0; k < trans_nblock; ++k) {
                                m_Z.var[k] = m_X.var[k] - trans_pmu * m_C[k] - tmp_mats[k];
                            }
                            m_Z.Avar = m_X.Avar - tmp_vec - mud;
                            comp_res_fp_impl(m_Z, m_FZ, m_X);
                        }
                    }
                } // if (doNEWT)
                else {
                    puts_verbose("mu update ADMM");
                    if (trans_hists_pvdorg.back() < mu_ADMM_ratio) {
                        ++trans_muADMM_prim_win;
                    } else {
                        ++trans_muADMM_dual_win;
                    }

                    puts_verbose("mu_iter");
                    Size mu_update_iter;
                    if (iter < 30) {
                        mu_update_iter = 3;
                    } else if (iter < 60) {
                        mu_update_iter = 6;
                    } else if (iter < 120) {
                        mu_update_iter = 12;
                    } else {
                        mu_update_iter = 25;
                    }

                    if (iter % mu_update_iter == 0) {
                        if (iter <= 250) {
                            if (trans_muADMM_prim_win >
                                max(One, mu_ADMM_mu_delta * trans_muADMM_dual_win)) {
                                trans_muADMM_prim_win = 0;
                                trans_pmu = min(mu_ADMM_mu_max, trans_pmu * mu_ADMM_mu_fact);
                            } else if (trans_muADMM_dual_win >
                                max(One, mu_ADMM_mu_delta * trans_muADMM_prim_win)) {
                                trans_muADMM_dual_win = 0;
                                trans_pmu = max(mu_ADMM_mu_min, trans_pmu / mu_ADMM_mu_fact);
                            }
                        } else {
                            Size prim_len = 0;
                            for (Index i = iter - 20; i < iter; ++i) {
                                if (trans_hists_pvdorg[i] <= mu_ADMM_ratio) {
                                    ++prim_len;
                                }
                            }
                            if (prim_len > 15) {
                                trans_pmu = min(mu_ADMM_mu_max, trans_pmu * mu_ADMM_mu_fact);
                            } else if (prim_len < 5) {
                                trans_pmu = max(mu_ADMM_mu_min, trans_pmu / mu_ADMM_mu_fact);
                            }
                        }
                    } // if (iter % mu_update_iter == 0)
                } // if (doNEWT)
            } // if (mu_adp_mu)
        } // main loop

        puts_verbose("generate output");
        {
            const RecoveredVar& rec = trans_rec;

            if (verbosity >= Verbosity::Info) {
                log_format(
                    "\n-------------------------------------------------------------------------------------------------\n"
                    "          pobj           dobj        gap       pinf       dinf         C1         K1     K1dual\n"
                    "%14.8e %14.8e %10.2e %10.2e %10.2e %10.2e %10.2e %10.2e\n"
                    "\n-------------------------------------------------------------------------------------------------\n",
                    rec.pobj, rec.dobj, rec.gap, rec.pinf, rec.dinf, rec.C1, rec.K1, rec.K1dual);
            }

            trans_hists_pobj.push_back(rec.pobj);
            trans_hists_dobj.push_back(rec.dobj);
            trans_hists_pinf.push_back(rec.pinf);
            trans_hists_pinforg.push_back(rec.pinf);
            trans_hists_dinf.push_back(rec.dinf);

            trans_hists_dinforg.push_back(rec.dinf);
            trans_hists_gap.push_back(rec.gap);
            trans_hists_gaporg.push_back(rec.gap);
            const Scalar pvd = rec.pinf / rec.dinf;
            const Scalar dvp = One / pvd;
            trans_hists_pvd.push_back(pvd);
            trans_hists_dvp.push_back(dvp);
            trans_hists_pvdorg.push_back(pvd);
            trans_hists_dvporg.push_back(dvp);

            out_iter = iter;
        }
    }

    SSNSDP_WEAK_INLINE
    CellArray<MatrixX> Model::cell_zeros() const {
        CellArray<MatrixX> A(trans_nblock);
        for (Size k = 0; k < trans_nblock; ++k) {
            const auto& kblk = m_blk[k];
            const auto n = kblk.n;
            if (kblk.is_linear()) {
                A[k].setZero(n, 1);
            } else {
                SSNSDP_ASSUME(kblk.is_sdp());
                A[k].setZero(n, n);
            }

        }
        return A;
    }

    SSNSDP_WEAK_INLINE
    CellArray<MatrixX> Model::cell_empty() const {
        CellArray<MatrixX> A(trans_nblock);
        for (Size k = 0; k < trans_nblock; ++k) {
            const auto& kblk = m_blk[k];
            const auto n = kblk.n;
            if (kblk.is_linear()) {
                A[k].resize(n, 1);
            } else {
                SSNSDP_ASSUME(kblk.is_sdp());
                A[k].resize(n, n);
            }
        }
        return A;
    }

    SSNSDP_WEAK_INLINE
    void Model::comp_res_fp_impl(const ZStruct& Z, FZStruct& FZ, XStruct& X) const {
        project2_impl(m_blk, Z.var, X.var, FZ.par);
        Amap_pair_impl(X.var, X.Avar, X.Avarorg);

        const VectorX& AX = X.Avar;

        // FZ.var = ATmap(mud) + trans_pmu * m_C + Z.var - X.var;
        ATmap_impl(2_s * AX - Z.Avar - trans_pmu * trans_AC - m_b, FZ.var);
        for (Index k = 0; k < trans_nblock; ++k) {
            FZ.var[k] += trans_pmu * m_C[k] + Z.var[k] - X.var[k];
        }
        FZ.res = FZ.var.norm();
        FZ.Avar = AX - m_b;
    }

    SSNSDP_WEAK_INLINE
    void Model::record_optimal_NEWT() {
        const Scalar bTy = m_b.dot(trans_AC + (m_Z.Avar - m_X.Avar) / trans_pmu);
        const Scalar trCX = inner_product(m_C, m_X.var);
        const Scalar dobj = trans_scale_objscale * bTy;
        const Scalar pobj = trans_scale_objscale * trCX;
        const Scalar gap = abs(trCX - bTy) / max(One, abs(trCX));
        const Scalar gaporg = abs(pobj - dobj) / max(One, abs(pobj));

        trans_bmAX = m_b - m_X.Avar;
        const VectorX& bmAX = trans_bmAX;
        const Scalar bmAXnrm = (m_X.Avarorg.cwiseQuotient(trans_scale_DA.diagonal()) *
            trans_scale_bscale - trans_borg).norm();

        const Scalar pinforg = bmAXnrm / trans_normborg;
        const Scalar pinf = bmAX.norm() / trans_normb;
        trans_AtymCSnrm = m_FZ.res / trans_pmu;
        const Scalar dinf = trans_AtymCSnrm / trans_normC;

        trans_hists_pobj.push_back(pobj);
        trans_hists_dobj.push_back(dobj);
        trans_hists_gaporg.push_back(gaporg);
        trans_hists_gap.push_back(gap);
        trans_hists_pinf.push_back(pinf);
        trans_hists_pinforg.push_back(pinforg);
        trans_hists_dinf.push_back(dinf);
        trans_hists_dinforg.push_back(dinf);
        const Scalar pvd = pinf / dinf;
        trans_hists_pvd.push_back(pvd);
        trans_hists_dvp.push_back(One / pvd);
        const Scalar pvdorg = pinforg / dinf;
        trans_hists_pvdorg.push_back(pvdorg);
        trans_hists_dvporg.push_back(One / pvdorg);
        trans_hists_cgiter.push_back(trans_cgiter);
        trans_hists_isNEWT.push_back(true);
        trans_hists_maxinf = max(pinf, dinf);
    }

    SSNSDP_WEAK_INLINE
    void Model::recover_var_chk_impl(
        CellArray<MatrixX>& Xnew,
        VectorX& ynew,
        CellArray<MatrixX>& Snew,
        Model::RecoveredVar& rec) const
    {
        if (trans_last_iter == IterType::NEWT) {
            ZStruct Znew;
            // Znew.var = Z.var - FZ.var;
            Znew.Avar = m_Z.Avar - m_FZ.Avar;
            ynew = trans_AC + (Znew.Avar - m_X.Avar) / trans_pmu;
            // Snew = (m_X.var - m_Z.var) / trans_pmu;
            for (Index k = 0; k < trans_nblock; ++k) {
                Snew[k] = (m_X.var[k] - m_Z.var[k]) / trans_pmu;
            }

            Xnew = m_X.var;
        } else {
            ynew = m_y.var;
            // Xnew = (m_S.var - m_W) * trans_pmu;
            for (Index k = 0; k < trans_nblock; ++k) {
                Xnew[k] = (m_S.var[k] - m_W[k]) * trans_pmu;
            }
            Snew = m_S.var;
        }

        Xnew *= trans_scale_bscale;
        trans_Lchol.bwsolve_in_place(ynew);
        ynew = trans_scale_Cscale * (trans_scale_DA * ynew);
        Snew *= trans_scale_Cscale;

        const VectorX AXnew = AXfun(m_blk, trans_Atorg, Xnew);
        const Scalar bmAXnrm = (AXnew - trans_borg).norm();
        rec.pinf = bmAXnrm / (One + trans_borg.norm());
        CellArray<MatrixX> AtymCS = Atyfun(trans_blkorg, trans_Atorg, ynew);
        for (Index k = 0; k < trans_nblock; ++k) {
            AtymCS[k] += Snew[k] - trans_Corg[k];
        }
        const Scalar AtymCSnrm = AtymCS.norm();
        rec.dinf = AtymCSnrm / (One + trans_Corg.norm());
        rec.pobj = inner_product(trans_Corg, Xnew);
        rec.dobj = trans_borg.dot(ynew);

        const Scalar& pobj = rec.pobj;
        const Scalar& dobj = rec.dobj;
        rec.gap = abs(pobj - dobj) / (One + abs(pobj) + abs(dobj));
        const Scalar trXS = inner_product(Xnew, Snew);

        const Scalar normX = Xnew.norm();
        const Scalar normS = Snew.norm();

        CellArray<Project2Info> Xpar, Spar;
        std::tie(std::ignore, Xpar) = project2(m_blk, Xnew);
        std::tie(std::ignore, Spar) = project2(m_blk, Snew);

        Scalar eigXnorm = Zero;
        Scalar eigSnorm = Zero;

        for (Index k = 0; k < trans_blkorg.size(); ++k) {
            if (trans_blkorg[k].is_sdp()) {
                eigXnorm += Xpar[k].dd.tail(Xpar[k].dd.size() - Xpar[k].poslen).squaredNorm();
                eigSnorm += Spar[k].dd.tail(Spar[k].dd.size() - Spar[k].poslen).squaredNorm();
            } else {
                SSNSDP_ASSUME(trans_blkorg[k].is_linear());
                eigXnorm += Xpar[k].Dsch12().cast<bool>().select(Zero, Xnew[k]).squaredNorm();
                eigSnorm += Spar[k].Dsch12().cast<bool>().select(Zero, Snew[k]).squaredNorm();
            }
        }

        eigXnorm = sqrt(eigXnorm);
        eigSnorm = sqrt(eigSnorm);

        rec.K1 = eigXnorm / (One + normX);
        rec.K1dual = eigSnorm / (One + normS);
        rec.C1 = abs(trXS) / (One + normX + normS);
    }
}

#endif
