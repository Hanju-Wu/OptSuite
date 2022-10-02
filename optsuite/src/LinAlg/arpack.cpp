/*
 * ==========================================================================
 *
 *       Filename:  arpack.cpp
 *
 *    Description:  wrapper for ARPACK-ng
 *
 *        Version:  1.0
 *        Created:  03/09/2021 02:10:51 PM
 *       Revision:  none
 *       Compiler:  g++
 *
 *         Author:  Haoyang Liu (@liuhy), liuhaoyang@pku.edu.cn
 *   Organization:  BICMR, PKU
 *
 * ==========================================================================
 */

#ifdef OPTSUITE_USE_ARPACK
#include "arpack.hpp"
#include "OptSuite/LinAlg/arpack.h"
#include "OptSuite/Utils/logger.h"
#include "OptSuite/Utils/optionchecker.h"

namespace OptSuite { namespace LinAlg {
    inline arpack::which to_arpack_which(const std::string& s){
        if (s == "LA" || s == "la")
            return arpack::which::largest_algebraic;
        else if (s == "SA" || s == "sa")
            return arpack::which::smallest_algebraic;
        else if (s == "LM" || s == "lm")
            return arpack::which::largest_magnitude;
        else if (s == "SM" || s == "sm")
            return arpack::which::smallest_magnitude;
        else if (s == "BE" || s == "be")
            return arpack::which::both_ends;

        // default to LA
        return arpack::which::largest_algebraic;
    }

    inline void set_common_options(a_int *iparam, Utils::OptionList& options){
        iparam[2] = options.get_integer("maxit");
        iparam[3] = 1;
    }

    template <typename dtype, int self_adjoint>
    ARPACK<dtype, self_adjoint>::ARPACK(){
        register_options();
    }

    template <typename dtype, int self_adjoint>
    ARPACK<dtype, self_adjoint>::ARPACK(
            const Ref<const mat_t> A, Index nev, const std::string& wh, bool no_evec){
        register_options();
        this->compute(A, nev, wh, no_evec);
    }

    template <typename dtype, int self_adjoint>
    ARPACK<dtype, self_adjoint>::ARPACK(
            const Ref<const spmat_t> A, Index nev, const std::string& wh, bool no_evec){
        register_options();
        this->compute(A, nev, wh, no_evec);
    }

    template <typename dtype, int self_adjoint>
    ARPACK<dtype, self_adjoint>::ARPACK(
            const matop_t& A, Index nev, const std::string& wh, bool no_evec){
        register_options();
        this->compute(A, nev, wh, no_evec);
    }



    template <typename dtype, int self_adjoint>
    Index ARPACK<dtype, self_adjoint>::get_ncv(Index n, Index nev) const {
        Index ncv = options_.get_integer("ncv");
        
        if (ncv == -1)
            ncv = std::min(std::max(20_i, 2_i * nev + 1), n);
        else
            ncv = std::min(std::max(nev, ncv), n);

        return ncv;
    }

    template <typename dtype, int self_adjoint>
    void ARPACK<dtype, self_adjoint>::register_options(){
        using namespace Utils;
        using OptionChecker_ptr = std::shared_ptr<OptionChecker>;
        using OptSuite::Utils::BoundCheckerSense;
        std::vector<RegOption> v;

        OptionChecker_ptr pos_scalar_checker =
            std::make_shared<BoundChecker<Scalar>>(0, BoundCheckerSense::Strict, 0, BoundCheckerSense::None);

        OptionChecker_ptr dimension_checker = 
            std::make_shared<BoundChecker<Index>>(-1, BoundCheckerSense::Standard, 0, BoundCheckerSense::None);
        OptionChecker_ptr pos_int_checker = 
            std::make_shared<BoundChecker<Index>>(0, BoundCheckerSense::Strict, 0, BoundCheckerSense::None);

        v.push_back({"tol", 1e-12_s, "Desired relative accuracy of computed eigenvalues.",
                pos_scalar_checker});
        v.push_back({"maxit", 300_i, "Max number of iterations. Default: 300",
                pos_int_checker});
        v.push_back({"ncv",  -1_i, "Dimension of Krylov subspace",
                dimension_checker});
        // v is constructed, now initialize options_
        this->options_ = Utils::OptionList(v);

    }

    template <typename dtype, int self_adjoint>
    const typename ARPACK<dtype, self_adjoint>::vec_t& ARPACK<dtype, self_adjoint>::d() const {
        return d_;
    }

    template <typename dtype, int self_adjoint>
    const typename ARPACK<dtype, self_adjoint>::mat_t& ARPACK<dtype, self_adjoint>::U() const {
        return U_;
    }

    template <typename dtype, int self_adjoint>
    const int& ARPACK<dtype, self_adjoint>::info() const { return info_; }

    template <typename dtype, int self_adjoint>
    const Utils::OptionList& ARPACK<dtype, self_adjoint>::options() const { return options_; }

    template <typename dtype, int self_adjoint>
    Utils::OptionList& ARPACK<dtype, self_adjoint>::options(){ return options_; }

    template <typename dtype, int self_adjoint>
    ARPACK<dtype, self_adjoint>& ARPACK<dtype, self_adjoint>::compute(
            const Ref<const mat_t> A,
            Index nev, const std::string& wh, bool no_evec){
        if (self_adjoint == 1){
            __proxy_mat_op Aop(A);
            compute_self_adjoint_normal(Aop, nev, wh, no_evec);
        }
        return *this;
    }


    template <typename dtype, int self_adjoint>
    ARPACK<dtype, self_adjoint>& ARPACK<dtype, self_adjoint>::compute(
            const Ref<const spmat_t> A,
            Index nev, const std::string& wh, bool no_evec){
        if (self_adjoint == 1){
            __proxy_spmat_op Aop(A);
            compute_self_adjoint_normal(Aop, nev, wh, no_evec);
        }
        return *this;
    }

    template <typename dtype, int self_adjoint>
    ARPACK<dtype, self_adjoint>& ARPACK<dtype, self_adjoint>::compute(
            const matop_t& A, Index nev, const std::string& wh, bool no_evec){
        if (self_adjoint == 1){
            compute_self_adjoint_normal(A, nev, wh, no_evec);
        }
        return *this;
    }

    template <typename dtype, int self_adjoint>
    int ARPACK<dtype, self_adjoint>::compute_self_adjoint_normal(
            const matop_t& Aop, Index nev, const std::string& wh, bool no_evec){
        using mat_t = Eigen::Matrix<dtype, Dynamic, Dynamic>;
        a_int ido = 0, info = 0;
        a_int iparam[11] = {0}, ipntr[11];

        // common options
        set_common_options(iparam, options_);
        a_int n = static_cast<a_int>(Aop.cols());
        a_int ncv = static_cast<a_int>(get_ncv(n, nev));

        // special options
        iparam[0] = 1; // exact shifts
        iparam[6] = 1; // normal

        // prepare storage
        resid.resize(n);
        a_int lworkl = static_cast<a_int>((ncv + 8) * ncv);
        workd.resize(3 * n);
        workl.resize(lworkl);
        V_.resize(n, ncv);
        d_.resize(nev, 1);

        // invoke saupd
        while (ido != 99){
            arpack::saupd(ido,
                    arpack::bmat::identity,
                    n,
                    to_arpack_which(wh),
                    static_cast<a_int>(nev),
                    options_.get_scalar("tol"),
                    resid.data(),
                    ncv,
                    V_.data(),
                    V_.outerStride(),
                    iparam,
                    ipntr,
                    workd.data(),
                    workl.data(),
                    lworkl,
                    info);
            if (ido == -1 || ido == 1){
                // data of X: workd.data() + ipntr[0] - 1
                // data of Y: workd.data() + ipntr[1] - 1
                Map<mat_t> x(workd.data() + ipntr[0] - 1, n, 1);
                Map<mat_t> y(workd.data() + ipntr[1] - 1, n, 1);
                Aop.apply(x, y);
            } // for this case, ido cannot be 2 or 3
        }

        // check return code
        if (info != 0){
            Utils::Global::logger_e.log_format("ARPACK: saupd returned with %d.\n",
                    static_cast<int>(info));
            if (info == 1)
                Utils::Global::logger_e.log_info("ARPACK: exceed max iteration.\n");
            if (info == 3)
                Utils::Global::logger_e.log_info("ARPACK: no shifts could be applied. Try larger NCV.\n");
            info_ = info;
        }

        // invoke seupd
        a_int rvec;
        if (!no_evec){
            rvec = 1;
            U_.resize(n, nev);
        } else {
            rvec = 0;
        }
        std::vector<a_int> select(ncv);
        arpack::seupd(rvec,
                arpack::howmny::ritz_vectors,
                select.data(),
                d_.data(),
                U_.data(),
                U_.outerStride(),
                0_s,
                arpack::bmat::identity,
                n,
                to_arpack_which(wh),
                static_cast<a_int>(nev),
                options_.get_scalar("tol"),
                resid.data(),
                ncv,
                V_.data(),
                V_.outerStride(),
                iparam,
                ipntr,
                workd.data(),
                workl.data(),
                lworkl,
                info);

        // check return code
        if (info != 0){
            Utils::Global::logger_e.log_format("ARPACK: seupd returned with %d.\n",
                    static_cast<int>(info));
        }

        // copy ipntr to class storage
        this->ipntr.resize(11);
        for (int i = 0; i < 11; ++i){
            this->ipntr[i] = static_cast<Index>(ipntr[i]);
        }
        return static_cast<int>(info);
    }

    // instantiate
    template class ARPACK<Scalar, ARPACK_Sym>; // real symmetric
}}

#endif
