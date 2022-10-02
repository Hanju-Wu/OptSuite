/*
 * @Author: Wu Hanju 
 * @Date: 2022-09-10 10:00:45 
 * @Last Modified by: Wu Hanju
 * @Last Modified time: 2022-09-12 11:50:37
 */

#include <algorithm>
#include <cmath>
#include "OptSuite/core_n.h"
#include "OptSuite/Base/mat_wrapper.h"
#include "OptSuite/Composite/Model/base.h"
#include "OptSuite/Composite/Solver/FISTA.h"
#include "OptSuite/Utils/logger.h"
#include "OptSuite/Utils/tictoc.h"

namespace OptSuite { namespace Composite { namespace Solver {
    inline Scalar to_real(ComplexScalar c) { return c.real(); }
    inline Scalar to_real(Scalar c) { return c; }

    template<typename T>
    OptionList& FISTA<T>::options(){ return options_; }

    template<typename T>
    const OptionList& FISTA<T>::options() const { return options_; }

    template<typename T>
    std::shared_ptr<const Structure> FISTA<T>::workspace() const {
        return workspace_;
    }

    template<typename T>
    void FISTA<T>::register_options(){
        using OptionChecker_ptr = std::shared_ptr<OptionChecker>;
        using OptSuite::Utils::BoundCheckerSense;
        using OptSuite::eps;
        std::vector<RegOption> v;
        OptionChecker_ptr pos_scalar_checker =
            std::make_shared<BoundChecker<Scalar>>(0, BoundCheckerSense::Strict, 0, BoundCheckerSense::None);
        OptionChecker_ptr zero_one_checker =
            std::make_shared<BoundChecker<Scalar>>(0, BoundCheckerSense::Strict, 1, BoundCheckerSense::Strict);
        OptionChecker_ptr pos_eq_int_checker =
            std::make_shared<BoundChecker<Index>>(0, BoundCheckerSense::Standard, 1, BoundCheckerSense::None);

        Scalar xtol, gtol, ftol;
        if (eps < 1e-12_s) { // double precision
            xtol = 1e-8_s;
            gtol = 1e-6_s;
            ftol = 1e-14_s;
        } else { // single precision
            xtol = 1e-4_s;
            gtol = 1e-3_s;
            ftol = 1e-7_s;
        }
        v.emplace_back("xtol", xtol, "Tolerance of xdiff (>0)", pos_scalar_checker);
        v.emplace_back("gtol", gtol, "Tolerance of nrmG (>0)", pos_scalar_checker);
        v.emplace_back("ftol", ftol, "Tolerance of objdiff (>0)", pos_scalar_checker);
        v.emplace_back("tau", 1e-3_s, "Initial step size (>0)", pos_scalar_checker);
        v.emplace_back("rhols", 1e-4_s, "Line search parameter for Armijo rule (0, 1)", zero_one_checker);
        v.emplace_back("eta", 0.1_s, "Diminishing step factor in line search (0, 1)", zero_one_checker);
        v.emplace_back("gamma", 0.85_s, "Parameter in Hongchao & Hanger rule (0, 1)", zero_one_checker);
        v.emplace_back("maxit", 1000_i, "Max number of iterations (>=0)", pos_eq_int_checker);
        v.emplace_back("maxnls", 5_i, "Max number of line search iterations (>=0)", pos_eq_int_checker);
        v.emplace_back("log_iter", 10_i, "Print iteration information every this many iterations. Zero means only the first and the last iteration is printed (>=0)", pos_eq_int_checker);
        v.emplace_back("verbosity", (Index)Verbosity::Info, "Verbosity level (>=0)", pos_eq_int_checker);
        v.emplace_back("log_file", "", "Redirect the iteration log into file (when not empty)");
        v.emplace_back("continuation", true, "Enable the continuation strategy (Default: true)");
        v.emplace_back("cont_alpha", 0.535_s, "Continuation parameter alpha (see doc)", zero_one_checker);
        v.emplace_back("cont_beta", 0.65_s, "Continuation parameter beta (see doc)", zero_one_checker);
        v.emplace_back("cont_tol", 1_s, "Continuation tolerance", pos_scalar_checker);
        v.emplace_back("adapls", false, "Enable the adaptive pre-determined step selection strategy (Default: true)");
        v.emplace_back("adapls_window", 10_i, "Adaptivew step selection parameter (see doc)");
        v.emplace_back("bb_variant", 0_i, "Specifies which variant of BB step size to use.", pos_eq_int_checker);
        v.emplace_back("ls_variant", 2_i, "Specifies which variant of line search criterion to use.", pos_eq_int_checker);

        // v is constructed, now initialize options_
        this->options_ = OptionList(v, logger);

    }

    template<typename T>
    void FISTA<T>::initialize_output(){
        out.num_feval = 0;
        out.num_geval = 0;
        out.message = "exceed max iteration";
        out.flag = 99;
    }

    template<typename T>
    void FISTA<T>::initialize_local_variables(const var_t& x0){
        // init x, xp, y, yp
        x->set_zero_like(x0);
        xp->set_zero_like(x0);
        y->set_zero_like(x0);
        yp->set_zero_like(x0);

        // optionally init gx, gxp, gy, gyp
        mat_wrapper_t* x_ptr = dynamic_cast<mat_wrapper_t*>(x.get());
        mat_wrapper_t* gx_ptr = dynamic_cast<mat_wrapper_t*>(gx.get());
        mat_wrapper_t* y_ptr = dynamic_cast<mat_wrapper_t*>(y.get());
        mat_wrapper_t* gy_ptr = dynamic_cast<mat_wrapper_t*>(gy.get());

        if (x_ptr && gx_ptr && y_ptr && gy_ptr){
            gx->set_zero_like(x0);
            gy->set_zero_like(x0);
            gyp->set_zero_like(x0);
        }
    }

    template<typename T>
    Scalar FISTA<T>::compute_obj_h(const Model::Base<T>& model){
        if (model.prox_h->has_objective_cache())
            return model.prox_h->cached_objective();
        else
            return (*model.h)(*x);
    }

    template<typename dtype>
    void FISTA<dtype>::set_y(var_t_ptr x, var_t_ptr xp, var_t_ptr y, Scalar theta)
    {
        mat_t A = getmat(x) + theta * (getmat(x) - getmat(xp));
        const var_t& a=mat_wrapper_t(A);
        y->assign(a);
    }

    template<typename dtype>
    void FISTA<dtype>::solve(const Model::Base<dtype>& model, const Ref<const mat_t> x0){
        this->solve(model, MatWrapper<dtype>(x0));
    }

    template<typename dtype>
    void FISTA<dtype>::solve(const Model::Base<dtype>& model, const var_t& x0){
        initialize_output();
        // set logger verbosity and output file
        logger->verbosity = (OptSuite::Verbosity)options_.get_integer("verbosity");
        logger->redirect_to_file(options_.get_string("log_file"));


        // initialize local variables
        initialize_local_variables(x0);

        // initialize workspace && bind to functions
        workspace_ = std::make_shared<Structure>();
        model.f->bind(workspace_);
        model.h->bind(workspace_);
        model.prox_h->bind(workspace_);
        if (custom_stop_checker != nullptr)
            custom_stop_checker->bind(workspace_);

        Scalar obj_orig, obj, objp, Q, Qp, Cval, tau, obj_fx, obj_h, obj_fy,ftol_c = 1_s,
               nrmG, nrm_dx, nrm2_dx, nrm2_dg, nrm2_dxy, nrm2_dy,
               objdiff, Xdiff = 0;
        Scalar mu_factor0 = 1, mu_factor = 1;
        Scalar theta;
        Scalar k;
        dtype x_x, x_xp, xp_xp, g_g, g_gp, gp_gp, y_g, x_gp = 0, yp_g, yp_yp, x_yp, yp_gp, y_gp, y_y, y_yp;
        Index iter, iter_cbegin;
        bool mu_changed = false;

        std::string header = "iter     step    objorig     obj     objdiff    Xdiff     nrmG    nls    mu_f          time(f|g|atot)";
        const char *format = "%4d  %+2.1e  %+2.1e  %+2.1e  %+2.1e  %+2.1e  %+2.1e  %2d   %2.1e  %2.1e|%2.1e|%2.1e";

        std::vector<Scalar> step_hist;

        // import from options_
        tau = options_.get_scalar("tau");
        bool cont = options_.get_bool("continuation");
        bool adapls = options_.get_bool("adapls");
        Scalar ftol = options_.get_scalar("ftol");
        Scalar xtol = options_.get_scalar("xtol");
        Scalar gtol = options_.get_scalar("gtol");
        Scalar eta = options_.get_scalar("eta");
        Scalar gamma = options_.get_scalar("gamma");
        Scalar rhols = options_.get_scalar("rhols");
        Index maxit = options_.get_integer("maxit");
        Index maxnls = options_.get_integer("maxnls");
        Index log_iter = options_.get_integer("log_iter");
        Index adap_w = options_.get_integer("adapls_window");
        Index bb_variant = options_.get_integer("bb_variant");
        Index ls_variant = options_.get_integer("ls_variant");
        Scalar c_alpha = options_.get_scalar("cont_alpha");
        Scalar c_beta = options_.get_scalar("cont_beta");
        Scalar c_tol = options_.get_scalar("cont_tol");

        // timers
        auto tstart = tic();
        Scalar t_local_acc, t_local_f, t_local_prox_h;

        // print option details
        logger->log_info("=========== FISTA Solver ============\n");
        logger->log_info(options_, "\n");

        // initialize local variables
        k = 1;
        theta = (k-2)/(k+1);
        x->assign(x0);
        xp->assign(x0);
        set_y(x, xp, y, theta);
        logger->log_info("Computing f, g ...");
        auto tstart_init = tic();
        obj_fx = (*model.f)(*x, *gx);
        obj_fy = (*model.f)(*y, *gy);
        logger->log_format("done. (%.2f sec)\n", toc(tstart_init));
        //++out.num_feval; ++out.num_geval;
        logger->log_info("Computing h ...");
        tstart_init = tic();
        obj_h = (*model.h)(*x);
        logger->log_format("done. (%.2f sec)\n", toc(tstart_init));
        obj = obj_fx + mu_factor * obj_h;
        obj_orig = obj_fx + obj_h;
        if (ls_variant == 2) Cval = obj;
        else Cval = obj_fy;
        Q = 1;
        nrmG = -1;
        x_x = x->dot(*x);
        g_g = gy->dot(*gy);
        y_g = y->dot(*gy);

        // continuation
        if (cont){
            if (!model.enable_continuation()){
                logger->log_info("Option \'continuation\' is set to true but the model ",
                        model.name, " does not support continuation.\n");
                logger->log_info("Disabling the strategy.\n");
                cont = false;
            } else {

                mu_factor0 = model.mu_factor_init();
                mu_factor = mu_factor0;
                ftol_c = c_tol * std::sqrt(ftol);
                logger->log_info("Using continuation strategy:\n");
                logger->log_info("  mu_factor_init = ", mu_factor0, "\n");

            }
        }

        // adaptive pre-determined step
        if (adapls)
            step_hist.resize(adap_w);

        // main loop
        for (iter = 0, iter_cbegin = 0; iter < maxit; ++iter){

            // copy vars from previous iteration
            xp->assign(*x);
            yp->assign(*y);
            gyp->assign(*gy);
            objp = obj;
            xp_xp = x_x; gp_gp = g_g; yp_gp = y_g;
            Index nls = 1;

            // write to workspace
            workspace_->set("iter", iter);
            workspace_->set("iter_c", iter - iter_cbegin);
            workspace_->set("mu_factor", mu_factor);

            auto tstart_local = tic();
            (*model.prox_h)(*yp, tau, *gyp, tau * mu_factor, *x);
            t_local_prox_h = toc(tstart_local);
            t_local_f = 0_s;

            // line search
            while (true){
                // we need to explicitly specify *g (though grad f is not needed)
                // this will help model.f to find the right call of operator()
                tstart_local = tic();
                obj_fx = (*model.f)(*x, *gx, false);
                ++out.num_feval;
                t_local_f += toc(tstart_local);
                // compute the squared norm of x - xp
                // x_x + xp_xp - 2x_xp or x.squared_norm_diff(xp)
                // note: when possible, perfer `squared_norm_diff` for better
                // numerical stability
                x_x = x->dot(*x);
                yp_yp = yp->dot(*yp);
                if (x->has_squared_norm_diff()){
                    nrm2_dxy = x->squared_norm_diff(*yp);
                    nrm2_dx = x->squared_norm_diff(*xp);
                    nrm_dx = sqrt(nrm2_dx);
                } else {
                    x_yp = x->dot(*yp);
                    x_x = x->dot(*x);
                    x_xp = x->dot(*xp);

                    nrm2_dxy = std::fabs(to_real(x_x + yp_yp - 2.0_s * x_yp));
                    nrm2_dx = std::fabs(to_real(x_x + xp_xp - 2.0_s * x_xp));
                    nrm_dx = sqrt(nrm2_dx);
                }

                // compute line search rule
                bool ls_result = false;
                switch (ls_variant){
                    // ls_variant = 1, use "standard" rule
                    case 1: {
                        // compute <gp, x - yp> = x_gp - yp_gp
                        x_gp = x->dot(*gyp);
                        Scalar deriv = to_real(x_gp - yp_gp) + 0.5_s / tau * nrm2_dxy;
                        ls_result = obj_fx <= Cval + deriv;
                        break;
                    }
                    // ls_variant = 2, use "Zhang, Hager" rule
                    // note: we need obj_h here
                    case 2:
                        // compute obj_h
                        obj_h = compute_obj_h(model);

                        // rule: psi(x^+) <= Cval - c * tau/2 * ||x^+ - y||^2
                        ls_result = obj_fx + mu_factor * obj_h <= Cval - 0.5_s * rhols * tau * nrm2_dxy;
                        break;
                    // other cases: no rules are applied
                    default:
                        ls_result = true;
                }

                if (ls_result || nls == maxnls)
                    break;

                tau = eta * tau; ++nls;

                tstart_local = tic();
                (*model.prox_h)(*yp, tau, *gyp, tau * mu_factor, *x);
                t_local_prox_h += toc(tstart_local);
            }

            if (ls_variant != 2) // for some ls variants, obj_h is not computed
                obj_h = compute_obj_h(model);

            // record tau
            if (adapls)
                step_hist[iter % adap_w] = tau;

            obj = obj_fx + mu_factor * obj_h;
            obj_orig = obj_fx + obj_h;

            // compute new function value and gradient
            // use cached gradient if possible
            tstart_local = tic();
            obj_fx = (*model.f)(*x, *gx, true, true);
            ++out.num_geval;
            t_local_f += toc(tstart_local);

            objdiff = std::fabs(objp - obj) / std::max(1_s, std::fabs(objp));
            Xdiff = nrm_dx / std::max(1_s, sqrt(to_real(x_x)));
            nrmG = nrm_dx / tau;

            // record accumulated etime
            t_local_acc = toc(tstart);

            // write to workspace
            workspace_->set("nrmG", nrmG);
            workspace_->set("tau", tau);

            // terminate flag
            bool flag_f = iter > 0 && objdiff < ftol;
            bool flag_X = iter > 0 && Xdiff < xtol;
            bool flag_g = nrmG < gtol;
            bool flag_c = mu_factor < 1 + 1e-12;
            bool flag_u = false;
            if (custom_stop_checker != nullptr)
                flag_u = (*custom_stop_checker)();

            bool cstop = ((flag_f && flag_X) || flag_g || flag_u) && flag_c;

            // logging
            bool it_print = log_iter > 0 && iter % log_iter == 0;
            if (cstop || iter == 0 || it_print || iter == maxit - 1){
                if (iter == 0){
                    logger->log(header);
                    logger->log_info(model.extra_msg_h(), "\n");
                }
                logger->log_format(format, iter, tau, obj_orig, obj, objdiff, Xdiff, nrmG, nls, mu_factor,
                        t_local_f, t_local_prox_h, t_local_acc);
                logger->log_info(model.extra_msg(), "\n");
            }

            // termination
            if (cstop){
                if (flag_g){
                    out.message = "solved: optimal nrmG";
                    out.flag = 0;
                } else if (flag_f){
                    out.message = "solved: optimal objdiff/Xdiff";
                    out.flag = 1;
                } else if (flag_u){
                    out.message = "solved: user-defined custom stop rule";
                    out.flag = 98;
                }
                break;
            }

            // continuation
            if (cont){
                if (mu_factor > 1_s && objdiff < std::max(ftol_c, ftol)){
                    Scalar r = std::max(0.15_s, (Scalar)(1 - c_alpha * std::pow(c_beta, std::log10(mu_factor0/mu_factor))));
                    mu_factor *= r;
                    mu_factor = std::max(mu_factor, 1.0_s);
                    mu_changed = true;
                    iter_cbegin = iter + 1;
                } else {
                    mu_changed = false;
                }
            }

            if (mu_changed)
            {
                k = 1;
                y->assign(*x);
            }
            else
            {
                k++;
                theta = (k-2)/(k+1);
                set_y(x, xp, y, theta);
            }
            
            obj_fy = (*model.f)(*y, *gy);

            // compute BB step size
            // dyg = <y - yp, g - gp>
            //     = y_g + yp_gp - y_gp - yp_g
            y_g = y->dot(*gy);
            yp_g = yp->dot(*gy);
            y_gp = y->dot(*gyp);
            Scalar dyg = to_real(y_g + yp_gp - y_gp - yp_g);

            // dg.squaredNorm()
            //     = <g - gp, g - gp>
            if (gy->has_squared_norm_diff()){
                nrm2_dg = gy->squared_norm_diff(*gyp);
            } else {
                g_g = gy->dot(*gy);
                g_gp = gy->dot(*gyp);
                nrm2_dg = std::fabs(to_real(g_g + gp_gp - 2.0_s * g_gp));
            }

            // dy.squaredNorm()
            //     = <y - yp, y - yp>
            if (y->has_squared_norm_diff()){
                nrm2_dy = y->squared_norm_diff(*yp);
            } else {
                y_y = y->dot(*y);
                y_yp = y->dot(*yp);
                nrm2_dy = std::fabs(to_real(y_y + yp_yp - 2.0_s * y_yp));
            }

            if (dyg > 0){
                if ((bb_variant == 0 && iter % 2 == 0) || bb_variant == 1)
                    tau = nrm2_dy / dyg;
                else if ((bb_variant == 0 && iter % 1 == 0) || bb_variant == 2)
                    tau = dyg / nrm2_dg;
            }

            // safeguarding tau
            tau = std::max(std::min(tau, 1e20_s), 1e-20_s);

            // adaptive step selection
            if (adapls && iter >= adap_w){
                auto res = std::max_element(step_hist.begin(), step_hist.end());
                // we want adap. step smaller than res / eta
                while (tau > *res / eta) {
                    tau *= eta;
                    logger->log_debug("adap. tau: ", tau, "\n");
                }
            }
            

            // update Hongchao & Hager constant
            if (ls_variant == 2){
                if (mu_changed){ // obj has changed, reinit
                    Q = 1;
                    Cval = obj;
                } else {
                    Qp = Q; Q = gamma * Qp + 1;
                    Cval = (gamma * Qp * Cval + obj) / Q;
                }
            } else {
                Cval = obj_fy;
            }
        }
        // write to output struct
        out.iter = iter;
        out.Xdiff = Xdiff;
        out.nrmG = nrmG;
        out.obj = obj_orig;
        out.xx = x;
        out.etime = toc(tstart);

        // unbinding
        model.f->unbind();
        model.h->unbind();
        model.prox_h->unbind();
        if (custom_stop_checker != nullptr)
            custom_stop_checker->unbind();
    }

    // template instantiate
    template class FISTA<Scalar>;
    template class FISTA<ComplexScalar>;
}}}
