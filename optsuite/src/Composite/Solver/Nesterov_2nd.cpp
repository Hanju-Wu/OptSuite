/*
 * @Author: Wu Hanju 
 * @Date: 2022-09-12 10:35:07 
 * @Last Modified by: Wu Hanju
 * @Last Modified time: 2022-09-15 12:26:21
 */
#include <algorithm>
#include <cmath>
#include "OptSuite/core_n.h"
#include "OptSuite/Base/mat_wrapper.h"
#include "OptSuite/Composite/Model/base.h"
#include "OptSuite/Composite/Solver/Nesterov_2nd.h"
#include "OptSuite/Utils/logger.h"
#include "OptSuite/Utils/tictoc.h"

namespace OptSuite { namespace Composite { namespace Solver {
    inline Scalar to_real(ComplexScalar c) { return c.real(); }
    inline Scalar to_real(Scalar c) { return c; }

    template<typename T>
    OptionList& Nesterov_2nd<T>::options(){ return options_; }

    template<typename T>
    const OptionList& Nesterov_2nd<T>::options() const { return options_; }

    template<typename T>
    std::shared_ptr<const Structure> Nesterov_2nd<T>::workspace() const {
        return workspace_;
    }

    template<typename T>
    void Nesterov_2nd<T>::register_options(){
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
        v.emplace_back("gamma", 0.1_s, "Parameter in Hongchao & Hanger rule (0, 1)", zero_one_checker);
        v.emplace_back("maxit", 10000_i, "Max number of iterations (>=0)", pos_eq_int_checker);
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
        v.emplace_back("ls_variant", 1_i, "Specifies which variant of line search criterion to use.", pos_eq_int_checker);

        // v is constructed, now initialize options_
        this->options_ = OptionList(v, logger);

    }

    template<typename T>
    void Nesterov_2nd<T>::initialize_output(){
        out.num_feval = 0;
        out.num_geval = 0;
        out.message = "exceed max iteration";
        out.flag = 99;
    }

    template<typename T>
    void Nesterov_2nd<T>::initialize_local_variables(const var_t& x0){
        // init x, xp, y, yp, z
        x->set_zero_like(x0);
        xp->set_zero_like(x0);
        y->set_zero_like(x0);
        yp->set_zero_like(x0);
        z->set_zero_like(x0);

        // optionally init gx, gz
        mat_wrapper_t* x_ptr = dynamic_cast<mat_wrapper_t*>(x.get());
        mat_wrapper_t* gx_ptr = dynamic_cast<mat_wrapper_t*>(gx.get());
        mat_wrapper_t* z_ptr = dynamic_cast<mat_wrapper_t*>(z.get());
        mat_wrapper_t* gz_ptr = dynamic_cast<mat_wrapper_t*>(gz.get());

        if (x_ptr && gx_ptr && z_ptr && gz_ptr){
            gx->set_zero_like(x0);
            gz->set_zero_like(x0);
            gxp->set_zero_like(x0);
        }
    }

    template<typename T>
    Scalar Nesterov_2nd<T>::compute_obj_h(const Model::Base<T>& model){
        if (model.prox_h->has_objective_cache())
            return model.prox_h->cached_objective();
        else
            return (*model.h)(*x);
    }
    
    //set_x and set_z can be written as one function
    template<typename dtype>
    void Nesterov_2nd<dtype>::set_x(var_t_ptr xp, var_t_ptr y, var_t_ptr x, Scalar theta)
    {
        mat_t A = (1 - theta) * getmat(xp) + theta * getmat(y);
        const var_t& a=mat_wrapper_t(A);
        x->assign(a);
    }

    template<typename dtype>
    void Nesterov_2nd<dtype>::set_z(var_t_ptr xp, var_t_ptr yp, var_t_ptr z, Scalar theta)
    {
        mat_t A = (1 - theta) * getmat(xp) + theta * getmat(yp);
        const var_t& a=mat_wrapper_t(A);
        z->assign(a);
    }

    template<typename dtype>
    void Nesterov_2nd<dtype>::solve(const Model::Base<dtype>& model, const Ref<const mat_t> x0, const Scalar tau0){
        this->solve(model, MatWrapper<dtype>(x0), tau0);
    }

    template<typename dtype>
    void Nesterov_2nd<dtype>::solve(const Model::Base<dtype>& model, const var_t& x0, const Scalar tau0){
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

        Scalar obj_orig, obj, objp, tau, obj_fx, obj_h, ftol_c = 1_s,
               nrmG, nrm_dx, nrm2_dx,
               objdiff, Xdiff = 0;
        Scalar mu_factor0 = 1, mu_factor = 1;
        Scalar theta;
        Scalar k;
        dtype x_x, x_xp, xp_xp;
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
        Index maxit = options_.get_integer("maxit");
        Index log_iter = options_.get_integer("log_iter");
        Index adap_w = options_.get_integer("adapls_window");
        Scalar c_alpha = options_.get_scalar("cont_alpha");
        Scalar c_beta = options_.get_scalar("cont_beta");
        Scalar c_tol = options_.get_scalar("cont_tol");

        // timers
        auto tstart = tic();
        Scalar t_local_acc, t_local_f, t_local_prox_h;

        // print option details
        logger->log_info("=========== Nesterov_2nd Solver ============\n");
        logger->log_info(options_, "\n");

        // initialize local variables
        k = 1;
        tau=tau0;
        x->assign(x0);
        y->assign(x0);
        logger->log_info("Computing f, g ...");
        auto tstart_init = tic();
        obj_fx = (*model.f)(*x, *gx);
        logger->log_format("done. (%.2f sec)\n", toc(tstart_init));
        //++out.num_feval; ++out.num_geval;
        logger->log_info("Computing h ...");
        tstart_init = tic();
        obj_h = (*model.h)(*x);
        logger->log_format("done. (%.2f sec)\n", toc(tstart_init));
        obj = obj_fx + mu_factor * obj_h;
        obj_orig = obj_fx + obj_h;
        nrmG = -1;
        x_x = x->dot(*x);

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
            gxp->assign(*gx);
            objp = obj;
            xp_xp = x_x;
            Index nls = 1;

            // write to workspace
            workspace_->set("iter", iter);
            workspace_->set("iter_c", iter - iter_cbegin);
            workspace_->set("mu_factor", mu_factor);

            theta = 2/(k+1);
            set_z(xp, yp, z, theta);
            (*model.f)(*z, *gz);
            auto tstart_local = tic();
            (*model.prox_h)(*yp, tau / theta, *gz, tau * mu_factor / theta, *y);
            t_local_prox_h = toc(tstart_local);
            t_local_f = 0_s;

            set_x(xp, y, x, theta);

            x_x = x->dot(*x);
            if (x->has_squared_norm_diff()){
                nrm2_dx = x->squared_norm_diff(*xp);
                nrm_dx = sqrt(nrm2_dx);
            } else {                    
                x_xp = x->dot(*xp);
                nrm2_dx = std::fabs(to_real(x_x + xp_xp - 2.0_s * x_xp));
                nrm_dx = sqrt(nrm2_dx);
            }

            // record tau
            if (adapls){
                step_hist[iter % adap_w] = tau;
            }

            // compute new function value and gradient
            // use cached gradient if possible
            tstart_local = tic();
            obj_fx = (*model.f)(*x, *gx, true, true);
            ++out.num_geval;
            t_local_f += toc(tstart_local);

            obj = obj_fx + mu_factor * obj_h;
            obj_orig = obj_fx + obj_h;

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

            if (mu_changed){
                k = 1;
                y->assign(*x);
            }else{
                k++;
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
    template class Nesterov_2nd<Scalar>;
    template class Nesterov_2nd<ComplexScalar>;
}}}
