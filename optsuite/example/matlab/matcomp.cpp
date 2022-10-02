/*
 * ===========================================================================
 *
 *       Filename:  matcomp.cpp
 *
 *    Description:
 *
 *        Version:  1.0
 *        Created:  12/05/2020 02:41:41 PM
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  Haoyang Liu (), liuhaoyang@pku.edu.cn
 *   Organization:  BICMR, Peking University
 *      Copyright:  Copyright (c) 2020, Haoyang Liu
 *
 * ===========================================================================
 */

#include <iostream>
#include <memory>
#include "OptSuite/core_n.h"
#include "OptSuite/Base/functional.h"
#include "OptSuite/Base/spmat_wrapper_mkl.h"
#include "OptSuite/Base/factorized_mat.h"
#include "OptSuite/Composite/Model/mat_comp.h"
#include "OptSuite/Composite/Solver/proxgbb.h"
#include "OptSuite/LinAlg/rng_wrapper.h"
#include "OptSuite/Interface/matlab.h"

using namespace OptSuite;
using namespace OptSuite::LinAlg;
using namespace OptSuite::Base;
using namespace OptSuite::Composite;
using namespace OptSuite::Utils;
using namespace OptSuite::Interface;

static OptionList options_m;

void register_options(){
    std::vector<RegOption> v;

    options_m.register_option({"m", 100_i, "m"});
    options_m.register_option({"n", 100_i, "n"});
    options_m.register_option({"r", 5_i, "r"});
    options_m.register_option({"sr", 0.1_s, "Sample ratio"});
    options_m.register_option({"mu_scaling", 1e-4_s, "mu_scaling"});
    options_m.register_option({"mkl", false, "use MKL for SpMV backend"});
    options_m.register_option({"prefix", "/tmp", "prefix for extern file"});
    options_m.register_option({"config", "../example/mc.conf", "path to configuration file"});
    options_m.register_option({"hconfig", "../example/mc_h.conf", "path to configuration file for prox_h"});
}

int do_random(int argc, char **argv){
    // set main options from cmd line
    register_options();
    options_m.set_from_cmd_line(argc, argv, "--mopt");
    std::cerr << options_m << std::endl;

    // extract options
    Index m = options_m.get_integer("m");
    Index n = options_m.get_integer("n");
    Index r = options_m.get_integer("r");
    Scalar sr = options_m.get_scalar("sr");
    Scalar mu_scaling = options_m.get_scalar("mu_scaling");
    bool mkl = options_m.get_bool("mkl");
    std::string prefix = options_m.get_string("prefix");
    std::string config = options_m.get_string("config");

    // set rng seed
    rng(233);

    Eigen::BDCSVD<Mat> svd;

    Mat UU, VV;
    SpMat omega;

    // read from file
    std::string filename = str_format("%s/NNLS_m%d_n%d_r%d_sr%d.mat",
            prefix.c_str(), m, n, r, int(sr*100));
    std::cerr << "Reading file \"" << filename << "\"" << std::endl;
    get_matrix_from_file(filename.c_str(), "U", UU);
    get_matrix_from_file(filename.c_str(), "V", VV);
    get_matrix_from_file(filename.c_str(), "B", omega);

    // exit if mat file does not exist
    if (UU.rows() == 0)
        return -1;

    FactorizedMat<Scalar> X(UU.transpose(), VV.transpose());
    Mat U = Mat::Zero(1, n);
    Mat V = Mat::Zero(1, n);
    FactorizedMat<Scalar> X0(U, V);

    // Logger logger("X0.dat");

    Composite::Model::MatComp model(omega);
    Composite::Solver::ProxGBB<> alg;

    // set final mu for model
    model.set_mu(model.mu_init() * mu_scaling);

    // set model and algorithm parameters
    if (!config.empty())
        alg.options().set_from_file(config);
    alg.options().set_from_cmd_line(argc, argv); // cmd-line takes higher prio

    // decide the gradient type
    if (sr > 0.15_s){
        std::cerr << "Selected Dense G" << std::endl;
        alg.set_variable_type<FactorizedMat<Scalar>, MatWrapper<Scalar>>();
    } else if (mkl){
        std::cerr << "Selected Sparse G with MKL" << std::endl;
        alg.set_variable_type<FactorizedMat<Scalar>, SpMatWrapper_mkl<Scalar>>();
    } else {
        std::cerr << "Selected Sparse G with builtin" << std::endl;
        alg.set_variable_type<FactorizedMat<Scalar>, SpMatWrapper<Scalar>>();
    }

    // solve the model
    alg.solve(model, X0);

    auto& xx = alg.get_sol<FactorizedMat<Scalar>>();
    Scalar nrmX_sqr = X.squared_norm();
    Scalar sqr_nrm_diff = nrmX_sqr + xx.squared_norm() - 2 * X.dot(xx);
    Scalar rerr = std::sqrt(sqr_nrm_diff / nrmX_sqr);
    std::cerr << "# of feval: " << alg.out.num_feval << "    " <<
                 "# of geval: " << alg.out.num_geval << std::endl;
    std::cerr << "Elapsed time is " << alg.out.etime << " sec." << std::endl;
    std::cerr << "Solver returned with message " << alg.out.message << std::endl;
    std::cerr << "Relative error is " << rerr << std::endl;
    std::cerr << alg.out.etime / (alg.out.iter + 1) << " sec/it." << std::endl;

    std::cout <<
        str_format(R"(& %4d & %4d & %12.6e & %6.1f & %6.2e \\ \hline)",
                alg.out.iter, xx.rank(), alg.out.obj, alg.out.etime, rerr) << std::endl;
    // std::cerr << "obj(GT) = " << (*model.f)(X) << " + " << (*model.h)(X) << std::endl;
    // std::cerr << "obj(SL) = " << (*model.f)(xx.mat()) << " + " <<  (*model.h)(xx.mat()) << std::endl;

    //if (m <= 2000_i && n <= 2000_i){
    //    logger.log(svd.compute(X.mat()).singularValues());
    //    logger.redirect_to_file("XX.dat");
    //    logger.log(svd.compute(xx.mat()).singularValues());
    //}

    return 0;
}

int do_real(int argc, char **argv){
    if (argc < 3){
        std::cerr << "No input data file." << std::endl;
        return 1;
    }
    // set main options from cmd line
    register_options();
    options_m.set_from_cmd_line(argc, argv, "--mopt");
    std::cerr << options_m << std::endl;

    // extract options
    Scalar mu_scaling = options_m.get_scalar("mu_scaling");
    std::string prefix = options_m.get_string("prefix");
    std::string config = options_m.get_string("config");
    std::string hconfig = options_m.get_string("hconfig");
    bool mkl = options_m.get_bool("mkl");

    // set rng seed
    rng(233);

    SpMat omega, M, M1;

    // read from file
    std::string filename = str_format("%s/%s.mat", prefix.c_str(), argv[2]);
    std::cerr << "Reading file \"" << filename << "\"" << std::endl;
    get_matrix_from_file(filename.c_str(), "M", M);
    get_matrix_from_file(filename.c_str(), "M1", M1);
    get_matrix_from_file(filename.c_str(), "B", omega);

    // exit if mat file does not exist
    if (omega.rows() == 0)
        return -1;

    Index m = omega.rows(), n = omega.cols();
    Mat U = Mat::Zero(1, m);
    Mat V = Mat::Zero(1, n);
    FactorizedMat<Scalar> X0(U, V);

    // Logger logger("X0.dat");

    Composite::Model::MatComp model(omega);
    Composite::Solver::ProxGBB<> alg;

    // set final mu for model
    model.set_mu(model.mu_init() * mu_scaling);

    // set model and algorithm parameters
    if (!config.empty())
        alg.options().set_from_file(config);
    alg.options().set_from_cmd_line(argc, argv); // cmd-line takes higher prio

    if (!hconfig.empty())
        model.prox_h_ptr()->options().set_from_file(hconfig);
    model.prox_h_ptr()->options().set_from_cmd_line(argc, argv, "--hopt");

    // decide the gradient type
    if (mkl)
        alg.set_variable_type<FactorizedMat<Scalar>, SpMatWrapper_mkl<Scalar>>();
    else
        alg.set_variable_type<FactorizedMat<Scalar>, SpMatWrapper<Scalar>>();

    // solve the model
    alg.solve(model, X0);

    // obtain solution
    auto& xx = alg.get_sol<FactorizedMat<Scalar>>();

    // compute RMAE
    FactorizedMat<Scalar> zero_f(m, n, 0);
    ProjectionOmega<Scalar> Amap(M);
    Amap(xx);
    Mat Rvec = -Amap.rvec();
    Amap(zero_f);
    Mat Mvec = Amap.rvec();
    Scalar MAE = Rvec.array().abs().mean();
    Scalar range = Mvec.maxCoeff() - Mvec.minCoeff();

    std::cerr << "# of feval: " << alg.out.num_feval << "    " <<
                 "# of geval: " << alg.out.num_geval << std::endl;
    std::cerr << "Elapsed time is " << alg.out.etime << " sec." << std::endl;
    std::cerr << "Solver returned with message " << alg.out.message << std::endl;
    std::cerr << alg.out.etime / (alg.out.iter + 1) << " sec/it." << std::endl;

    std::cout <<
        str_format(R"(& %4d & %4d & %12.6e & %6.1f & %6.2e \\ \hline)",
                alg.out.iter, xx.rank(), alg.out.obj, alg.out.etime, MAE/range) << std::endl;

    return 0;
}

int main(int argc, char **argv){
    if (argc < 2){
        std::cerr << "Usage: ./matcomp <random|real> [OPTIONS]" << std::endl;
        return 1;
    }

    if (!std::strcmp(argv[1], "random"))
        return do_random(argc, argv);
    else if (!std::strcmp(argv[1], "real"))
        return do_real(argc, argv);
    else
        std::cerr << "Unknown dataset " << argv[1] << std::endl;
    return 0;
}
