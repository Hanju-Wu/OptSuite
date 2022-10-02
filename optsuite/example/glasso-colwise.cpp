/*
 * ==========================================================================
 *
 *       Filename:  glasso_col.cpp
 *
 *    Description:
 *
 *        Version:  1.0
 *        Created:  2/19/2021 08:47:03 PM
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  Yiyang Liu, yiyeung_lau@pku.edu.cn
 *   Organization:  BICMR, PKU
 *
 * ==========================================================================
 */

#include <iostream>
#include "OptSuite/core_n.h"
#include "OptSuite/Base/functional.h"
#include "OptSuite/Base/mat_array.h"
#include "OptSuite/Utils/tictoc.h"
#include "OptSuite/Composite/Model/lasso.h"
#include "OptSuite/Composite/Solver/proxgbb.h"
#include "OptSuite/LinAlg/rng_wrapper.h"

using namespace OptSuite;
using namespace OptSuite::Base;
using namespace OptSuite::Utils;
using namespace OptSuite::LinAlg;
using namespace OptSuite::Composite;

int main(int argc, char **argv){
    int m = 2048;
    int n = 1024;
    int l = 2;

    // set rng seed
    rng(0);

    Mat A = randn(m, n);
    SpMat u = sprandn_c(n, l, 0.1);
    Mat b = A * u;
    Scalar mu = 0.1;
    Mat x0 = randn(n, l);
    Mat y(n, l);

    Logger logger("u.dat");

    Composite::Model::LASSO model(A, b, mu, Model::LASSOType::Colwise);
    Composite::Solver::ProxGBB<> alg;

    alg.options().set_from_cmd_line(argc, argv);

    logger.log(u);

    auto tstart = tic();
    alg.solve(model, x0);
    auto elapsed = toc(tstart);
    std::cerr << "Elapsed time is " << elapsed << " sec." << std::endl;
    std::cerr << "Solver returned with message " << alg.out.message << std::endl;

    logger.redirect_to_file("x.dat");
    logger.log(alg.get_sol().mat());

    return 0;
}
