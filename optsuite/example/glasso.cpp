/*
 * ==========================================================================
 *
 *       Filename:  glasso.cpp
 *
 *    Description:
 *
 *        Version:  1.0
 *        Created:  11/14/2020 05:22:50 PM
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  Haoyang Liu (@liuhy), liuhaoyang@pku.edu.cn
 *   Organization:  BICMR, PKU
 *
 * ==========================================================================
 */

#include <iostream>
#include "OptSuite/core_n.h"
#include "OptSuite/Base/functional.h"
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
    int m = 256;
    int n = 512;
    int l = 2;

    // set rng seed
    rng(233);

    Mat A = randn(m, n);
    SpMat u = sprandn_c(n, l, 0.1);
    Mat b = A * u;
    Scalar mu = 0.001;
    Mat x0 = randn(n, l);
    Mat y(n, l);

    Composite::Model::LASSO model(A, b, mu, Model::LASSOType::Rowwise);
    Composite::Solver::ProxGBB<> alg;

    alg.options().set_from_cmd_line(argc, argv);

    auto tstart = tic();
    alg.solve(model, x0);
    auto elapsed = toc(tstart);
    std::cout << "Elapsed time is " << elapsed << " sec." << std::endl;
    std::cout << "Solver returned with message " << alg.out.message << std::endl;

    Mat err = alg.get_sol().mat() - u;
    std::cout << "Relative error: " << err.norm() / (u.norm() + 1.0) << std::endl;

    return 0;
}
