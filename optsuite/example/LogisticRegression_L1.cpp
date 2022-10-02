#include <iostream>
#include "OptSuite/core_n.h"
#include "OptSuite/Base/functional.h"
#include "OptSuite/Utils/tictoc.h"
#include "OptSuite/Composite/Model/logistic_regression_l1.h"
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
    int l = 1;

    // set rng seed
    rng(233);

    Mat A = randn(n, m);
    Mat b = randn(m, l);
    for (Index i = 0; i < m; i++){
        b.coeffRef(i,0) = (rand() % 2 == 0 ? 1 : -1);
    }
    Scalar mu = 0.001;
    Mat x0 = randn(n, l);

    Composite::Model::LogisticRegression_L1 model(A, b, mu);
    Composite::Solver::ProxGBB<> alg;

    alg.options().set_from_cmd_line(argc, argv);

    auto tstart = tic();
    alg.solve(model, x0);
    auto elapsed = toc(tstart);
    std::cout << "Elapsed time is " << elapsed << " sec." << std::endl;
    std::cout << "Solver returned with message " << alg.out.message << std::endl;

    return 0;
}