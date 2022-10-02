/*
 * @Author: Wu Hanju 
 * @Date: 2022-09-11 19:13:25 
 * @Last Modified by: Wu Hanju
 * @Last Modified time: 2022-09-15 11:36:54
 */
#include <iostream>
#include "OptSuite/core_n.h"
#include "OptSuite/Base/functional.h"
#include "OptSuite/Utils/tictoc.h"
#include "OptSuite/Composite/Model/lasso.h"
#include "OptSuite/Composite/Solver/FISTA.h"
#include "OptSuite/Composite/Solver/proxgbb.h"
#include "OptSuite/Composite/Solver/Nesterov_2nd.h"
#include "OptSuite/LinAlg/rng_wrapper.h"
#include "OptSuite/Base/mat_wrapper.h"
#include "OptSuite/Base/variable.h"
#include "OptSuite/LinAlg/power_method.h"

using namespace OptSuite;
using namespace OptSuite::Base;
using namespace OptSuite::Utils;
using namespace OptSuite::LinAlg;
using namespace OptSuite::Composite;
using namespace std;
using var_t = Variable<Scalar>;
using mat_wrapper_t = MatWrapper<Scalar>;
using var_t_ptr = std::shared_ptr<Variable<Scalar>>;

int main(int argc, char **argv){
    int m = 256;
    int n = 512;
    int l = 1;

    // set rng seed
    rng(233);

    Mat A = randn(m, n);
    SpMat u = sprandn_c(n, l, 0.1);
    Mat b = A * u;
    Scalar mu = 0.001;
    Mat x0 = randn(n, l);

    Composite::Model::LASSO model(A, b, mu, Model::LASSOType::Standard);
    //Composite::Solver::ProxGBB<> alg;
    //Composite::Solver::FISTA<> alg;
    Composite::Solver::Nesterov_2nd<> alg;
    

    alg.options().set_from_cmd_line(argc, argv);

    Scalar L;
    if(A.rows()<A.cols())
        L=maxeigvalue(A*A.transpose());
    else
        L=maxeigvalue(A.transpose()*A);
    Scalar t=1/L;

    auto tstart = tic();
    alg.solve(model, x0, t);
    auto elapsed = toc(tstart);
    std::cout << "Elapsed time is " << elapsed << " sec." << std::endl;
    std::cout << "Solver returned with message " << alg.out.message << std::endl;

    Mat err = alg.get_sol().mat() - u;
    std::cout << "Relative error: " << err.norm() / (u.norm() + 1.0) << std::endl;
    
    return 0;
}
