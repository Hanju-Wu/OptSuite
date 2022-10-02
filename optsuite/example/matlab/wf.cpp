/*
 * ==========================================================================
 *
 *       Filename:  wf.cpp
 *
 *    Description:
 *
 *        Version:  1.0
 *        Created:  11/27/2020 03:26:09 PM
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  Haoyang Liu (@liuhy), liuhaoyang@pku.edu.cn
 *   Organization:  BICMR, PKU
 *
 * ==========================================================================
 */

#include <iostream>
#include <fstream>
#include "OptSuite/core_n.h"
#include "OptSuite/Base/functional.h"
#include "OptSuite/Utils/tictoc.h"
#include "OptSuite/Composite/Model/lasso.h"
#include "OptSuite/Composite/Solver/proxgbb.h"
#include "OptSuite/LinAlg/rng_wrapper.h"
#include "OptSuite/LinAlg/fftw_wrapper.h"
#include "OptSuite/Interface/matlab.h"

using namespace OptSuite;
using namespace OptSuite::LinAlg;
using namespace OptSuite::Base;
using namespace OptSuite::Composite;
using namespace OptSuite::Utils;
using namespace OptSuite::Interface;

class PhaseLS : public FuncGrad<ComplexScalar> {
    public:
        PhaseLS(const Ref<const CMat> D, const Ref<const Mat> Y){
            this->D = D;
            this->Y = Y;
            m = D.size();
        }

        ~PhaseLS() = default;

        Scalar operator()(const Ref<const CMat> z, Ref<CMat> y, bool compute_grad = true, bool cached_grad = false){
            // compute Az
            if (cached_grad){
                Az = fft(D.conjugate().array().colwise() * z.col(0).array());
                res_mat = Az.cwiseAbs2() - Y;
                fun = res_mat.squaredNorm() / (4_s * (Scalar)m);
            }

            if (compute_grad){
                y = (D.array() * ifft(res_mat.array() * Az.array()).array()).
                    rowwise().sum() / (Scalar)D.cols();
            }

            return fun;
        }
    private:
        CMat D;
        Mat Y;
        Index m;
        CMat Az;
        Mat res_mat;
        Scalar fun;
};

void gen_y_C(const Ref<const CMat> z, Ref<CMat> D, Ref<Mat> Y){
    Index n = D.rows();
    Index L = D.cols();
    Mat b = LinAlg::rand(n, L);
    CMat b1(n, L);
    Mat b2(n, L);


    b1 = (b.array() < 0.25).select(1_s + 0_ii, b1);
    b1 = (b.array() >= 0.25 && b.array() < 0.5).select(-1_s + 0_ii, b1);
    b1 = (b.array() >= 0.5 && b.array() < 0.75).select(1_ii, b1);
    b1 = (b.array() >= 0.75).select(-1_ii, b1);

    b = LinAlg::rand(n, L);

    b2 = (b.array() < 0.8).select(sqrt(2.0_s) / 2, b2);
    b2 = (b.array() >= 0.8).select(sqrt(3.0_s), b2);

    D = b1.array() * b2.array();

    Y = fft(D.conjugate().array().colwise() * z.col(0).array()).cwiseAbs2();
}

inline std::complex<Scalar> matrix_dot(const Ref<const CMat> x, const Ref<const CMat> y){
    return (x.array().conjugate() * y.array()).sum();
}

Scalar recover_error(const Ref<const CMat> z, const Ref<const CMat> x, bool is_relative = true){
    Scalar abs_error = (x - std::exp(-1_ii * std::arg(matrix_dot(x, z))) * z).norm();
    return is_relative ? abs_error / x.norm() : abs_error;
}


int main(int argc, char **argv){
    // read data
    Mat X;
#if OPTSUITE_SCALAR_TOKEN == 0
    get_matrix_from_file("../example/data/tower_d.mat", "tower", X);
#elif OPTSUITE_SCALAR_TOKEN == 1
    get_matrix_from_file("../example/data/tower_s.mat", "tower", X);
#else
#error "unsupported type"
#endif

    // check the input
    if (X.size() == 0){
        std::cerr << "Failed to load tower_s.mat file." << std::endl;
        return 1;
    } else {
        std::cout << "tower_s.mat loaded: " << X.rows() << " x " << X.cols()
            << " = " << X.size() << " pixels." << std::endl;
    }

    X.resize(X.size(), 1);

    Index n = X.rows(), L = 15;
    CMat D(n, L);
    Mat Y(n, L);
    Logger file_logger("x0.dat");

    // set random seed
    rng(23333);
    CVec x = (1.0_s + 0_ii) * X;
    CVec z0 = randn(n, 1) + 1_ii * randn(n, 1);

    file_logger.log(x);

    std::cout << "Set of masks: " << L << std::endl;
    std::cout << "Generating masks and observations... ";
    gen_y_C(x, D, Y);
    std::cout << "Done." << std::endl;

    Composite::Model::Base<ComplexScalar> model;
    Composite::Solver::ProxGBB<ComplexScalar> alg;

    model.f = std::make_shared<PhaseLS>(D, Y);
    model.h = std::make_shared<Zero<ComplexScalar>>();
    model.prox_h = std::make_shared<IdentityProx<ComplexScalar>>();

    alg.options().set_from_cmd_line(argc, argv);

    auto tstart = tic();
    alg.solve(model, z0);
    auto elapsed = toc(tstart);

    auto& xx = alg.get_sol();
    std::cerr << "Elapsed time is " << elapsed << " sec." << std::endl;
    std::cerr << "Solver returned with message " << alg.out.message << std::endl;
    std::cerr << "Recover error is " << recover_error(xx.mat(), x) << std::endl;

    file_logger.redirect_to_file("recovered.dat");
    file_logger.log(xx.mat());



    return 0;
}
