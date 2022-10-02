/*
 * ===========================================================================
 *
 *       Filename:  mkl_mul.cpp
 *
 *    Description:  
 *
 *        Version:  1.0
 *        Created:  02/04/2021 03:52:58 PM
 *       Revision:  none
 *       Compiler:  g++
 *
 *         Author:  Haoyang Liu (), liuhaoyang@pku.edu.cn
 *   Organization:  BICMR, Peking University
 *      Copyright:  Copyright (c) 2021, Haoyang Liu
 *
 * ===========================================================================
 */

#include <iostream>
#include <cstdlib>
#include "OptSuite/core_n.h"
#include "OptSuite/LinAlg/mkl_sparse.h"
#include "OptSuite/LinAlg/rng_wrapper.h"
#include "OptSuite/Utils/tictoc.h"
#include "OptSuite/Base/spmat_wrapper.h"
#include "OptSuite/Base/spmat_wrapper_mkl.h"

using namespace OptSuite;
using namespace OptSuite::LinAlg;
using namespace OptSuite::Base;

int main(int argc, char **argv){
    if (argc != 2){
        std::cerr << "Usage: ./mkl_mul N" << std::endl;
    }
    Index n = std::strtol(argv[1], NULL, 10);
    SpMat A = sprandn(n, n, 0.1);
    Vec b = randn(n, 1);
    Vec tmp = randn(n, 1);
    Mat B = randn(n, std::max(2_i, n / 100));
    Mat C = randn(std::max(2_i, n / 100), n);
    Vec c = Mat::Zero(n, 1);
    Vec c1 = Mat::Zero(n, 1);
    Mat X = Mat::Zero(n, std::max(2_i, n / 100));
    Mat X1 = Mat::Zero(n, std::max(2_i, n / 100));
    Mat Y = Mat::Zero(std::max(2_i, n / 100), n);
    Mat Y1 = Mat::Zero(std::max(2_i, n / 100), n);

    // call native mult
    auto t1 = Utils::tic();
    for (int i = 0; i < 10; ++i){
        c = A.transpose() * b;
        X = A.transpose() * B;
    }
    std::cout << Utils::toc(t1) << std::endl;
    // call mkl
    SpMatWrapper_mkl<Scalar> m_A(A);
    m_A.multiplyT(tmp, c1);
    t1 = Utils::tic();
    for (int i = 0; i < 10; ++i){
        m_A.multiplyT(b, c1);
        m_A.multiplyT(B, X1);
    }
    std::cout << Utils::toc(t1) << std::endl;

    // check norm error
    std::cout << (c - c1).norm() << std::endl;
    std::cout << (X - X1).norm() << std::endl;

    // call native mult transpose
    t1 = Utils::tic();
    for (int i = 0; i < 10; ++i){
        Y = (A * C.transpose()).transpose();
    }
    std::cout << Utils::toc(t1) << std::endl;
    // call mkl
    t1 = Utils::tic();
    for (int i = 0; i < 10; ++i){
        m_A.multiply(C, Y1, MulOp::Transpose);
    }
    std::cout << Utils::toc(t1) << std::endl;

    // check norm error
    std::cout << (Y - Y1).norm() << std::endl;
    return 0;
}
