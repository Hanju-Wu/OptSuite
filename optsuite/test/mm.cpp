/*
 * ==========================================================================
 *
 *       Filename:  mm.cpp
 *
 *    Description:
 *
 *        Version:  1.0
 *        Created:  03/11/2021 07:49:41 PM
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  Haoyang Liu (@liuhy), liuhaoyang@pku.edu.cn
 *   Organization:  BICMR, PKU
 *
 * ==========================================================================
 */

#include <iostream>
#include <cstdlib>
#include <unistd.h>
#include "OptSuite/core_n.h"
#include "OptSuite/Utils/tictoc.h"

using namespace OptSuite;
using namespace Utils;
#define N 5000

int main(/*int argc, char **argv*/){
    Mat A, A1, B, C, D, E;
    A = Mat::Zero(N, N);
    A1 = Mat::Zero(N, N);
    B = Mat::Random(N, N);
    C = Mat::Random(N, N);
    E = Mat::Random(N, N);
    D = Mat::Random(N, N);
    sleep(5);

    auto t = tic();
    A.noalias() = B * C + D * E;
    std::cout << "Method 1: elapsed: " << toc(t) << std::endl;

    //t = tic();
    //A1.noalias() = B * C;
    //A1.noalias() += D * E;
    //std::cout << "Method 2: elapsed: " << toc(t) << std::endl;

    //std::cout << "Compare results: " << (A - A1).norm() << std::endl;

    return 0;
}
