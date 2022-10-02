/*
 * ==========================================================================
 *
 *       Filename:  fft.cpp
 *
 *    Description:
 *
 *        Version:  1.0
 *        Created:  11/26/2020 01:22:04 PM
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  Haoyang Liu (@liuhy), liuhaoyang@pku.edu.cn
 *   Organization:  BICMR, PKU
 *
 * ==========================================================================
 */

#include <iostream>
#include <complex>
#include <vector>
#include "OptSuite/core_n.h"
#include "OptSuite/LinAlg/fftw_wrapper.h"

using namespace OptSuite;
using namespace OptSuite::LinAlg;


int main(/*int argc, char **argv*/) {
    CMat in(16, 2);

    in(0, 0) = 1;
    in(3, 0) = {0, 2};

    in(1, 1) = 2;
    in(4, 1) = {2, 5};

    CMat out = fft(in);
    in = ifft(out);

    std::cout << out << std::endl;
    std::cout << in << std::endl;

    Mat A = Mat::Zero(16, 2);
    A(0, 0) = 1;
    A(1, 1) = 1;
    A(3, 0) = 2;
    A(4, 1) = 2;
    A(8, 0) = 7.5;
    A(9, 1) = 7.5;
    A(15, 0) = -3;
    A(15, 1) = -3;
    Mat B = dct(A);
    Mat C = dct(A, -1, DCT_Type::TYPE_IV);
    Mat D = dct(A, -1, DCT_Type::TYPE_I);

    std::cout << A << std::endl;
    std::cout << B << std::endl;
    std::cout << C << std::endl;
    std::cout << D << std::endl;
    std::cout << idct(B) << std::endl;
    std::cout << dct(C, -1, DCT_Type::TYPE_IV) << std::endl;
    std::cout << dct(D, -1, DCT_Type::TYPE_I) << std::endl;

    return 0;
}
