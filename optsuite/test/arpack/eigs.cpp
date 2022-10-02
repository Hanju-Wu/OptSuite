/*
 * ==========================================================================
 *
 *       Filename:  eigs.cpp
 *
 *    Description:  
 *
 *        Version:  1.0
 *        Created:  03/09/2021 06:05:12 PM
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  Haoyang Liu (@liuhy), liuhaoyang@pku.edu.cn
 *   Organization:  BICMR, PKU
 *
 * ==========================================================================
 */

#include "OptSuite/core_n.h"
#include "OptSuite/LinAlg/arpack.h"

using namespace OptSuite;
using namespace OptSuite::LinAlg;

int main(){
    Mat A = Mat::Random(50, 50);
    Mat B = A + A.transpose();

    ARPACK<Scalar, ARPACK_Sym> eigs;
    Mat ev = eigs.compute(A, 15, "LA").d();
    std::cout << ev << std::endl;
    return 0;
}
