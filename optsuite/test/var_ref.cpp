/*
 * ==========================================================================
 *
 *       Filename:  var_ref.cpp
 *
 *    Description:
 *
 *        Version:  1.0
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  Dong Xu (@taroxd), taroxd@pku.edu.cn
 *   Organization:  BICMR, PKU
 *
 * ==========================================================================
 */

#include <iostream>
#include "OptSuite/core_n.h"
#include "OptSuite/Base/variable_ref.h"
#include "OptSuite/Base/variable_ref_cwiseop.h"
#include "OptSuite/LinAlg/rng_wrapper.h"

using namespace OptSuite;
using namespace OptSuite::Base;
using namespace OptSuite::LinAlg;

int main() {
    using varref_t = VariableRef<Scalar>;
    Mat A = Mat::Random(3, 3);
    varref_t refA(A);
    SpMat B = sprandn(3, 3, 0.2);
    varref_t refB(B);

    Mat C = A + B;
    varref_t refC2 = refA + refB;
    varref_t refC(C);

    std::cout << (refC - refC2)->squared_norm() << std::endl;  // expected 0

    varref_t refA2 = refA;
    refA2 *= 3.0;   // does not affect refA
    std::cout << (refA * 3.0 - refA2)->squared_norm() << std::endl;  // expected 0

    varref_t refE1 = refA * 2.0;
    varref_t refE2 { Mat { A * 2.0 } };

    std::cout << (refE1 - refE2)->squared_norm() << std::endl;  // expected 0

    Mat negA = -A;
    std::cout << (varref_t(negA) - (-refA))->squared_norm() << std::endl; // expected 0

    Mat ACmin = A.cwiseMin(C);
    std::cout << (min(refA, refC) - varref_t(ACmin))->squared_norm() << std::endl; // expected 0

    // simple test on VariableRef<void>
    VariableRef<void> obj;
    obj.rebind(new ObjectWrapper<std::string> { "hello world" });
    std::cout << obj << std::endl;
    obj.rebind(new ObjectWrapper<std::vector<int>> { std::vector<int> { 1, 2, 3 }});
    std::cout << obj << std::endl;

    return 0;
}
