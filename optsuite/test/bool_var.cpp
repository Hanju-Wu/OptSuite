
/*
 * ==========================================================================
 *
 *       Filename:  bool_var.cpp
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
#include "OptSuite/LinAlg/rng_wrapper.h"

using namespace OptSuite;
using namespace OptSuite::Base;
using namespace OptSuite::LinAlg;

int main() {
    VariableRef<Scalar> A { Mat::NullaryExpr(2, 2, [](int i, int j) { return i + j; }) };
    std::cout << A << "\n\n";  // expected [0 1; 1 2]

    VariableRef<bool> b1 = A.where([](Scalar s) { return s > 0_s; });  // expected [0 1; 1 1]
    std::cout << b1 << "\n\n";

    VariableRef<bool> b2 = (A < 2.0_s);  // expected [1 1; 1 0]
    std::cout << b2 << "\n\n";

    std::cout << ~(b1 & b2) << "\n\n";  // expected [1 0; 0 1]

    std::cout << A(A > 0_s) << "\n\n";  // expected [1 1 2]

    std::cout << (b1.clone().equals(b1)) << "\n\n";  // expected 1

    A.slice_set(A > 0_s, -1.0_s);
    std::cout << A << "\n\n";  // expected [0 -1; -1 -1]

    VariableRef<Scalar> A2 = b1.map_to_scalar<Scalar>([](bool b) { return b ? 3.0_s : 2.0_s; });
    std::cout << A2 << "\n\n"; // expected [2 3; 3 3]
    return 0;
}
