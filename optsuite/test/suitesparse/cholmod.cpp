
/*
 * ==========================================================================
 *
 *       Filename:  cholmod.cpp
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
#include "OptSuite/LinAlg/cholmod_wrapper.h"

using namespace OptSuite;
using namespace OptSuite::LinAlg;

int main() {

    // MATLAB:
    // A = sparse([3,4,2,2,4,1,5],[1,1,3,5,5,4,2],[0.2,0.3,0.4,0.5,0.6,0.7,0.8]);
    std::vector< Eigen::Triplet<Scalar> > triplets;
    triplets.emplace_back(2, 0, 0.2);
    triplets.emplace_back(3, 0, 0.3);
    triplets.emplace_back(1, 2, 0.4);
    triplets.emplace_back(1, 4, 0.5);
    triplets.emplace_back(3, 4, 0.6);
    triplets.emplace_back(0, 3, 0.7);
    triplets.emplace_back(4, 1, 0.8);

    constexpr Index N = 5;
    SpMat A(N, N);
    A.setFromTriplets(triplets.begin(), triplets.end());

    // b = [1,2,3,4,5]';
    Vec b(N);
    for (Index i = 0; i < N; ++i) {
        b(i) = static_cast<Scalar>(i + 1);
    }

    CholmodWrapper cholmod;

    // P1 = analyze(A);
    const PermutationMatrix P1 = cholmod.analyze(A, CholmodWrapper::AnalyzeType::Normal);
    std::cout << P1.indices() << "\n\n";  // expected 3 2 0 1 4

    // P2 = analyze(A, 'row');
    const PermutationMatrix P2 = cholmod.analyze(A, CholmodWrapper::AnalyzeType::Row);
    std::cout << P2.indices() << "\n\n";  // expected 0 2 1 3 4

    // [LD1, p] = ldlchol(A)
    Index p;
    const SpMat LD1 = cholmod.ldlchol(A, nullptr, p);
    std::cout << p << "\n\n"; // expected 0 (failed)

    constexpr Scalar beta = 0.01;
    // [LD2, p] = ldlchol(A, 0.01);
    const SpMat LD2 = cholmod.ldlchol(A, &beta, p);
    SpMat L2;
    DiagMat D2;
    cholmod.ldlsplit(LD2, L2, D2);
    const SpMat AAtbeta = A * A.adjoint() + Mat::Identity(N, N) * beta;
    const SpMat AAtbeta2 = L2 * D2 * L2.adjoint();
    std::cout << (AAtbeta2 - AAtbeta).norm() << "\n\n";  // expected near 0

    // x = ldlsolve(LD2, b);
    Mat x = cholmod.ldlsolve(LD2, b);
    std::cout << x << "\n\n"; // expected 2.0000 8.9912 67.1053 -5.9211 7.6923

    // test raw version
    auto LD2_raw = cholmod.ldlchol_raw(A, &beta, p);
    x = cholmod.ldlsolve(LD2_raw, b);
    std::cout << x << "\n\n"; // expected 2.0000 8.9912 67.1053 -5.9211 7.6923

    // test reuse version
    // MATLAB:
    // [LD2, p] = ldlchol(A * 2, 0.01);
    // x = ldlsolve(LD2, b);
    cholmod.ldlchol_raw(A * 2.0, &beta, LD2_raw);
    x = cholmod.ldlsolve(LD2_raw, b);
    std::cout << x << "\n\n"; // expected 0.5076 3.2657 21.6335 -2.8237 1.9455

    return 0;
}
