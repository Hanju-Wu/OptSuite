/*
 * ==========================================================================
 *
 *       Filename:  variable_ref_cwiseop.h
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

#ifndef OPTSUITE_BASE_VARIABLE_REF_CWISEOP_H
#define OPTSUITE_BASE_VARIABLE_REF_CWISEOP_H

#include "OptSuite/Base/variable_ref.h"
#include <functional>

namespace OptSuite { namespace Base {
    // NONE of the function in this file supports variable of different types!
    // If min(Dense, Sparse) is necessary in the future, implement it as a normal binary op (instead of component-wise one)
    inline VariableRef<Scalar> min(const VariableRef<Scalar>& lhs, const VariableRef<Scalar>& rhs) {
        return lhs.binary_map(rhs, [](const Scalar& a, const Scalar& b) { return std::min(a, b); });
    }

    inline VariableRef<Scalar> min(const VariableRef<Scalar>& lhs, const Scalar& rhs) {
        return lhs.unary_map([&rhs](const Scalar& a) { return std::min(a, rhs); });
    }

    inline VariableRef<Scalar> max(const VariableRef<Scalar>& lhs, const VariableRef<Scalar>& rhs) {
        return lhs.binary_map(rhs, [](const Scalar& a, const Scalar& b) { return std::max(a, b); });
    }

    inline VariableRef<Scalar> max(const VariableRef<Scalar>& lhs, const Scalar& rhs) {
        return lhs.unary_map([&rhs](const Scalar& a) { return std::max(a, rhs); });
    }

    template <typename dtype>
    inline VariableRef<dtype> cwise_product(const VariableRef<dtype>& lhs, const VariableRef<dtype>& rhs) {
        return lhs.binary_map(rhs, std::multiplies<dtype>());
    }

    template <typename dtype>
    inline VariableRef<dtype> zeros_like(const VariableRef<dtype>& var) {
        return var.unary_map([](const dtype&) { return dtype(); });
    }
}}

#endif
