/*
 * ==========================================================================
 *
 *       Filename:  geomean.h
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

#ifndef OPTSUITE_GEOMEAN_H
#define OPTSUITE_GEOMEAN_H

#include "OptSuite/core_n.h"

namespace OptSuite { namespace Utils {
    template <typename Derived>
    Scalar geomean(const MatrixBase<Derived>& v_) {
        EIGEN_STATIC_ASSERT_VECTOR_ONLY(Derived);
        const Derived& v = v_.derived();
        return exp(v.array().log().sum() / v.size());
    }
}}

#endif
