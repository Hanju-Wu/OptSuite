/*
 * @Author: Wu Hanju 
 * @Date: 2022-09-15 10:46:17 
 * @Last Modified by: Wu Hanju
 * @Last Modified time: 2022-09-15 11:22:13
 */

#include "OptSuite/LinAlg/power_method.h"
#include "OptSuite/LinAlg/rng_wrapper.h"
#include <ctime>

using Eigen::Infinity;

namespace OptSuite { namespace LinAlg {
    Scalar maxeigvalue(Mat A)
    {
        OPTSUITE_ASSERT(A.rows() == A.cols());
        rng(time(0));

        Index n = A.rows();
        Vec u0, u1, y;
        double mu0, mu1;
        u0 = randn(n, 1);
        mu0 = 0;
        y = A * u0;
        mu1 = y.lpNorm<Infinity>();
        u1 = y / mu1;
        while ((u0 - u1).lpNorm<Infinity>() > 1e-10 && abs(mu0 - mu1) > 1e-10)
        {
            mu0 = mu1;
            y = A * u1;
            u0 = u1;
            mu1 = y.lpNorm<Infinity>();
            u1 = y / mu1;
        }
        return mu1;
    }
}}




