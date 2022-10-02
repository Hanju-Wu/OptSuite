#ifndef SSNSDP_GEOMEAN_H
#define SSNSDP_GEOMEAN_H

#include "core.h"

namespace ssnsdp {
    template <typename Derived>
    Scalar geomean(const MatrixBase<Derived>& v_) {
        EIGEN_STATIC_ASSERT_VECTOR_ONLY(Derived);
        const Derived& v = v_.derived();
        return exp(v.array().log().sum() / v.size());
    }
}

#endif
