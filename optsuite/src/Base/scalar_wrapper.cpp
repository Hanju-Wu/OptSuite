/*
 * ==========================================================================
 *
 *       Filename:  scalar_wrapper.cpp
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

#include "OptSuite/Base/scalar_wrapper.h"
#include "OptSuite/Base/mat_wrapper.h"
#include "OptSuite/Base/spmat_wrapper.h"

namespace OptSuite { namespace Base {
    template <typename dtype>
    typename ScalarWrapper<dtype>::VarPtr
    ScalarWrapper<dtype>::accept_op_add(const Variable<dtype>& lhs) const {
        return lhs.eval_op_add(data_);
    }

    template <typename dtype>
    typename ScalarWrapper<dtype>::VarPtr
    ScalarWrapper<dtype>::eval_op_add(const dtype& rhs) const {
        std::unique_ptr<ScalarWrapper<dtype>> ans { new ScalarWrapper<dtype> };
        ans->data() = data_ + rhs;
        return VarPtr { std::move(ans) };
    }

    template <typename dtype>
    typename ScalarWrapper<dtype>::VarPtr
    ScalarWrapper<dtype>::accept_op_sub(const Variable<dtype>& lhs) const {
        return lhs.eval_op_sub(data_);
    }

    template <typename dtype>
    typename ScalarWrapper<dtype>::VarPtr
    ScalarWrapper<dtype>::eval_op_sub(const dtype& rhs) const {
        std::unique_ptr<ScalarWrapper<dtype>> ans { new ScalarWrapper<dtype> };
        ans->data() = data_ - rhs;
        return VarPtr { std::move(ans) };
    }

    template <typename dtype>
    typename ScalarWrapper<dtype>::VarPtr
    ScalarWrapper<dtype>::accept_op_mul(const Variable<dtype>& lhs) const {
        return lhs.eval_op_mul(data_);
    }

    template <typename dtype>
    typename ScalarWrapper<dtype>::VarPtr
    ScalarWrapper<dtype>::eval_op_mul(const Ref<const mat_t>& rhs) const {
        return VarPtr { new MatWrapper<dtype> { data_ * rhs } };
    }

    template <typename dtype>
    typename ScalarWrapper<dtype>::VarPtr
    ScalarWrapper<dtype>::eval_op_mul(const Ref<const spmat_t>& rhs) const {
        return VarPtr { new SpMatWrapper<dtype> { data_ * rhs } };
    }

    template <typename dtype>
    typename ScalarWrapper<dtype>::VarPtr
    ScalarWrapper<dtype>::eval_op_mul(const dtype& rhs) const {
        return VarPtr { new ScalarWrapper<dtype> { data_ * rhs } };
    }

    template <typename dtype>
    typename ScalarWrapper<dtype>::VarPtr
    ScalarWrapper<dtype>::accept_op_div(const Variable<dtype>& lhs) const {
        return lhs.eval_op_div(data_);
    }

    template <typename dtype>
    typename ScalarWrapper<dtype>::VarPtr
    ScalarWrapper<dtype>::eval_op_div(const dtype& rhs) const {
        return VarPtr { new ScalarWrapper<dtype> { data_ / rhs } };
    }

    // template <typename dtype>
    // typename ScalarWrapper<dtype>::BoolVarPtr
    // ScalarWrapper<dtype>::eval_op_gt(const dtype& rhs) const {
    //     using std::real;
    //     if (std::is_same<dtype, ComplexScalar>::value) {
    //         return BaseClass::eval_op_##opname(rhs);
    //     }
    //     return BoolVarPtr { new ScalarWrapper<dtype> { real(data_) > real(rhs) } };
    // }

    // template <typename dtype>
    // typename ScalarWrapper<dtype>::BoolVarPtr
    // ScalarWrapper<dtype>::eval_op_gt(const Variable<dtype>& rhs) const {
    //     // data_ > rhs  <=>  rhs < data_
    //     return rhs.eval_op_lt(data_);
    // }

#define OPTSUITE_SCALAR_DEFINE_COMP_OP_BY_OPPOSITE(interface_name, opposite_interface_name) \
    template <typename dtype> \
    typename ScalarWrapper<dtype>::BoolVarPtr \
    ScalarWrapper<dtype>::eval_op_##interface_name(const Variable<dtype>& rhs) const { \
        return rhs.eval_op_##opposite_interface_name(data_); \
    } \

#define OPTSUITE_SCALAR_DEFINE_COMP_OP_REAL_ONLY(interface_name, reducer, opposite_interface_name) \
    template <typename dtype> \
    typename ScalarWrapper<dtype>::BoolVarPtr \
    ScalarWrapper<dtype>::eval_op_##interface_name(const dtype& rhs) const { \
        using std::real; \
        if (std::is_same<dtype, ComplexScalar>::value) { \
            return BaseClass::eval_op_##interface_name(rhs); \
        } \
        return BoolVarPtr { new ScalarWrapper<bool> { real(data_) reducer real(rhs) } }; \
    } \
    OPTSUITE_SCALAR_DEFINE_COMP_OP_BY_OPPOSITE(interface_name, opposite_interface_name)


#define OPTSUITE_SCALAR_DEFINE_COMP_OP_WITH_COMPLEX(interface_name, reducer, opposite_interface_name) \
    template <typename dtype> \
    typename ScalarWrapper<dtype>::BoolVarPtr \
    ScalarWrapper<dtype>::eval_op_##interface_name(const dtype& rhs) const { \
        return BoolVarPtr { new ScalarWrapper<bool> { data_ reducer rhs } }; \
    } \
    OPTSUITE_SCALAR_DEFINE_COMP_OP_BY_OPPOSITE(interface_name, opposite_interface_name)

    OPTSUITE_SCALAR_DEFINE_COMP_OP_REAL_ONLY(gt, >, lt)
    OPTSUITE_SCALAR_DEFINE_COMP_OP_REAL_ONLY(ge, >=, le)
    OPTSUITE_SCALAR_DEFINE_COMP_OP_REAL_ONLY(lt, <, gt)
    OPTSUITE_SCALAR_DEFINE_COMP_OP_REAL_ONLY(le, <=, ge)
    OPTSUITE_SCALAR_DEFINE_COMP_OP_WITH_COMPLEX(eq, ==, eq)
    OPTSUITE_SCALAR_DEFINE_COMP_OP_WITH_COMPLEX(ne, !=, ne)

#undef OPTSUITE_SCALAR_DEFINE_COMP_OP_REAL_ONLY
#undef OPTSUITE_SCALAR_DEFINE_COMP_OP_WITH_COMPLEX
#undef OPTSUITE_SCALAR_DEFINE_COMP_OP_BY_OPPOSITE

    template class ScalarWrapper<Scalar>;
    template class ScalarWrapper<ComplexScalar>;
}}
