/*
 * ==========================================================================
 *
 *       Filename:  spmat_wrapper.cpp
 *
 *    Description:  source for SpMatWrapper
 *
 *        Version:  1.0
 *        Created:  10/13/2021 02:26:19 PM
 *       Revision:  none
 *       Compiler:  g++
 *
 *         Author:  Haoyang Liu (@liuhy), liuhaoyang@pku.edu.cn
 *   Organization:  BICMR, PKU
 *
 * ==========================================================================
 */

#include "OptSuite/Base/mat_wrapper.h"
#include "OptSuite/Base/spmat_wrapper.h"

namespace OptSuite { namespace Base {

    // arithmetic ops related to MatWrapper
    template <typename dtype>
    auto SpMatWrapper<dtype>::eval_op_add(const Ref<const mat_t>& rhs) const -> VarPtr {
        return VarPtr { new MatWrapper<dtype> { data_ + rhs } };
    }

    template <typename dtype>
    auto SpMatWrapper<dtype>::eval_op_sub(const Ref<const mat_t>& rhs) const -> VarPtr {
        return VarPtr { new MatWrapper<dtype> { data_ - rhs } };
    }

    template <typename dtype>
    auto SpMatWrapper<dtype>::eval_op_mul(const Ref<const mat_t>& rhs) const -> VarPtr {
        return VarPtr { new MatWrapper<dtype> { data_ * rhs } };
    }

    template <typename dtype>
    auto SpMatWrapper<dtype>::eval_op_tmul(const Ref<const mat_t>& rhs) const -> VarPtr {
        return VarPtr { new MatWrapper<dtype> { data_.adjoint() * rhs } };
    }

    template <typename dtype>
    auto SpMatWrapper<dtype>::eval_op_mult(const Ref<const mat_t>& rhs) const -> VarPtr {
        return VarPtr { new MatWrapper<dtype> { data_ * rhs.adjoint() } };
    }

    template <typename dtype>
    auto SpMatWrapper<dtype>::eval_op_tmult(const Ref<const mat_t>& rhs) const -> VarPtr {
        return VarPtr { new MatWrapper<dtype> { data_.adjoint() * rhs.adjoint() } };
    }

    // instantiation
    template class SpMatWrapper<Scalar>;
    template class SpMatWrapper<ComplexScalar>;
}}
