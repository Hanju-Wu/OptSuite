/*
 * ==========================================================================
 *
 *       Filename:  mat_wrapper.cpp
 *
 *    Description:  source for mat_wrapper
 *
 *        Version:  1.0
 *        Created:  10/13/2021 02:20:04 PM
 *       Revision:  none
 *       Compiler:  g++
 *
 *         Author:  Haoyang Liu (@liuhy), liuhaoyang@pku.edu.cn
 *   Organization:  BICMR, PKU
 *
 * ==========================================================================
 */

#include "OptSuite/Base/mat_wrapper.h"

namespace OptSuite { namespace Base {
    std::unique_ptr<Variable<bool>>
    MatWrapper<bool>::map_to_scalar(const std::function<bool(bool)>& op) const {
        return std::unique_ptr<Variable<bool>> {
            new MatWrapper<bool> { data_.unaryExpr(op) }
        };
    }

    std::unique_ptr<Variable<Scalar>>
    MatWrapper<bool>::map_to_scalar(const std::function<Scalar(bool)>& op) const {
        return std::unique_ptr<Variable<Scalar>> {
            new MatWrapper<Scalar> { data_.unaryExpr(op) }
        };
    }

    std::unique_ptr<Variable<ComplexScalar>>
    MatWrapper<bool>::map_to_scalar(const std::function<ComplexScalar(bool)>& op) const {
        return std::unique_ptr<Variable<ComplexScalar>> {
            new MatWrapper<ComplexScalar> { data_.unaryExpr(op) }
        };
    }

    // instantiation
    template class MatWrapper<Scalar>;
    template class MatWrapper<ComplexScalar>;
}}
