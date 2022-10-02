/*
 * ==========================================================================
 *
 *       Filename:  object_wrapper_finder.h
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

#ifndef OPTSUITE_OBJECT_WRAPPER_FINDER_H
#define OPTSUITE_OBJECT_WRAPPER_FINDER_H

#include <type_traits>
#include "OptSuite/core_n.h"
#include "OptSuite/Base/object_wrapper.h"
#include "OptSuite/Base/fwddecl.h"

namespace OptSuite { namespace Base { namespace Internal {
    template <typename T>
    struct find_wrapper_impl {
        using type = ObjectWrapper<T>;
    };

    template <>
    struct find_wrapper_impl<Scalar> {
        using type = ScalarWrapper<Scalar>;
    };

    template <>
    struct find_wrapper_impl<ComplexScalar> {
        using type = ScalarWrapper<ComplexScalar>;
    };

    template <>
    struct find_wrapper_impl<bool> {
        using type = ScalarWrapper<bool>;
    };

    template <typename dtype>
    struct find_wrapper_impl<Eigen::Matrix<dtype, Dynamic, Dynamic>> {
        using type = MatWrapper<dtype>;
    };

    static_assert(std::is_same<find_wrapper_impl<Mat>::type, MatWrapper<Scalar>>::value,
        "template specialization not as expected");

    template <typename dtype>
    struct find_wrapper_impl<Eigen::SparseMatrix<dtype, ColMajor, SparseIndex>> {
        using type = SpMatWrapper<dtype>;
    };

    template <typename T>
    struct find_wrapper {
        using WrappedType = typename std::remove_const<typename std::remove_reference<T>::type>::type;
        using type = typename find_wrapper_impl<WrappedType>::type;
    };
}}}

#endif
