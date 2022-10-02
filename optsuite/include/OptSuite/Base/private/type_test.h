/*
 * ==========================================================================
 *
 *       Filename:  type_test.h
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

#ifndef OPTSUITE_BASE_PRIVATE_TYPE_TEST_H
#define OPTSUITE_BASE_PRIVATE_TYPE_TEST_H

namespace OptSuite { namespace Base { namespace Internal {
    template <typename Derived>
    constexpr bool is_eigen_type_f(const EigenBase<Derived>*) {
        return true;
    }
    constexpr bool is_eigen_type_f(...) {
        return false;
    }

    // is_eigen_type<T>: true if T is derived from EigenBase<T>, false otherwise
    template <typename T>
    struct is_eigen_type {
        static constexpr bool value =
            is_eigen_type_f(static_cast<typename std::add_pointer<T>::type>(nullptr));
    };

    // is_optsuite_scalar_type<T>: true if T == Scalar or ComplexScalar
    template <typename T>
    struct is_optsuite_scalar_type {
        static constexpr bool value =
            std::is_same<T, Scalar>::value || std::is_same<T, ComplexScalar>::value;
    };

    // is_scalar_type<T>: true if T is integer or floating-point number
    template <typename T>
    struct is_scalar_type {
        static constexpr bool value = std::is_arithmetic<T>::value;
    };

    // is_value_type<T>: true if T is either eigen type or scalar type
    template <typename T>
    struct is_value_type {
        static constexpr bool value = is_eigen_type<T>::value || is_scalar_type<T>::value;
    };
}}}
#endif
