/*
 * ==========================================================================
 *
 *       Filename:  variable.cpp
 *
 *    Description:
 *
 *        Version:  1.0
 *        Created:  10/13/2021 02:00:44 PM
 *       Revision:  none
 *       Compiler:  g++
 *
 *         Author:  Haoyang Liu (@liuhy), liuhaoyang@pku.edu.cn
 *   Organization:  BICMR, PKU
 *
 * ==========================================================================
 */

#include "OptSuite/Base/variable.h"

namespace OptSuite { namespace Base {
    template <typename dtype>
    const dtype& Variable<dtype>::slice(const Index& i) const {
        static dtype dummy_dtype_val = dtype();
        Index ptrsize = 0;
        const dtype* dptr = value_span(ptrsize);
        if (dptr) {
            OPTSUITE_ASSERT(i >= 0 && i < ptrsize);
            return dptr[i];
        } else {
            not_implemented();
            return dummy_dtype_val;
        }
    }

    template <typename dtype>
    dtype& Variable<dtype>::slice(const Index& i) {
        const auto& cthis = *this;
        return const_cast<dtype&>(cthis.slice(i));
    }

    template <typename dtype>
    const dtype& Variable<dtype>::slice(const Index&, const Index&) const {
        static dtype dummy_dtype_val = dtype();
        not_implemented();
        return dummy_dtype_val;
    }

    template <typename dtype>
    dtype& Variable<dtype>::slice(const Index& i, const Index& j) {
        const auto& cthis = *this;
        return const_cast<dtype&>(cthis.slice(i, j));
    }

    template <typename dtype>
    typename Variable<dtype>::vec_t Variable<dtype>::slice(const std::vector<Index>& indices) const {
        Index ptrsize = 0;
        const dtype* dptr = value_span(ptrsize);
        OPTSUITE_ASSERT(ptrsize == 0 || dptr);
#if EIGEN_WORLD_VERSION >= 3 && EIGEN_MAJOR_VERSION >= 4
        // Eigen 3.4 API, untested yet
        Map<const vec_t> vec(dptr, ptrsize);
        return vec(indices);
#else
        Index ans_size = static_cast<Index>(indices.size());
        vec_t ans(ans_size);
        for (Index i = 0; i < ans_size; ++i) {
            OPTSUITE_ASSERT(indices[i] >= 0 && indices[i] < ptrsize);
            ans(i) = dptr[indices[i]];
        }
        return ans;
#endif
    }


    template <typename dtype>
    void Variable<dtype>::slice_set(const std::vector<Index>& indices, const Variable<dtype>& rhs)
    {
        Index ptrsize = 0;
        dtype* dptr = value_span(ptrsize);
        OPTSUITE_ASSERT(ptrsize == 0 || dptr);

        Index rptrsize = 0;
        const dtype* rdptr = rhs.value_span(rptrsize);
        OPTSUITE_ASSERT(rptrsize == 0 || rdptr);

        Index isize = static_cast<Index>(indices.size());
        OPTSUITE_ASSERT(rptrsize == isize);

        for (Index i = 0; i < isize; ++i) {
            OPTSUITE_ASSERT(indices[i] >= 0 && indices[i] < ptrsize);
            dptr[indices[i]] = rdptr[i];
        }
    }


    template <typename dtype>
    void Variable<dtype>::slice_set(const std::vector<Index>& indices, const dtype& val)
    {
        Index ptrsize = 0;
        dtype* dptr = value_span(ptrsize);
        OPTSUITE_ASSERT(ptrsize == 0 || dptr);

        for (Index index : indices) {
            OPTSUITE_ASSERT(index >= 0 && index < ptrsize);
            dptr[index] = val;
        }
    }

    template <typename dtype>
    void Variable<dtype>::slice_set(const Variable<bool>& boolvar, const Variable<dtype>& rhs)
    {
        Index ptrsize = 0;
        dtype* dptr = value_span(ptrsize);
        OPTSUITE_ASSERT(ptrsize == 0 || dptr);

        auto bool_vec = boolvar.vector_view();
        OPTSUITE_ASSERT(bool_vec.size() == ptrsize);

        Index rptrsize = 0;
        const dtype* rdptr = rhs.value_span(rptrsize);
        OPTSUITE_ASSERT(rptrsize == 0 || rdptr);

        Index j = 0;
        for (Index i = 0; i < ptrsize; ++i) {
            if (bool_vec(i)) {
                OPTSUITE_ASSERT(j < rptrsize);
                dptr[i] = rdptr[j];
                ++j;
            }
        }
    }

    template <typename dtype>
    void Variable<dtype>::slice_set(const Variable<bool>& boolvar, const dtype& val)
    {
        Index ptrsize = 0;
        dtype* dptr = value_span(ptrsize);

        OPTSUITE_ASSERT(ptrsize == 0 || dptr);

        auto bool_vec = boolvar.vector_view();
        OPTSUITE_ASSERT(bool_vec.size() == ptrsize);

        for (Index i = 0; i < ptrsize; ++i) {
            if (bool_vec(i)) {
                dptr[i] = val;
            }
        }
    }

    // instantiation
    template class Variable<Scalar>;
    template class Variable<ComplexScalar>;

}}
