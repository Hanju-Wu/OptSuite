#ifndef OPTSUITE_CELL_ARRAY_H
#define OPTSUITE_CELL_ARRAY_H

#include "core_n.h"
#include <vector>
#include <initializer_list>
#include <utility>

namespace OptSuite {

    // general squared norm
    namespace {
        template <typename T>
        typename std::enable_if<is_eigen_type<T>, Scalar>::type
        squared_norm(const T& elem) {
            return elem.squaredNorm();
        }

        template <typename T>
        typename std::enable_if<is_optsuite_scalar_type<T>, Scalar>::type
        squared_norm(const T& elem) {
            return std::norm(elem);
        }

    }

    template <typename T>
    class CellArray {
        std::vector<T> m_elems;

    public:
        const T& operator[](Index i) const { return m_elems[i]; }
        T& operator[](Index i) { return m_elems[i]; }
        Size size() const { return m_elems.size(); }

        CellArray() = default;
        explicit CellArray(const Size n) : m_elems(n) {}
        explicit CellArray(const Size n, const T& val) : m_elems(n, val) {}
        CellArray(std::initializer_list<T> init) : m_elems(init) {}
        explicit CellArray(const std::vector<T>& vec) : m_elems(vec) {}
        explicit CellArray(std::vector<T>&& vec) : m_elems(std::move(vec)) {}
        CellArray(const CellArray<T>&) = default;
        CellArray(CellArray<T>&&) = default;
        CellArray& operator=(const CellArray<T>&) = default;
        CellArray& operator=(CellArray<T>&&) = default;
        ~CellArray() = default;

        CellArray& operator=(const std::vector<T>& vec) {
            m_elems = vec;
            return *this;
        }
        CellArray& operator=(std::vector<T>&& vec) {
            m_elems = std::move(vec);
            return *this;
        }
        const std::vector<T>& vector() const { return m_elems; }
        std::vector<T>& vector() { return m_elems; }
        explicit operator std::vector<T>() const { return m_elems; }

        void resize(const Size n) { m_elems.resize(n); }
        void reserve(const Size n) { m_elems.reserve(n); }
        auto begin() const { return m_elems.begin(); }
        auto begin() { return m_elems.begin(); }
        auto end() const { return m_elems.end(); }
        auto end() { return m_elems.end(); }

        // Squared norm of numerical array
        Scalar squaredNorm() const {
            Scalar sum_of_square = 0;
            for (Size i = 0; i < this->size(); ++i) {
                sum_of_square += OptSuite::squared_norm(m_elems[i]);
            }
            return sum_of_square;
        }

        // Norm of numerical cell array
        Scalar norm() const {
            return sqrt(this->squared_norm());
        }

        // *= Scalar
        template <typename _Scalar,
            std::enable_if_t<is_scalar_type<_Scalar>, int> = 0>
        CellArray<T>& operator*=(const _Scalar& other) {
            for (auto& elem : m_elems) {
                elem *= other;
            }
            return *this;
        }

        // /= Scalar
        template <typename _Scalar,
            std::enable_if_t<is_scalar_type<_Scalar>, int> = 0>
        CellArray<T>& operator/=(const _Scalar& other) {
            for (auto& elem : m_elems) {
                elem /= other;
            }
            return *this;
        }

        template <typename Q>
        CellArray<Q> cast() const {
            CellArray<Q> result(this->size());
            for (Index i = 0; i < this->size(); ++i) {
                result[i] = static_cast<Q>(m_elems[i]);
            }
            return result;
        }
    };

    template <typename T>
    OPTSUITE_STRONG_INLINE
    std::enable_if_t<is_value_type<T>, Scalar>
    squared_norm(const CellArray<T>& ary) {
        return ary.squared_norm();
    }
}

#endif
