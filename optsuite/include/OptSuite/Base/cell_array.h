/*
 * ==========================================================================
 *
 *       Filename:  cell_array.h
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

#ifndef OPTSUITE_BASE_CELL_ARRAY_H
#define OPTSUITE_BASE_CELL_ARRAY_H

#include <vector>
#include <iterator>
#include <initializer_list>
#include "OptSuite/core_n.h"
#include "OptSuite/Base/base.h"
#include "OptSuite/Base/variable_ref.h"
#include "OptSuite/Base/private/object_wrapper_finder.h"

namespace OptSuite { namespace Base {
    class CellArray: public OptSuiteBase {
        using VecType = std::vector<std::shared_ptr<OptSuiteBase>>;
        VecType data_;

        public:
            using iterator = VecType::iterator;
            using const_iterator = VecType::const_iterator;

            CellArray() = default;
            ~CellArray() = default;
            CellArray(const CellArray&) = default;
            CellArray(CellArray&&) = default;
            CellArray& operator=(const CellArray&) = default;
            CellArray& operator=(CellArray&&) = default;

            CellArray(Index n) : data_(n) {}

            template <typename It>
            CellArray(It begin, It end) {
                // reserve when possible
                using category = typename std::iterator_traits<It>::iterator_category;
                if (std::is_base_of<std::random_access_iterator_tag, category>::value) {
                    data_.reserve(std::distance(begin, end));
                }
                while (begin != end) {
                    push_back(*begin);
                    ++begin;
                }
            }

            template <typename T>
            CellArray(std::initializer_list<T> l) : CellArray(l.begin(), l.end()) {}

            inline std::string classname() const override { return "CellArray"; }
            inline iterator begin() noexcept { return data_.begin(); }
            inline iterator end() noexcept { return data_.end(); }
            inline const_iterator begin() const noexcept { return data_.begin(); }
            inline const_iterator end() const noexcept { return data_.end(); }
            inline const_iterator cbegin() const noexcept { return data_.cbegin(); }
            inline const_iterator cend() const noexcept { return data_.cend(); }
            inline Index size() const noexcept { return static_cast<Index>(data_.size()); }
            inline void resize(Index new_size) { data_.resize(new_size); }
            inline void reserve(Index n) { data_.reserve(n); }
            inline void clear() { data_.clear(); }
            inline void pop_back() { data_.pop_back(); }
            inline bool empty() { return data_.empty(); }

            std::string to_string() const override {
                std::string ans;
                for (const auto& value : data_) {
                    ans += value->to_string();
                    ans += '\n';
                }
                return ans;
            }


            // find a tag.
            // Returns a pointer that points to the result if tag is not out-of-bound and type matches.
            // Otherwise returns nullptr.
            template <typename T>
            inline T* find(Index tag) const noexcept {
                return get_impl<T, false>(tag);
            }

            // generic getter/setter
            // but use specific ones when they are provided
            template <typename T>
            inline T& get(Index tag) {
                return *get_impl<T>(tag);
            }

            template <typename T>
            inline const T& get(Index tag) const {
                return *get_impl<T>(tag);
            }

            template <typename T>
            inline void set(Index tag, T&& v) {
                emplace<T>(tag, std::forward<T>(v));
            }

            template <typename T>
            inline void push_back(T&& obj) {
                emplace_back<T>(std::forward<T>(obj));
            }

            template <typename T, typename... Args>
            inline T& emplace_back(Args&&... args) {
                auto ptr = std::make_shared<typename Internal::find_wrapper<T>::type>(std::forward<Args>(args)...);
                data_.push_back(ptr);
                return ptr->data();
            }

            // Construct the element in-place and do the same as set(tag, T(args...))
            // note: This is very different from the `emplace` in std::vector.
            template <typename T, typename... Args>
            inline T& emplace(Index tag, Args&&... args) {
                OPTSUITE_ASSERT_MSG(tag < 0 || tag > data_.size(), "tag out of bound");
                auto ptr = std::make_shared<typename Internal::find_wrapper<T>::type>(std::forward<Args>(args)...);
                data_[tag] = ptr;
                return ptr->data();
            }

        private:
            template <typename T, bool do_assert = true>
            inline T* get_impl(Index tag) const {
                if (tag < 0 || tag > data_.size()) {
                    OPTSUITE_ASSERT_MSG(!do_assert, "tag not found");
                    return nullptr;
                }
                auto p = dynamic_cast<typename Internal::find_wrapper<T>::type*>(data_[tag].get());
                if (!p) {
                    OPTSUITE_ASSERT_MSG(!do_assert, "type mismatch");
                    return nullptr;
                }
                return &p->data();
            }

    };
}}

#endif
