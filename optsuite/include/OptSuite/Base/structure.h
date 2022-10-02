/*
 * ==========================================================================
 *
 *       Filename:  structure.h
 *
 *    Description:
 *
 *        Version:  1.0
 *        Created:  04/02/2021 03:25:36 PM
 *       Revision:  none
 *       Compiler:  g++
 *
 *         Author:  Haoyang Liu (@liuhy), liuhaoyang@pku.edu.cn
 *   Organization:  BICMR, PKU
 *
 * ==========================================================================
 */

#ifndef OPTSUITE_BASE_STRUCTURE_H
#define OPTSUITE_BASE_STRUCTURE_H

#include <unordered_map>
#include "OptSuite/core_n.h"
#include "OptSuite/Base/base.h"
#include "OptSuite/Base/variable_ref.h"
#include "OptSuite/Base/private/object_wrapper_finder.h"

namespace OptSuite { namespace Base {
    class Structure : public OptSuiteBase {
        using MapType = std::unordered_map<std::string, std::shared_ptr<OptSuiteBase>>;
        MapType data_;

        public:
            using iterator = MapType::iterator;
            using const_iterator = MapType::const_iterator;

            Structure() = default;
            ~Structure() = default;
            Structure(const Structure&) = default;
            Structure(Structure&&) = default;
            Structure& operator=(const Structure&) = default;
            Structure& operator=(Structure&&) = default;

            std::string to_string() const override {
                std::string ans;
                for (const auto& kvpair : data_) {
                    ans += kvpair.first;
                    ans += ": ";
                    ans += kvpair.second->to_string();
                    ans += "\n";
                }
                return ans;
            }

            inline std::string classname() const override { return "Structure"; }
            inline iterator begin() noexcept { return data_.begin(); }
            inline iterator end() noexcept { return data_.end(); }
            inline const_iterator begin() const noexcept { return data_.begin(); }
            inline const_iterator end() const noexcept { return data_.end(); }
            inline const_iterator cbegin() const noexcept { return data_.cbegin(); }
            inline const_iterator cend() const noexcept { return data_.cend(); }
            inline Index size() const noexcept { return static_cast<Index>(data_.size()); }
            inline bool empty() { return data_.empty(); }

            // check the existence of a field
            //   when T != void, also check type
            template <typename T = void>
            inline bool is_field(const std::string& tag) const noexcept {
                return get_impl<T, false>(tag) != nullptr;
            }

            inline void clear() { data_.clear(); }

            // erase a field
            inline void erase(const std::string& tag) { data_.erase(tag); }

            // find a tag.
            // Returns a pointer that points to the result if it exists and type matches.
            // Otherwise returns nullptr.
            template <typename T>
            inline T* find(const std::string& tag) const noexcept {
                return get_impl<T, false>(tag);
            }

            template <typename T>
            inline T& get(const std::string& tag) {
                return *get_impl<T>(tag);
            }

            template <typename T>
            inline const T& get(const std::string& tag) const {
                return *get_impl<T>(tag);
            }

            template <typename T>
            inline void set(const std::string& tag, T&& v) {
                emplace<T>(tag, std::forward<T>(v));
            }

            // note: unlike STL, the key is overriden when already exists
            template <typename T, typename... Args>
            inline T& emplace(const std::string& tag, Args&&... args) {
                auto ptr = std::make_shared<typename Internal::find_wrapper<T>::type>(std::forward<Args>(args)...);
                data_[tag] = ptr;
                return ptr->data();
            }

            template <typename dtype = void>
            inline VariableRef<dtype> get_var(const std::string& tag) const {
                return VariableRef<dtype>{ get_var_impl<dtype>(tag) };
            }

            template <typename dtype = void>
            inline bool is_var(const std::string& tag) const noexcept {
                return get_var_impl<dtype, false>(tag) != nullptr;
            }

            // returns VariableRef if it exists , otherwise returns empty VariableRef
            template <typename dtype = void>
            inline VariableRef<dtype> find_var(const std::string& tag) const noexcept {
                return VariableRef<dtype>{ get_var_impl<dtype, false>(tag) };
            }


            template <typename dtype = void>
            inline void set_var(const std::string& tag, const VariableRef<dtype>& var) {
                data_[tag] = var.get_shared();
            }

            // note: unlike STL, the key is overriden when already exists
            template <typename dtype, typename... Args>
            inline void emplace_var(const std::string& tag, Args&&... args) {
                data_[tag] = VariableRef<dtype>(std::forward<Args>(args)...).get_shared();
            }
        private:
            template <typename T, bool do_assert = true>
            inline T* get_impl(const std::string& tag) const {
                auto it = data_.find(tag);
                if (it == data_.end()) {
                    OPTSUITE_ASSERT_MSG(!do_assert, "tag not found");
                    return nullptr;
                }
                auto p = dynamic_cast<typename Internal::find_wrapper<T>::type*>(it->second.get());
                if (!p) {
                    OPTSUITE_ASSERT_MSG(!do_assert, "type mismatch");
                    return nullptr;
                }
                return &p->data();
            }

            template <typename dtype, bool do_assert = true>
            inline std::shared_ptr<Variable<dtype>> get_var_impl(const std::string& tag) const {
                auto it = data_.find(tag);
                if (it == data_.end()) {
                    OPTSUITE_ASSERT_MSG(!do_assert, "tag not found");
                    return nullptr;
                }
                auto p = std::dynamic_pointer_cast<Variable<dtype>>(it->second);
                if (!p) {
                    OPTSUITE_ASSERT_MSG(!do_assert, "type mismatch");
                    return nullptr;
                }
                return p;
            }

            std::string to_json_impl(Index indent_level) const;
    };

    template <>
    inline bool Structure::is_field<void>(const std::string& tag) const noexcept {
        return data_.find(tag) != data_.end();
    }

}}

#endif
