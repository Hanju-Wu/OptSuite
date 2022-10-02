/*
 * ==========================================================================
 *
 *       Filename:  object_wrapper.h
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

#ifndef OPTSUITE_OBJECT_WRAPPER_H
#define OPTSUITE_OBJECT_WRAPPER_H

#include <type_traits>
#include <iomanip>
#include "OptSuite/core_n.h"
#include "OptSuite/Base/variable.h"
#include "OptSuite/Base/private/to_string.h"

namespace OptSuite { namespace Base {
    template <typename T, typename dtype = void>
    class ObjectWrapper : public Variable<dtype> {
        using ThisType = ObjectWrapper<T, dtype>;
        public:
            using typename Variable<dtype>::VarPtr;
            using WrappedType = T;
            ObjectWrapper() = default;
            ObjectWrapper(const ThisType&) = default;
            ObjectWrapper(ThisType&&) = default;
            template <typename... Args>
            explicit ObjectWrapper(Args&&... args) : data_(std::forward<Args>(args)...) {}
            ~ObjectWrapper() = default;

            ObjectWrapper& operator=(const ThisType&) = default;
            ObjectWrapper& operator=(ThisType&&) = default;

            T& data() { return data_; }
            const T& data() const { return data_; }

            inline VarPtr clone() const override {
                return clone_impl<T>(data_);
            }

            inline std::string classname() const override {
                return "ObjectWrapper";
            }

            inline std::string to_string() const override {
                using conv_t = Internal::StringConverter<T>;
                if (conv_t::is_implemented) {
                    return conv_t::to_string(data_);
                } else {
                    std::stringstream ss;
                    // ObjectWrapper<TypeName>#0x1234ab
                    ss << "ObjectWrapper<" << typeid(T).name() << ">#"
                      << std::hex << std::addressof(data_);
                    return ss.str();
                }
            }

            inline void assign(const Variable<dtype>& other) override {
                const ThisType* other_ptr = dynamic_cast<const ThisType*>(&other);
                OPTSUITE_ASSERT(other_ptr);
                assign_impl<T>(other_ptr->data_);
            }

        protected:
            T data_;

        private:
            template <typename U>
            inline
            typename std::enable_if<!std::is_copy_constructible<U>::value, VarPtr>::type
            clone_impl(const U&) const {
                OPTSUITE_ASSERT_MSG(false, "type is not copy constructible");
                return VarPtr {};
            }

            template <typename U>
            inline
            typename std::enable_if<std::is_copy_constructible<U>::value, VarPtr>::type
            clone_impl(const U& data_u) const {
                return VarPtr { new ThisType(data_u) };
            }

            template <typename U>
            inline
            typename std::enable_if<!std::is_copy_constructible<U>::value, void>::type
            assign_impl(const U&) {
                OPTSUITE_ASSERT_MSG(false, "type is not copy constructible");
            }

            template <typename U>
            inline
            typename std::enable_if<std::is_copy_constructible<U>::value, void>::type
            assign_impl(const U& data_u) {
                data_ = data_u;
            }
    };
}}

#endif
