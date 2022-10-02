/*
 * ==========================================================================
 *
 *       Filename:  to_string.h
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

#ifndef OPTSUITE_BASE_PRIVATE_TO_STRING_H
#define OPTSUITE_BASE_PRIVATE_TO_STRING_H

#include <type_traits>
#include <utility>
#include <string>
#include <sstream>

namespace OptSuite { namespace Base { namespace Internal {

    // https://stackoverflow.com/questions/22758291/how-can-i-detect-if-a-type-can-be-streamed-to-an-stdostream
    template<typename S, typename T>
    class is_streamable
    {
        template<typename SS, typename TT>
        static auto test(int)
        -> decltype( std::declval<SS&>() << std::declval<TT>(), std::true_type() );

        template<typename, typename>
        static auto test(...) -> std::false_type;

    public:
        static const bool value = decltype(test<S,T>(0))::value;
    };

    template <typename T, typename Enabled = void>
    struct StringConverter {
        constexpr static bool is_implemented = false;
        static inline std::string to_string(const T&) { return ""; }
    };

    template <typename T>
    struct StringConverter<T, typename std::enable_if<is_streamable<std::stringstream, T>::value>::type> {
        constexpr static bool is_implemented = true;
        static inline std::string to_string(const T& t) {
            std::stringstream ss;
            ss << t;
            return ss.str();
        }
    };

    template <>
    struct StringConverter<bool> {
        constexpr static bool is_implemented = true;
        static inline std::string to_string(const bool& b) { return b ? "true" : "false"; }
    };

    // template <>
    // struct StringConverter<std::string> {
    //     constexpr static bool is_implemented = true;
    //     static inline std::string to_string(const std::string& s) {
    //         return std::string{s};
    //     }
    // };
#define OPTSUITE_SPECIALIZE_CONVERTIBLE_TO_STRING(T) \
    template <> \
    struct StringConverter<T> { \
        constexpr static bool is_implemented = true; \
        static inline std::string to_string(T const& s) { \
            return std::string{s}; \
        } \
    };

    OPTSUITE_SPECIALIZE_CONVERTIBLE_TO_STRING(std::string)
    OPTSUITE_SPECIALIZE_CONVERTIBLE_TO_STRING(char*)
    OPTSUITE_SPECIALIZE_CONVERTIBLE_TO_STRING(const char*)
#undef OPTSUITE_SPECIALIZE_CONVERTIBLE_TO_STRING
}}}

#endif
