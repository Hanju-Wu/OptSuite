#ifndef SSNSDP_LOG_H
#define SSNSDP_LOG_H

#include <memory>
#include <iostream>
#include <string>
#include <cstdio>
#include <type_traits>

#include "core.h"

namespace ssnsdp {
    template <Verbosity loglevel = Verbosity::Info>
    SSNSDP_STRONG_INLINE
    void f_log(std::ostream& out) {}

    // log to file with verbosity control
    // f_log<loglevel = Verbosity::Info>(out, obj1, obj2, obj3, ...)
    //   is basically the same as out << obj1 << obj2 << obj3 << ...
    template <Verbosity loglevel = Verbosity::Info, typename T, typename... Rest>
    SSNSDP_STRONG_INLINE
    void f_log(std::ostream& out, const T& obj, const Rest&... rest) {
        if constexpr (CurrentVerbosity >= loglevel) {
            out << obj;
            f_log<loglevel>(out, rest...);
        }
    }

    template <Verbosity loglevel = Verbosity::Info>
    SSNSDP_STRONG_INLINE
    void f_log_l(std::ostream& out) {}

    // log to file with verbosity control, inserting line break
    // f_log_l<loglevel = Verbosity::Info>(out, obj1, obj2, ...)
    //   basically the same as out << obj1 << '\n' << obj2 << '\n' << ...
    template <Verbosity loglevel = Verbosity::Info, typename T, typename... Rest>
    SSNSDP_STRONG_INLINE
    void f_log_l(std::ostream& out, const T& obj, const Rest&... rest) {
        f_log<loglevel>(out, obj, '\n');
        f_log_l<loglevel>(out, rest...);
    }

    // output debug information
    // See f_log with more information.
    template <Verbosity loglevel = Verbosity::Debug, typename... Args>
    SSNSDP_STRONG_INLINE
    void log_debug(const Args&... args) {
        f_log<loglevel>(std::cout, args...);
    }

    // output debug information with a line break
    // see f_log_l with more information.
    template <Verbosity loglevel = Verbosity::Debug, typename... Args>
    SSNSDP_STRONG_INLINE
    void log_debug_l(const Args&... args) {
        f_log_l<loglevel>(std::cout, args...);
    }

    // similar to std::puts, but with a verbosity control
    template <Verbosity loglevel = Verbosity::Debug>
    SSNSDP_STRONG_INLINE
    void puts_debug(const char* str) {
        if constexpr (CurrentVerbosity < loglevel) {
            return;
        }
        if constexpr (UseCStyleIO) {
            std::puts(str);
        } else {
            f_log_l<loglevel>(std::cout, str);
        }
    }

    // similar to std::putchar, but with a verbosity control
    template <Verbosity loglevel = Verbosity::Debug>
    SSNSDP_STRONG_INLINE
    void putchar_debug(const int ch) {
        if constexpr (CurrentVerbosity < loglevel) {
            return;
        }
        if constexpr (UseCStyleIO) {
            std::putchar(ch);
        } else {
            f_log_l<loglevel>(std::cout, static_cast<const unsigned char>(ch));
        }
    }

    // output generic information
    // See f_log with more information.
    template <typename... Args>
    SSNSDP_STRONG_INLINE
    void log_info(const Args&... args) {
        f_log<Verbosity::Info, Args...>(std::cout, args...);
    }

    // output generic information with a line break
    // see f_log_l with more information.
    template <typename... Args>
    SSNSDP_STRONG_INLINE
    void log_info_l(const Args&... args) {
        f_log_l<Verbosity::Info, Args...>(std::cout, args...);
    }

    // similar to std::puts, but with a verbosity control
    SSNSDP_STRONG_INLINE
    void puts_info(const char* str) {
        puts_debug<Verbosity::Info>(str);
    }

    // similar to std::putchar, but with a verbosity control
    SSNSDP_STRONG_INLINE
    void putchar_info(const int ch) {
        putchar_debug<Verbosity::Info>(ch);
    }

    // output verbose information
    // log_verbose("message\n") may serve as a comment "message"
    // See f_log with more information.
    template <typename... Args>
    SSNSDP_STRONG_INLINE
    void log_verbose(const Args&... args) {
        f_log<Verbosity::Verbose, Args...>(std::cout, args...);
    }

    // output verbose with a line break
    // see f_log_l with more information.
    template <typename... Args>
    SSNSDP_STRONG_INLINE
    void log_verbose_l(const Args&... args) {
        f_log_l<Verbosity::Verbose, Args...>(std::cout, args...);
    }

    // similar to std::puts, but with a verbosity control
    SSNSDP_STRONG_INLINE
    void puts_verbose(const char* str) {
        puts_debug<Verbosity::Verbose>(str);
    }

    // similar to std::putchar, but with a verbosity control
    SSNSDP_STRONG_INLINE
    void putchar_verbose(const int ch) {
        putchar_debug<Verbosity::Verbose>(ch);
    }

    // sprintf with buffer length auto detected
    template <typename... Args>
    std::unique_ptr<char[]> cstr_format(const char* format, Args&&... args) {
        using std::snprintf;
        auto size = snprintf(nullptr, 0, format, args...) + 1; // Extra space for '\0'
        auto buf = std::make_unique<char[]>(size);
        snprintf(buf.get(), size, format, args...);
        return buf;
    }

    // sprintf to std::string
    // Original source: https://stackoverflow.com/a/26221725/11722
    template <typename... Args>
    std::string str_format(const char* format, Args&&... args) {
        using std::snprintf;
        auto size = snprintf(nullptr, 0, format, args...) + 1; // Extra space for '\0'
        auto buf = std::make_unique<char[]>(size);
        snprintf(buf.get(), size, format, args...);
        return std::string(buf.get(), buf.get() + size - 1); // We don't want the '\0' inside
    }

    // fprintf with global verbosity level control
    template <Verbosity loglevel = Verbosity::Info, typename... Args>
    SSNSDP_STRONG_INLINE
    void log_format(std::ostream& out, const char* format, Args&&... args) {
        f_log<loglevel>(out, cstr_format(format, std::forward<Args>(args)...).get());
    }

    // printf with global verbosity level control
    template <Verbosity loglevel = Verbosity::Info, typename... Args>
    SSNSDP_STRONG_INLINE
    void log_format(const char* format, Args&&... args) {
        if constexpr (CurrentVerbosity < loglevel) {
            return;
        }
        if constexpr (UseCStyleIO) {
            std::printf(format, std::forward<Args>(args)...);
        } else {
            log_format<loglevel>(std::cout, format, std::forward<Args>(args)...);
        }
    }

    // log error message to std::cerr
    template <Verbosity loglevel = Verbosity::Info>
    SSNSDP_STRONG_INLINE
    void log_error(const char* str) {
        if constexpr (CurrentVerbosity < loglevel) {
            return;
        }
        if constexpr (UseCStyleIO) {
            std::fputs(str, stderr);
        } else {
            f_log<loglevel>(std::cerr, str);
        }
    }

    // log error message to std::cerr, appending new line
    template <Verbosity loglevel = Verbosity::Info>
    SSNSDP_STRONG_INLINE
    void log_error_l(const char* str) {
        if constexpr (CurrentVerbosity < loglevel) {
            return;
        }
        if constexpr (UseCStyleIO) {
            std::fputs(str, stderr);
            std::fputc('\n', stderr);
        } else {
            f_log_l<loglevel>(std::cerr, str);
        }
    }

    // log formatted error message to std::cerr
    template <Verbosity loglevel = Verbosity::Info, typename... Args>
    SSNSDP_STRONG_INLINE
    void log_error_format(const char* format, Args&&... args) {
        if constexpr (CurrentVerbosity < loglevel) {
            return;
        }
        if constexpr (UseCStyleIO) {
            std::fprintf(stderr, format, std::forward<Args>(args)...);
        } else {
            log_format<loglevel>(std::cerr, format, std::forward<Args>(args)...);
        }
    }

    // disable sync with stdio, only if not using C-style IO
    SSNSDP_STRONG_INLINE
    void try_disable_sync_with_stdio() {
        if constexpr (!UseCStyleIO) {
            std::ios_base::sync_with_stdio(false);
        }
    }
}

#endif
