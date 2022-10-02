/*
 * ==========================================================================
 *
 *       Filename:  json_formatter.h
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
#ifndef OPTSUITE_UTILS_JSON_FORMATTER_H
#define OPTSUITE_UTILS_JSON_FORMATTER_H

#include "OptSuite/core_n.h"
#include "OptSuite/Base/fwddecl.h"

namespace OptSuite { namespace Utils {
    class JsonFormatter {
        public:
            using OptSuiteBase = OptSuite::Base::OptSuiteBase;
            using Structure = OptSuite::Base::Structure;
            using CellArray = OptSuite::Base::CellArray;

            JsonFormatter() = default;
            ~JsonFormatter() = default;

            // This also unnecessarily copys `buff_'. Change that when it becomes a problem.
            JsonFormatter(const JsonFormatter&) = default;
            JsonFormatter(JsonFormatter&&) = default;
            JsonFormatter& operator=(const JsonFormatter&) = default;
            JsonFormatter& operator=(JsonFormatter&&) = default;

            inline Index& indent_size() { return indent_size_; }
            inline const Index& indent_size() const { return indent_size_; }

            std::string format(const Scalar&) const;
            std::string format(const Index&) const;
            std::string format(const bool&) const;
            std::string format(const std::string&) const;

            // Note: each value in the Structure must be a type that can be formatted by JsonFormatter.
            //   That is, Scalar/Index/bool/string/Structure/CellArray
            //   int is NOT supported when int != Index.
            inline std::string format(const Structure& st) const {
                return format_impl(st, 0);
            }

            // Note: each value in the CellArray must be a type that can be formatted by JsonFormatter.
            //   That is, Scalar/Index/bool/string/Structure/CellArray
            //   int is NOT supported when int != Index.
            inline std::string format(const CellArray& ca) const {
                return format_impl(ca, 0);
            }
        private:
            std::string format_generic(const OptSuiteBase* obj, Index indent_level) const;
            std::string format_impl(const Structure& st, Index indent_level) const;
            std::string format_impl(const CellArray& ca, Index indent_level) const;

            mutable std::vector<char> buff_;
            Index indent_size_ = 4;
    };
}}


#endif
