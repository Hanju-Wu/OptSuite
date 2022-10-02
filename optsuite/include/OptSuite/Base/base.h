/*
 * ==========================================================================
 *
 *       Filename:  base.h
 *
 *    Description:
 *
 *        Version:  1.0
 *        Created:  04/02/2021 01:13:55 PM
 *       Revision:  none
 *       Compiler:  g++
 *
 *         Author:  Haoyang Liu (@liuhy), liuhaoyang@pku.edu.cn
 *   Organization:  BICMR, PKU
 *
 * ==========================================================================
 */

#ifndef OPTSUITE_BASE_BASE_H
#define OPTSUITE_BASE_BASE_H

#include <string>

namespace OptSuite { namespace Base {
    class OptSuiteBase {
        public:
            virtual ~OptSuiteBase() = default;
            virtual std::string to_string() const;

            // maybe used to determine an object's class at runtime
            virtual inline std::string classname() const { return "OptSuiteBase"; }
        protected:
            OptSuiteBase() = default;
    };

    std::ostream& operator<<(std::ostream& out, const OptSuiteBase& var);
}}

#endif
