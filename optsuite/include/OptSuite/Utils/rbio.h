/*
 * ===========================================================================
 *
 *       Filename:  rbio.h
 *
 *    Description:  
 *
 *        Version:  1.0
 *        Created:  02/20/2021 05:32:37 PM
 *       Revision:  none
 *       Compiler:  g++
 *
 *         Author:  Haoyang Liu (), liuhaoyang@pku.edu.cn
 *   Organization:  BICMR, Peking University
 *
 * ===========================================================================
 */

#ifndef OPTSUITE_UTILS_RBIO
#define OPTSUITE_UTILS_RBIO

#include <string>
#include "OptSuite/core_n.h"

namespace OptSuite { namespace Utils {
    class RBio {
    public:
        int read(const std::string&, Ref<SpMat>);
        int write(const Ref<const SpMat>);
        int write(const std::string&, const Ref<const SpMat>);
    };
}}

#endif
