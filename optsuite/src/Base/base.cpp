/*
 * ==========================================================================
 *
 *       Filename:  base.cpp
 *
 *    Description:
 *
 *        Version:  1.0
 *       Revision:  none
 *       Compiler:  g++
 *
 *         Author:  Dong Xu (@taroxd), taroxd@pku.edu.cn
 *   Organization:  BICMR, PKU
 *
 * ==========================================================================
 */

#include "OptSuite/Base/base.h"
#include "OptSuite/Utils/str_format.h"

namespace OptSuite { namespace Base {
    std::string OptSuiteBase::to_string() const {
        // OptSuiteBase#0x001234ab
        return OptSuite::Utils::str_format("OptSuiteBase#%p", this);
    }

    std::ostream& operator<<(std::ostream &out, const OptSuiteBase &var) {
        out << var.to_string();
        return out;
    }
}}
