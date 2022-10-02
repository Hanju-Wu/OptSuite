
/*
 * ==========================================================================
 *
 *       Filename:  fwddecl.h
 *
 *    Description:
 *
 *        Version:  1.0
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  Dong Xu (@taroxd), taroxd@pku.edu.cn
 *   Organization:  BICMR, PKU
 *    Description:  Common forward declarations
 *
 * ==========================================================================
 */

#ifndef OPTSUITE_BASE_FWDDECL_H
#define OPTSUITE_BASE_FWDDECL_H

namespace OptSuite { namespace Base {
    class OptSuiteBase;
    class Structure;
    class CellArray;
    template<typename dtype> class MatWrapper;
    template<typename dtype> class SpMatWrapper;
    template<typename dtype> class ScalarWrapper;
    template<typename dtype> class MatArray_t;
    template<typename dtype> class FactorizedMat;
    template<typename dtype> class Variable;
    template<typename dtype> class VariableRef;
    template<typename dtype> class MatOp;
}}
#endif
