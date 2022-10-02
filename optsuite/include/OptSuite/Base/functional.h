/*
 * ==========================================================================
 *
 *       Filename:  functional.h
 *
 *    Description:  
 *
 *        Version:  1.0
 *        Created:  11/04/2020 05:07:52 PM
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  Haoyang Liu (@liuhy), liuhaoyang@pku.edu.cn
 *   Organization:  BICMR, PKU
 *
 * ==========================================================================
 */
#ifndef OPTSUITE_BASE_FUNCTIONAL_H
#define OPTSUITE_BASE_FUNCTIONAL_H

// base class
#include "OptSuite/Base/func/base.h"

// Func
#include "OptSuite/Base/func/simple.h"
#include "OptSuite/Base/func/lp_norm.h"
#include "OptSuite/Base/func/nuclear_norm.h"

// Proximal
#include "OptSuite/Base/func/shrinkage_lp.h"
#include "OptSuite/Base/func/shrinkage_nuclear.h"
#include "OptSuite/Base/func/ball_lp.h"
#include "OptSuite/Base/func/probability_simplex.h"
#include "OptSuite/Base/func/maximal.h"

// FuncGrad
#include "OptSuite/Base/func/axmb_norm_sqr.h"
#include "OptSuite/Base/func/projection_omega.h"
#include "OptSuite/Base/func/LogisticRegression.h"

#endif
