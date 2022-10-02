/*
 * ===========================================================================
 *
 *       Filename:  rbio.cpp
 *
 *    Description:  
 *
 *        Version:  1.0
 *        Created:  02/20/2021 05:47:42 PM
 *       Revision:  none
 *       Compiler:  g++
 *
 *         Author:  Haoyang Liu (), liuhaoyang@pku.edu.cn
 *   Organization:  BICMR, Peking University
 *      Copyright:  Copyright (c) 2021, Haoyang Liu
 *
 * ===========================================================================
 */

#include <iostream>
#include <fstream>
#include "OptSuite/Utils/rbio.h"

namespace OptSuite { namespace Utils {
    int RBio::read(const std::string&, Ref<SpMat>){
        return 0;
    }
    int RBio::write(const Ref<const SpMat>){
        return 0;
    }
    int RBio::write(const std::string&, const Ref<const SpMat>){
        return 0;
    }

}}
