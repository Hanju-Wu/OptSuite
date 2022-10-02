/*
 * ==========================================================================
 *
 *       Filename:  fftw_wrapper.h
 *
 *    Description:  wrapper for fftw library
 *
 *        Version:  2.0
 *        Created:  11/26/2020 01:29:11 PM
 *       Revision:  27/02/2021 19:54
 *       Compiler:  g++
 *
 *         Author:  Haoyang Liu (@liuhy), liuhaoyang@pku.edu.cn
 *   Organization:  BICMR, PKU
 *
 * ==========================================================================
 */

#ifdef OPTSUITE_USE_FFTW

#ifndef OPTSUITE_LINALG_FFTW_WRAPPER_H
#define OPTSUITE_LINALG_FFTW_WRAPPER_H

#include "OptSuite/core_n.h"

namespace OptSuite { namespace LinAlg {
    enum class FFTW_R2R_Type {
        DCT_I = 0,
        DCT_II = 1,
        DCT_III = 2,
        DCT_IV = 3,
        DST_I = 4,
        DST_II = 5,
        DST_III = 6,
        DST_IV = 7
    };

    enum class DCT_Type {
        TYPE_I = 0,
        TYPE_II = 1,
        TYPE_III = 2,
        TYPE_IV = 3
    };

    enum class DST_Type {
        TYPE_I = 4,
        TYPE_II = 5,
        TYPE_III = 6,
        TYPE_IV = 7
    };

    CMat fft(const Ref<const CMat>, Index = -1);
    CMat ifft(const Ref<const CMat>, Index = -1);

    Mat dct(const Ref<const Mat>, Index = -1, DCT_Type = DCT_Type::TYPE_II);
    Mat idct(const Ref<const Mat>, Index = -1);

}}

#endif

#endif // OPTSUITE_USE_FFTW
