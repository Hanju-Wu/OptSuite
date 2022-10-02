/*
 * ==========================================================================
 *
 *       Filename:  fftw_wrapper.cpp
 *
 *    Description:
 *
 *        Version:  1.0
 *        Created:  11/26/2020 07:04:06 PM
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  Haoyang Liu (@liuhy), liuhaoyang@pku.edu.cn
 *   Organization:  BICMR, PKU
 *
 * ==========================================================================
 */

#ifdef OPTSUITE_USE_FFTW

#include "OptSuite/core_n.h"
#include "OptSuite/LinAlg/private/fftw_impl.h"

namespace OptSuite { namespace LinAlg {
    FFTWPlan& FFTWManager::get_plan(Index nfft, bool inverse, const void * in, void * out){
        bool inplace = (in == out);
        bool aligned = ( (reinterpret_cast<size_t>(in)&15) | (reinterpret_cast<size_t>(out)&15) ) == 0;
        uint64_t key = ( (nfft<<3 ) | (inverse<<2) | (inplace<<1) | aligned ) << 1;
        return map_dft_1d[key];
    }

    FFTWPlan& FFTWManager::get_plan(Index nfft, int type, const Scalar* in, Scalar* out){
        bool inplace = (in == out);
        bool aligned = ( (reinterpret_cast<size_t>(in)&15) | (reinterpret_cast<size_t>(out)&15) ) == 0;
        uint64_t key = ( (nfft<<5 ) | (type<<2) | (inplace<<1) | aligned ) << 1;
        return map_r2r_1d[key];
    }

    void FFTWManager::forward(const std::vector<ComplexScalar>& in,
                                    std::vector<ComplexScalar>& out,
                                    Index nfft){
        if (nfft == -1) nfft = in.size();
        if (out.size() < (size_t)nfft) out.resize(nfft);
        get_plan(nfft, false, in.data(), out.data()).
            forward(nfft, fftw_cast(in.data()), fftw_cast(out.data()));
    }
    void FFTWManager::backward(const std::vector<ComplexScalar>& in,
                                     std::vector<ComplexScalar>& out,
                                     Index nfft){
        if (nfft == -1) nfft = in.size();
        if (out.size() < (size_t)nfft) out.resize(nfft);
        get_plan(nfft, true, in.data(), out.data()).
            backward(nfft, fftw_cast(in.data()), fftw_cast(out.data()));
    }

    void FFTWManager::forward(const Ref<const CMat> in, Ref<CMat> out, Index nfft){
        if (nfft == -1) nfft = in.rows();
        Index howmany = in.cols();
        Index idist = in.outerStride(), odist = out.outerStride();
        CMat tmp;

        auto in_ptr = in.data();
        auto out_ptr = out.data();

        // check whether we need a temporary mat
        if (in.innerStride() != 1 || in.rows() < nfft){
            tmp.resize(nfft, howmany);
            tmp.setZero();
            tmp.block(0, 0, std::min(in.rows(), nfft), howmany) =
                in.block(0, 0, std::min(in.rows(), nfft), howmany);
            idist = tmp.outerStride();
            in_ptr = tmp.data();
        }

        if (howmany == 1){
            get_plan(nfft, false, in_ptr, out_ptr).
                forward(nfft, fftw_cast(in_ptr), fftw_cast(out_ptr));
        } else {
            // use temporary plan
            FFTWPlan fft_many;
            fft_many.forward_many(nfft, howmany, idist, odist, fftw_cast(in_ptr), fftw_cast(out_ptr));
        }

    }
    void FFTWManager::backward(const Ref<const CMat> in, Ref<CMat> out, Index nfft){
        if (nfft == -1) nfft = in.rows();
        Index howmany = in.cols();
        Index idist = in.outerStride(), odist = out.outerStride();
        CMat tmp;

        auto in_ptr = in.data();
        auto out_ptr = out.data();

        // check whether we need a temporary mat
        if (in.innerStride() != 1 || in.rows() < nfft){
            tmp.resize(nfft, howmany);
            tmp.setZero();
            tmp.block(0, 0, std::min(in.rows(), nfft), howmany) =
                in.block(0, 0, std::min(in.rows(), nfft), howmany);
            idist = tmp.outerStride();
            in_ptr = tmp.data();
        }

        if (howmany == 1){
            get_plan(nfft, true, in_ptr, out_ptr).
                backward(nfft, fftw_cast(in_ptr), fftw_cast(out_ptr));
        } else {
            // use temporary plan
            FFTWPlan fft_many;
            fft_many.backward_many(nfft, howmany, idist, odist, fftw_cast(in_ptr), fftw_cast(out_ptr));
        }

        // scaling
#ifndef OPTSUITE_UNSCALED_IFFT
        out /= (Scalar)nfft;
#endif

    }

    void FFTWManager::real_to_real(const Ref<const Mat> in, Ref<Mat> out,
            Index nfft, FFTW_R2R_Type type){
        if (nfft == -1) nfft = in.rows();
        Index howmany = in.cols();
        Index idist = in.outerStride(), odist = out.outerStride();
        Mat tmp;

        auto in_ptr = in.data();
        auto out_ptr = out.data();

        // check whether we need a temporary mat
        if (in.innerStride() != 1 || in.rows() < nfft){
            tmp.resize(nfft, howmany);
            tmp.setZero();
            tmp.block(0, 0, std::min(in.rows(), nfft), howmany) =
                in.block(0, 0, std::min(in.rows(), nfft), howmany);
            idist = tmp.outerStride();
            in_ptr = tmp.data();
        }

        if (howmany == 1){
            get_plan(nfft, static_cast<int>(type), in_ptr, out_ptr).
                real_to_real(nfft, fftw_cast(in_ptr), out_ptr, to_r2r_kind(type));
        } else {
            // use temporary plan
            FFTWPlan pl;
            pl.real_to_real_many(nfft, howmany, idist, odist, fftw_cast(in_ptr), out_ptr, to_r2r_kind(type));
        }
    }

    namespace {
        static FFTWManager manager;
    }

    CMat fft(const Ref<const CMat> in, Index nfft){
        if (nfft == -1) nfft = in.rows();
        CMat out(nfft, in.cols());
        manager.forward(in, out, nfft);
        return out;
    }

    CMat ifft(const Ref<const CMat> in, Index nfft){
        if (nfft == -1) nfft = in.rows();
        CMat out(nfft, in.cols());
        manager.backward(in, out, nfft);
        return out;
    }

    Mat dct(const Ref<const Mat> in, Index nfft, DCT_Type t){
        if (nfft == -1) nfft = in.rows();
        Mat out(nfft, in.cols());
        manager.real_to_real(in, out, nfft, static_cast<FFTW_R2R_Type>(t));
        // the definition of dct is different from matlab
        switch (t){
            case DCT_Type::TYPE_I:
                out.array().rowwise() += (std::sqrt(2_s) - 1_s) * in.row(0).array();
                for (Index i = 0; i < out.rows(); i+=2)
                    out.row(i).array() += (std::sqrt(2_s) - 1_s) * in.row(in.rows() - 1).array();
                for (Index i = 1; i < out.rows(); i+=2)
                    out.row(i).array() -= (std::sqrt(2_s) - 1_s) * in.row(in.rows() - 1).array();
                out.row(0) /= std::sqrt(2_s);
                out.row(out.rows() - 1) /= std::sqrt(2_s);
                out /= std::sqrt(2_s * (nfft - 1));
                break;
            case DCT_Type::TYPE_II:
                out.row(0) /= std::sqrt(2_s);
                out /= std::sqrt(2_s * nfft);
                break;
            case DCT_Type::TYPE_III:
                out.array().rowwise() += (std::sqrt(2_s) - 1_s) * in.row(0).array();
                out /= std::sqrt(2_s * nfft);
                break;
            case DCT_Type::TYPE_IV:
                out /= std::sqrt(2_s * nfft);
                break;
        }
        return out;
    }

    Mat idct(const Ref<const Mat> in, Index nfft){
        if (nfft == -1) nfft = in.rows();
        Mat out(nfft, in.cols());
        manager.real_to_real(in, out, nfft, FFTW_R2R_Type::DCT_III);
        // the definition of idct is different from matlab
        out.array() += (std::sqrt(2_s) - 1_s) * in(0, 0);
        out /= std::sqrt(2_s * nfft);
        return out;
    }

}}

#endif 
