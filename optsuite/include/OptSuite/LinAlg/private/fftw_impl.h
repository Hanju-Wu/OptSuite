/*
 * ===========================================================================
 *
 *       Filename:  fftw_impl.h
 *
 *    Description:  
 *
 *        Version:  1.0
 *        Created:  03/04/2021 07:26:11 PM
 *       Revision:  none
 *       Compiler:  g++
 *
 *         Author:  Haoyang Liu (), liuhaoyang@pku.edu.cn
 *   Organization:  BICMR, Peking University
 *
 * ===========================================================================
 */

#ifndef OPTSUITE_LINALG_PRIVATE_FFTW_IMPL_H
#define OPTSUITE_LINALG_PRIVATE_FFTW_IMPL_H

#include <iostream>
#include <complex>
#include <fftw3.h>
#include "OptSuite/core_n.h"
#include "OptSuite/LinAlg/fftw_wrapper.h"

#if OPTSUITE_SCALAR_TOKEN == 0
#define OPTSUITE_FFTW(name) fftw_ ## name
#elif OPTSUITE_SCALAR_TOKEN == 1
#define OPTSUITE_FFTW(name) fftwf_ ## name
#else
#error "unsupported fftw type"
#endif

namespace OptSuite { namespace LinAlg {
    inline OPTSUITE_FFTW(complex) * fftw_cast(const std::complex<Scalar> *p){ 
        return const_cast<OPTSUITE_FFTW(complex)*>( 
                reinterpret_cast<const OPTSUITE_FFTW(complex)*>(p) ); 
    }

    inline Scalar* fftw_cast(const Scalar *p){
        return const_cast<Scalar*>(p);
    }

    inline fftw_r2r_kind to_r2r_kind(FFTW_R2R_Type t){
        switch (t){
            case FFTW_R2R_Type::DCT_I:
                return FFTW_REDFT00;
            case FFTW_R2R_Type::DCT_II:
                return FFTW_REDFT10;
            case FFTW_R2R_Type::DCT_III:
                return FFTW_REDFT01;
            case FFTW_R2R_Type::DCT_IV:
                return FFTW_REDFT11;
            case FFTW_R2R_Type::DST_I:
                return FFTW_RODFT00;
            case FFTW_R2R_Type::DST_II:
                return FFTW_RODFT10;
            case FFTW_R2R_Type::DST_III:
                return FFTW_RODFT01;
            case FFTW_R2R_Type::DST_IV:
                return FFTW_RODFT11;
            default:
                return FFTW_REDFT10;
        }
    }

    class FFTWPlan {
        using ctype = OPTSUITE_FFTW(complex);
        OPTSUITE_FFTW(plan) plan;

        public:
            inline FFTWPlan() : plan(NULL) {}
            inline ~FFTWPlan(){
                if (plan != NULL)
                    OPTSUITE_FFTW(destroy_plan)(plan);
            }

            inline void execute(){
                if (plan != NULL)
                    OPTSUITE_FFTW(execute)(plan);
            }

            inline void forward(Index nfft, ctype *in, ctype *out){
                if (plan == NULL)
                    plan = OPTSUITE_FFTW(plan_dft_1d)(nfft, in, out, FFTW_FORWARD,
                            FFTW_ESTIMATE | FFTW_PRESERVE_INPUT);
                OPTSUITE_FFTW(execute_dft)(plan, in, out);
            }

            inline void forward_many(Index nfft, Index howmany, Index idist, Index odist,
                                     ctype *in, ctype *out){
                int n = nfft;
                if (plan == NULL){
                    plan = OPTSUITE_FFTW(plan_many_dft)(1, &n, howmany, in, &n, 1, idist, out, &n, 1, odist, FFTW_FORWARD, FFTW_ESTIMATE | FFTW_PRESERVE_INPUT);
                }
                OPTSUITE_FFTW(execute_dft)(plan, in, out);
            }

            inline void forward(Index nfft, ComplexScalar *in, ComplexScalar *out){
                this->forward(nfft, fftw_cast(in), fftw_cast(out));
            }

            inline void forward2(Index n0, Index n1, ctype *in, ctype *out){
                if (plan == NULL)
                    plan = OPTSUITE_FFTW(plan_dft_2d)(n0, n1, in, out, FFTW_FORWARD,
                            FFTW_ESTIMATE | FFTW_PRESERVE_INPUT);
                OPTSUITE_FFTW(execute_dft)(plan, in, out);
            }

            inline void backward(Index nfft, ctype *in, ctype *out){
                if (plan == NULL)
                    plan = OPTSUITE_FFTW(plan_dft_1d)(nfft, in, out, FFTW_BACKWARD,
                            FFTW_ESTIMATE | FFTW_PRESERVE_INPUT);
                OPTSUITE_FFTW(execute_dft)(plan, in, out);
            }

            inline void backward_many(Index nfft, Index howmany, Index idist, Index odist,
                                     ctype *in, ctype *out){
                int n = nfft;
                if (plan == NULL){
                    plan = OPTSUITE_FFTW(plan_many_dft)(1, &n, howmany, in, &n, 1, idist, out, &n, 1, odist, FFTW_BACKWARD, FFTW_ESTIMATE | FFTW_PRESERVE_INPUT);
                }
                OPTSUITE_FFTW(execute_dft)(plan, in, out);
            }

            inline void backward(Index nfft, ComplexScalar *in, ComplexScalar *out){
                this->backward(nfft, fftw_cast(in), fftw_cast(out));
            }

            inline void backward2(Index n0, Index n1, ctype *in, ctype *out){
                if (plan == NULL)
                    plan = OPTSUITE_FFTW(plan_dft_2d)(n0, n1, in, out, FFTW_BACKWARD,
                            FFTW_ESTIMATE | FFTW_PRESERVE_INPUT);
                OPTSUITE_FFTW(execute_dft)(plan, in, out);
            }

            inline void real_to_real(Index nfft, Scalar *in, Scalar *out, fftw_r2r_kind kind){
                if (plan == NULL)
                    plan = OPTSUITE_FFTW(plan_r2r_1d)(nfft, in, out, kind,
                            FFTW_ESTIMATE | FFTW_PRESERVE_INPUT);
                OPTSUITE_FFTW(execute_r2r)(plan, in, out);
            }

            inline void real_to_real_many(Index nfft, Index howmany, Index idist, Index odist,
                                     Scalar *in, Scalar *out, fftw_r2r_kind kind){
                int n = nfft;
                if (plan == NULL){
                    plan = OPTSUITE_FFTW(plan_many_r2r)(1, &n, howmany, in, &n, 1, idist, out, &n, 1, odist, &kind, FFTW_ESTIMATE | FFTW_PRESERVE_INPUT);
                }
                OPTSUITE_FFTW(execute_r2r)(plan, in, out);
            }

    };

    class FFTWManager {
        using ctype = OPTSUITE_FFTW(complex);
        std::map<uint64_t, FFTWPlan> map_dft_1d;
        std::map<uint64_t, FFTWPlan> map_r2r_1d;

        FFTWPlan& get_plan(Index, bool, const void *, void *);
        FFTWPlan& get_plan(Index, int, const Scalar*, Scalar*);

        public:
            inline void clear() { map_dft_1d.clear(); }
            void forward(const std::vector<ComplexScalar>&,
                               std::vector<ComplexScalar>&,
                               Index = -1);
            void backward(const std::vector<ComplexScalar>&,
                                std::vector<ComplexScalar>&,
                                Index = -1);

            void forward(const Ref<const CMat>, Ref<CMat>, Index = -1);
            void backward(const Ref<const CMat>, Ref<CMat>, Index = -1);

            void real_to_real(const Ref<const Mat>, Ref<Mat>, Index = -1, FFTW_R2R_Type = FFTW_R2R_Type::DCT_II);

    };

}}

#endif

