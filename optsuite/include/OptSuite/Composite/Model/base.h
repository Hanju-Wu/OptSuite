/*
 * ==========================================================================
 *
 *       Filename:  Composite/Model/base.h
 *
 *    Description:
 *
 *        Version:  1.0
 *        Created:  11/04/2020 04:16:07 PM
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  Haoyang Liu (@liuhy), liuhaoyang@pku.edu.cn
 *   Organization:  BICMR, PKU
 *
 * ==========================================================================
 */
#ifndef OPTSUITE_COMPOSITE_MODEL_BASE_H
#define OPTSUITE_COMPOSITE_MODEL_BASE_H

#include "OptSuite/Base/model.h"
#include "OptSuite/Base/functional.h"

namespace OptSuite { namespace Composite { namespace Model {
    using std::shared_ptr;
    using namespace OptSuite::Base;
    template<typename dtype = Scalar>
    class Base : public ModelBase {
        public:
            Base() : ModelBase("composite") {}
            ~Base() = default;

            shared_ptr<FuncGrad<dtype>> f = nullptr;
            shared_ptr<Func<dtype>> h = nullptr;
            shared_ptr<Proximal<dtype>> prox_h = nullptr;

            virtual bool enable_continuation() const { return false; }
            virtual Scalar mu_factor_init() const { return 1_s; }
    };
}}}

#endif
