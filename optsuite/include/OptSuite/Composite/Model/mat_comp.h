/*
 * ===========================================================================
 *
 *       Filename:  mat_comp.h
 *
 *    Description:  
 *
 *        Version:  1.0
 *        Created:  01/02/2021 08:44:19 PM
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  Haoyang Liu (), liuhaoyang@pku.edu.cn
 *   Organization:  BICMR, Peking University
 *
 * ===========================================================================
 */

#ifdef OPTSUITE_USE_PROPACK

#ifndef OPTSUITE_COMPOSITE_MODEL_MAT_COMP_H
#define OPTSUITE_COMPOSITE_MODEL_MAT_COMP_H

#include "OptSuite/core_n.h"
#include "OptSuite/Composite/Model/base.h"
#include "OptSuite/Utils/optionlist.h"

namespace OptSuite { namespace Composite { namespace Model {
    
    class MatComp : public Base<Scalar> {
        public:
            MatComp(const Ref<const SpMat>, Scalar = 1_s);
            ~MatComp() = default;
            inline Scalar get_mu() { return mu; }

            void set_obs(const Ref<const SpMat>);
            void set_mu(Scalar);

            bool enable_continuation() const override;
            Scalar mu_factor_init() const override;
            Scalar mu_init() const;

            std::string extra_msg_h() const override;
            std::string extra_msg() const override;

            std::shared_ptr<ShrinkageNuclear> prox_h_ptr() const;

        private:
            void create_functional_f(const Ref<const SpMat>);
            void create_functional_h(Scalar);
            void compute_mu_max();
            Scalar mu;
            Scalar mu_max;

            Index rows_;
            Index cols_;
    };
}}}

#endif

#endif // OPTSUITE_USE_PROPACK

