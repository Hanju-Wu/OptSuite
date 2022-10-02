/*
 * ===========================================================================
 *
 *       Filename:  mat_comp.cpp
 *
 *    Description:  
 *
 *        Version:  1.0
 *        Created:  01/02/2021 09:14:21 PM
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  Haoyang Liu (), liuhaoyang@pku.edu.cn
 *   Organization:  BICMR, Peking University
 *      Copyright:  Copyright (c) 2021, Haoyang Liu
 *
 * ===========================================================================
 */

#ifdef OPTSUITE_USE_PROPACK

#include "OptSuite/Composite/Model/mat_comp.h"
#include "OptSuite/LinAlg/lansvd.h"

namespace OptSuite { namespace Composite { namespace Model {
    MatComp::MatComp(const Ref<const SpMat> obs, Scalar mu_){
        OPTSUITE_ASSERT(mu_ > 0);
        this->mu = mu_;

        create_functional_f(obs);
        create_functional_h(mu_);
        compute_mu_max();
    }

    void MatComp::set_obs(const Ref<const SpMat> obs){
        create_functional_f(obs);
        compute_mu_max();
    }

    void MatComp::set_mu(Scalar mu){
        OPTSUITE_ASSERT(mu > 0);
        this->mu = mu;
        create_functional_h(mu);
    }

    void MatComp::create_functional_f(const Ref<const SpMat> obs){
        this->f = std::make_shared<ProjectionOmega<>>(obs);
        this->rows_ = obs.rows();
        this->cols_ = obs.cols();
    }

    void MatComp::create_functional_h(Scalar mu){
        this->h = std::make_shared<NuclearNorm>(mu);
        this->prox_h = std::make_shared<ShrinkageNuclear>(mu);
    }

    std::shared_ptr<ShrinkageNuclear> MatComp::prox_h_ptr() const {
        return std::dynamic_pointer_cast<ShrinkageNuclear>(this->prox_h);
    }

    bool MatComp::enable_continuation() const {
        return true;
    }

    Scalar MatComp::mu_factor_init() const {
        return mu_max > mu ? mu_max / mu : 1_s;
    }

    Scalar MatComp::mu_init() const {
        return mu_max;
    }

    void MatComp::compute_mu_max(){
        using OptSuite::LinAlg::LANSVD;
        const ProjectionOmega<>* fp = dynamic_cast<const ProjectionOmega<>*>(f.get());
        LANSVD<Scalar> svd;

        const Map<const SpMat> B(rows_, cols_, fp->innerIndexPtr.size(),
                fp->outerIndexPtr.data(), fp->innerIndexPtr.data(), fp->b.data());

        mu_max = svd.compute(B, 1).d()(0);
    }

    std::string MatComp::extra_msg_h() const {
        return "       sv|svp";
    }

    std::string MatComp::extra_msg() const {
        const ShrinkageNuclear* hp = dynamic_cast<const ShrinkageNuclear*>(prox_h.get());
        return Utils::str_format("%6d|%d", hp->sv(), hp->svp());
    }

}}}

#endif

