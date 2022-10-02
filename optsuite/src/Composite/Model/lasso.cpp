/*
 * ==========================================================================
 *
 *       Filename:  Composite/Model/lasso.cpp
 *
 *    Description:  
 *
 *        Version:  1.0
 *        Created:  11/01/2020 06:46:25 PM
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  Haoyang Liu (@liuhy), liuhaoyang@pku.edu.cn
 *   Organization:  BICMR, PKU
 *
 * ==========================================================================
 */

#include <cmath>
#include <memory>
#include "OptSuite/Composite/Model/lasso.h"

namespace OptSuite { namespace Composite { namespace Model {
    LASSO::LASSO(Scalar mu, LASSOType type){
        OPTSUITE_ASSERT(mu > 0);
        this->mu = mu;
        this->type = type;
        create_functional_h(mu, type);
    }

    LASSO::LASSO(const Ref<const Mat> A, const Ref<const Mat> b, Scalar mu, LASSOType type){
        lasso_init(A, b, mu, type);
    }

    LASSO::LASSO(const Ref<const SpMat> A, const Ref<const Mat> b, Scalar mu, LASSOType type){
        lasso_init(A, b, mu, type);
    }

    LASSO::LASSO(const MatOp<Scalar>& Aop, const Ref<const Mat> b, Scalar mu, LASSOType type){
        lasso_init(Aop, b, mu, type);
    }

    template <typename AT>
    void LASSO::lasso_init(const AT& A, const Ref<const Mat> b, Scalar mu, LASSOType lassoType){
        OPTSUITE_ASSERT(A.rows() == b.rows());
        OPTSUITE_ASSERT(mu > 0);
        this->mu = mu;
        this->type = lassoType;
        create_functional_f(A, b);
        create_functional_h(mu, lassoType);
    }

    void LASSO::set_mu(Scalar mu){
        OPTSUITE_ASSERT(mu > 0);
        this->mu = mu;
        create_functional_h(mu, type);
    }

    void LASSO::create_functional_h(Scalar mu_, LASSOType lassoType){
        switch (lassoType){
        case LASSOType::Standard:
            this->h = std::make_shared<L1Norm>(mu_);
            this->prox_h = std::make_shared<ShrinkageL1>(mu_);
            break;
        case LASSOType::Rowwise:
            this->h = std::make_shared<L1_2Norm>(mu_);
            this->prox_h = std::make_shared<ShrinkageL2Rowwise>(mu_);
            break;
        case LASSOType::Colwise:
            /* OPTSUITE_ASSERT(0); */
            this->h = std::make_shared<L2_1Norm>(mu_);
            this->prox_h = std::make_shared<ShrinkageL2Colwise>(mu_);
            break;
        }
    }

    bool LASSO::enable_continuation() const {
        return true;
    }

    Scalar LASSO::mu_factor_init() const {
        std::shared_ptr<AxmbNormSqr<>> pf = std::dynamic_pointer_cast<AxmbNormSqr<>>(f);
        Scalar bsup = pf->get_b().lpNorm<Eigen::Infinity>();
        Scalar mu0 = std::min(0.01_s * pf->cols(),
                2.2_s * (Scalar)std::pow(bsup / mu, -1.0/3) * bsup);
        return std::max(mu0 / mu, 1.0_s);
    }

    std::string LASSO::extra_msg_h() const {
        if (type == LASSOType::Standard)
            return "       nnz|nnz_t";
        else
            return "";
    }

    std::string LASSO::extra_msg() const {
        switch (type) {
            case LASSOType::Standard: {
                auto hp = dynamic_cast<const ShrinkageL1 *>(prox_h.get());
                return Utils::str_format("%7d|%d", hp->nnz(), hp->nnz_t());
            }
            default: {
                return "";
            }
        }
    }

    std::shared_ptr<ShrinkageL1> LASSO::prox_h_ptr() const {
        return std::dynamic_pointer_cast<ShrinkageL1>(this->prox_h);
    }

    std::shared_ptr<L1Norm> LASSO::h_ptr() const {
        return std::dynamic_pointer_cast<L1Norm>(this->h);
    }
}}}
