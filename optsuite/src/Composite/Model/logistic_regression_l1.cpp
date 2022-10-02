/*
 * @Author: Wu Hanju 
 * @Date: 2022-09-03 18:58:17 
 * @Last Modified by: Wu Hanju
 * @Last Modified time: 2022-09-05 19:09:24
 */
#include <cmath>
#include <memory>
#include "OptSuite/Composite/Model/logistic_regression_l1.h"

namespace OptSuite { namespace Composite { namespace Model {
    LogisticRegression_L1::LogisticRegression_L1(Scalar mu){
        OPTSUITE_ASSERT(mu > 0);
        this->mu = mu;
        create_functional_h(mu);
    }

    LogisticRegression_L1::LogisticRegression_L1(const Ref<const Mat> A, const Ref<const Mat> b, Scalar mu){
        logistic_regression_init(A, b, mu);
    }

    template <typename AT>
    void LogisticRegression_L1::logistic_regression_init(const AT& A, const Ref<const Mat> b, Scalar mu_){
        OPTSUITE_ASSERT(A.cols() == b.rows());
        OPTSUITE_ASSERT(mu_ > 0);
        this->mu = mu_;
        create_functional_f(A, b);
        create_functional_h(mu);
    }

    void LogisticRegression_L1::set_mu(Scalar mu){
        OPTSUITE_ASSERT(mu > 0);
        this->mu = mu;
        create_functional_h(mu);
    }

    void LogisticRegression_L1::create_functional_h(Scalar mu_){
        this->h = std::make_shared<L1Norm>(mu_);
        this->prox_h = std::make_shared<ShrinkageL1>(mu_);
    }

    bool LogisticRegression_L1::enable_continuation() const {
        return true;
    }

    Scalar LogisticRegression_L1::mu_factor_init() const {
        std::shared_ptr<LogisticRegression<>> pf = std::dynamic_pointer_cast<LogisticRegression<>>(f);
        Scalar bsup = pf->get_b().lpNorm<Eigen::Infinity>();
        Scalar mu0 = std::min(0.01_s * pf->cols(),
                2.2_s * (Scalar)std::pow(bsup / mu, -1.0/3) * bsup);
        return std::max(mu0 / mu, 1.0_s);
    }

    std::string LogisticRegression_L1::extra_msg_h() const {
        return "       nnz|nnz_t";
    }

    std::string LogisticRegression_L1::extra_msg() const {
        auto hp = dynamic_cast<const ShrinkageL1 *>(prox_h.get());
        return Utils::str_format("%7d|%d", hp->nnz(), hp->nnz_t());
    }

    std::shared_ptr<ShrinkageL1> LogisticRegression_L1::prox_h_ptr() const {
        return std::dynamic_pointer_cast<ShrinkageL1>(this->prox_h);
    }

    std::shared_ptr<L1Norm> LogisticRegression_L1::h_ptr() const {
        return std::dynamic_pointer_cast<L1Norm>(this->h);
    }
}}}