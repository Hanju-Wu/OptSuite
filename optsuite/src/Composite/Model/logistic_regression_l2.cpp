/*
 * @Author: Wu Hanju 
 * @Date: 2022-09-05 19:01:17 
 * @Last Modified by: Wu Hanju
 * @Last Modified time: 2022-09-05 22:27:27
 */
#include <cmath>
#include <memory>
#include "OptSuite/Composite/Model/logistic_regression_l2.h"

namespace OptSuite { namespace Composite { namespace Model {
    LogisticRegression_L2::LogisticRegression_L2(Scalar mu){
        OPTSUITE_ASSERT(mu > 0);
        this->mu = mu;
        create_functional_h(mu);
    }

    LogisticRegression_L2::LogisticRegression_L2(const Ref<const Mat> A, const Ref<const Mat> b, Scalar mu){
        logistic_regression_init(A, b, mu);
    }

    template <typename AT>
    void LogisticRegression_L2::logistic_regression_init(const AT& A, const Ref<const Mat> b, Scalar mu_){
        OPTSUITE_ASSERT(A.cols() == b.rows());
        OPTSUITE_ASSERT(mu_ > 0);
        this->mu = mu_;
        create_functional_f(A, b);
        create_functional_h(mu);
    }

    void LogisticRegression_L2::set_mu(Scalar mu){
        OPTSUITE_ASSERT(mu > 0);
        this->mu = mu;
        create_functional_h(mu);
    }

    void LogisticRegression_L2::create_functional_h(Scalar mu_){
        this->h = std::make_shared<L2Norm>(mu_);
        this->prox_h = std::make_shared<ShrinkageL2>(mu_);
    }

    bool LogisticRegression_L2::enable_continuation() const {
        return true;
    }

    Scalar LogisticRegression_L2::mu_factor_init() const {
        std::shared_ptr<LogisticRegression<>> pf = std::dynamic_pointer_cast<LogisticRegression<>>(f);
        Scalar bsup = pf->get_b().lpNorm<Eigen::Infinity>();
        Scalar mu0 = std::min(0.01_s * pf->cols(),
                2.2_s * (Scalar)std::pow(bsup / mu, -1.0/3) * bsup);
        return std::max(mu0 / mu, 1.0_s);
    }

    std::shared_ptr<ShrinkageL2> LogisticRegression_L2::prox_h_ptr() const {
        return std::dynamic_pointer_cast<ShrinkageL2>(this->prox_h);
    }

    std::shared_ptr<L2Norm> LogisticRegression_L2::h_ptr() const {
        return std::dynamic_pointer_cast<L2Norm>(this->h);
    }
}}}