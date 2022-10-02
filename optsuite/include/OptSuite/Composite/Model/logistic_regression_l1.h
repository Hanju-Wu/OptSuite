/*
 * @Author: Wu Hanju 
 * @Date: 2022-09-03 18:58:15 
 * @Last Modified by: Wu Hanju
 * @Last Modified time: 2022-09-05 19:08:14
 */
#include "OptSuite/core_n.h"
#include "OptSuite/Base/mat_op.h"
#include "OptSuite/Composite/Model/base.h"

namespace OptSuite { namespace Composite { namespace Model {
    using namespace OptSuite::Base;

    class LogisticRegression_L1 : public Base<Scalar> {
        public:
            LogisticRegression_L1(Scalar);
            LogisticRegression_L1(const Ref<const Mat>, const Ref<const Mat>, Scalar);
            ~LogisticRegression_L1() = default;

            inline Scalar get_mu() { return mu; }

            template <typename AT>
            void set_A_b(const AT& A, const Ref<const Mat> b){
                OPTSUITE_ASSERT(b.cols() == 1);
                OPTSUITE_ASSERT(b.rows() == A.cols());
                create_functional_f(A, b);
            }

            void set_mu(Scalar);

            bool enable_continuation() const override;
            Scalar mu_factor_init() const override;

            std::string extra_msg_h() const override;
            std::string extra_msg() const override;

            std::shared_ptr<ShrinkageL1> prox_h_ptr() const;
            std::shared_ptr<L1Norm> h_ptr() const;

        private:
            Scalar mu;
            template <typename AT>
            void logistic_regression_init(const AT&, const Ref<const Mat>, Scalar);

            template <typename AT>
            void create_functional_f(const AT& A, const Ref<const Mat> b){
                this->f = std::make_shared<LogisticRegression<>>(A, b);
            }

            void create_functional_h(Scalar);
    };
}}}