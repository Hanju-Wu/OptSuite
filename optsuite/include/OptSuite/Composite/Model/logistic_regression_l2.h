/*
 * @Author: Wu Hanju 
 * @Date: 2022-09-05 19:00:24 
 * @Last Modified by: Wu Hanju
 * @Last Modified time: 2022-09-05 22:27:43
 */
#include "OptSuite/core_n.h"
#include "OptSuite/Base/mat_op.h"
#include "OptSuite/Composite/Model/base.h"

namespace OptSuite { namespace Composite { namespace Model {
    using namespace OptSuite::Base;

    class LogisticRegression_L2 : public Base<Scalar> {
        public:
            LogisticRegression_L2(Scalar);
            LogisticRegression_L2(const Ref<const Mat>, const Ref<const Mat>, Scalar);
            ~LogisticRegression_L2() = default;

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

            std::shared_ptr<ShrinkageL2> prox_h_ptr() const;
            std::shared_ptr<L2Norm> h_ptr() const;

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