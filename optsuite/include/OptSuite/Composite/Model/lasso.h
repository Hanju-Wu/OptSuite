/*
 * ==========================================================================
 *
 *       Filename:  lasso.h
 *
 *    Description:  header file for LASSO model
 *
 *        Version:  2.0
 *        Created:  11/01/2020 05:49:01 PM
 *       Revision:  28/02/2021 14:10
 *       Compiler:  g++
 *
 *         Author:  Haoyang Liu (@liuhy), liuhaoyang@pku.edu.cn
 *   Organization:  BICMR, PKU
 *
 * ==========================================================================
 */

#include "OptSuite/core_n.h"
#include "OptSuite/Base/mat_op.h"
#include "OptSuite/Composite/Model/base.h"

namespace OptSuite { namespace Composite { namespace Model {
    using namespace OptSuite::Base;
    enum class LASSOType {
        Standard,
        Rowwise,
        Colwise
    };

    class LASSO : public Base<Scalar> {
        public:
            LASSO(Scalar, LASSOType = LASSOType::Standard);
            LASSO(const Ref<const Mat>, const Ref<const Mat>, Scalar, LASSOType = LASSOType::Standard);
            LASSO(const Ref<const SpMat>, const Ref<const Mat>, Scalar, LASSOType = LASSOType::Standard);
            LASSO(const MatOp<Scalar>&, const Ref<const Mat>, Scalar, LASSOType = LASSOType::Standard);
            ~LASSO() = default;

            inline Scalar get_mu() { return mu; }

            template <typename AT>
            void set_A_b(const AT& A, const Ref<const Mat> b){
                OPTSUITE_ASSERT(A.rows() == b.rows());
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
            void lasso_init(const AT&, const Ref<const Mat>, Scalar, LASSOType = LASSOType::Standard);

            template <typename AT>
            void create_functional_f(const AT& A, const Ref<const Mat> b){
                this->f = std::make_shared<AxmbNormSqr<>>(A, b);
            }

            void create_functional_h(Scalar, LASSOType);
            LASSOType type = LASSOType::Standard;
    };
}}}
