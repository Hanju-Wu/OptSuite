/*
 * @Author: Wu Hanju 
 * @Date: 2022-09-10 10:00:53 
 * @Last Modified by: Wu Hanju
 * @Last Modified time: 2022-09-11 23:03:58
 */
#ifndef OPTSUITE_ALGO_FISTA
#define OPTSUITE_ALGO_FISTA

#include "OptSuite/core_n.h"
#include "OptSuite/Base/solver.h"
#include "OptSuite/Base/structure.h"
#include "OptSuite/Base/variable.h"
#include "OptSuite/Base/mat_wrapper.h"
#include "OptSuite/Composite/Model/base.h"
#include "OptSuite/Utils/logger.h"
#include "OptSuite/Utils/optionlist.h"

namespace OptSuite { namespace Composite { namespace Solver {
    using namespace OptSuite::Base;
    using namespace OptSuite::Utils;
    template<typename dtype = Scalar>
    class FISTA : public SolverBase {
        public:
            using mat_t = Eigen::Matrix<dtype, Dynamic, Dynamic>;
            using mat_array_t = MatArray_t<dtype>;
            using mat_wrapper_t = MatWrapper<dtype>;
            using var_t = Variable<dtype>;
            using var_t_ptr = std::shared_ptr<Variable<dtype>>;

            struct Output {
                var_t_ptr xx;
                Scalar nrmG;
                Scalar obj;
                Scalar Xdiff;
                Scalar etime;
                Index iter;
                std::string message;
                Index flag;

                Index num_feval;
                Index num_geval;
            };
            inline FISTA() : SolverBase("FISTA") {
                logger = std::make_shared<Logger>();
                register_options();
                set_variable_type<>();
                // set_default_options();
            }
            ~FISTA() = default;

            template<typename xtype = mat_wrapper_t, typename gtype = mat_wrapper_t>
            void set_variable_type(){
                x = std::shared_ptr<xtype>(new xtype{});
                xp = std::shared_ptr<xtype>(new xtype{});
                
                y = std::shared_ptr<xtype>(new xtype{});
                yp = std::shared_ptr<xtype>(new xtype{});

                gx = std::shared_ptr<gtype>(new gtype{});
                gy = std::shared_ptr<gtype>(new gtype{});
                gyp = std::shared_ptr<gtype>(new gtype{});
            }
            template<typename xtype = mat_wrapper_t>
            const xtype& get_sol() const {
                const xtype* sol = dynamic_cast<const xtype*>(out.xx.get());
                OPTSUITE_ASSERT(sol);
                return *sol;
            }
            void solve(const Model::Base<dtype>&, const Ref<const mat_t>);
            void solve(const Model::Base<dtype>&, const var_t&);
            OptionList& options();
            const OptionList& options() const;
            std::shared_ptr<const Structure> workspace() const;

            Output out;
            std::shared_ptr<StopRuleChecker> custom_stop_checker = nullptr;
        private:
            // local storage
            // mat_array_t x, xp, gx, gxp, y, yp, gy, gyp;
            var_t_ptr x, xp, gx, gy, gyp, y, yp;
            std::shared_ptr<Logger> logger;
            // private methods
            void register_options();
            void initialize_output();
            void initialize_local_variables(const var_t&);
            Scalar compute_obj_h(const Model::Base<dtype>&);

            // optionlist object
            OptionList options_;

            // workspace object
            std::shared_ptr<Structure> workspace_;

            void set_y(var_t_ptr, var_t_ptr, var_t_ptr, Scalar);

            template<typename xtype = mat_wrapper_t>
            mat_t getmat(var_t_ptr x_)  {
                xtype* sol = dynamic_cast<xtype*>(x_.get());
                OPTSUITE_ASSERT(sol);
                return (*sol).mat();
            }
    };
}}}

#endif
