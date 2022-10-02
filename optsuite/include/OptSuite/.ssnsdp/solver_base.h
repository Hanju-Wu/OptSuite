/*
 * ==========================================================================
 *
 *       Filename:  solver_base.h
 *
 *    Description:  
 *
 *        Version:  1.0
 *        Created:  11/01/2020 07:51:43 PM
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  Haoyang Liu (@liuhy), liuhaoyang@pku.edu.cn
 *   Organization:  BICMR, PKU
 *
 * ==========================================================================
 */
#include <string>

namespace OptSuite {
    class Solver_Base{
        public:
            Solver_Base(std::string);
            ~Solver_Base() = default;

            std::string name;
        private:
    };
}
