/*
 * ==========================================================================
 *
 *       Filename:  opt.cpp
 *
 *    Description:
 *
 *        Version:  1.0
 *        Created:  11/20/2020 03:46:26 PM
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  Haoyang Liu (@liuhy), liuhaoyang@pku.edu.cn
 *   Organization:  BICMR, PKU
 *
 * ==========================================================================
 */

#include <vector>
#include "OptSuite/core_n.h"
#include "OptSuite/Utils/logger.h"
#include "OptSuite/Utils/optionlist.h"
#include "OptSuite/Utils/optionchecker.h"

using namespace OptSuite;
using namespace OptSuite::Utils;

int main(int argc, char **argv){
    std::vector<RegOption> v;
    std::shared_ptr<OptionChecker> checker(new StrEnumChecker
            {"very poor", "poor", "normal", "rich", "very rich"});

    v.emplace_back("name", "taroxd", "input your name");
    v.emplace_back("cuxia", true, "is cuxia or not");
    v.emplace_back("gpa", 4.0, "current gpa, must in [0, 4]",
            std::make_shared<BoundChecker<Scalar>>(0, BoundCheckerSense::Standard, 4, BoundCheckerSense::Standard));
    v.emplace_back("age", (Index)9, "Age. Must be positive",
            std::make_shared<BoundChecker<Index>>(0, BoundCheckerSense::Strict, 0, BoundCheckerSense::None));
    v.emplace_back("wealth", "poor", "Level of wealth", checker);

    OptionList ol(v, std::make_shared<Logger>());

    // register on-the-fly
    ol.register_option({"gf", "nmzn", "girl friend name"});
    ol.register_option({"favorite_game", "OW", "his/her favorite PC game"});

    // options from cmd line have higher priority
    ol.set_from_file("cuxia.conf");
    ol.set_from_cmd_line(argc, argv);

    // print out
    ol.show(true);

    return 0;
}
