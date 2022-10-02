/*
 * ==========================================================================
 *
 *       Filename:  struct.cpp
 *
 *    Description:
 *
 *        Version:  1.0
 *        Created:  04/02/2021 03:46:30 PM
 *       Revision:  none
 *       Compiler:  g++
 *
 *         Author:  Haoyang Liu (@liuhy), liuhaoyang@pku.edu.cn
 *   Organization:  BICMR, PKU
 *
 * ==========================================================================
 */

#include <iostream>
#include "OptSuite/Base/structure.h"

using namespace OptSuite;
using namespace OptSuite::Base;


class NonMove {
    public:
        NonMove(int data) : data_(data) {}
        NonMove(const NonMove&) = delete;
        NonMove(NonMove&&) = delete;
        inline int data() { return data_; }
    private:
        int data_;
};


#define EXPECT(expr) \
    if (expr) { \
        std::cout << "test success: " << #expr << std::endl; \
    } else { \
        std::cout << "test fail: " << #expr << std::endl; \
    }

int main() {
    Structure work;

    std::cout << "Testing get<Index>\n";
    work.set("iter", 0_i);
    EXPECT(work.get<Index>("iter") == 0);

    work.set("iter", 100_i);
    EXPECT(work.get<Index>("iter") == 100);

    std::cout << "\nTesting is_field<Index>\n";
    EXPECT(work.is_field<Index>("iter"));
    EXPECT(!work.is_field<Scalar>("iter"));

    std::cout << "\nTesting get<string>\n";
    work.emplace<std::string>("Z", "Hello world");
    EXPECT(work.get<std::string>("Z") == "Hello world");

    std::cout << "\nTesting get<Mat>\n";
    work.emplace<Mat>("X", Mat::Random(3, 3));
    std::cout << work.get<Mat>("X") << "\n\n";
    work.get<Mat>("X")(1, 1) = 42;
    std::cout << work.get<Mat>("X") << "\n\n";
    EXPECT(work.get<Mat>("X")(1, 1) == 42);

    std::cout << "\nTesting set_var and get_var\n";
    work.set_var<Scalar>("Y", VariableRef<Scalar>{ Mat::Random(2, 2) });
    EXPECT(work.is_var<Scalar>("Y"));
    std::cout << work.get_var<Scalar>("Y") << "\n";

    std::cout << "\nTesting set<Mat> and get_var\n";
    EXPECT(work.is_var<Scalar>("X"));
    std::cout << work.get_var<Scalar>("X") << "\n";

    std::cout << "\nTesting set<string> and get_var\n";
    EXPECT(work.is_var<void>("Z"));
    std::cout << work.get_var<>("Z") << "\n";

    std::cout << "\nTesting set_var and get<Mat>\n";
    EXPECT(work.is_field<Mat>("Y"));
    std::cout << work.get<Mat>("Y") << "\n";

    std::cout << "\nTesting emplace<NonMove> and get<NonMove>\n";
    EXPECT(work.emplace<NonMove>("NonMove", 42).data() == 42);
    EXPECT(work.get<NonMove>("NonMove").data() == 42);

    return 0;
}
