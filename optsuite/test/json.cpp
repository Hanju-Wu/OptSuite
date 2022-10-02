
#include <iostream>
#include "OptSuite/Base/structure.h"
#include "OptSuite/Base/cell_array.h"
#include "OptSuite/Utils/json_formatter.h"

using namespace OptSuite;
using namespace OptSuite::Base;
using namespace OptSuite::Utils;

int main() {
    Structure st;

    st.set("i", -42_i);
    st.set("b", true);
    st.emplace<Scalar>("d", M_PI);
    st.emplace<std::string>("s", "Hello world");

    std::vector<Scalar> iter_hist { 1.0_s, 2.0_s, 3.0_s };
    auto& ca = st.emplace<CellArray>("ca", iter_hist.begin(), iter_hist.end());
    ca.emplace_back<std::string>("I just want a value of another type");

    JsonFormatter formatter;
    std::cout << formatter.format(st) << std::endl;

    return 0;
}
