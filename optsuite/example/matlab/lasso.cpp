/*
 * ==========================================================================
 *
 *       Filename:  lasso.cpp
 *
 *    Description:  integrated tests for LASSO problem
 *
 *        Version:  2.0
 *        Created:  11/04/2020 07:10:10 PM
 *       Revision:  02/28/2021 14:54
 *       Compiler:  g++
 *
 *         Author:  Haoyang Liu (@liuhy), liuhaoyang@pku.edu.cn
 *   Organization:  BICMR, PKU
 *
 * ==========================================================================
 */

#include <iostream>
#include <cstring>
#include "OptSuite/core_n.h"
#include "OptSuite/Base/functional.h"
#include "OptSuite/Base/mat_op.h"
#include "OptSuite/Utils/tictoc.h"
#include "OptSuite/Composite/Model/lasso.h"
#include "OptSuite/Composite/Solver/proxgbb.h"
#include "OptSuite/LinAlg/fftw_wrapper.h"
#include "OptSuite/Interface/matlab.h"

using namespace OptSuite;
using namespace OptSuite::Base;
using namespace OptSuite::Composite;
using namespace OptSuite::Utils;
using namespace OptSuite::Interface;

using RowVec = Eigen::Matrix<Scalar, 1, Dynamic>;

static const std::string help_txt = R"(
Usage:
    ./lasso synth [OPTIONS]
    ./lasso real filename [OPTIONS]
)";

static OptionList options_m;

void register_options(){
    std::vector<RegOption> v;

    options_m.register_option({"n", 262144_i, "n"});
    options_m.register_option({"d", 20_s, "dyna"});
    options_m.register_option({"scaling", false, "use scaled A as input"});
    options_m.register_option({"prefix", "/tmp", "prefix for extern file"});
    options_m.register_option({"config", "../example/lasso.conf", "path to configuration file"});
    options_m.register_option({"mu", 1e-3_s, "mu factor (real case only)"});
}


// create MatOp object
// A = dct(.)_J
#ifdef OPTSUITE_USE_FFTW
using namespace OptSuite::LinAlg;

class DCT_J : public MatOp<Scalar> {
    std::vector<Index> J_ind;
    mutable Mat r;
    public:
        DCT_J(Index n, const std::vector<Index>& J) : MatOp<Scalar>(J.size(), n) {
            J_ind = J;
        }

        void apply(const Ref<const Mat>& x, Ref<Mat> y) const override {
            r = dct(x, cols());
            for (size_t i = 0; i < J_ind.size(); ++i){
                y.row(i) = r.row(J_ind[i]);
            }
        }

        void apply_transpose(const Ref<const Mat>& x, Ref<Mat> y) const override {
            r.resize(cols(), x.cols());
            r.setZero();
            for (size_t i = 0; i < J_ind.size(); ++i){
                r.row(J_ind[i]) = x.row(i);
            }
            y = idct(r, cols());
        }
};
#endif

class NZChecker : public StopRuleChecker {
    public:
        inline bool operator()() const override {
            if (workspace != nullptr){
                if (workspace->is_field<Index>("nnz_p") && workspace->is_field<Index>("nnz") && workspace->is_field<Index>("iter_c")){
                    return workspace->get<Index>("iter_c") > 99 &&
                        workspace->get<Index>("nnz") < 0.75 * workspace->get<Index>("nnz_p");
                }
            }
            return false;
        }
};


#if OPTSUITE_USE_FFTW
int do_synth(int argc, char **argv){
    register_options();
    options_m.set_from_cmd_line(argc, argv, "--mopt");

    Index n = options_m.get_integer("n");
    Index m = n / 8;
    Index k = n / 40;
    Scalar d = options_m.get_scalar("d");
    std::string prefix = options_m.get_string("prefix");

    std::string filename = str_format("%s/ssnlasso_N%d_M%d_K%d_dyna%d.mat",
            prefix.c_str(), n, m, k, static_cast<int>(d));

    Mat b, J_orig, mu_orig, fopt_orig;
    get_matrix_from_file(filename.c_str(), "b", b);
    get_matrix_from_file(filename.c_str(), "J", J_orig);
    get_matrix_from_file(filename.c_str(), "mu", mu_orig);
    get_matrix_from_file(filename.c_str(), "fopt", fopt_orig);

    if (b.size() == 0)
        return 1;

    Scalar mu = mu_orig(0), fopt = fopt_orig(0);

    // convert J_orig to J
    std::vector<Index> J_ind(m);
    for (Index i = 0; i < m; ++i){
        J_ind[i] = static_cast<Index>(J_orig(0, i) - 1);
    }

    DCT_J Aop(n, J_ind);

    Logger logger("x.dat");

    Vec x0 = Vec::Zero(n, 1);
    Composite::Model::LASSO model(Aop, b, mu);
    Composite::Solver::ProxGBB<> alg;

    model.prox_h_ptr()->options().set_from_cmd_line(argc, argv, "--hopt");
    alg.options().set_from_cmd_line(argc, argv);

    auto tstart = tic();
    alg.solve(model, x0);
    auto elapsed = toc(tstart);
    std::cout << str_format("& %8.1e & %6.2f ",
            (alg.out.obj - fopt) / std::max(1_s, fopt),
            elapsed);
    std::cerr << str_format("Obj: %.6e\n", alg.out.obj);
    std::cerr << "Elapsed time is " << elapsed << " sec." << std::endl;
    std::cerr << "Solver returned with message " << alg.out.message << std::endl;

    logger.log(alg.get_sol().mat());
#else
int do_synth(int, char **){
    std::cerr << "do_synth is not supported because fftw is disabled." << std::endl;
    std::cerr << "Please remove -DUSE_FFTW=OFF and rebuild OptSuite." << std::endl;
#endif
    return 0;
}

int do_real(int argc, char **argv){
    register_options();
    options_m.set_from_cmd_line(argc, argv, "--mopt");
    char *filename = argv[2];

    Scalar mu_fac = options_m.get_scalar("mu");
    std::string config = options_m.get_string("config");
    bool scaling = options_m.get_bool("scaling");

    Mat b, A, Asub, mu_orig, xx, xx_sub, xx0;
    RowVec colnorms;
    SpMat spA;
    get_matrix_from_file(filename, "b", b);
    get_matrix_from_file(filename, "mu", mu_orig);

    bool sparse_A = false;
    get_matrix_from_file(filename, "B", A);
    if (A.cols() == 0){ // B is not dense, try sparse
        get_matrix_from_file(filename, "B", spA);
        sparse_A = true;
    }

    if (b.size() == 0)
        return 1;

    Index n = sparse_A ? spA.cols() : A.cols();
    Scalar mu = mu_fac * mu_orig(0);

    // generate output filename
    std::string path(filename);
    std::string out_xx_base = path.substr(path.find_last_of("/\\") + 1);
    auto p = out_xx_base.find_last_of('.');
    std::string out_xx = p > 0 && p != std::string::npos ? out_xx_base.substr(0, p) : out_xx_base;
    out_xx.append(".dat");

    Logger logger(out_xx);

    Vec x0 = Vec::Zero(n, 1);
    Composite::Model::LASSO model(mu), model_sub(mu);
    if (sparse_A)
        model.set_A_b(spA, b);
    else {
        if (scaling){
            colnorms.array() = A.colwise().norm().array().max(1_s);
            A.array().rowwise() /= colnorms.array();
        }
        model.set_A_b(A, b);
        if (scaling){
            model.prox_h_ptr()->weights() = colnorms.array().inverse().transpose();
            model.h_ptr()->weights() = colnorms.array().inverse().transpose();
        }
    }
    Composite::Solver::ProxGBB<> alg;

    model.prox_h_ptr()->options().set_from_cmd_line(argc, argv, "--hopt");
    if (!config.empty())
        alg.options().set_from_file(config);
    alg.options().set_from_cmd_line(argc, argv);
    alg.custom_stop_checker = std::make_shared<NZChecker>();

    // main solve process
    auto tstart = tic();
    std::vector<Index> ind_keep;
    ind_keep.reserve(n);
    Index reduction_itr = 0;
    bool phase2 = false;

    // initial solve: probably with continuation
    alg.solve(model, x0);

    // model reduction loop
    while (true){
        Scalar ratio;
        xx_sub = alg.get_sol().mat();

        // recover xx in R^n
        if (xx_sub.rows() == n)
            xx = xx_sub;
        else {
            xx = Mat::Zero(n, 1);
            for (size_t i = 0; i < ind_keep.size(); ++i){
                xx(ind_keep[i]) = xx_sub(i);
            }
        }

        if ((phase2 && (alg.out.flag == 0 || (alg.out.flag == 1 && reduction_itr > 50))) || reduction_itr > 300)
            break;

        // compute gradient gg
        Mat gg(xx.rows(), 1);
        (*model.f)(xx, gg, true);

        // variable reduction: keep the following indices:
        // 1) abs(xx(i)) > eps
        // 2) abs(gg(i)) > ratio * max(min(mu, abs(gg(J)))), where J satisfies 1)
        ind_keep.clear();
        for (Index i = 0; i < xx.rows(); ++i){
            if (std::fabs(xx(i)) > OptSuite::eps)
                ind_keep.push_back(i);
        }
        Scalar threshold = 0;
        for (auto &i : ind_keep){
            Scalar v = std::min(mu, std::fabs(gg(i)));
            if (v > threshold)
                threshold = v;
        }

        if (alg.out.nrmG < 100 * alg.options().get_scalar("gtol")){
            ratio = 0.99;

            for (Index i = 0; i < gg.rows(); ++i){
                if (std::fabs(gg(i)) > ratio * threshold && std::fabs(xx(i)) <= OptSuite::eps)
                    ind_keep.push_back(i);
            }
            phase2 = true;
        }

        // logging
        std::cerr << "Next model dim: " <<
            str_format("%d -> %d", xx_sub.rows(), ind_keep.size()) << std::endl;

        // construct a new model
        Asub.resize(b.rows(), ind_keep.size());
        xx0 = Mat::Zero(ind_keep.size(), 1);
        for (size_t i = 0; i < ind_keep.size(); ++i){
            Asub.col(i) = A.col(ind_keep[i]);
            xx0(i) = xx(ind_keep[i]);
        }

        // generate new model
        model_sub.set_A_b(Asub, b);
        model_sub.prox_h_ptr()->options().set_from_cmd_line(argc, argv, "--hopt");

        // call alg
        alg.options().set_bool("continuation", false);
        alg.solve(model_sub, xx0);

        // increase itr
        ++reduction_itr;
    }

    auto elapsed = toc(tstart);

    Index nnz = 0;
    for (Index i = 0; i < xx.rows(); ++i){
        if (std::fabs(xx(i)) > OptSuite::eps) ++nnz;
    }

    Mat g(n, 1), x_new(n, 1);
    (*model.f)(xx, g, true); // compute g
    (*model.prox_h)(xx - g, 1_s, x_new);

    std::cout << str_format("& %13.6e & %6.2e & %6.2f & %5d",
            alg.out.obj, (xx - x_new).norm(), elapsed, nnz) << std::endl;
    std::cerr << "Elapsed time is " << elapsed << " sec." << std::endl;
    std::cerr << "Solver returned with message " << alg.out.message << std::endl;

    logger.log(xx);

    return 0;
}

int main(int argc, char **argv){
    if (argc < 2){
        std::cerr << help_txt << std::endl;
        return 1;
    }

    if (std::strcmp(argv[1], "synth") == 0)
        return do_synth(argc, argv);
    else if (std::strcmp(argv[1], "real") == 0){
        if (argc >= 3)
            return do_real(argc, argv);
    }

    std::cerr << help_txt << std::endl;
    return 1;
}
