// libgp - Gaussian process library for Machine Learning
// Copyright (c) 2011, Manuel Blum <mblum@informatik.uni-freiburg.de>
// All rights reserved.

#include "gp.h"
#include "gp_utils.h"

#include <cmath>
#include <iostream>
#include <gtest/gtest.h>

double test_gp_regression(libgp::GaussianProcess * gp)
{
  int input_dim = gp->get_input_dim();
  int n = libgp::Utils::randi(400)+100;
  Eigen::MatrixXd X(n, input_dim);
  X.setRandom();
  Eigen::VectorXd y = gp->covf().draw_random_sample(X);
  for(size_t i = 0; i < n*0.8; ++i) {
    double x[input_dim];
    for(int j = 0; j < input_dim; ++j) x[j] = X(i,j);
    gp->add_pattern(x, y(i));
  }
  double tss = 0;
  for(int i = n*0.8+1; i < n; ++i) {
    double x[input_dim];
    for(int j = 0; j < input_dim; ++j) x[j] = X(i,j);
    double f = gp->f(x);
    double error = f - y(i);
    tss += error*error;
  }
  return tss/(n*0.2-1);
}

void run_regression_test(std::string covf_str)
{
  double mss = 0.0;
  int n=20;
  for (int i=0; i<n; ++i) {
    int input_dim = libgp::Utils::randi(2) + 2;
    libgp::GaussianProcess * gp = new libgp::GaussianProcess(input_dim, covf_str);
    Eigen::VectorXd params(gp->covf().get_param_dim());
    params.setZero();
    params(gp->covf().get_param_dim()-1) = -2;
    gp->covf().set_loghyper(params);
    mss += test_gp_regression(gp);    
    delete gp;
  }
  ASSERT_TRUE(mss/n < 0.05);
}

TEST(GPRegressionTest, SEiso) {
  std::string covf_str("CovSum ( CovSEiso, CovNoise)");
  run_regression_test(covf_str);
}

TEST(GPRegressionTest, Matern3iso) {
  std::string covf_str("CovSum ( CovMatern3iso, CovNoise)");
  run_regression_test(covf_str);
}

TEST(GPRegressionTest, Matern5iso) {
  std::string covf_str("CovSum ( CovMatern5iso, CovNoise)");
  run_regression_test(covf_str);
}

TEST(GPRegressionTest, CovSEard) {
  std::string covf_str("CovSum ( CovSEard, CovNoise)");
  run_regression_test(covf_str);
}

TEST(GPRegressionTest, CovRQiso) {
  std::string covf_str("CovSum ( CovRQiso, CovNoise)");
  run_regression_test(covf_str);
}

TEST(GPRegressionTest, UpdateL) {
  int input_dim = 2;
  libgp::GaussianProcess * gp = new libgp::GaussianProcess(input_dim, "CovSum ( CovSEiso, CovNoise)");
  Eigen::VectorXd params(gp->covf().get_param_dim());
  params << 0, 0, -2;
  gp->covf().set_loghyper(params);
  size_t n = 10;
  Eigen::MatrixXd X(n, input_dim);
  X.setRandom();
  Eigen::VectorXd y = gp->covf().draw_random_sample(X);
  for(size_t i = 0; i < n; ++i) {
    double x[2];
    for(int j = 0; j < input_dim; ++j) x[j] = X(i,j);
    gp->add_pattern(x, y(i));
  }
  double x[2] = {0,0};
  gp->f(x);
}
