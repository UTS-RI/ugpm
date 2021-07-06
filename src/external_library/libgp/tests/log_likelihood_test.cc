// libgp - Gaussian process library for Machine Learning
// Copyright (c) 2011, Manuel Blum <mblum@informatik.uni-freiburg.de>
// All rights reserved.

#include "gp.h"
#include "gp_utils.h"

#include <cmath>
#include <iostream>
#include <gtest/gtest.h>

TEST(LogLikelihoodTest, CheckGradients) 
{
  int input_dim = 3, param_dim = 3;
  libgp::GaussianProcess * gp = new libgp::GaussianProcess(input_dim, "CovSum ( CovSEiso, CovNoise)");
  Eigen::VectorXd params(param_dim);
  params << 0, 0, -2;
  gp->covf().set_loghyper(params);
  size_t n = 500;
  Eigen::MatrixXd X(n, input_dim);
  X.setRandom();
  Eigen::VectorXd y = gp->covf().draw_random_sample(X);
  for(size_t i = 0; i < n; ++i) {
    double x[input_dim];
    for(int j = 0; j < input_dim; ++j) x[j] = X(i,j);
    gp->add_pattern(x, y(i));
  }

  double e = 1e-4;

  Eigen::VectorXd grad = gp->log_likelihood_gradient();
  
  for (int i=0; i<param_dim; ++i) {
    double theta = params(i);
    params(i) = theta - e;
    gp->covf().set_loghyper(params);
    double j1 = gp->log_likelihood();
    params(i) = theta + e;
    gp->covf().set_loghyper(params);
    double j2 = gp->log_likelihood();
    params(i) = theta;
    ASSERT_NEAR((j2-j1)/(2*e), grad(i), 1e-5);
  }

  delete gp;
}

