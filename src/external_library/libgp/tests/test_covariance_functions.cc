// libgp - Gaussian process library for Machine Learning
// Copyright (c) 2011, Manuel Blum <mblum@informatik.uni-freiburg.de>
// All rights reserved.

#include "cov_factory.h"

#include <Eigen/Dense>
#include <gtest/gtest.h>

#if GTEST_HAS_PARAM_TEST

using ::testing::TestWithParam;
using ::testing::Values;

class GradientTest : public TestWithParam<std::string> {
  protected:
    virtual void SetUp() {
      n = 3;
      e = 1e-8;
      covf = factory.create(n, GetParam());
      param_dim = covf->get_param_dim();
      params = Eigen::VectorXd::Random(param_dim);
      x1 = Eigen::VectorXd::Random(n);
      x2 = Eigen::VectorXd::Random(n);
      covf->set_loghyper(params);
    }
    virtual void TearDown() {
      delete covf;
    }
    int n, param_dim;
    libgp::CovFactory factory;
    libgp::CovarianceFunction * covf;
    double e;    
    Eigen::VectorXd params;
    Eigen::VectorXd x1;
    Eigen::VectorXd x2; 
    Eigen::VectorXd gradient() {
      Eigen::VectorXd grad(param_dim);
      covf->grad(x1, x2, grad);
      return grad;
    }
    double numerical_gradient(int i) {
      double theta = params(i);
      params(i) = theta - e;
      covf->set_loghyper(params);
      double j1 = covf->get(x1, x2);
      params(i) = theta + e;
      covf->set_loghyper(params);
      double j2 = covf->get(x1, x2);
      params(i) = theta;
      return (j2-j1)/(2*e);
    }
};

TEST_P(GradientTest, EqualToNumerical) {
  Eigen::VectorXd grad = gradient();
  for (int i=0; i<param_dim; ++i) {
    if (grad(i) == 0.0) ASSERT_NEAR(numerical_gradient(i), 0.0, 1e-2);
    else ASSERT_NEAR((numerical_gradient(i)-grad(i))/grad(i), 0.0, 1e-2);
  }
}

INSTANTIATE_TEST_CASE_P(CovarianceFunction, GradientTest, Values(
          "CovLinearard",
          "CovLinearone",
          "CovMatern3iso",
          "CovMatern5iso",
          "CovNoise",
          "CovProd(CovSEiso, CovMatern3iso)",
          "CovRQiso",
          "CovSEard",
          "CovSEiso",
          "CovSum(CovSEiso, CovNoise)",
          "CovSum(CovLinearard, CovNoise)",
          "InputDimFilter(1/CovSEiso)",
          "InputDimFilter(0/CovSum(CovSEiso, CovNoise))"
          ));

TEST(FilterTest, EqualToVector) {



}


#else

// Google Test may not support value-parameterized tests with some
// compilers. If we use conditional compilation to compile out all
// code referring to the gtest_main library, MSVC linker will not link
// that library at all and consequently complain about missing entry
// point defined in that library (fatal error LNK1561: entry point
// must be defined). This dummy test keeps gtest_main linked in.
TEST(DummyTest, ValueParameterizedTestsAreNotSupportedOnThisPlatform) {}

#endif  // GTEST_HAS_PARAM_TEST
