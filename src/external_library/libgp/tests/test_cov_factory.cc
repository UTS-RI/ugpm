// libgp - Gaussian process library for Machine Learning
// Copyright (c) 2013, Manuel Blum <mblum@informatik.uni-freiburg.de>
// All rights reserved.

#include "cov_factory.h"

#include <Eigen/Dense>
#include <gtest/gtest.h>

TEST(CovFactoryTest, Parser) {
  
  libgp::CovFactory factory;
  libgp::CovarianceFunction * covf;

  covf = factory.create(4, "CovSum ( CovSum(CovLinearone, CovNoise), CovMatern3iso)");
  ASSERT_EQ(covf->to_string().compare("CovSum(CovSum(CovLinearone, CovNoise), CovMatern3iso)"), 0);
  delete covf;

  covf = factory.create(4, "CovSum(CovMatern3iso, CovSum(CovLinearone, CovNoise))");
  ASSERT_EQ(covf->to_string().compare("CovSum(CovMatern3iso, CovSum(CovLinearone, CovNoise))"), 0);
  delete covf;

  covf = factory.create(4, "CovSum(CovLinearone,CovNoise)"); 
  ASSERT_EQ(covf->to_string().compare("CovSum(CovLinearone, CovNoise)"), 0);
  delete covf;

  covf = factory.create(4, "CovSum ( CovLinearone , CovNoise )"); 
  ASSERT_EQ(covf->to_string().compare("CovSum(CovLinearone, CovNoise)"), 0);
  delete covf;

  covf = factory.create(4, "CovLinearone"); 
  ASSERT_EQ(covf->to_string().compare("CovLinearone"), 0);
  delete covf;

  covf = factory.create(4, "InputDimFilter(2/CovSEiso)"); 
  ASSERT_EQ(covf->to_string().compare("InputDimFilter(2/CovSEiso)"), 0);
  delete covf;
}

