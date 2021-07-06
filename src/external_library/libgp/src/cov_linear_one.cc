// libgp - Gaussian process library for Machine Learning
// Copyright (c) 2013, Manuel Blum <mblum@informatik.uni-freiburg.de>
// All rights reserved.

#include "cov_linear_one.h"
#include <cmath>

namespace libgp
{
  
  CovLinearone::CovLinearone() {}
  
  CovLinearone::~CovLinearone() {}
  
  bool CovLinearone::init(int n)
  {
    input_dim = n;
    param_dim = 1;
    loghyper.resize(param_dim);
    loghyper.setZero();
    return true;
  }
  
  double CovLinearone::get(const Eigen::VectorXd &x1, const Eigen::VectorXd &x2)
  {
    return it2*(1+x1.dot(x2));
  }
  
  void CovLinearone::grad(const Eigen::VectorXd &x1, const Eigen::VectorXd &x2, Eigen::VectorXd &grad)
  {
    grad << -2*it2*(1+x1.dot(x2));
  }
  
  void CovLinearone::set_loghyper(const Eigen::VectorXd &p)
  {
    CovarianceFunction::set_loghyper(p);
    it2 = exp(-2*loghyper(0));
  }
  
  std::string CovLinearone::to_string()
  {
    return "CovLinearone";
  }
  
}
