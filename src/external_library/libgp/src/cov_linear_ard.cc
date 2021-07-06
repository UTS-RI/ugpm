// libgp - Gaussian process library for Machine Learning
// Copyright (c) 2013, Manuel Blum <mblum@informatik.uni-freiburg.de>
// All rights reserved.

#include "cov_linear_ard.h"
#include <cmath>

namespace libgp
{
  
  CovLinearard::CovLinearard() {}
  
  CovLinearard::~CovLinearard() {}
  
  bool CovLinearard::init(int n)
  {
    input_dim = n;
    param_dim = n;
    ell.resize(input_dim);
    loghyper.resize(param_dim);
    return true;
  }
  
  double CovLinearard::get(const Eigen::VectorXd &x1, const Eigen::VectorXd &x2)
  { 
    return x1.cwiseQuotient(ell).dot(x2.cwiseQuotient(ell));
  }
  
  void CovLinearard::grad(const Eigen::VectorXd &x1, const Eigen::VectorXd &x2, Eigen::VectorXd &grad)
  {
    grad = -2*x1.cwiseQuotient(ell).cwiseProduct(x2.cwiseQuotient(ell));
  }
  
  void CovLinearard::set_loghyper(const Eigen::VectorXd &p)
  {
    CovarianceFunction::set_loghyper(p);
    for(size_t i = 0; i < input_dim; ++i) ell(i) = exp(loghyper(i));
  }
  
  std::string CovLinearard::to_string()
  {
    return "CovLinearard";
  }
}

