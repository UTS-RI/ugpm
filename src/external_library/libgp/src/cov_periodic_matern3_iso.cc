// libgp - Gaussian process library for Machine Learning
// Copyright (c) 2013, Manuel Blum <mblum@informatik.uni-freiburg.de>
// All rights reserved.

#include "cov_periodic_matern3_iso.h"
#include <cmath>

namespace libgp
{
  
  CovPeriodicMatern3iso::CovPeriodicMatern3iso() {}
  
  CovPeriodicMatern3iso::~CovPeriodicMatern3iso() {}
  
  bool CovPeriodicMatern3iso::init(int n)
  {
    input_dim = n;
    param_dim = 3;
    loghyper.resize(param_dim);
    loghyper.setZero();
    sqrt3 = sqrt(3);
    return true;
  }
  
  double CovPeriodicMatern3iso::get(const Eigen::VectorXd &x1, const Eigen::VectorXd &x2)
  {
    double s = sqrt3*fabs((sin(M_PI * (x1-x2).norm() / T) / ell));
    return sf2*(1+s)*exp(-s);
  }
  
  void CovPeriodicMatern3iso::grad(const Eigen::VectorXd &x1, const Eigen::VectorXd &x2, Eigen::VectorXd &grad)
  {
    double k = M_PI * (x1-x2).norm() / T;
    double s = sqrt3*fabs((sin(k) / ell));
    grad << sf2*s*s*exp(-s), 2*sf2*(1+s)*exp(-s), sf2*exp(-s)*s*sqrt3*k*cos(k)/ell/T;
  }
  
  void CovPeriodicMatern3iso::set_loghyper(const Eigen::VectorXd &p)
  {
    CovarianceFunction::set_loghyper(p);
    ell = exp(loghyper(0));
    sf2 = exp(2*loghyper(1));
    T = loghyper(2);
  }
  
  std::string CovPeriodicMatern3iso::to_string()
  {
    return "CovPeriodicMatern3iso";
  }
  
}
