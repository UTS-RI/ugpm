// libgp - Gaussian process library for Machine Learning
// Copyright (c) 2013, Manuel Blum <mblum@informatik.uni-freiburg.de>
// All rights reserved.

#include "cov_matern5_iso.h"
#include <cmath>

namespace libgp
{
  
  CovMatern5iso::CovMatern5iso() {}
  
  CovMatern5iso::~CovMatern5iso() {}
  
  bool CovMatern5iso::init(int n)
  {
    input_dim = n;
    param_dim = 2;
    loghyper.resize(param_dim);
    loghyper.setZero();
    sqrt5 = sqrt(5);
    return true;
  }
  
  double CovMatern5iso::get(const Eigen::VectorXd &x1, const Eigen::VectorXd &x2)
  {
    double z = ((x1-x2)*sqrt5/ell).norm();
    return sf2*exp(-z)*(1+z+z*z/3);
  }
  
  void CovMatern5iso::grad(const Eigen::VectorXd &x1, const Eigen::VectorXd &x2, Eigen::VectorXd &grad)
  {
    double z = ((x1-x2)*sqrt5/ell).norm();
    double k = sf2*exp(-z);
    double z_square = z*z;
    grad << k*(z_square + z_square*z)/3, 2*k*(1+z+z_square/3);
  }
  
  void CovMatern5iso::set_loghyper(const Eigen::VectorXd &p)
  {
    CovarianceFunction::set_loghyper(p);
    ell = exp(loghyper(0));
    sf2 = exp(2*loghyper(1));
  }
  
  std::string CovMatern5iso::to_string()
  {
    return "CovMatern5iso";
  }
  
}
