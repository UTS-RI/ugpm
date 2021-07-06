// libgp - Gaussian process library for Machine Learning
// Copyright (c) 2013, Manuel Blum <mblum@informatik.uni-freiburg.de>
// All rights reserved.

#include "cov_rq_iso.h"
#include <cmath>

namespace libgp
{
  
  CovRQiso::CovRQiso() {}
  
  CovRQiso::~CovRQiso() {}
  
  bool CovRQiso::init(int n)
  {
    input_dim = n;
    param_dim = 3;
    loghyper.resize(param_dim);
    loghyper.setZero();
    return true;
  }
  
  double CovRQiso::get(const Eigen::VectorXd &x1, const Eigen::VectorXd &x2)
  {
    double z = ((x1-x2)/ell).squaredNorm();
    return sf2*pow(1+0.5*z/alpha, -alpha);
  }
  
  void CovRQiso::grad(const Eigen::VectorXd &x1, const Eigen::VectorXd &x2, Eigen::VectorXd &grad)
  {
    double z = ((x1-x2)/ell).squaredNorm();
    double k = 1+0.5*z/alpha;
    double sf2_k = sf2*pow(k, -alpha);
    grad << sf2*z*pow(k, -alpha-1), 2*sf2_k, sf2_k*(0.5*z/k-alpha*log(k));
  }
  
  void CovRQiso::set_loghyper(const Eigen::VectorXd &p)
  {
    CovarianceFunction::set_loghyper(p);
    ell = exp(loghyper(0));
    sf2 = exp(2*loghyper(1));
    alpha = exp(loghyper(2));
  }
  
  std::string CovRQiso::to_string()
  {
    return "CovRQiso";
  }
  
}
