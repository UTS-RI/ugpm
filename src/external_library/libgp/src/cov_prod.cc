// libgp - Gaussian process library for Machine Learning
// Copyright (c) 2013, Manuel Blum <mblum@informatik.uni-freiburg.de>
// All rights reserved.

#include "cov_prod.h"
#include "cmath"

namespace libgp
{
  
  CovProd::CovProd()
  {
  }
  
  CovProd::~CovProd()
  {
    delete first;
    delete second;
  }
  
  bool CovProd::init(int n, CovarianceFunction * first, CovarianceFunction * second)
  {
    this->input_dim = n;
    this->first = first;
    this->second = second;
    param_dim_first = first->get_param_dim();
    param_dim_second = second->get_param_dim();
    param_dim = param_dim_first + param_dim_second;
    loghyper.resize(param_dim);
    loghyper.setZero();
    return true;
  }
  
  double CovProd::get(const Eigen::VectorXd &x1, const Eigen::VectorXd &x2)
  {
    return first->get(x1, x2) * second->get(x1, x2);
  }
  
  void CovProd::grad(const Eigen::VectorXd &x1, const Eigen::VectorXd &x2, Eigen::VectorXd &grad)
  {
    Eigen::VectorXd grad_first(param_dim_first);
    Eigen::VectorXd grad_second(param_dim_second);
    first->grad(x1, x2, grad_first);
    second->grad(x1, x2, grad_second);
    grad.head(param_dim_first) = grad_first * second->get(x1, x2);
    grad.tail(param_dim_second) = grad_second * first->get(x1, x2);
  }
  
  void CovProd::set_loghyper(const Eigen::VectorXd &p)
  {
    CovarianceFunction::set_loghyper(p);
    first->set_loghyper(p.head(param_dim_first));
    second->set_loghyper(p.tail(param_dim_second));
  }
  
  std::string CovProd::to_string()
  {
    return "CovProd("+first->to_string()+", "+second->to_string()+")";
  }
}
