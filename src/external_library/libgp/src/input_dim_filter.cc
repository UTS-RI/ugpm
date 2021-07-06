// libgp - Gaussian process library for Machine Learning
// Copyright (c) 2013, Manuel Blum <mblum@informatik.uni-freiburg.de>
// All rights reserved.

#include "input_dim_filter.h"
#include <cmath>
#include <cassert>

namespace libgp
{
  
  InputDimFilter::InputDimFilter() {}
  
  InputDimFilter::~InputDimFilter() {}
  
  bool InputDimFilter::init(int input_dim, int filter, CovarianceFunction * covf)
  {
    this->input_dim = input_dim;
    this->nested = covf;
    this->param_dim = nested->get_param_dim();
    this->filter = filter;
    assert(filter < input_dim && filter >=0);
    loghyper.resize(param_dim);
    return true;
  }
  
  double InputDimFilter::get(const Eigen::VectorXd &x1, const Eigen::VectorXd &x2)
  {
    return nested->get(x1.segment(filter, 1), x2.segment(filter, 1));
  }
  
  void InputDimFilter::grad(const Eigen::VectorXd &x1, const Eigen::VectorXd &x2, Eigen::VectorXd &grad)
  {
    nested->grad(x1.segment(filter, 1), x2.segment(filter, 1), grad);
  }
  
  void InputDimFilter::set_loghyper(const Eigen::VectorXd &p)
  {
    CovarianceFunction::set_loghyper(p);
    nested->set_loghyper(p);
  }
  
  std::string InputDimFilter::to_string()
  {
    std::ostringstream is;
    is <<  "InputDimFilter(" << filter << "/" << nested->to_string() << ")";
    return is.str();
  }
}

