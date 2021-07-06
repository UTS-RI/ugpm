// libgp - Gaussian process library for Machine Learning
// Copyright (c) 2013, Manuel Blum <mblum@informatik.uni-freiburg.de>
// All rights reserved.

#ifndef __COV_INPUT_DIM_FILTER__
#define __COV_INPUT_DIM_FILTER__

#include "cov.h"

namespace libgp
{
  
  /** Linear covariance function.
   *  @ingroup cov_group
   *  @author Manuel Blum
   */
  class InputDimFilter : public CovarianceFunction
  {
  public:
    InputDimFilter ();
    virtual ~InputDimFilter ();
    bool init(int input_dim, int filter, CovarianceFunction * covf);
    double get(const Eigen::VectorXd &x1, const Eigen::VectorXd &x2);
    void grad(const Eigen::VectorXd &x1, const Eigen::VectorXd &x2, Eigen::VectorXd &grad);
    void set_loghyper(const Eigen::VectorXd &p);
    virtual std::string to_string();
  private:
    int filter;
    CovarianceFunction *nested;
  };
  
}

#endif /* __COV_LINEAR_ONE__ */
