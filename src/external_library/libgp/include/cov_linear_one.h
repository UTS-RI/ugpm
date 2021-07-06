// libgp - Gaussian process library for Machine Learning
// Copyright (c) 2013, Manuel Blum <mblum@informatik.uni-freiburg.de>
// All rights reserved.

#ifndef __COV_LINEAR_ONE__
#define __COV_LINEAR_ONE__

#include "cov.h"

namespace libgp
{
  
  /** Linear covariance function.
   *  Parameter: \f$\theta^2\f$
   *  @ingroup cov_group
   *  @author Manuel Blum
   */
  class CovLinearone : public CovarianceFunction
  {
  public:
    CovLinearone ();
    virtual ~CovLinearone ();
    bool init(int n);
    double get(const Eigen::VectorXd &x1, const Eigen::VectorXd &x2);
    void grad(const Eigen::VectorXd &x1, const Eigen::VectorXd &x2, Eigen::VectorXd &grad);
    void set_loghyper(const Eigen::VectorXd &p);
    virtual std::string to_string();
  private:
    double it2;
  };
  
}

#endif /* __COV_LINEAR_ONE__ */
