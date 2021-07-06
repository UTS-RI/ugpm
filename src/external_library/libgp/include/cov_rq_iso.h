// libgp - Gaussian process library for Machine Learning
// Copyright (c) 2013, Manuel Blum <mblum@informatik.uni-freiburg.de>
// All rights reserved.

#ifndef __COV_RQ_ISO_H__
#define __COV_RQ_ISO_H__

#include "cov.h"

namespace libgp
{
  /** Isotropic rational quadratic covariance function.
   *  Parameters: \f$l^2, \sigma_f^2, \alpha\f$
   *  @ingroup cov_group
   *  @author Manuel Blum
   */
  class CovRQiso : public CovarianceFunction
  {
  public:
    CovRQiso ();
    virtual ~CovRQiso ();
    bool init(int n);
    double get(const Eigen::VectorXd &x1, const Eigen::VectorXd &x2);
    void grad(const Eigen::VectorXd &x1, const Eigen::VectorXd &x2, Eigen::VectorXd &grad);
    void set_loghyper(const Eigen::VectorXd &p);
    virtual std::string to_string();
  private:
    double ell;
    double sf2;
    double alpha;
  };
  
}

#endif /* __COV_RQ_ISO_H__ */
