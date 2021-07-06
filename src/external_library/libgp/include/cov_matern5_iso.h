// libgp - Gaussian process library for Machine Learning
// Copyright (c) 2013, Manuel Blum <mblum@informatik.uni-freiburg.de>
// All rights reserved.

#ifndef __COV_MATERN5_ISO_H__
#define __COV_MATERN5_ISO_H__

#include "cov.h"

namespace libgp
{
  /** Matern covariance function with \f$\nu = \frac{5}{2}\f$ and isotropic distance measure.
   *  Parameter: \f$l^2, \sigma_f^2\f$
   *  @ingroup cov_group
   *  @author Manuel Blum
   */
  class CovMatern5iso : public CovarianceFunction
  {
  public:
    CovMatern5iso ();
    virtual ~CovMatern5iso ();
    bool init(int n);
    double get(const Eigen::VectorXd &x1, const Eigen::VectorXd &x2);
    void grad(const Eigen::VectorXd &x1, const Eigen::VectorXd &x2, Eigen::VectorXd &grad);
    void set_loghyper(const Eigen::VectorXd &p);
    virtual std::string to_string();
  private:
    double ell;
    double sf2;
    double sqrt5;
  };
  
}

#endif /* __COV_MATERN5_ISO_H__ */
