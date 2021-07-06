// libgp - Gaussian process library for Machine Learning
// Copyright (c) 2013, Manuel Blum <mblum@informatik.uni-freiburg.de>
// All rights reserved.

#ifndef __COV_SE_ARD_H__
#define __COV_SE_ARD_H__

#include "cov.h"

namespace libgp
{
  
  /** Squared exponential covariance function with automatic relevance detection.
   *  Computes the squared exponential covariance
   *  \f$k_{SE}(x, y) := \alpha^2 \exp(-\frac{1}{2}(x-y)^T\Lambda^{-1}(x-y))\f$,
   *  with \f$\Lambda = diag(l_1^2, \dots, l_n^2)\f$ being the characteristic
   *  length scales and \f$\alpha\f$ describing the variability of the latent
   *  function. The parameters \f$l_1^2, \dots, l_n^2, \alpha\f$ are expected
   *  in this order in the parameter array.
   *  @ingroup cov_group
   *  @author Manuel Blum
   */
  class CovSEard : public CovarianceFunction
  {
  public:
    CovSEard ();
    virtual ~CovSEard ();
    bool init(int n);
    double get(const Eigen::VectorXd &x1, const Eigen::VectorXd &x2);
    void grad(const Eigen::VectorXd &x1, const Eigen::VectorXd &x2, Eigen::VectorXd &grad);
    void set_loghyper(const Eigen::VectorXd &p);
    virtual std::string to_string();
  private:
    Eigen::VectorXd ell;
    double sf2;
  };
  
}

#endif /* __COV_SE_ARD_H__ */

