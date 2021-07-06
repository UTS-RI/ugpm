// libgp - Gaussian process library for Machine Learning
// Copyright (c) 2013, Manuel Blum <mblum@informatik.uni-freiburg.de>
// All rights reserved.

#ifndef __COV_NOISE_H__
#define __COV_NOISE_H__

#include "cov.h"

namespace libgp
{
  
  /** Independent covariance function (white noise). 
   *  Parameters: signal noise, \f$\sigma^2\f$
   *  @author Manuel Blum
   *  @ingroup cov_group
   */
  class CovNoise : public CovarianceFunction
  {
  public:
    CovNoise ();
    virtual ~CovNoise ();
    bool init(int n);
    double get(const Eigen::VectorXd &x1, const Eigen::VectorXd &x2);
    void grad(const Eigen::VectorXd &x1, const Eigen::VectorXd &x2, Eigen::VectorXd &grad);
    void set_loghyper(const Eigen::VectorXd &p);
    virtual std::string to_string();
    virtual double get_threshold();
    virtual void set_threshold(double threshold);
  private:
    double s2;
  };
  
}

#endif /* __COV_NOISE_H__ */
