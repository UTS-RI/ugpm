// libgp - Gaussian process library for Machine Learning
// Copyright (c) 2013, Manuel Blum <mblum@informatik.uni-freiburg.de>
// All rights reserved.

#ifndef __COV_PERIODIC_MATERN3_ISO_H__
#define __COV_PERIODIC_MATERN3_ISO_H__

#include "cov.h"

namespace libgp
{
  class CovPeriodicMatern3iso : public CovarianceFunction
  {
  public:
    CovPeriodicMatern3iso ();
    virtual ~CovPeriodicMatern3iso ();
    bool init(int n);
    double get(const Eigen::VectorXd &x1, const Eigen::VectorXd &x2);
    void grad(const Eigen::VectorXd &x1, const Eigen::VectorXd &x2, Eigen::VectorXd &grad);
    void set_loghyper(const Eigen::VectorXd &p);
    virtual std::string to_string();
  private:
    double ell;
    double sf2;
    double sqrt3;
    double T;
  };
  
}

#endif
