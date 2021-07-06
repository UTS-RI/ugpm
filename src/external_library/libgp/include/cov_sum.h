// libgp - Gaussian process library for Machine Learning
// Copyright (c) 2013, Manuel Blum <mblum@informatik.uni-freiburg.de>
// All rights reserved.

#ifndef __COV_SUM_H__
#define __COV_SUM_H__

#include "cov.h"

namespace libgp
{
  /** Sums of covariance functions.
   *  @author Manuel Blum 
   *  @ingroup cov_group */
  class CovSum : public CovarianceFunction
  {
  public:
    CovSum ();
    virtual ~CovSum ();
    bool init(int n, CovarianceFunction * first, CovarianceFunction * second);
    double get(const Eigen::VectorXd &x1, const Eigen::VectorXd &x2);
    void grad(const Eigen::VectorXd &x1, const Eigen::VectorXd &x2, Eigen::VectorXd &grad);
    void set_loghyper(const Eigen::VectorXd &p);
    virtual std::string to_string();
  private:
    size_t param_dim_first;
    size_t param_dim_second;
    CovarianceFunction * first;
    CovarianceFunction * second;
  };
  
}

#endif /* __COV_SUM_H__ */
