// libgp - Gaussian process library for Machine Learning
// Copyright (c) 2013, Manuel Blum <mblum@informatik.uni-freiburg.de>
// All rights reserved.

#include "cov.h"
#include "gp_utils.h"

namespace libgp
{
  
  size_t CovarianceFunction::get_param_dim()
  {
    return param_dim;
  }
  
  size_t CovarianceFunction::get_input_dim()
  {
    return input_dim;
  }
  
  Eigen::Vector3d CovarianceFunction::get_loghyper3()
  {
    Eigen::Vector3d out;
    out << loghyper(0), loghyper(1), loghyper(2);
    return out;
  }
  Eigen::VectorXd CovarianceFunction::get_loghyper()
  {
    return loghyper;
  }
  
  void CovarianceFunction::set_loghyper(const Eigen::VectorXd &p)
  {
    assert(p.size() == loghyper.size());
    loghyper = p;
    loghyper_changed = true;
  }
  
  void CovarianceFunction::set_loghyper(const double p[])
  {
    Eigen::Map<const Eigen::VectorXd> p_vec_map(p, param_dim);
    set_loghyper(p_vec_map);
  }

  
  Eigen::VectorXd CovarianceFunction::draw_random_sample(Eigen::MatrixXd &X)
  {
    assert (X.cols() == int(input_dim));  
    int n = X.rows();
    Eigen::MatrixXd K(n, n);
    Eigen::LLT<Eigen::MatrixXd> solver;
    Eigen::VectorXd y(n);
    // compute kernel matrix (lower triangle)
    for(int i = 0; i < n; ++i) {
      for(int j = i; j < n; ++j) {
        K(j, i) = get(X.row(j), X.row(i));
      }
      y(i) = Utils::randn();
    }
    // perform cholesky factorization
    solver = K.llt();  
    return solver.matrixL() * y;
  }
}
