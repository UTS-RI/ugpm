// libgp - Gaussian process library for Machine Learning
// Copyright (c) 2011, Manuel Blum <mblum@informatik.uni-freiburg.de>
// All rights reserved.

#ifndef __GP_SPARSE_H__
#define __GP_SPARSE_H__

#include <Eigen/Sparse>
#include <Eigen/SparseCholesky>

#include "gp.h"

namespace libgp {
  
  /** Sparse Gaussian process regression.
   *  @author Manuel Blum */
  class SparseGaussianProcess : public GaussianProcess
  {
  public:
    
    /** Create an instance of SparseGaussianProcess with given input dimensionality and covariance function. */
    SparseGaussianProcess (size_t input_dim, std::string covf_def);
    
    /** Create an instance of SparseGaussianProcess from file. */
    SparseGaussianProcess (const char * filename);
    
    virtual ~SparseGaussianProcess ();
    
    virtual void compute();

  protected:
    
  };
}

#endif /* __GP_SPARSE_H__ */
