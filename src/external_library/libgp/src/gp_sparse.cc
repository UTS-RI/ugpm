// libgp - Gaussian process library for Machine Learning
// Copyright (c) 2013, Manuel Blum <mblum@informatik.uni-freiburg.de>
// All rights reserved.

#include "gp_sparse.h"

namespace libgp {
  
  SparseGaussianProcess::SparseGaussianProcess (size_t input_dim, std::string covf_def) : GaussianProcess(input_dim, covf_def) {}
  
  SparseGaussianProcess::SparseGaussianProcess (const char * filename) : GaussianProcess(filename) {}
  
  SparseGaussianProcess::~SparseGaussianProcess () {}  
  
  void SparseGaussianProcess::compute()
  {    
    if (cf->get_threshold() == INFINITY) {
      std::cerr << "warning: no threshold defined, computation will be slow." << std::endl
        << "Use full GP or define distance threshold!" << std::endl;
    }
    if (sampleset->empty()) return; 
    Eigen::SparseMatrix<double> K(sampleset->size(), sampleset->size());
    alpha.resize(sampleset->size());
    // compute kernel matrix (lower triangle)
    for(size_t i = 0; i < sampleset->size(); ++i) {
      K.startVec(i);
      for(size_t j = i; j < sampleset->size(); ++j) {
        double cov = cf->get(sampleset->x(i), sampleset->x(j));
        if (cov != 0) K.insertBack(j,i) = cov;
      }
      alpha(i) = sampleset->y(i);
    }
    K.finalize();
    // perform cholesky factorization
    solver.compute(K);
    alpha = solver.solve(alpha);
  }
  
}
