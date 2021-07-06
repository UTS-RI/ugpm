// libgp - Gaussian process library for Machine Learning
// Copyright (c) 2013, Manuel Blum <mblum@informatik.uni-freiburg.de>
// All rights reserved.

#include "sampleset.h"
#include <Eigen/StdVector>

namespace libgp {
  
  SampleSet::SampleSet (int input_dim)
  {
    this->input_dim = input_dim;
    n = 0;
  }
  
  SampleSet::SampleSet ( const SampleSet& ss )
  {
    // shallow copies
    n = ss.n;
    input_dim = ss.input_dim;
    targets = ss.targets;

    // deep copy needed for vector of pointers
    for (size_t i=0; i<ss.inputs.size(); ++i)
    {
      Eigen::VectorXd * sample_to_store = new Eigen::VectorXd(input_dim);
      *sample_to_store = *ss.inputs.at(i);
      inputs.push_back(sample_to_store);
    }
  }

  SampleSet::~SampleSet() 
  {
    clear();
  }
  
  void SampleSet::add(const double x[], double y)
  {
    Eigen::VectorXd * v = new Eigen::VectorXd(input_dim);
    for (size_t i=0; i<input_dim; ++i) (*v)(i) = x[i];
    inputs.push_back(v);
    targets.push_back(y);
    assert(inputs.size()==targets.size());
    n = inputs.size();
  }
  
  void SampleSet::add(const Eigen::VectorXd x, double y)
  {
    Eigen::VectorXd * v = new Eigen::VectorXd(x);
    inputs.push_back(v);
    targets.push_back(y);
    assert(inputs.size()==targets.size());
    n = inputs.size();
  }
  
  const Eigen::VectorXd & SampleSet::x(size_t k)
  {
    return *inputs.at(k);
  }

  double SampleSet::y(size_t k)
  {
    return targets.at(k);
  }

  const std::vector<double>& SampleSet::y() 
  {
    return targets;
  }

  bool SampleSet::set_y(size_t i, double y)
  {
    if (i>=n) return false;
    targets[i] = y;
    return true;
  }
  
  size_t SampleSet::size()
  {
    return n;
  }
  
  void SampleSet::clear()
  {
    while (!inputs.empty()) {
      delete inputs.back();
      inputs.pop_back();
    }    
    n = 0;
    targets.clear();
  }
  
  bool SampleSet::empty ()
  {
    return n==0;
  }
}
