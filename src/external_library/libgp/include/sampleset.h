// libgp - Gaussian process library for Machine Learning
// Copyright (c) 2013, Manuel Blum <mblum@informatik.uni-freiburg.de>
// All rights reserved.

#ifndef __SAMPLESET_H__
#define __SAMPLESET_H__

#include <Eigen/Dense>
#include <vector>

namespace libgp {
  
  /** Container holding training patterns.
   *  @author Manuel Blum */
  class SampleSet
  {
  public:
    /** Constructor.
     *  @param input_dim dimensionality of input vectors */
    SampleSet (int input_dim);

    /** Copy constructor */
    SampleSet ( const SampleSet& ss );

    /** Destructor. */    
    virtual ~SampleSet();
    
    /** Add input-output pattern to sample set.
     *  @param x input array
     *  @param y target value */
    void add(const double x[], double y);
    void add(const Eigen::VectorXd x, double y);
    
    /** Get input vector at index k. */
    const Eigen::VectorXd & x (size_t k);

    /** Get target value at index k. */
    double y (size_t k);

    /** Set target value at index i. */
    bool set_y(size_t i, double y);

    /** Get reference to vector of target values. */
    const std::vector<double>& y();
    
    /** Get number of samples. */
    size_t size();
    
    /** Clear sample set. */
    void clear();
    
    /** Check if sample set is empty. */
    bool empty ();


  private:

    /** Container holding input vectors. */
    std::vector<Eigen::VectorXd *> inputs;
    
    /** Container holding target values. */
    std::vector<double> targets;
    
    /** Dimensionality of input vectors. */
    size_t input_dim;
    
    /** Number of samples. */
    size_t n;
  };
}

#endif /* __SAMPLESET_H__ */
