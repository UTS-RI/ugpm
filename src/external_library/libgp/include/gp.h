// libgp - Gaussian process library for Machine Learning
// Copyright (c) 2013, Manuel Blum <mblum@informatik.uni-freiburg.de>
// All rights reserved.

/*! 
 *  
 *   \page licence Licensing
 *    
 *     libgp - Gaussian process library for Machine Learning
 *
 *      \verbinclude "../COPYING"
 */

#ifndef __GP_H__
#define __GP_H__

#define _USE_MATH_DEFINES
#include <cmath>
#include <Eigen/Dense>

#include "cov.h"
#include "sampleset.h"

namespace libgp {
  
  /** Gaussian process regression.
   *  @author Manuel Blum */
  class GaussianProcess
  {
  public:

    /** Empty initialization */
    GaussianProcess ();
    
    /** Create and instance of GaussianProcess with given input dimensionality 
     *  and covariance function. */
    GaussianProcess (size_t input_dim, std::string covf_def);
    
    /** Create and instance of GaussianProcess from file. */
    GaussianProcess (const char * filename);
    
    /** Copy constructor */
    GaussianProcess (const GaussianProcess& gp);
    
    virtual ~GaussianProcess ();
    
    /** Write current gp model to file. */
    void write(const char * filename);
    
    /** Predict target value for given input.
     *  @param x input vector
     *  @return predicted value */
    virtual double f(const double x[]);
    
    /** MODIFICATION CEDRIC 08/09/2019 Predict the integral of target value for given interval.
     *  @param a Lower limit of the integral interval
     *  @param b Upper limit of the integral interval
     *  @return predicted value */
    virtual double Integral(const double a, const double b);

    /** MODIFICATION CEDRIC 08/09/2019 Predict the double integral of target value for given interval.
     *  @param a Lower limit of the integral interval
     *  @param b Upper limit of the integral interval
     *  @return predicted value */
    virtual double Integral2(const double a, const double b);


    /** Predict variance of prediction for given input.
     *  @param x input vector
     *  @return predicted variance */
    virtual double var(const double x[]);
    
    /** Add input-output-pair to sample set.
     *  Add a copy of the given input-output-pair to sample set.
     *  @param x input array
     *  @param y output value
     */
    void add_pattern(const double x[], double y);


    bool set_y(size_t i, double y);

    /** Get number of samples in the training set. */
    size_t get_sampleset_size();
    
    /** Clear sample set and free memory. */
    void clear_sampleset();
    
    /** Get reference on currently used covariance function. */
    CovarianceFunction & covf();
    
    /** Get log of covariance hyperparameter. */
    Eigen::Vector3d get_loghyper();
      
    /** Get input vector dimensionality. */
    size_t get_input_dim();

    double log_likelihood();
    
    Eigen::VectorXd log_likelihood_gradient();

    // CEDRIC
    //std::vector<double> getAlpha();
    Eigen::VectorXd getAlpha();
    Eigen::MatrixXd getL();

  protected:
    
    // CEDRIC
    //std::vector<double> inverted_bit_vector_;



    /** The covariance function of this Gaussian process. */
    CovarianceFunction * cf;
    
    /** The training sample set. */
    SampleSet * sampleset;
    
    /** Alpha is cached for performance. */ 
    Eigen::VectorXd alpha;
    
    /** Last test kernel vector. */
    Eigen::VectorXd k_star;

    /** Linear solver used to invert the covariance matrix. */
//    Eigen::LLT<Eigen::MatrixXd> solver;
    Eigen::MatrixXd L;
    
    /** Input vector dimensionality. */
    size_t input_dim;
    
    /** Update test input and cache kernel vector. */
    void update_k_star(const Eigen::VectorXd &x_star);

    void update_alpha();

    /** Compute covariance matrix and perform cholesky decomposition. */
    virtual void compute();
    
    bool alpha_needs_update;

  private:

    /** No assignement */
    GaussianProcess& operator=(const GaussianProcess&);

  };
}

#endif /* __GP_H__ */
