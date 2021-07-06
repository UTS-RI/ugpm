// libgp - Gaussian process library for Machine Learning
// Copyright (c) 2013, Manuel Blum <mblum@informatik.uni-freiburg.de>
// All rights reserved.

#ifndef __COV_H__
#define __COV_H__

#include <iostream>
#include <vector>

#include <Eigen/Dense>

namespace libgp
{

  /** Covariance function base class.
   *  @author Manuel Blum
   *  @ingroup cov_group 
   *  @todo implement more covariance functions */
  class CovarianceFunction
  {
    public:
      /** Constructor. */
      CovarianceFunction() {};

      /** Destructor. */
      virtual ~CovarianceFunction() {};

      /** Initialization method for atomic covariance functions. 
       *  @param input_dim dimensionality of the input vectors */
      virtual bool init(int input_dim) 
      { 
        return false;
      };

      /** Initialization method for compound covariance functions. 
       *  @param input_dim dimensionality of the input vectors 
       *  @param first first covariance function of compound
       *  @param second second covariance function of compound */
      virtual bool init(int input_dim, CovarianceFunction * first, CovarianceFunction * second)
      {
        return false;
      };

      virtual bool init(int input_dim, int filter, CovarianceFunction * covf)
      {
        return false;
      };

      /** Computes the covariance of two input vectors.
       *  @param x1 first input vector
       *  @param x2 second input vector
       *  @return covariance of x1 and x2 */
      virtual double get(const Eigen::VectorXd &x1, const Eigen::VectorXd &x2) = 0;

      /** Covariance gradient of two input vectors with respect to the hyperparameters.
       *  @param x1 first input vector
       *  @param x2 second input vector
       *  @param grad covariance gradient */
      virtual void grad(const Eigen::VectorXd &x1, const Eigen::VectorXd &x2, Eigen::VectorXd &grad) = 0;

      /** Update parameter vector.
       *  @param p new parameter vector */
      virtual void set_loghyper(const Eigen::VectorXd &p);

      /** Update parameter vector.
       *  @param p new parameter vector */
      virtual void set_loghyper(const double p[]);

      /** Get number of parameters for this covariance function.
       *  @return parameter vector dimensionality */
      size_t get_param_dim();

      /** Get input dimensionality.
       *  @return input dimensionality */
      size_t get_input_dim();

      /** Get log-hyperparameter of covariance function.
       *  @return log-hyperparameter */
      Eigen::VectorXd get_loghyper();
      Eigen::Vector3d get_loghyper3();

      /** Returns a string representation of this covariance function.
       *  @return string containing the name of this covariance function */
      virtual std::string to_string() = 0;

      /** Draw random target values from this covariance function for input X. */
      Eigen::VectorXd draw_random_sample(Eigen::MatrixXd &X);

      bool loghyper_changed;

    protected:
      /** Input dimensionality. */
      size_t input_dim;

      /** Size of parameter vector. */
      size_t param_dim;

      /** Parameter vector containing the log hyperparameters of the covariance function.
       *  The number of necessary parameters is given in param_dim. */
      Eigen::VectorXd loghyper;

  };

}

#endif /* __COV_H__ */

/** Covariance functions available for Gaussian process models. 
 *  There are atomic and composite covariance functions. 
 *  @defgroup cov_group Covariance Functions */
