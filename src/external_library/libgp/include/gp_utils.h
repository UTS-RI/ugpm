// libgp - Gaussian process library for Machine Learning
// Copyright (c) 2013, Manuel Blum <mblum@informatik.uni-freiburg.de>
// All rights reserved.

#ifndef __GP_UTILS_H__
#define __GP_UTILS_H__

#include <cstdlib>
#include <cmath>
#include <cassert>

namespace libgp {
  
  /** Various auxiliary functions. */
  class Utils {
  public:
    /** Generate independent standard normally distributed (zero mean, 
     *  unit variance) random numbers using the Box–Muller transform. */
    static double randn();
    
    /** Generate random permutation of array 0, ..., n-1 using 
     *  Fisher–Yates shuffle algorithm. */
    static int * randperm(int n);
    
    /** Pseudorandom integers from a uniform discrete distribution.
     *  Returns a random integer drawn from the discrete uniform 
     *  distribution on 0, ..., n-1. */
    static size_t randi(size_t n);
    
    /** Double precision numerical approximation to the standard normal 
     *  cumulative distribution function. The cumulative standard normal 
     *  integral is the function: 
     *  \f$ cdf\_norm(x)=\frac{1}{\sqrt{2\pi}}\int_{-\infty}^{x}
     *  \exp{\frac{-x^2}{2}}dx \f$ */
    static double cdf_norm (double x);
    
    /** Computes values of first Friedman dataset from 10-D input.
     *  \f$ f(\mathbf{x}) = 10\sin(\pi x_1 x_2) + 20(x_3 -0.5)^2 + 10x_4+5x_5 \f$ */
    static double friedman (double x[]);
    
    /** Hill function. 
     *  \f$ f(\mathbf{x}) = \sin(x-y)+0.2y^3 + \cos(xy - 0.5y) \f$  */
    static double hill(double x, double y);    
    
    /** Sign function.
     */
    static double sign(double x);

    // drand48() is not aviable under win
#if defined(WIN32) || defined(WIN64)
    inline double drand48()
    {
      return (rand() / (RAND_MAX + 1.0));
    }
#endif

  };  
}

#endif /* __GP_UTILS_H__ */
