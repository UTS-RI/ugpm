// libgp - Gaussian process library for Machine Learning
// Copyright (c) 2013, Manuel Blum <mblum@informatik.uni-freiburg.de>
// All rights reserved.

#include "gp_utils.h"
#include <Eigen/Dense>
#include <gtest/gtest.h>
#include <algorithm>

TEST(Utils, randn) {
  // Kolmogorow-Smirnow-Test
  int n = 10e5;
  double D = 0.0;
  std::vector<double> array;
  for (int i=0; i<n; ++i) array.push_back(libgp::Utils::randn());
  std::sort(array.begin(), array.end());
  for (int i=0; i<n; ++i) {
    double F = libgp::Utils::cdf_norm(array[i]);
    D = std::max(D, std::max(F-(i-1.0)/n, 1.0*i/n-F));
  }
  ASSERT_TRUE(1.63/sqrt(n) > D);
}

TEST(Utils, randperm) {
  // number of samples
  int m = 10e5;
  // number of categories
  int n=31;
  Eigen::MatrixXd count(n,n);
  count.setZero();
  int fails = 0;
  double np = 1.0*m/n;
  count.setZero();
  for (int i=0; i<m; ++i) {
    int * array = libgp::Utils::randperm(n);
    for (int j=0; j<n; ++j) {
      count(j, array[j]) += 1; 
    }
    delete[] array;
  }
  for (int i=0; i<n; ++i) {  
    // chi-square test
    double V = 0.0;
    for (int j=0; j<n; ++j) V += pow(count(j, i) - np, 2) / np;
    if (18.49 > V || 43.77 < V) fails++;
  }
  
  ASSERT_TRUE(10 >= fails);
}

TEST(Utils, randi) {
  // number of samples
  int m = 10e5;
  // number of categories
  int n=31;
  // number of iterations
  int k=30;
  Eigen::VectorXd count(n);
  int fails = 0;
  double np = 1.0*m/n;
  for (int j=0; j<k; ++j) {
    count.setZero();
    for (int i=0; i<m; ++i) count(libgp::Utils::randi(n)) += 1;
    // chi-square test
    double V = 0.0;
    for (int i=0; i<n; ++i) V += pow(count(i) - np, 2) / np;
    if (18.49 > V || 43.77 < V) fails++;
  }
  ASSERT_TRUE(10 >= fails);
}

TEST(Utils, cdf_norm) {
  double err = 10e-16;
  double cdf[65] = {0.000000000000000622, 0.000000000000004595,
    0.000000000000031909, 0.000000000000208386, 0.000000000001279813,
    0.000000000007392258, 0.000000000040160006, 0.000000000205226343,
    0.000000000986587645, 0.000000004462172454, 0.000000018989562466,
    0.000000076049605165, 0.000000286651571879, 0.000001017083242569,
    0.000003397673124730, 0.000010688525774934, 0.000031671241833120,
    0.000088417285200804, 0.000232629079035525, 0.000577025042390767,
    0.001349898031630096, 0.002979763235054556, 0.006209665325776138,
    0.012224472655044704, 0.022750131948179216, 0.040059156863817079,
    0.066807201268858085, 0.105649773666855282, 0.158655253931457046,
    0.226627352376868207, 0.308537538725986937, 0.401293674317076299,
    0.500000000000000000, 0.598706325682923701, 0.691462461274013007,
    0.773372647623131737, 0.841344746068542926, 0.894350226333144760,
    0.933192798731141915, 0.959940843136182886, 0.977249868051820791,
    0.987775527344955329, 0.993790334674223841, 0.997020236764945444,
    0.998650101968369897, 0.999422974957609234, 0.999767370920964460,
    0.999911582714799185, 0.999968328758166880, 0.999989311474225095,
    0.999996602326875261, 0.999998982916757484, 0.999999713348428076,
    0.999999923950394831, 0.999999981010437522, 0.999999995537827591,
    0.999999999013412300, 0.999999999794773609, 0.999999999959840014,
    0.999999999992607691, 0.999999999998720135, 0.999999999999791611,
    0.999999999999968137, 0.999999999999995448, 0.999999999999999334};
  for (int i=0; i<65; ++i) ASSERT_NEAR(cdf[i], libgp::Utils::cdf_norm(i*0.25-8.0), err);
}
