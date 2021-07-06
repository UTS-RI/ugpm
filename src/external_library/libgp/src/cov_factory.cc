// libgp - Gaussian process library for Machine Learning
// Copyright (c) 2013, Manuel Blum <mblum@informatik.uni-freiburg.de>
// All rights reserved.

#include <string>
#include <cassert>

#include "cov_factory.h"

#include "cov_noise.h"
#include "cov_linear_ard.h"
#include "cov_linear_one.h"
#include "cov_se_ard.h"
#include "cov_se_iso.h"
#include "cov_matern3_iso.h"
#include "cov_matern5_iso.h"
#include "cov_rq_iso.h"
#include "cov_sum.h"
#include "cov_prod.h"
#include "cov_periodic_matern3_iso.h"
#include "cov_periodic.h"
#include "input_dim_filter.h"

namespace libgp {
  
  CovFactory::CovFactory () {
    registry["CovLinearard"] = & create_func<CovLinearard>;
    registry["CovLinearone"] = & create_func<CovLinearone>;
    registry["CovMatern3iso"] = & create_func<CovMatern3iso>;
    registry["CovMatern5iso"] = & create_func<CovMatern5iso>;
    registry["CovNoise"] = & create_func<CovNoise>;
    registry["CovRQiso"] = & create_func<CovRQiso>;
    registry["CovSEard"] = & create_func<CovSEard>;
    registry["CovSEiso"] = & create_func<CovSEiso>;
    registry["CovSum"] = & create_func<CovSum>;
    registry["CovProd"] = & create_func<CovProd>;
    registry["CovPeriodicMatern3iso"] = & create_func<CovPeriodicMatern3iso>;
    registry["CovPeriodic"] = & create_func<CovPeriodic>;
    registry["InputDimFilter"] = & create_func<InputDimFilter>;
  }
  
  CovFactory::~CovFactory () {};
  
  libgp::CovarianceFunction* CovFactory::create(size_t input_dim, const std::string key) {

    CovarianceFunction * covf = NULL;

    //remove whitespace 
    std::string trimmed = key;
    for(size_t i=0; i<trimmed.length(); i++) if(trimmed[i] == ' ') trimmed.erase(i,1);
    
    // find parenthesis
    size_t left = trimmed.find_first_of('(');
    size_t right = trimmed.find_last_of(')');
    std::string func = trimmed.substr(0,left);
    std::string arg;
    int sep = 0;
    if (left != right) {
      arg = trimmed.substr(left);
      size_t i = 0, pos = 0;
      while ((pos = arg.find_first_of("(,)", pos)) != std::string::npos) {
        if (arg.at(pos) == '(') i++;
        else if (arg.at(pos) == ')') i--;
        else if (arg.at(pos) == ',' && i == 1) sep = pos;
        pos++;
      }
    }
    std::map<std::string , CovFactory::create_func_def>::iterator it = registry.find(func);
    if (it == registry.end()) {
      std::cerr << "fatal error while parsing covariance function: " << func << " not found" << std::endl;
      exit(0);
    } 
    covf = registry.find(func)->second();
    if (left == right) {
      bool res = covf->init(input_dim);
      assert(res);
    } else if (sep == 0) {
      size_t sep = arg.find_first_of('/');
      int filter = atoi(arg.substr(1,sep-1).c_str());
      std::string second = arg.substr(sep+1, arg.length() - sep - 2);
      bool res = covf->init(input_dim, filter, create(1, second));
      assert(res);
    } else {
      bool res = covf->init(input_dim, 
            create(input_dim, arg.substr(1,sep-1)), 
            create(input_dim, arg.substr(sep+1, arg.length()-sep-2)));
      assert(res);
    }
    return covf;
  }

  std::vector<std::string> CovFactory::list()
  {
    std::vector<std::string> products;
    std::map<std::string , CovFactory::create_func_def>::iterator it;
    for (it = registry.begin(); it != registry.end(); ++it) {
      products.push_back((*it).first);
    }
    return products;
  }
}
