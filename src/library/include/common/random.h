/**
 *  Author: Cedric LE GENTIL 
 *
 *  Copyright 2021 Cedric LE GENTIL
 * 
 *  For any further question, recommendation or contribution
 *  le.gentil.cedric@gmail.com
 **/

#ifndef CELIB_RANDOM_H
#define CELIB_RANDOM_H

#include <random>


namespace celib
{

    class RandomGenerator{
        public:
            RandomGenerator():random_eng_(std::random_device()())
            {
            }

            double randUniform(double from, double to)
            {
                std::uniform_real_distribution<double> distr(from, to);
                return distr(random_eng_);
            }
            double randGauss(double mean, double std_dev)
            {
                std::normal_distribution<double> distr(mean, std_dev);
                return distr(random_eng_);
            }

        private:

            std::default_random_engine random_eng_;


    };


}

#endif