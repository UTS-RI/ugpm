/**
 *  Author: Cedric LE GENTIL 
 *
 *  Copyright 2021 Cedric LE GENTIL
 *
 *  For any further question, recommendation or contribution
 *  le.gentil.cedric@gmail.com
 **/

#include "common/math_utils.h"
#include "Eigen/unsupported/SpecialFunctions"

namespace celib
{
    Eigen::MatrixXd seKernel(const Eigen::VectorXd& x1, const Eigen::VectorXd& x2, const double l2, const double sf2)
    {
        Eigen::MatrixXd output(x1.size(), x2.size());
        Eigen::MatrixXd X1(x1.size(), x2.size());
        X1 = x1.replicate(1,x2.size());
        Eigen::MatrixXd X2(x1.size(), x2.size());
        X2 = x2.transpose().replicate(x1.size(),1);
        output = (((X1-X2).array().pow(2) * (-0.5/l2)).exp() * sf2).matrix();
        return output;
    }



    Eigen::MatrixXd seKernelIntegral(const double a, const Eigen::VectorXd& b, const Eigen::VectorXd& x2, const double l2, const double sf2)
    {
        double sqrt_inv_l2 = std::sqrt(1.0/l2);

        Eigen::MatrixXd output(b.size(), x2.size());
        Eigen::MatrixXd B(b.size(), x2.size());
        B = b.replicate(1,x2.size());
        Eigen::MatrixXd X2(b.size(), x2.size());
        X2 = x2.transpose().replicate(b.size(),1);
        output = (-kSqrt2*sf2*kSqrtPi*( (kSqrt2*(-X2.array()+a)*sqrt_inv_l2/2.0).erf() - (kSqrt2*(-X2+B)*sqrt_inv_l2/2.0).array().erf() )/(2.0*sqrt_inv_l2)).matrix();
        return output;
    }



    Eigen::MatrixXd seKernelIntegralDt(const double a, const Eigen::VectorXd& b, const Eigen::VectorXd& x2, const double l2, const double sf2)
    {
        double sqrt_inv_l2 = std::sqrt(1.0/l2);


        Eigen::MatrixXd output(b.size(), x2.size());
        Eigen::MatrixXd B(b.size(), x2.size());
        B = b.replicate(1,x2.size());
        Eigen::MatrixXd X2(b.size(), x2.size());
        X2 = x2.transpose().replicate(b.size(),1);
        output = (
            sf2*((B - X2).array().pow(2)/(-2.0*l2)).exp()
            - sf2*((X2.array() - a).pow(2)/(-2.0*l2)).exp()
            ).matrix();
        return output;
    }



    Eigen::MatrixXd seKernelIntegral2(const double a, const Eigen::VectorXd& b, const Eigen::VectorXd& x2, const double l2, const double sf2)
    {
        double sqrt_inv_l2 = std::sqrt(1.0/l2);

        Eigen::MatrixXd output(b.size(), x2.size());
        Eigen::MatrixXd B(b.size(), x2.size());
        B = b.replicate(1,x2.size());
        Eigen::MatrixXd X2(b.size(), x2.size());
        X2 =x2.transpose().replicate(b.size(),1);

        output = 
                (kSqrt2*sf2*kSqrtPi*( (kSqrt2*(-X2.array()+a)*sqrt_inv_l2/2.0).erf()*(-B.array()+a))/(2.0*sqrt_inv_l2)
                - (kSqrt2*sf2*kSqrtPi*(                        
                    (kSqrt2*(-X2.array()+a)*sqrt_inv_l2/2.0).erf()*(-X2.array()+a)
                    - (kSqrt2*(-X2+B)*sqrt_inv_l2/2.0).array().erf()*((-X2+B).array())
                    + kSqrt2*((-X2.array()+a).pow(2)/(-2.0*l2)).exp()/(kSqrtPi*sqrt_inv_l2)
                    - kSqrt2*((B-X2).array().pow(2)/(-2.0*l2)).exp()/(kSqrtPi*sqrt_inv_l2)
                    )
                )/(2.0*sqrt_inv_l2) ).matrix();

        return output;
    }



    Eigen::MatrixXd seKernelIntegral2Dt(const double a, const Eigen::VectorXd& b, const Eigen::VectorXd& x2, const double l2, const double sf2)
    {
        double sqrt_inv_l2 = std::sqrt(1.0/l2);


        Eigen::MatrixXd output(b.size(), x2.size());
        Eigen::MatrixXd B(b.size(), x2.size());
        B = b.replicate(1,x2.size());
        Eigen::MatrixXd X2(b.size(), x2.size());
        X2 = x2.transpose().replicate(b.size(),1);
        output = ((
                    (( kSqrt2*(-X2.array()+a)*sqrt_inv_l2*0.5).erf()
                    - ( kSqrt2*(B-X2).array()*sqrt_inv_l2*0.5).erf() )*kSqrt2*sf2*kSqrtPi
                   + 2.0*sqrt_inv_l2*(B.array()-a)*sf2*( (-X2.array()+a).pow(2)/(-2.0*l2)).exp()

                )/(-2.0*sqrt_inv_l2)).matrix();
        return output;
    }
}