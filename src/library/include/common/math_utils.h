/**
 *  Author: Cedric LE GENTIL 
 *
 *  Copyright 2021 Cedric LE GENTIL
 *
 *  For any further question, recommendation or contribution
 *  le.gentil.cedric@gmail.com
 **/

#ifndef COMMON_MATH_UTILS_H
#define COMMON_MATH_UTILS_H

#include "common/types.h"
#include "Eigen/Dense"

namespace celib
{

    const double kExpNormTolerance = 1e-14;
    const double kLogTraceTolerance = 3.0 - kExpNormTolerance;


    const double kSqrt2 = std::sqrt(2.0);
    const double kSqrtPi = std::sqrt(M_PI);


    inline celib::Mat3 eulToRotMat(double eul_z, double eul_y, double eul_x)
    {

        celib::Mat3 transform;
        double c1 = std::cos(eul_x);
        double c2 = std::cos(eul_y);
        double c3 = std::cos(eul_z);
        double s1 = std::sin(eul_x);
        double s2 = std::sin(eul_y);
        double s3 = std::sin(eul_z);
        transform << c1*c2, c1*s2*s3 - c3*s1, s1*s3 + c1*c3*s2,
                  c2*s1, c1*c3 + s1*s2*s3, c3*s1*s2 - c1*s3,
                -s2, c2*s3, c2*c3;
    
        return transform;
    }

    inline celib::Mat3 eulToRotMat(std::vector<double> eul)
    {
        if(eul.size() != 3) throw std::range_error("Wrong vector size for Euler to Rotation matrix conversion");
        return eulToRotMat(eul[2], eul[1], eul[0]);
    }


    // SO3 Log mapping
    inline Vec3 LogMap(const Mat3& rot_mat){
        double trace_mat = rot_mat.trace();
        if(trace_mat < kLogTraceTolerance){
            double phi = std::acos( (trace_mat-1) * 0.5);
            Mat3 skew_mat = (phi/(2.0*std::sin(phi))) * (rot_mat - rot_mat.transpose());
            Vec3 output;
            output << skew_mat(2,1), skew_mat(0,2), skew_mat(1,0);
            return output;
        }else{
            return Vec3::Zero();
        }
    }   
    
    // SO3 Exp mapping
    inline Mat3 ExpMap(const Vec3& vec){
        double vec_norm = vec.norm();
        if(vec_norm > kExpNormTolerance){
            Mat3 skew_mat;
            skew_mat << 0.0, -vec(2), vec(1),
                        vec(2), 0.0, -vec(0),
                        -vec(1), vec(0), 0.0;
            return  Mat3::Identity()
                + ( (std::sin(vec_norm)/vec_norm) * skew_mat)
                + ( ( (1 - std::cos(vec_norm))/(vec_norm * vec_norm)) * skew_mat * skew_mat);


        }else{
            return Mat3::Identity();
        }
    }

    inline Mat3 ToSkewSymMat(const Vec3& rot_vec)
    {
        Mat3 skew_mat;
        skew_mat << 0.0, -rot_vec(2), rot_vec(1),
                    rot_vec(2), 0.0, -rot_vec(0),
                    -rot_vec(1), rot_vec(0), 0.0;
        return skew_mat;

    }



    // Righthand Jacobian of Exp mapping
    template<typename T>
    inline Eigen::Matrix<T, 3, 3> JacobianRighthandExpMap( Eigen::Matrix<T, 3, 1> rot_vec)
    {
        Eigen::Matrix<T, 3, 3> output = Eigen::Matrix<T, 3, 3>::Identity();
        T vec_norm = rot_vec.norm();

        
        Eigen::Matrix<T, 3, 1> vec = rot_vec;

        if(vec_norm > kExpNormTolerance)
        {

            Eigen::Matrix<T, 3, 3> skew_mat;
            skew_mat << T(0.0), T(-vec(2)), T(vec(1)),
                        T(vec(2)), T(0.0), T(-vec(0)),
                        T(-vec(1)), T(vec(0)), T(0.0);
            
            output += ( (vec_norm - sin(vec_norm)) / (vec_norm*vec_norm*vec_norm) )*skew_mat*skew_mat  - ( (1.0 - cos(vec_norm))/(vec_norm*vec_norm) )*skew_mat;
        }
        return output;
    }


    // Jacobian of the Exp mapping
    inline celib::Mat9_3 JacobianExpMap(const celib::Vec3& rot_vec){

        celib::Mat9_3 output;
        double vec_norm = rot_vec.norm();

        if(vec_norm > kExpNormTolerance)
        {                                                                           
                                                                                                                    
            double r1_2 = rot_vec(0) * rot_vec(0);
            double r2_2 = rot_vec(1) * rot_vec(1);
            double r3_2 = rot_vec(2) * rot_vec(2);
            double r12_22_23_15 = std::pow(r1_2+r2_2+r3_2, 1.5);
            double r12_22_23_2 = std::pow(r1_2+r2_2+r3_2, 2);
            double r_norm = std::sqrt(r1_2+r2_2+r3_2);
            double r1 = rot_vec(0);
            double r2 = rot_vec(1);
            double r3 = rot_vec(2);
                                                                                                                    
            // Equation from MATLAB symbolic toolbox (might have a better formualtion, to inspect later)            
            output(0,0) = - (r1*std::sin(r_norm)*(r2_2 + r3_2))/r12_22_23_15 - (2*r1*(r2_2 + r3_2)*(std::cos(r_norm) - 1))/r12_22_23_2;
            output(0,1) = (2*r2*(std::cos(r_norm) - 1))/(r1_2 + r2_2 + r3_2) - (r2*std::sin(r_norm)*(r2_2 + r3_2))/r12_22_23_15 - (2*r2*(r2_2 + r3_2)*(std::cos(r_norm) - 1))/r12_22_23_2;
            output(0,2) = (2*r3*(std::cos(r_norm) - 1))/(r1_2 + r2_2 + r3_2) - (r3*std::sin(r_norm)*(r2_2 + r3_2))/r12_22_23_15 - (2*r3*(r2_2 + r3_2)*(std::cos(r_norm) - 1))/r12_22_23_2;
            output(1,0) = (r1*r3*std::cos(r_norm))/(r1_2 + r2_2 + r3_2) - (r2*(std::cos(r_norm) - 1))/(r1_2 + r2_2 + r3_2) - (r1*r3*std::sin(r_norm))/r12_22_23_15 + (r1_2*r2*std::sin(r_norm))/r12_22_23_15 + (2*r1_2*r2*(std::cos(r_norm) - 1))/r12_22_23_2;
            output(1,1) = (r2*r3*std::cos(r_norm))/(r1_2 + r2_2 + r3_2) - (r1*(std::cos(r_norm) - 1))/(r1_2 + r2_2 + r3_2) - (r2*r3*std::sin(r_norm))/r12_22_23_15 + (r1*r2_2*std::sin(r_norm))/r12_22_23_15 + (2*r1*r2_2*(std::cos(r_norm) - 1))/r12_22_23_2;
            output(1,2) = std::sin(r_norm)/r_norm - (r3_2*std::sin(r_norm))/r12_22_23_15 + (r3_2*std::cos(r_norm))/(r1_2 + r2_2 + r3_2) + (r1*r2*r3*std::sin(r_norm))/r12_22_23_15 + (2*r1*r2*r3*(std::cos(r_norm) - 1))/r12_22_23_2;
            output(2,0) = (r1*r2*std::sin(r_norm))/r12_22_23_15 - (r1*r2*std::cos(r_norm))/(r1_2 + r2_2 + r3_2) - (r3*(std::cos(r_norm) - 1))/(r1_2 + r2_2 + r3_2) + (r1_2*r3*std::sin(r_norm))/r12_22_23_15 + (2*r1_2*r3*(std::cos(r_norm) - 1))/r12_22_23_2;
            output(2,1) = (r2_2*std::sin(r_norm))/r12_22_23_15 - std::sin(r_norm)/r_norm - (r2_2*std::cos(r_norm))/(r1_2 + r2_2 + r3_2) + (r1*r2*r3*std::sin(r_norm))/r12_22_23_15 + (2*r1*r2*r3*(std::cos(r_norm) - 1))/r12_22_23_2;
            output(2,2) = (r2*r3*std::sin(r_norm))/r12_22_23_15 - (r2*r3*std::cos(r_norm))/(r1_2 + r2_2 + r3_2) - (r1*(std::cos(r_norm) - 1))/(r1_2 + r2_2 + r3_2) + (r1*r3_2*std::sin(r_norm))/r12_22_23_15 + (2*r1*r3_2*(std::cos(r_norm) - 1))/r12_22_23_2;
            output(3,0) = (r1*r3*std::sin(r_norm))/r12_22_23_15 - (r1*r3*std::cos(r_norm))/(r1_2 + r2_2 + r3_2) - (r2*(std::cos(r_norm) - 1))/(r1_2 + r2_2 + r3_2) + (r1_2*r2*std::sin(r_norm))/r12_22_23_15 + (2*r1_2*r2*(std::cos(r_norm) - 1))/r12_22_23_2;
            output(3,1) = (r2*r3*std::sin(r_norm))/r12_22_23_15 - (r2*r3*std::cos(r_norm))/(r1_2 + r2_2 + r3_2) - (r1*(std::cos(r_norm) - 1))/(r1_2 + r2_2 + r3_2) + (r1*r2_2*std::sin(r_norm))/r12_22_23_15 + (2*r1*r2_2*(std::cos(r_norm) - 1))/r12_22_23_2;
            output(3,2) = (r3_2*std::sin(r_norm))/r12_22_23_15 - std::sin(r_norm)/r_norm - (r3_2*std::cos(r_norm))/(r1_2 + r2_2 + r3_2) + (r1*r2*r3*std::sin(r_norm))/r12_22_23_15 + (2*r1*r2*r3*(std::cos(r_norm) - 1))/r12_22_23_2;
            output(4,0) = (2*r1*(std::cos(r_norm) - 1))/(r1_2 + r2_2 + r3_2) - (r1*std::sin(r_norm)*(r1_2 + r3_2))/r12_22_23_15 - (2*r1*(r1_2 + r3_2)*(std::cos(r_norm) - 1))/r12_22_23_2;
            output(4,1) = - (r2*std::sin(r_norm)*(r1_2 + r3_2))/r12_22_23_15 - (2*r2*(r1_2 + r3_2)*(std::cos(r_norm) - 1))/r12_22_23_2;
            output(4,2) = (2*r3*(std::cos(r_norm) - 1))/(r1_2 + r2_2 + r3_2) - (r3*std::sin(r_norm)*(r1_2 + r3_2))/r12_22_23_15 - (2*r3*(r1_2 + r3_2)*(std::cos(r_norm) - 1))/r12_22_23_2;
            output(5,0) = std::sin(r_norm)/r_norm - (r1_2*std::sin(r_norm))/r12_22_23_15 + (r1_2*std::cos(r_norm))/(r1_2 + r2_2 + r3_2) + (r1*r2*r3*std::sin(r_norm))/r12_22_23_15 + (2*r1*r2*r3*(std::cos(r_norm) - 1))/r12_22_23_2;
            output(5,1) = (r1*r2*std::cos(r_norm))/(r1_2 + r2_2 + r3_2) - (r3*(std::cos(r_norm) - 1))/(r1_2 + r2_2 + r3_2) - (r1*r2*std::sin(r_norm))/r12_22_23_15 + (r2_2*r3*std::sin(r_norm))/r12_22_23_15 + (2*r2_2*r3*(std::cos(r_norm) - 1))/r12_22_23_2;
            output(5,2) = (r1*r3*std::cos(r_norm))/(r1_2 + r2_2 + r3_2) - (r2*(std::cos(r_norm) - 1))/(r1_2 + r2_2 + r3_2) - (r1*r3*std::sin(r_norm))/r12_22_23_15 + (r2*r3_2*std::sin(r_norm))/r12_22_23_15 + (2*r2*r3_2*(std::cos(r_norm) - 1))/r12_22_23_2;
            output(6,0) = (r1*r2*std::cos(r_norm))/(r1_2 + r2_2 + r3_2) - (r3*(std::cos(r_norm) - 1))/(r1_2 + r2_2 + r3_2) - (r1*r2*std::sin(r_norm))/r12_22_23_15 + (r1_2*r3*std::sin(r_norm))/r12_22_23_15 + (2*r1_2*r3*(std::cos(r_norm) - 1))/r12_22_23_2;
            output(6,1) = std::sin(r_norm)/r_norm - (r2_2*std::sin(r_norm))/r12_22_23_15 + (r2_2*std::cos(r_norm))/(r1_2 + r2_2 + r3_2) + (r1*r2*r3*std::sin(r_norm))/r12_22_23_15 + (2*r1*r2*r3*(std::cos(r_norm) - 1))/r12_22_23_2;
            output(6,2) = (r2*r3*std::cos(r_norm))/(r1_2 + r2_2 + r3_2) - (r1*(std::cos(r_norm) - 1))/(r1_2 + r2_2 + r3_2) - (r2*r3*std::sin(r_norm))/r12_22_23_15 + (r1*r3_2*std::sin(r_norm))/r12_22_23_15 + (2*r1*r3_2*(std::cos(r_norm) - 1))/r12_22_23_2;
            output(7,0) = (r1_2*std::sin(r_norm))/r12_22_23_15 - std::sin(r_norm)/r_norm - (r1_2*std::cos(r_norm))/(r1_2 + r2_2 + r3_2) + (r1*r2*r3*std::sin(r_norm))/r12_22_23_15 + (2*r1*r2*r3*(std::cos(r_norm) - 1))/r12_22_23_2;
            output(7,1) = (r1*r2*std::sin(r_norm))/r12_22_23_15 - (r1*r2*std::cos(r_norm))/(r1_2 + r2_2 + r3_2) - (r3*(std::cos(r_norm) - 1))/(r1_2 + r2_2 + r3_2) + (r2_2*r3*std::sin(r_norm))/r12_22_23_15 + (2*r2_2*r3*(std::cos(r_norm) - 1))/r12_22_23_2;
            output(7,2) = (r1*r3*std::sin(r_norm))/r12_22_23_15 - (r1*r3*std::cos(r_norm))/(r1_2 + r2_2 + r3_2) - (r2*(std::cos(r_norm) - 1))/(r1_2 + r2_2 + r3_2) + (r2*r3_2*std::sin(r_norm))/r12_22_23_15 + (2*r2*r3_2*(std::cos(r_norm) - 1))/r12_22_23_2;
            output(8,0) = (2*r1*(std::cos(r_norm) - 1))/(r1_2 + r2_2 + r3_2) - (r1*std::sin(r_norm)*(r1_2 + r2_2))/r12_22_23_15 - (2*r1*(r1_2 + r2_2)*(std::cos(r_norm) - 1))/r12_22_23_2;
            output(8,1) = (2*r2*(std::cos(r_norm) - 1))/(r1_2 + r2_2 + r3_2) - (r2*std::sin(r_norm)*(r1_2 + r2_2))/r12_22_23_15 - (2*r2*(r1_2 + r2_2)*(std::cos(r_norm) - 1))/r12_22_23_2;
            output(8,2) = - (r3*std::sin(r_norm)*(r1_2 + r2_2))/r12_22_23_15 - (2*r3*(r1_2 + r2_2)*(std::cos(r_norm) - 1))/r12_22_23_2;
        }else{
            output <<   0,   0,   0,
                        0,   0,   1,
                        0,  -1,   0,
                        0,   0,  -1,
                        0,   0,   0,
                        1,   0,   0,
                        0,   1,   0,
                       -1,   0,   0,
                        0,   0,   0;
        }
        return output;
    }


    inline celib::Mat3_9 JacobianLogMap(const celib::Mat3& rot_mat)
    {
        celib::Mat3_9 output;
        double trace_mat = rot_mat.trace();
        
        if(trace_mat < kLogTraceTolerance){

            // Equation from MATLAB symbolic toolbox (might have a better formualtion, to inspect later)
            output(0,0) = - (rot_mat(1,2) - rot_mat(2,1))/(4*(std::pow(rot_mat(0,0)/2.0
                            + rot_mat(1,1)/2.0 + rot_mat(2,2)/2.0 - 0.5,2) - 1)) - 
                            (std::acos(rot_mat(0,0)/2.0 + rot_mat(1,1)/2.0 + rot_mat(2,2)/2.0 - 0.5)*
                             (rot_mat(1,2) - rot_mat(2,1))*(rot_mat(0,0)/2.0 + rot_mat(1,1)/2.0
                                 + rot_mat(2,2)/2.0 - 0.5))/(4*std::pow(1 - std::pow(rot_mat(0,0)/2.0 + rot_mat(1,1)/2.0 
                                     + rot_mat(2,2)/2.0 - 0.5,2),1.5));
            output(0,1) = 0.0;
            output(0,2) = 0.0;
            output(0,3) = 0.0;
            output(0,4) = - (rot_mat(1,2) - rot_mat(2,1))/
                (4*(std::pow(rot_mat(0,0)/2.0 + rot_mat(1,1)/2.0 + rot_mat(2,2)/2.0 - 0.5,2) - 1)) -
                (std::acos(rot_mat(0,0)/2.0 + rot_mat(1,1)/2.0 + rot_mat(2,2)/2.0 - 0.5)*
                 (rot_mat(1,2) - rot_mat(2,1))*(rot_mat(0,0)/2.0 + rot_mat(1,1)/2.0 + rot_mat(2,2)/2.0 - 0.5))/
                (4*std::pow(1 - std::pow(rot_mat(0,0)/2.0 + rot_mat(1,1)/2.0 + rot_mat(2,2)/2.0 - 0.5,2),1.5));
            output(0,5) = std::acos(rot_mat(0,0)/2.0 + rot_mat(1,1)/2.0 + rot_mat(2,2)/2.0 - 0.5)
                /(2*std::pow(1 - std::pow(rot_mat(0,0)/2.0 + rot_mat(1,1)/2.0 + rot_mat(2,2)/2.0 - 0.5,2),0.5));
            output(0,6) = 0.0;
            output(0,7) = -std::acos(rot_mat(0,0)/2.0 + rot_mat(1,1)/2.0 + rot_mat(2,2)/2.0 - 0.5)/
                (2*std::pow(1 - std::pow(rot_mat(0,0)/2.0 + rot_mat(1,1)/2.0 + rot_mat(2,2)/2.0 - 0.5,2),0.5));
            output(0,8) = - (rot_mat(1,2) - rot_mat(2,1))/
                (4*(std::pow(rot_mat(0,0)/2.0 + rot_mat(1,1)/2.0 + rot_mat(2,2)/2.0 - 0.5,2) - 1)) -
                (std::acos(rot_mat(0,0)/2.0 + rot_mat(1,1)/2.0 + rot_mat(2,2)/2.0 - 0.5)*
                 (rot_mat(1,2) - rot_mat(2,1))*(rot_mat(0,0)/2.0 + rot_mat(1,1)/2.0 + rot_mat(2,2)/2.0 - 0.5))/
                (4*std::pow(1 - std::pow(rot_mat(0,0)/2.0 + rot_mat(1,1)/2.0 + rot_mat(2,2)/2.0 - 0.5,2),1.5));
            output(1,0) = (rot_mat(0,2) - rot_mat(2,0))/
                (4*(std::pow(rot_mat(0,0)/2.0 + rot_mat(1,1)/2.0 + rot_mat(2,2)/2.0 - 0.5,2) - 1)) + 
                (std::acos(rot_mat(0,0)/2.0 + rot_mat(1,1)/2.0 + rot_mat(2,2)/2.0 - 0.5)*
                 (rot_mat(0,2) - rot_mat(2,0))*(rot_mat(0,0)/2.0 + rot_mat(1,1)/2.0 + rot_mat(2,2)/2.0 - 0.5))/
                (4*std::pow(1 - std::pow(rot_mat(0,0)/2.0 + rot_mat(1,1)/2.0 + rot_mat(2,2)/2.0 - 0.5,2),1.5));
            output(1,1) = 0.0;
            output(1,2) = -std::acos(rot_mat(0,0)/2.0 + rot_mat(1,1)/2.0 + rot_mat(2,2)/2.0 - 0.5)/
                (2*std::pow(1 - std::pow(rot_mat(0,0)/2.0 + rot_mat(1,1)/2.0 + rot_mat(2,2)/2.0 - 0.5,2),0.5));
            output(1,3) = 0.0;
            output(1,4) = (rot_mat(0,2) - rot_mat(2,0))/
                (4*(std::pow(rot_mat(0,0)/2.0 + rot_mat(1,1)/2.0 + rot_mat(2,2)/2.0 - 0.5,2) - 1)) + 
                (std::acos(rot_mat(0,0)/2.0 + rot_mat(1,1)/2.0 + rot_mat(2,2)/2.0 - 0.5)*
                 (rot_mat(0,2) - rot_mat(2,0))*(rot_mat(0,0)/2.0 + rot_mat(1,1)/2.0 + rot_mat(2,2)/2.0 - 0.5))/(
                 4*std::pow(1 - std::pow(rot_mat(0,0)/2.0 + rot_mat(1,1)/2.0 + rot_mat(2,2)/2.0 - 0.5,2),1.5));
            output(1,5) = 0.0;
            output(1,6) = std::acos(rot_mat(0,0)/2.0 + rot_mat(1,1)/2.0 + rot_mat(2,2)/2.0 - 0.5)/
                (2*std::pow(1 - std::pow(rot_mat(0,0)/2.0 + rot_mat(1,1)/2.0 + rot_mat(2,2)/2.0 - 0.5,2),0.5));
            output(1,7) = 0.0;
            output(1,8) = (rot_mat(0,2) - rot_mat(2,0))/
                (4*(std::pow(rot_mat(0,0)/2.0 + rot_mat(1,1)/2.0 + rot_mat(2,2)/2.0 - 0.5,2) - 1)) + 
                (std::acos(rot_mat(0,0)/2.0 + rot_mat(1,1)/2.0 + rot_mat(2,2)/2.0 - 0.5)*(rot_mat(0,2) - rot_mat(2,0))*
                 (rot_mat(0,0)/2.0 + rot_mat(1,1)/2.0 + rot_mat(2,2)/2.0 - 0.5))/
                (4*std::pow(1 - std::pow(rot_mat(0,0)/2.0 + rot_mat(1,1)/2.0 + rot_mat(2,2)/2.0 - 0.5,2),1.5));
            output(2,0) = - (rot_mat(0,1) - rot_mat(1,0))/
                (4*(std::pow(rot_mat(0,0)/2.0 + rot_mat(1,1)/2.0 + rot_mat(2,2)/2.0 - 0.5,2) - 1)) - 
                (std::acos(rot_mat(0,0)/2.0 + rot_mat(1,1)/2.0 + rot_mat(2,2)/2.0 - 0.5)*(rot_mat(0,1) - rot_mat(1,0))*
                 (rot_mat(0,0)/2.0 + rot_mat(1,1)/2.0 + rot_mat(2,2)/2.0 - 0.5))/
                (4*std::pow(1 - std::pow(rot_mat(0,0)/2.0 + rot_mat(1,1)/2.0 + rot_mat(2,2)/2.0 - 0.5,2),1.5));
            output(2,1) = std::acos(rot_mat(0,0)/2.0 + rot_mat(1,1)/2.0 + rot_mat(2,2)/2.0 - 0.5)/
                (2*std::pow(1 - std::pow(rot_mat(0,0)/2.0 + rot_mat(1,1)/2.0 + rot_mat(2,2)/2.0 - 0.5,2),0.5));
            output(2,2) = 0.0;
            output(2,3) = -std::acos(rot_mat(0,0)/2.0 + rot_mat(1,1)/2.0 + rot_mat(2,2)/2.0 - 0.5)/
                (2*std::pow(1 - std::pow(rot_mat(0,0)/2.0 + rot_mat(1,1)/2.0 + rot_mat(2,2)/2.0 - 0.5,2),0.5));
            output(2,4) = - (rot_mat(0,1) - rot_mat(1,0))/
                (4*(std::pow(rot_mat(0,0)/2.0 + rot_mat(1,1)/2.0 + rot_mat(2,2)/2.0 - 0.5,2) - 1)) - 
                (std::acos(rot_mat(0,0)/2.0 + rot_mat(1,1)/2.0 + rot_mat(2,2)/2.0 - 0.5)*(rot_mat(0,1) - rot_mat(1,0))*
                 (rot_mat(0,0)/2.0 + rot_mat(1,1)/2.0 + rot_mat(2,2)/2.0 - 0.5))/
                (4*std::pow(1 - std::pow(rot_mat(0,0)/2.0 + rot_mat(1,1)/2.0 + rot_mat(2,2)/2.0 - 0.5,2),1.5));
            output(2,5) = 0.0;
            output(2,6) = 0.0;
            output(2,7) = 0.0;
            output(2,8) = - (rot_mat(0,1) - rot_mat(1,0))/
                (4*(std::pow(rot_mat(0,0)/2.0 + rot_mat(1,1)/2.0 + rot_mat(2,2)/2.0 - 0.5,2) - 1)) - 
                (std::acos(rot_mat(0,0)/2.0 + rot_mat(1,1)/2.0 + rot_mat(2,2)/2.0 - 0.5)*(rot_mat(0,1) - rot_mat(1,0))*
                 (rot_mat(0,0)/2.0 + rot_mat(1,1)/2.0 + rot_mat(2,2)/2.0 - 0.5))/
                (4*std::pow(1 - std::pow(rot_mat(0,0)/2.0 + rot_mat(1,1)/2.0 + rot_mat(2,2)/2.0 - 0.5,2),1.5));
        }else{
            output <<   0,   0,    0,    0, 0, 0.5,   0, -0.5, 0,
                        0,   0, -0.5,    0, 0,   0, 0.5,    0, 0,
                        0, 0.5,    0, -0.5, 0,   0,   0,    0, 0;
        }
        return output;
    }

    inline celib::Row9 Mat3ToRow(celib::Mat3 R)
    {
        celib::Row9 output;
        output = Eigen::Map<celib::Row9>(R.data());
        return output;
    }


    inline  celib::Mat3_9 JacobianYXv(const celib::Mat3& Y, const celib::Vec3& v)
    {
        celib::Mat3_9 output;
        output <<
            Mat3ToRow((v * Y.row(0)).transpose()) ,
            Mat3ToRow((v * Y.row(1)).transpose()) ,
            Mat3ToRow((v * Y.row(2)).transpose());
        return output;
    }

    inline  celib::Mat9 JacobianYXW(celib::Mat3& Y, celib::Mat3& W)
    {
        celib::Mat9 output;
        output <<
            Mat3ToRow((W.col(0) * Y.row(0)).transpose()) ,
            Mat3ToRow((W.col(0) * Y.row(1)).transpose()) ,
            Mat3ToRow((W.col(0) * Y.row(2)).transpose()) ,
            Mat3ToRow((W.col(1) * Y.row(0)).transpose()) ,
            Mat3ToRow((W.col(1) * Y.row(1)).transpose()) ,
            Mat3ToRow((W.col(1) * Y.row(2)).transpose()) ,
            Mat3ToRow((W.col(2) * Y.row(0)).transpose()) ,
            Mat3ToRow((W.col(2) * Y.row(1)).transpose()) ,
            Mat3ToRow((W.col(2) * Y.row(2)).transpose());
        return output;
    }

    inline  celib::Mat9 JacobianYX(const celib::Mat3& Y)
    {
        celib::Mat9 output;
        output = celib::Mat9::Zero();
        output.block<3,3>(0,0) = Y;
        output.block<3,3>(3,3) = Y;
        output.block<3,3>(6,6) = Y;
        return output;
    }

    inline celib::Mat3 PropagateJacobianRp(const celib::Mat3& R, const celib::Mat3& d_r, const celib::Vec3& p, const celib::Mat3& d_p){

        celib::Mat3 output;
        Mat9_3 d_R = JacobianYX(R)*JacobianExpMap(Vec3::Zero())*d_r;
        output <<
                R.row(0) * d_p + d_R.row(0)*p(0)
                               + d_R.row(3)*p(1)
                               + d_R.row(6)*p(2),
                R.row(1) * d_p + d_R.row(1)*p(0)
                               + d_R.row(4)*p(1)
                               + d_R.row(7)*p(2),
                R.row(2) * d_p + d_R.row(2)*p(0)
                               + d_R.row(5)*p(1)
                               + d_R.row(8)*p(2);

        return output;
    }

    inline celib::Mat3 PropagateJacobianRR(const celib::Mat3& R1, const celib::Mat3& d_r1, const celib::Mat3& R2, const celib::Mat3& d_r2){

        celib::Mat3 output;
        Mat9_3 d_R1 = JacobianYX(R1)*JacobianExpMap(Vec3::Zero())*d_r1;
        Mat9_3 d_R2 = JacobianYX(R2)*JacobianExpMap(Vec3::Zero())*d_r2;
        Mat9_3 d_RR;
        d_RR <<
                R1.row(0) * d_R2.block<3,3>(0,0) + d_R1.row(0)*R2(0,0)
                                                 + d_R1.row(3)*R2(1,0)
                                                 + d_R1.row(6)*R2(2,0),
                R1.row(1) * d_R2.block<3,3>(0,0) + d_R1.row(1)*R2(0,0)
                                                 + d_R1.row(4)*R2(1,0)
                                                 + d_R1.row(7)*R2(2,0),
                R1.row(2) * d_R2.block<3,3>(0,0) + d_R1.row(2)*R2(0,0)
                                                 + d_R1.row(5)*R2(1,0)
                                                 + d_R1.row(8)*R2(2,0),
                R1.row(0) * d_R2.block<3,3>(3,0) + d_R1.row(0)*R2(0,1)
                                                 + d_R1.row(3)*R2(1,1)
                                                 + d_R1.row(6)*R2(2,1),
                R1.row(1) * d_R2.block<3,3>(3,0) + d_R1.row(1)*R2(0,1)
                                                 + d_R1.row(4)*R2(1,1)
                                                 + d_R1.row(7)*R2(2,1),
                R1.row(2) * d_R2.block<3,3>(3,0) + d_R1.row(2)*R2(0,1)
                                                 + d_R1.row(5)*R2(1,1)
                                                 + d_R1.row(8)*R2(2,1),
                R1.row(0) * d_R2.block<3,3>(6,0) + d_R1.row(0)*R2(0,2)
                                                 + d_R1.row(3)*R2(1,2)
                                                 + d_R1.row(6)*R2(2,2),
                R1.row(1) * d_R2.block<3,3>(6,0) + d_R1.row(1)*R2(0,2)
                                                 + d_R1.row(4)*R2(1,2)
                                                 + d_R1.row(7)*R2(2,2),
                R1.row(2) * d_R2.block<3,3>(6,0) + d_R1.row(2)*R2(0,2)
                                                 + d_R1.row(5)*R2(1,2)
                                                 + d_R1.row(8)*R2(2,2);

        output = JacobianLogMap(R1*R2)*d_RR;
        return output;
    }

    inline celib::Vec3 PropagateJacobianRp(const celib::Mat3& R, const celib::Vec3& d_r, const celib::Vec3& p, const celib::Vec3& d_p){

        celib::Vec3 output;
        Vec9 d_R = JacobianYX(R)*JacobianExpMap(Vec3::Zero())*d_r;
        output <<
                R.row(0) * d_p + d_R(0)*p(0)
                               + d_R(3)*p(1)
                               + d_R(6)*p(2),
                R.row(1) * d_p + d_R(1)*p(0)
                               + d_R(4)*p(1)
                               + d_R(7)*p(2),
                R.row(2) * d_p + d_R(2)*p(0)
                               + d_R(5)*p(1)
                               + d_R(8)*p(2);

        return output;
    }

    inline celib::Vec3 PropagateJacobianRR(const celib::Mat3& R1, const celib::Vec3& d_r1, const celib::Mat3& R2, const celib::Vec3& d_r2){

        celib::Vec3 output;
        Vec9 d_R1 = JacobianYX(R1)*JacobianExpMap(Vec3::Zero())*d_r1;
        Vec9 d_R2 = JacobianYX(R2)*JacobianExpMap(Vec3::Zero())*d_r2;
        Vec9 d_RR;
        d_RR <<
                R1.row(0) * d_R2.block<3,1>(0,0) + d_R1(0)*R2(0,0)
                                                 + d_R1(3)*R2(1,0)
                                                 + d_R1(6)*R2(2,0),
                R1.row(1) * d_R2.block<3,1>(0,0) + d_R1(1)*R2(0,0)
                                                 + d_R1(4)*R2(1,0)
                                                 + d_R1(7)*R2(2,0),
                R1.row(2) * d_R2.block<3,1>(0,0) + d_R1(2)*R2(0,0)
                                                 + d_R1(5)*R2(1,0)
                                                 + d_R1(8)*R2(2,0),
                R1.row(0) * d_R2.block<3,1>(3,0) + d_R1(0)*R2(0,1)
                                                 + d_R1(3)*R2(1,1)
                                                 + d_R1(6)*R2(2,1),
                R1.row(1) * d_R2.block<3,1>(3,0) + d_R1(1)*R2(0,1)
                                                 + d_R1(4)*R2(1,1)
                                                 + d_R1(7)*R2(2,1),
                R1.row(2) * d_R2.block<3,1>(3,0) + d_R1(2)*R2(0,1)
                                                 + d_R1(5)*R2(1,1)
                                                 + d_R1(8)*R2(2,1),
                R1.row(0) * d_R2.block<3,1>(6,0) + d_R1(0)*R2(0,2)
                                                 + d_R1(3)*R2(1,2)
                                                 + d_R1(6)*R2(2,2),
                R1.row(1) * d_R2.block<3,1>(6,0) + d_R1(1)*R2(0,2)
                                                 + d_R1(4)*R2(1,2)
                                                 + d_R1(7)*R2(2,2),
                R1.row(2) * d_R2.block<3,1>(6,0) + d_R1(2)*R2(0,2)
                                                 + d_R1(5)*R2(1,2)
                                                 + d_R1(8)*R2(2,2);

        output = JacobianLogMap(R1*R2)*d_RR;
        return output;
    }


    inline celib::Mat3 UncertaintyRp(celib::Mat3& R, const celib::Mat3& cov_r, celib::Vec3& p, const celib::Mat3& cov_p)
    {
        celib::Mat3 output;

        celib::Mat6 cov_rp = celib::Mat6::Zero();
        cov_rp.block<3,3>(0,0) = cov_r;
        cov_rp.block<3,3>(3,3) = cov_p;
        celib::Mat3_6 d_Rp_d_rp;
        d_Rp_d_rp.block<3,3>(0,0) = JacobianYXv(R, p) * JacobianExpMap(celib::Vec3::Zero());
        d_Rp_d_rp.block<3,3>(0,3) = R;

        output = d_Rp_d_rp*cov_rp*d_Rp_d_rp.transpose();

        return output;
    }
    
    inline celib::Mat3 UncertaintyRR(celib::Mat3& R1, const celib::Mat3& cov_r1, celib::Mat3& R2, const celib::Mat3& cov_r2)
    {
        celib::Mat3 output;

        celib::Mat3 RR = R1*R2;
        celib::Mat9_6 d_RR_d_rr;
        celib::Mat6 cov_rr = celib::Mat6::Zero();
        cov_rr.block<3,3>(0,0) = cov_r1;
        cov_rr.block<3,3>(3,3) = cov_r2;

        d_RR_d_rr.block<9,3>(0,0) = JacobianYXW(R1, R2) * JacobianExpMap(celib::Vec3::Zero());
        d_RR_d_rr.block<9,3>(0,3) = JacobianYX(RR) * JacobianExpMap(celib::Vec3::Zero());

        celib::Mat9 cov_RR = d_RR_d_rr*cov_rr*d_RR_d_rr.transpose();

        celib::Mat3_9 d_log_RR_d_RR = JacobianLogMap(RR);
        output = d_log_RR_d_RR*cov_RR*d_log_RR_d_RR.transpose();


        return output;
    }



    Eigen::MatrixXd seKernel(const Eigen::VectorXd& x1, const Eigen::VectorXd& x2, const double l2, const double sf2);



    Eigen::MatrixXd seKernelIntegral(const double a, const Eigen::VectorXd& b, const Eigen::VectorXd& x2, const double l2, const double sf2);



    Eigen::MatrixXd seKernelIntegralDt(const double a, const Eigen::VectorXd& b, const Eigen::VectorXd& x2, const double l2, const double sf2);



    Eigen::MatrixXd seKernelIntegral2(const double a, const Eigen::VectorXd& b, const Eigen::VectorXd& x2, const double l2, const double sf2);


    Eigen::MatrixXd seKernelIntegral2Dt(const double a, const Eigen::VectorXd& b, const Eigen::VectorXd& x2, const double l2, const double sf2);




}

#endif