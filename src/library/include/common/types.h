/**
 *  Author: Cedric LE GENTIL 
 *
 *  Copyright 2021 Cedric LE GENTIL
 *
 *  For any further question, recommendation or contribution
 *  le.gentil.cedric@gmail.com
 **/

#ifndef COMMON_TYPES_H
#define COMMON_TYPES_H

#include <vector>
#include <string>
#include <iostream>
#include "Eigen/Dense"




namespace celib{

    // Data structure to store individual measurements (can be acc or gyr data)
    struct ImuSample
    {
        double t;
        double data[3];
    };

    // Data structure to store a collection of IMU measurements
    struct ImuData
    {
        // Offset in second for every sample timestamps
        double t_offset;

        // Vectors of samples
        std::vector<ImuSample> acc;
        std::vector<ImuSample> gyr;

        // Noise specification (assuming same on 3 axis)
        double acc_var;
        double gyr_var;

        // Return the collection of samples in between the two timestamps
        ImuData get(double from, double to)
        {
            // Check if the query interval makes sense
            if(from < to)
            {
                ImuData output;

                output.t_offset = t_offset;
                output.acc_var = acc_var;
                output.gyr_var = gyr_var;
                output.acc = get(acc, acc_current_index_, from, to);
                output.gyr = get(gyr, gyr_current_index_, from, to);
                return output;
            }
            // If the query inteval does not make sense throw an exception
            else
            {
                throw std::invalid_argument("The argument of ImuReader::Get are not consistent");
            }
        }

        void print()
        {
            std::cout << "Imu data with offset of " << t_offset << std::endl;
            std::cout << "  Accelerometer data (" << acc.size() << " samples with std = " << sqrt(acc_var) << ")" << std::endl;
            for(const auto& a : acc)
            {
                std::cout << "      t = " << a.t << ":    "
                        << a.data[0] << "    "
                        << a.data[1] << "    "
                        << a.data[2] << std::endl;
            }
            std::cout << "  Gyroscope data (" << gyr.size() << " samples with std = " << sqrt(gyr_var) << ")" << std::endl;
            for(const auto& g : gyr)
            {
                std::cout << "      t = " << g.t << ":    " 
                        << g.data[0] << "    "
                        << g.data[1] << "    "
                        << g.data[2] << std::endl;
            }
        }

        private:
            // To keep track of the "Get" and prevent the whole dataset parsing
            int acc_current_index_ = 0;
            int gyr_current_index_ = 0;

            // Return the collection of samples in between the two timestamps
            std::vector<ImuSample> get(const std::vector<ImuSample>& samples, int& current_index, double from, double to)
            {
                std::vector<ImuSample> output;

                int current_index_saved = current_index;
                bool loop = true;
                bool forward = false;


                while(loop)
                {
                    if(!forward)
                    {
                        if(samples[current_index].t > from)
                        {
                            if(samples[current_index].t < to)
                            {
                                output.push_back(samples[current_index]);
                            }
                            if(current_index > 0)
                            {
                                current_index--;
                            }
                            else if(current_index_saved < (samples.size()-1))
                            {
                                current_index = current_index_saved + 1;
                                forward = true;
                                std::reverse(output.begin(),output.end());
                            }
                            else
                            {
                                loop = false;
                                std::reverse(output.begin(),output.end());
                            }
                        }
                        else
                        {
                            if(current_index_saved < (samples.size()-1))
                            {
                                current_index = current_index_saved + 1;
                                forward = true;
                                std::reverse(output.begin(),output.end());
                            }
                            else
                            {
                                loop = false;
                                std::reverse(output.begin(),output.end());
                            }
                        }
                    }
                    else
                    {
                        if(samples[current_index].t < to)
                        {
                            if(samples[current_index].t > from)
                            {
                                output.push_back(samples[current_index]);
                            }
                            if(current_index < (samples.size()-1))
                            {
                                current_index++;
                            }
                            else
                            {
                                loop = false;
                            }
                        }
                        else
                        {
                            loop = false;
                        }
                    }
                }
                return output;
            }
    };

    // Data structure for the IMU
    struct ImuRosbagOption
    {
        std::string file_path;
        std::string topic_name;
        bool timestamp_interpolation;

    };



    typedef Eigen::Matrix<double, 3, 3> Mat3;
    typedef Eigen::Matrix<double, 3, 1> Vec3;
    typedef Eigen::Matrix<double, 1, 3> Row3;
    typedef Eigen::Matrix<double, 1, 2> Row2;
    typedef Eigen::Matrix<double, 6, 6> Mat6;
    typedef Eigen::Matrix<double, 4, 4> Mat4;
    typedef Eigen::Matrix<double, 9, 9> Mat9;
    typedef Eigen::Matrix<double, 12, 12> Mat12;
    typedef Eigen::Matrix<double, 2, 1> Vec2;
    typedef Eigen::Matrix<double, 6, 1> Vec6;
    typedef Eigen::Matrix<double, 9, 1> Vec9;
    typedef Eigen::Matrix<double, 1, 9> Row9;
    typedef Eigen::Matrix<double, 1, 12> Row12;
    typedef Eigen::Matrix<double, 3, 6> Mat3_6;
    typedef Eigen::Matrix<double, 3, 4> Mat3_4;
    typedef Eigen::Matrix<double, 2, 6> Mat2_6;
    typedef Eigen::Matrix<double, 9, 6> Mat9_6;
    typedef Eigen::Matrix<double, 3, 9> Mat3_9;
    typedef Eigen::Matrix<double, 2, 9> Mat2_9;
    typedef Eigen::Matrix<double, 2, 3> Mat2_3;
    typedef Eigen::Matrix<double, 9, 3> Mat9_3;
    typedef Eigen::Matrix<double, 3, 12> Mat3_12;
    typedef Eigen::Matrix<double, Eigen::Dynamic, 1> VecX;
    typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> MatX;




    inline Vec3 stdToCelibVec3(const std::vector<double> v)
    {
        Vec3 output;
        output << v[0], v[1], v[2];
        return output;
    }



    // Structure to store the different elements of a preintegrated measurement
    struct PreintMeasBasic{
        Mat3 delta_R;
        Vec3 delta_v;
        Vec3 delta_p;
        double dt; 
        double dt_sq_half;

        void print()
        {
            std::cout << "Preintegrated measurement" << std::endl;
            std::cout << "  dt = " << dt << std::endl;
            std::cout << "  dt_sq_half = " << dt_sq_half << std::endl;
            std::cout << "  Delta R = " << std::endl
                    << "    " << delta_R.row(0) << std::endl
                    << "    " << delta_R.row(1) << std::endl
                    << "    " << delta_R.row(2) << std::endl;
            std::cout << "  Delta v = " << std::endl 
                    << "    " << delta_v.row(0) << std::endl
                    << "    " << delta_v.row(1) << std::endl
                    << "    " << delta_v.row(2) << std::endl;
            std::cout << "  Delta p = " << std::endl
                    << "    " << delta_p.row(0) << std::endl
                    << "    " << delta_p.row(1) << std::endl
                    << "    " << delta_p.row(2) << std::endl;
        }
    };

    struct PreintMeas : PreintMeasBasic
    {
        Mat9 cov;

        Mat3 d_delta_R_d_bw;
        Vec3 d_delta_R_d_t;
     
        Mat3 d_delta_v_d_bw;
        Mat3 d_delta_v_d_bf;
        Vec3 d_delta_v_d_t;
     
        Mat3 d_delta_p_d_bw;
        Mat3 d_delta_p_d_bf;
        Vec3 d_delta_p_d_t;

        PreintMeas(Mat3 d_R,
                Vec3 d_v,
                Vec3 d_p,
                double dt, 
                double dt_sq_half,
                Mat9 cov_mat) :
                PreintMeasBasic{d_R, d_v, d_p, dt, dt_sq_half}, cov(cov_mat) {}; 

        PreintMeas(){};


        void printAll()
        {
            print();
            std::cout << "  Covariance = " << std::endl
                    << "    " << cov.row(0) << std::endl
                    << "    " << cov.row(1) << std::endl
                    << "    " << cov.row(2) << std::endl
                    << "    " << cov.row(3) << std::endl
                    << "    " << cov.row(4) << std::endl
                    << "    " << cov.row(5) << std::endl
                    << "    " << cov.row(6) << std::endl
                    << "    " << cov.row(7) << std::endl
                    << "    " << cov.row(8) << std::endl;
            std::cout << "  Jacobian Delta R / gyr bias = " << std::endl
                    << "    " << d_delta_R_d_bw.row(0) << std::endl
                    << "    " << d_delta_R_d_bw.row(1) << std::endl
                    << "    " << d_delta_R_d_bw.row(2) << std::endl;
            std::cout << "  Jacobian Delta R / time-shift = " << std::endl
                    << "    " << d_delta_R_d_t.transpose() << std::endl;
            std::cout << "  Jacobian Delta v / gyr bias = " << std::endl
                    << "    " << d_delta_v_d_bw.row(0) << std::endl
                    << "    " << d_delta_v_d_bw.row(1) << std::endl
                    << "    " << d_delta_v_d_bw.row(2) << std::endl;
            std::cout << "  Jacobian Delta v / acc bias = " << std::endl
                    << "    " << d_delta_v_d_bf.row(0) << std::endl
                    << "    " << d_delta_v_d_bf.row(1) << std::endl
                    << "    " << d_delta_v_d_bf.row(2) << std::endl;
            std::cout << "  Jacobian Delta v / time-shift = " << std::endl
                    << "    " << d_delta_v_d_t.transpose() << std::endl;
            std::cout << "  Jacobian Delta p / gyr bias = " << std::endl
                    << "    " << d_delta_p_d_bw.row(0) << std::endl
                    << "    " << d_delta_p_d_bw.row(1) << std::endl
                    << "    " << d_delta_p_d_bw.row(2) << std::endl;
            std::cout << "  Jacobian Delta p / acc bias = " << std::endl
                    << "    " << d_delta_p_d_bf.row(0) << std::endl
                    << "    " << d_delta_p_d_bf.row(1) << std::endl
                    << "    " << d_delta_p_d_bf.row(2) << std::endl;
            std::cout << "  Jacobian Delta p / time-shift = " << std::endl
                    << "    " << d_delta_p_d_t.transpose() << std::endl;


        }
    };
}

#endif
