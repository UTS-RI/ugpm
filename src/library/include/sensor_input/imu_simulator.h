/**
 *  Author: Cedric LE GENTIL 
 *
 *  Copyright 2021 Cedric LE GENTIL
 *
 *  This files contains the data structures and declarations
 *  related to the IMU simulator.
 * 
 *  For any further question, recommendation or contribution
 *  le.gentil.cedric@gmail.com
 **/

#ifndef IMU_SIMULATOR_H
#define IMU_SIMULATOR_H

#include "common/types.h"
#include "common/random.h"
#include <random>
#include <iostream>

namespace celib
{
    struct SineProperties3{
        std::vector<double> freq = {1, 1, 1};
        std::vector<double> amp = {1, 1, 1};

        void print() const
        {
            std::cout << "Sine properties" << std::endl;
            std::cout << "  Frequencies [Hz]: "
                << freq[0] << "   "
                << freq[1] << "   "
                << freq[2] << "   "
                << std::endl;
            std::cout << "  Amplitudes [Hz]: "
                << amp[0] << "   "
                << amp[1] << "   "
                << amp[2] << "   "
                << std::endl;
        }

    };


    // Data structure for the IMU simulator options
    struct ImuSimulatorOption{
        // Frequency of the simulated IMU (Hz)
        double imu_frequency = 100.0;
        // Length of the dataset in second
        double dataset_length = 60.0;

        // Noise specification
        double acc_std = 0.02;
        double gyr_std = 0.002;

        // Biases (constant) FUTURE_WORK: put brownian motion on biases
        std::vector<double> acc_bias = {0, 0, 0};
        std::vector<double> gyr_bias = {0, 0, 0};

        double g_cst = 9.8;

        std::string motion_type = "fast"; // "slow" | "fast"

        // FUTURE_WORK: Consider making a constructor from high-level option structure
    };


    class ImuSimulator{

        public:
            // Constructor that computes the simulated data `imu_data_`
            // according to the given options
            ImuSimulator(ImuSimulatorOption);

            // Get the simulated IMU data between the given times (in seconds)
            ImuData get(double start, double end);


            // Test a preintegrated measurements against the ground truth
            std::vector<double> testPreint(double start, double end, PreintMeas preint);

            // Get average velocity (at IMU timestamps)
            double getAvgVel();
            // Get average angular velocity (at IMU timestamps)
            double getAvgAngVel();

            // Get the trajectory length (numerically)
            double getTranslationDistance(const double from, const double to);
            double getOrientationDistance(const double from, const double to);

        private:
            // Stored the simulated data
            ImuData imu_data_;

            double kNumericalQuantum = 0.0001;

            // Store the trajectory parameters
            std::vector<SineProperties3> vel_sine_;
            SineProperties3 rot_sine_;
            celib::Vec3 g_vect_;

            // For the sake of potential visualisation
            std::vector<double> vel_offset_ = {0, 0, 0};

            // For random number generation
            celib::RandomGenerator rand_gen_;

            // Get the position values based on simulated sines
            std::vector<double> getPos(const double t);
            // Get the velocity values based on simulated sines
            std::vector<double> getVel(const double t);
            // Get the acceleration values based on simulated sines
            std::vector<double> getAcc(const double t);
            // Get the euler angle representation of orientation based on simulated sines
            std::vector<double> getEul(const double t);
            // Get angular velocity based on simulated sines
            std::vector<double> getAngVel(const double t);

    };


}


#endif
