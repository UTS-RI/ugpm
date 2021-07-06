/**
 *  Author: Cedric LE GENTIL 
 *
 *  Copyright 2021 Cedric LE GENTIL
 * 
 *  For any further question, recommendation or contribution
 *  le.gentil.cedric@gmail.com
 **/
#include "sensor_input/imu_simulator.h"
#include "common/types.h"
#include "common/math_utils.h"

namespace celib
{

    // Simulator constructor that fill up the IMU data structure
    ImuSimulator::ImuSimulator(ImuSimulatorOption opt)
    {
        imu_data_.t_offset = 0.0;
        imu_data_.acc_var = opt.acc_std*opt.acc_std;
        imu_data_.gyr_var = opt.gyr_std*opt.gyr_std;

        // Simulate a gravity vector in the IMU's first frame
        celib::Vec2 g_angles = celib::Vec2::Random();
        g_vect_ <<
            std::sin(g_angles(0))*std::cos(g_angles(1)),
            std::sin(g_angles(0))*std::sin(g_angles(1)),
            std::cos(g_angles(0));
        g_vect_ *= opt.g_cst;


        // Pick trajectory characteristics depending on the simulator's option
        int number_sines = 3;
        double vel_amplitude_min;
        double vel_amplitude_max;
        double vel_freq_min;
        double vel_freq_max;
        double rot_amplitude_min;
        double rot_amplitude_max;
        double rot_freq_min;
        double rot_freq_max;
        if(opt.motion_type == "slow")
        {
            vel_amplitude_min = 0.3;
            vel_amplitude_max = 2;
            vel_freq_min = 0.05;
            vel_freq_max = 0.1;
            rot_amplitude_min = 0.5;
            rot_amplitude_max = 0.75;
            rot_freq_min = 0.05;
            rot_freq_max = 0.3;
        }
        else if(opt.motion_type == "fast")
        {
            vel_amplitude_min = 3;
            vel_amplitude_max = 5;
            vel_freq_min = 0.7;
            vel_freq_max = 1.5;
            rot_amplitude_min = 0.75;
            rot_amplitude_max = 1.0;
            rot_freq_min = 0.5;
            rot_freq_max = 1;

        }
        else
        {
            throw std::range_error("Unkown 'motion_type' given to the IMU simulator");
        }
        


        // Creating the sine properties from the above characteristics
        // The velocities are the combination of multiple random sines
        // that rougthly cover the spectrum of the characteristics
        // The rotations are based on one sine only (simpler math ^^)
        double vel_freq_quantum = vel_freq_max - vel_freq_min;
        vel_sine_.resize(number_sines); 
        for(int axis = 0; axis < 3; ++axis)
        {
            rot_sine_.amp[axis] = rand_gen_.randUniform(rot_amplitude_min, rot_amplitude_max);
            rot_sine_.freq[axis] = rand_gen_.randUniform(rot_freq_min, rot_freq_max);
            for(int i = 0; i < number_sines; ++i)
            {
                double temp_vel_freq_min = vel_freq_min + i*vel_freq_quantum;
                double temp_vel_freq_max = vel_freq_min + (i+1)*vel_freq_quantum;
                vel_sine_[i].amp[axis] = rand_gen_.randUniform(vel_amplitude_min, vel_amplitude_max);
                vel_sine_[i].freq[axis] = rand_gen_.randUniform(temp_vel_freq_min, temp_vel_freq_max);
            }

        }


        // Create the simulated IMU measurements with the specified noise
        int nb_samples = opt.imu_frequency*opt.dataset_length;
        imu_data_.acc.resize(nb_samples);
        imu_data_.gyr.resize(nb_samples);
        for(int i = 0; i < nb_samples; ++i)
        {
            double t = i*opt.dataset_length/nb_samples;
            
            auto acc = stdToCelibVec3(getAcc(t));
            auto eul = getEul(t);
            auto ang_vel = getAngVel(t);
            celib::Vec3 temp_acc = (celib::eulToRotMat(getEul(t))).transpose() * (acc - g_vect_);
            for(int axis = 0; axis < 3; ++axis)
            {
                imu_data_.acc[i].t = t;
                imu_data_.acc[i].data[axis] = temp_acc(axis) + rand_gen_.randGauss(0, opt.acc_std);
                imu_data_.gyr[i].t = t;
                imu_data_.gyr[i].data[axis] = ang_vel[axis] + rand_gen_.randGauss(0, opt.gyr_std);
            }

        }
    }

    // Get the collection of IMU samples contained between the two 
    ImuData ImuSimulator::get(double from, double to)
    {
        // Check if the query interval makes sense
        if(from < to)
        {
            return imu_data_.get(from, to);
        }
        // If the query inteval does not make sense throw an exception
        else
        {
            throw std::invalid_argument("The argument of ImuReader::Get are not consistent");
        }
    }



    // Get the position values based on simulated sines
    std::vector<double> ImuSimulator::getPos(const double t)
    {
        // Initialise the output vector
        std::vector<double> output = vel_offset_;
        for(auto& p : output) p*=t;

        // Apply the simulation model
        for(int i = 0; i < vel_sine_.size(); ++i)
        {
            for(int j = 0; j < 3; ++j)
            {
                output[j] += vel_sine_[i].amp[j]*sin(2*M_PI*vel_sine_[i].freq[j]*t)/(2*M_PI*vel_sine_[i].freq[j]);
            }
        }
        return output;
    }

    // Get the velocity values based on simulated sines
    std::vector<double> ImuSimulator::getVel(const double t)
    {
        // Initialise the output vector
        std::vector<double> output = vel_offset_;

        // Apply the simulation model
        for(int i = 0; i < vel_sine_.size(); ++i)
        {
            for(int j = 0; j < 3; ++j)
            {
                output[j] += vel_sine_[i].amp[j]*cos(2*M_PI*vel_sine_[i].freq[j]*t);
            }
        }
        return output;
    }

    // Get the acceleration values based on simulated sines
    std::vector<double> ImuSimulator::getAcc(const double t)
    {
        // Initialise the output vector
        std::vector<double> output = {0, 0, 0};

        // Apply the simulation model
        for(int i = 0; i < vel_sine_.size(); ++i)
        {
            for(int j = 0; j < 3; ++j)
            {
                output[j] -= vel_sine_[i].amp[j]*sin(2*M_PI*vel_sine_[i].freq[j]*t)*(2*M_PI*vel_sine_[i].freq[j]);
            }
        }
        return output;
    }

    // Get the euler angle representation of orientation based on simulated sines
    std::vector<double> ImuSimulator::getEul(const double t)
    {
        // Initialise the output vector
        std::vector<double> output = {0, 0, 0};
        // Simulate sine on orientation
        for(int j = 0; j < 3; ++j)
        {
            output[j] += rot_sine_.amp[j]*sin(2*M_PI*rot_sine_.freq[j]*t);
        }
        return output;
    }


    std::vector<double> ImuSimulator::getAngVel(const double t)
    {
        // Initialise the output vector
        std::vector<double> output(3);
        // Sinulate from euler sine
        output[0] = 2*rot_sine_.amp[2]*rot_sine_.freq[2]*M_PI*cos(2*M_PI*rot_sine_.freq[2]*t) - 2*rot_sine_.amp[0]*rot_sine_.freq[0]*M_PI*sin(rot_sine_.amp[1]*sin(2*M_PI*rot_sine_.freq[1]*t))*cos(2*M_PI*rot_sine_.freq[0]*t);
        output[1] = 2*rot_sine_.amp[1]*rot_sine_.freq[1]*M_PI*cos(2*M_PI*rot_sine_.freq[1]*t)*cos(rot_sine_.amp[2]*sin(2*M_PI*rot_sine_.freq[2]*t)) + 2*rot_sine_.amp[0]*rot_sine_.freq[0]*M_PI*sin(rot_sine_.amp[2]*sin(2*M_PI*rot_sine_.freq[2]*t))*cos(2*M_PI*rot_sine_.freq[0]*t)*cos(rot_sine_.amp[1]*sin(2*M_PI*rot_sine_.freq[1]*t));
        output[2] = 2*rot_sine_.amp[0]*rot_sine_.freq[0]*M_PI*cos(2*M_PI*rot_sine_.freq[0]*t)*cos(rot_sine_.amp[1]*sin(2*M_PI*rot_sine_.freq[1]*t))*cos(rot_sine_.amp[2]*sin(2*M_PI*rot_sine_.freq[2]*t)) - 2*rot_sine_.amp[1]*rot_sine_.freq[1]*M_PI*sin(rot_sine_.amp[2]*sin(2*M_PI*rot_sine_.freq[2]*t))*cos(2*M_PI*rot_sine_.freq[1]*t);

        return output;
    }


    double ImuSimulator::getAvgVel()
    {
        double avg_vel = 0;
        int nb_samples = imu_data_.acc.size();
        double coeff = (imu_data_.acc.back().t - imu_data_.acc[0].t ) / double(nb_samples-1.0); 
        for(int i = 0; i < nb_samples; ++i)
        {
            double t = i*coeff;
            auto vel = getVel(t);
            Eigen::Map<Vec3> vel_temp(&(vel[0]));
            avg_vel += vel_temp.norm();

        }
        avg_vel /= double(nb_samples);
        return avg_vel;
    }
    double ImuSimulator::getAvgAngVel()
    {
        double avg_vel = 0;
        int nb_samples = imu_data_.acc.size();
        double coeff = (imu_data_.acc.back().t - imu_data_.acc[0].t ) / double(nb_samples-1.0); 
        for(int i = 0; i < nb_samples; ++i)
        {
            double t = i*coeff;
            auto vel = getAngVel(t);
            Eigen::Map<Vec3> vel_temp(&(vel[0]));
            avg_vel += vel_temp.norm();

        }
        avg_vel /= double(nb_samples);
        return avg_vel;
    }


    // Test a preintegrated measurements against the ground truth
    // Returns the L2 error in Rot/Vel/Pos order
    std::vector<double> ImuSimulator::testPreint(double start, double end, PreintMeas preint)
    {
        std::vector<double> output;


        Mat3 rot_start_tr = eulToRotMat(getEul(start)).transpose();
        output.push_back( LogMap(preint.delta_R.transpose()*rot_start_tr*eulToRotMat(getEul(end)) ).norm() );
        
        double dt = end - start;
        output.push_back( (rot_start_tr*( stdToCelibVec3(getVel(end)) - stdToCelibVec3(getVel(start)) - (dt*g_vect_) ) - preint.delta_v).norm());
        output.push_back( (rot_start_tr*( stdToCelibVec3(getPos(end)) - stdToCelibVec3(getPos(start)) - (dt*stdToCelibVec3(getVel(start))) -(0.5*dt*dt*g_vect_) ) - preint.delta_p).norm());
        return output;
    }


    double ImuSimulator::getTranslationDistance(const double from, const double to)
    {
        double output = 0;
        Vec3 prev_pos = stdToCelibVec3(getPos(from));
        for(double t = (from+kNumericalQuantum); t < to; t += kNumericalQuantum)
        {
            Vec3 temp_pos = stdToCelibVec3(getPos(t));
            output += (temp_pos-prev_pos).norm();
            prev_pos = temp_pos;
        }
        output += (stdToCelibVec3(getPos(to))-prev_pos).norm();

        return output;
    }

    double ImuSimulator::getOrientationDistance(const double from, const double to)
    {
        double output = 0;

        Mat3 prev_rot = eulToRotMat(getEul(from));
        for(double t = (from+kNumericalQuantum); t < to; t += kNumericalQuantum)
        {
            Mat3 temp_rot = eulToRotMat(getEul(t));
            output += (LogMap(temp_rot.transpose() * prev_rot)).norm();
            prev_rot = temp_rot;
        }
        output += (LogMap(eulToRotMat(getEul(to)).transpose() * prev_rot)).norm();

        return output;
    }


}//namespace