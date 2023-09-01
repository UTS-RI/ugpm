/**
 *  Author: Cedric LE GENTIL
 *
 *  Copyright 2021 Cedric LE GENTIL
 *
 *  This is a simple example of how to generate the UGPMs
 *  
 *  Disclaimer:
 *  This code is not optimised neither for performance neither for maintainability.
 *  There might still be errors in the code, logic or procedure. Every feedback is welcomed.
 *
 *  For any further question or if you see any problem in that code
 *  le.gentil.cedric@gmail.com
 **/


#include <iostream>
#include <string>


#include <boost/program_options.hpp>

#include "../preint/preint.h"
#include "../preint/simulation.h"



int main(int argc, char* argv[]){

    ugpm::PreintOption preint_opt;


    // Program options
    boost::program_options::options_description opt_description("Allowed options");
    opt_description.add_options()
        ("help,h", "Produce help message")
        ("method,m", boost::program_options::value< std::string >(), "LPM | UGPM")
        ("length,l", boost::program_options::value< double >(), "Length simulated integration (default = 2.0)")
        ("quantum,q", boost::program_options::value< double >(), "Time quantum for piece-wise integration (useful for long integration intervals, -1 to deactivate (default = 0.2)")
        ("correlate,c", boost::program_options::bool_switch(&preint_opt.correlate), "Flag to activate the correlation of the biases")
        ;

    boost::program_options::variables_map var_map;
    boost::program_options::store(boost::program_options::parse_command_line(argc, argv, opt_description), var_map);
    boost::program_options::notify(var_map);    

    if(var_map.count("help")) {
        std::cout << opt_description << std::endl;
        return 1;
    }

    preint_opt.type = ugpm::UGPM;
    if(var_map.count("method"))
    {
        std::string type = var_map["method"].as<std::string>();
        preint_opt.type = ugpm::strToPreintType(type);
    }
    std::string to_print;
    switch (preint_opt.type)
    {
    case ugpm::LPM:
        to_print = "LPM";
        break;
    
    case ugpm::UGPM:
        to_print = "UGPM";
        break;
    
    default:
        break;
    }
    std::cout << "Preintegration tests with " << to_print << std::endl;

    double integration_length = 2;
    if(var_map.count("length"))
    {
        integration_length = var_map["length"].as<double>();
    }

    preint_opt.quantum = 0.2;
    if(var_map.count("quantum"))
    {
        preint_opt.quantum = var_map["quantum"].as<double>();
    }



    // Create an IMU simulator
    ugpm::ImuSimulatorOption sim_opt;
    sim_opt.acc_std = 0.02;
    sim_opt.gyr_std = 0.002;
    sim_opt.motion_type = "slow";
    ugpm::ImuSimulator imu_sim(sim_opt);

    // Create some fake data
    double overlap = 0.25;
    ugpm::RandomGenerator rand_gen;
    double start_t = rand_gen.randUniform(overlap,sim_opt.dataset_length - integration_length - overlap);
    double end_t = start_t + integration_length;
    ugpm::ImuData data = imu_sim.get(start_t-overlap, end_t+overlap);

    // Create a preintegration object
    preint_opt.min_freq = 1000;
    preint_opt.state_freq = 50;
    ugpm::PreintPrior prior;
    std::vector<std::vector<double> > t;
    std::vector<double> temp_t;
    temp_t.push_back(end_t);
    t.push_back(temp_t);


    {
        std::cout << "\n\n\n\n" << std::endl;
        std::cout << "-- Test removing some measurements (asynchronous sensors) --" << std::endl;
        ugpm::ImuData temp_data = data;
        temp_data.gyr.clear();
        for(int i = 0; i < data.gyr.size(); i+=2)
        {
            temp_data.gyr.push_back(data.gyr[i]);
        }

        ugpm::StopWatch stop_watch;
        stop_watch.start();
        ugpm::ImuPreintegration preint(temp_data, start_t, end_t, preint_opt, prior);
        stop_watch.stop();
        stop_watch.print();

        ugpm::PreintMeas preint_meas = preint.get();

        std::vector<double> error = imu_sim.testPreint(start_t, end_t, preint.get());

        std::cout << "Preintegration errors over window of " << preint_meas.dt << " seconds:" << std::endl;
        std::cout << "  Rotation [deg] = " << error[0]*180.0/M_PI << std::endl;
        std::cout << "  Velocity [m/s] = " << error[1] << std::endl;
        std::cout << "  Position [m]   = " << error[2] << std::endl;
    }





    {
        std::cout << "\n\n\n\n" << std::endl;
        std::cout << "-- Test using the prior feature --" << std::endl;
        ugpm::ImuData temp_data = data;
        ugpm::PreintPrior temp_prior = prior;
        temp_prior.acc_bias = {0.1, -0.15, 0.03};
        for(int i = 0; i < temp_data.acc.size(); ++i)
        {
            temp_data.acc[i].data[0] += temp_prior.acc_bias[0];
            temp_data.acc[i].data[1] += temp_prior.acc_bias[1];
            temp_data.acc[i].data[2] += temp_prior.acc_bias[2];
        }
        
        ugpm::StopWatch stop_watch;
        stop_watch.start();
        ugpm::ImuPreintegration preint(temp_data, start_t, t, preint_opt, prior);
        stop_watch.stop();
        stop_watch.print();
        stop_watch.reset();

        ugpm::PreintMeas preint_meas = preint.get(0, 0, 0.0, 0.0);

        std::vector<double> error = imu_sim.testPreint(start_t, end_t, preint.get(0,0));

        std::cout << "Preintegration errors without prior over window of " << preint_meas.dt << " seconds:" << std::endl;
        std::cout << "  Rotation [deg] = " << error[0]*180.0/M_PI << std::endl;
        std::cout << "  Velocity [m/s] = " << error[1] << std::endl;
        std::cout << "  Position [m]   = " << error[2] << std::endl;
        std::cout << std::endl;
        

        stop_watch.start();
        ugpm::ImuPreintegration preint2(temp_data, start_t, t, preint_opt, temp_prior);
        stop_watch.stop();
        stop_watch.print();

        preint_meas = preint2.get(0, 0, 0.0, 0.0);

        error = imu_sim.testPreint(start_t, end_t, preint2.get(0,0));

        std::cout << "Preintegration errors with prior over window of " << preint_meas.dt << " seconds:" << std::endl;
        std::cout << "  Rotation [deg] = " << error[0]*180.0/M_PI << std::endl;
        std::cout << "  Velocity [m/s] = " << error[1] << std::endl;
        std::cout << "  Position [m]   = " << error[2] << std::endl;
    }



    {
        std::cout << "\n\n\n\n" << std::endl;
        std::cout << "-- Test the IMU frequency check --" << std::endl;
        ugpm::ImuData temp_data = data;
        temp_data.acc[4].t = 0.0;

        ugpm::ImuPreintegration preint(temp_data, start_t, t, preint_opt, prior);
    }



    return 0;
}
