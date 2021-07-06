/**
 *  Author: Cedric LE GENTIL
 *
 *  Copyright 2021 Cedric LE GENTIL
 *
 *  This run Monte Carlo simulation experiements for the paper
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
#include <yaml-cpp/yaml.h>
#include "sensor_input/imu_simulator.h"
#include "imu_preintegration/preintegration.h"
#include "common/random.h"
#include "common/utils.h"


celib::PreintMeas VanillaPreintegration(celib::ImuData data, double start_t, double t)
{
    celib::PreintMeas output;
    output.delta_R = celib::Mat3::Identity();
    output.delta_v = celib::Vec3::Zero();
    output.delta_p = celib::Vec3::Zero();
    output.cov = celib::Mat9::Zero();

    int counter = 0;
    while(start_t > data.gyr[counter].t)
    {
        counter++;
        if(counter == data.acc.size())
        {
            throw std::range_error("VanillaPreintegration: looks like integration window is out of data");
        }
    }

    double dt = data.gyr[counter].t - start_t;
    celib::Vec3 acc;
    acc << data.acc[counter-1].data[0],
           data.acc[counter-1].data[1],
           data.acc[counter-1].data[2];
    celib::Vec3 gyr;
    gyr << data.gyr[counter-1].data[0],
           data.gyr[counter-1].data[1],
           data.gyr[counter-1].data[2];


    celib::Mat6 cov_imu = celib::Mat6::Identity();
    cov_imu.block<3,3>(0,0) = data.gyr_var*celib::Mat3::Identity();
    cov_imu.block<3,3>(3,3) = data.acc_var*celib::Mat3::Identity();

    while(t > data.gyr[counter].t)
    {
        celib::Vec3 acc_rot = output.delta_R*acc;
        output.delta_p = output.delta_p + (output.delta_v*dt) + (acc_rot*dt*dt/2.0);
        output.delta_v = output.delta_v + (acc_rot*dt);
        celib::Mat3 e_R = celib::ExpMap(gyr*dt);
        celib::Mat3 j_r = celib::JacobianRighthandExpMap<double>(gyr*dt);
        celib::Mat9 A = celib::Mat9::Identity();
        celib::Mat9_6 B = celib::Mat9_6::Zero();
        celib::Mat3 skew_acc = celib::ToSkewSymMat(acc);
        A.block<3,3>(0,0) = e_R.transpose();
        A.block<3,3>(3,0) = -output.delta_R*skew_acc*dt;
        A.block<3,3>(6,0) = -output.delta_R*skew_acc*dt*dt/2.0;
        A.block<3,3>(6,3) = celib::Mat3::Identity();

        B.block<3,3>(0,0) = j_r*dt;
        B.block<3,3>(3,3) = output.delta_R*dt;
        B.block<3,3>(3,6) = output.delta_R*dt*dt/2.0;

        
        output.delta_R = output.delta_R*e_R;

        output.cov = (A*output.cov*A.transpose()) + (B*cov_imu*B.transpose());

        if(counter >= (data.acc.size()-1))
        {
            throw std::range_error("VanillaPreintegration: looks like integration window is out of data");
        }else{
            counter++;
            acc << data.acc[counter-1].data[0],
                data.acc[counter-1].data[1],
                data.acc[counter-1].data[2];
            gyr << data.gyr[counter-1].data[0],
                data.gyr[counter-1].data[1],
                data.gyr[counter-1].data[2];
            dt = data.acc[counter].t - data.acc[counter-1].t;
            if( (data.acc[counter-1].t < t) && (t < data.acc[counter].t) )
            {
                dt = t - data.acc[counter-1].t;
            }

        }
    }
    return output;
}



int main(int argc, char* argv[]){

    celib::PreintOption preint_opt;
    preint_opt.min_freq = 500;
    

    int nb_monte_carlo = 100;
    double overlap = 0.15;

    
    // Program options
    boost::program_options::options_description opt_description("Allowed options");
    opt_description.add_options()
        ("help,h", "Produce help message")
        ("experiment,e", boost::program_options::value< int >(), "1 = accuracy metrics (default), 2 = computation time, 3 = noise robustness, 4 = bias correction")
        ("nb_monte_carlo,n", boost::program_options::value< int >(), "(default = 100)")
        ;

    boost::program_options::variables_map var_map;
    boost::program_options::store(boost::program_options::parse_command_line(argc, argv, opt_description), var_map);
    boost::program_options::notify(var_map);    

    // Check help options
    if(var_map.count("help")) {
        std::cout << opt_description << std::endl;
        return 1;
    }
    
    int experiment_type = 1;
    if(var_map.count("experiment"))
    {
        experiment_type = var_map["experiment"].as<int>();
    }
    if(var_map.count("nb_monte_carlo"))
    {
        nb_monte_carlo = var_map["nb_monte_carlo"].as<int>();
    }


    if(experiment_type == 1)
    { // Metrics for the accuracy  with realistic noise

        std::cout << "ACCURACY EXPERIMENT" << std::endl;
        std::cout.precision(3);

        std::vector<double> durations = {0.05, 0.1, 0.5, 1.0};
        celib::ImuSimulatorOption sim_opt;
        sim_opt.acc_std = 0.04;
        sim_opt.gyr_std = 0.01;

        int nb_methods = 7;

        std::vector<std::vector<std::vector<double> > > rot_error_avg;
        std::vector<std::vector<std::vector<double> > > rot_error_std;
        std::vector<std::vector<std::vector<double> > > vel_error_avg;
        std::vector<std::vector<std::vector<double> > > vel_error_std;
        std::vector<std::vector<std::vector<double> > > pos_error_avg;
        std::vector<std::vector<std::vector<double> > > pos_error_std;
        std::vector<std::vector<std::vector<double> > > rot_rel_error_avg;
        std::vector<std::vector<std::vector<double> > > pos_rel_error_avg;

        std::vector<double> avg_ang_vel;
        std::vector<double> avg_vel;
        for(int type = 0; type < 2; ++type)
        {
            rot_error_avg.push_back(std::vector<std::vector<double> >());
            rot_error_std.push_back(std::vector<std::vector<double> >());
            vel_error_avg.push_back(std::vector<std::vector<double> >());
            vel_error_std.push_back(std::vector<std::vector<double> >());
            pos_error_avg.push_back(std::vector<std::vector<double> >());
            pos_error_std.push_back(std::vector<std::vector<double> >());
            rot_rel_error_avg.push_back(std::vector<std::vector<double> >());
            pos_rel_error_avg.push_back(std::vector<std::vector<double> >());

            
            avg_vel.push_back(0);
            avg_ang_vel.push_back(0);

            if(type == 1)
            {
                sim_opt.motion_type = "slow";
            }
            else
            {
                sim_opt.motion_type = "fast";
            }
            std::cout << "===============================================" << std::endl;
            std::cout << "Run results for " << sim_opt.motion_type << " motion" << std::endl;
            for(int d = 0; d < durations.size(); ++d)
            {

                rot_error_avg[type].push_back(std::vector<double>(nb_methods,0.0));
                rot_error_std[type].push_back(std::vector<double>(nb_methods,0.0));
                vel_error_avg[type].push_back(std::vector<double>(nb_methods,0.0));
                vel_error_std[type].push_back(std::vector<double>(nb_methods,0.0));
                pos_error_avg[type].push_back(std::vector<double>(nb_methods,0.0));
                pos_error_std[type].push_back(std::vector<double>(nb_methods,0.0));
                rot_rel_error_avg[type].push_back(std::vector<double>(nb_methods,0.0));
                pos_rel_error_avg[type].push_back(std::vector<double>(nb_methods,0.0));
                std::cout << "Duration : " << durations[d] << " s" << std::endl;


                std::vector<std::vector<double> > rot_errors(nb_methods);
                std::vector<std::vector<double> > vel_errors(nb_methods);
                std::vector<std::vector<double> > pos_errors(nb_methods);
                std::vector<std::vector<double> > rot_rel_errors(nb_methods);
                std::vector<std::vector<double> > pos_rel_errors(nb_methods);
                for(int i = 0; i < nb_monte_carlo; ++i)
                {
                    std::cout << i << "." << std::flush;
                    celib::ImuSimulator imu_sim(sim_opt);
                    avg_vel[type] += imu_sim.getAvgVel();
                    avg_ang_vel[type] += imu_sim.getAvgAngVel();
                    celib::RandomGenerator rand_gen;
                    double start_t = rand_gen.randUniform(overlap,sim_opt.dataset_length - durations[d] - overlap);
                    double end_t = start_t + durations[d];
                    auto data = imu_sim.get(start_t-overlap, end_t+overlap);

                    double pos_dist = imu_sim.getTranslationDistance(start_t, end_t);
                    double rot_dist = imu_sim.getOrientationDistance(start_t, end_t);


                    for(int method = 0; method < nb_methods; ++method)
                    {
                        celib::PreintMeas preint;
                        if(method == 0)
                        {
                            preint = VanillaPreintegration(data, start_t, end_t);
                        }
                        else
                        {
                            // Create a preintegration object
                            celib::PreintPrior prior;
                            std::vector<std::vector<double> > t;
                            std::vector<double> temp_t;
                            temp_t.push_back(end_t);
                            t.push_back(temp_t);
                            if(method == 1)
                            {
                                preint_opt.type = celib::LPM;
                                preint_opt.train_gpm = false;
                                preint_opt.quantum = -1;
                            }
                            else if(method == 2)
                            {
                                preint_opt.type = celib::GPM;
                                preint_opt.train_gpm = false;
                                preint_opt.quantum = -1;
                            }
                            else if(method == 3)
                            {
                                preint_opt.type = celib::GPM;
                                preint_opt.train_gpm = true;
                                preint_opt.quantum = -1;
                            }
                            else if(method == 4)
                            {
                                preint_opt.type = celib::UGPM;
                                preint_opt.train_gpm = false;
                                preint_opt.quantum = -1;
                            }
                            else if(method == 5)
                            {
                                preint_opt.type = celib::UGPM;
                                preint_opt.train_gpm = true;
                                preint_opt.quantum = -1;
                            }
                            else
                            {
                                preint_opt.type = celib::UGPM;
                                preint_opt.train_gpm = false;
                                preint_opt.quantum = 0.2;
                            }
                            celib::ImuPreintegration imu_preint(data, start_t, t, preint_opt, prior);
                            preint = imu_preint.get(0,0);

                        }
                        auto error = imu_sim.testPreint(start_t, end_t, preint);
                        rot_errors[method].push_back(error[0]);
                        vel_errors[method].push_back(error[1]);
                        pos_errors[method].push_back(error[2]);
                        rot_rel_errors[method].push_back(error[0]/rot_dist);
                        pos_rel_errors[method].push_back(error[2]/pos_dist);
                    }
                }
                std::cout << std::endl;

                for(int method = 0; method < nb_methods; ++method)
                {
                    for(int i = 0; i < nb_monte_carlo; ++i)
                    {
                        rot_error_avg[type][d][method] += rot_errors[method][i];
                        vel_error_avg[type][d][method] += vel_errors[method][i];
                        pos_error_avg[type][d][method] += pos_errors[method][i];
                        rot_rel_error_avg[type][d][method] += rot_rel_errors[method][i];
                        pos_rel_error_avg[type][d][method] += pos_rel_errors[method][i];
                    }
                    rot_error_avg[type][d][method] /= nb_monte_carlo;
                    vel_error_avg[type][d][method] /= nb_monte_carlo;
                    pos_error_avg[type][d][method] /= nb_monte_carlo;
                    rot_rel_error_avg[type][d][method] /= nb_monte_carlo;
                    pos_rel_error_avg[type][d][method] /= nb_monte_carlo;
                    for(int i = 0; i < nb_monte_carlo; ++i)
                    {
                        rot_error_std[type][d][method] += std::pow(rot_errors[method][i] - rot_error_avg[type][d][method],2);
                        vel_error_std[type][d][method] += std::pow(vel_errors[method][i] - vel_error_avg[type][d][method],2);
                        pos_error_std[type][d][method] += std::pow(pos_errors[method][i] - pos_error_avg[type][d][method],2);
                    }
                    rot_error_std[type][d][method] = std::sqrt(rot_error_std[type][d][method]/nb_monte_carlo);
                    vel_error_std[type][d][method] = std::sqrt(vel_error_std[type][d][method]/nb_monte_carlo);
                    pos_error_std[type][d][method] = std::sqrt(pos_error_std[type][d][method]/nb_monte_carlo);

                }
            }
        }


        std::cout << std::endl << "RESULT " << std::endl << "(Methods: 0 = PM, 1 = LPM, 2 = GPM, 3 = UGPM, 4 = GPM trained, 5 = UGPM trained, 6 = UGPM per chunk" << std::endl;
        for(int type = 0; type < 2; ++type)
        {
            avg_vel[type] /= double(nb_monte_carlo);
            avg_ang_vel[type] /= double(nb_monte_carlo);

            if(type == 1) std::cout << std::endl << std::endl << "Slow motion" << std::endl;
            if(type == 0) std::cout << std::endl << std::endl << "Fast motion" << std::endl;
            std::cout << "Avg velocity : " << avg_vel[type] << " m/s    Avg ang velocity : " << avg_ang_vel[type] << " rad/s" << std::endl;
            std::cout << " Rot and Pos error ====" << std::endl;
            std::cout << "Duration \\ method";
            for(int i = 0; i < nb_methods; ++i) std::cout << "      " << i;
            std::cout << std::endl;
            for(int d = 0; d < durations.size(); ++d)
            {
                std::cout << "\\\\ \\hline \\hline" << std::endl;
                std::cout << "\\multirow{4}{*}{" << durations[d] << "} & Rot abs. er. [$\\,\\mathrm{mrad}$]      ";
                for(int method = 0; method < nb_methods; ++method)
                {
                    std::cout << " &  " << "\\scriptsize " << rot_error_avg[type][d][method]*1000.0 << " $\\pm$ " << rot_error_std[type][d][method]*1000.0;

                }
                std::cout << std::endl;
                std::cout << "\\\\ & Rot rel. er.      ";
                for(int method = 0; method < nb_methods; ++method)
                {
                    std::cout << " &  " << "\\scriptsize " << rot_rel_error_avg[type][d][method]*100.0 << "\\%";

                }
                std::cout << std::endl;
                std::cout << "\\\\ \\cline{2-9}" << std::endl;
                std::cout << " & Pos abs. er. [$\\,\\mathrm{mm}$]      ";
                for(int method = 0; method < nb_methods; ++method)
                {
                    std::cout << " &  " << "\\scriptsize " << pos_error_avg[type][d][method]*1000.0 << " $\\pm$ " << pos_error_std[type][d][method]*1000.0;
                }
                std::cout << std::endl;
                std::cout << "\\\\ & Pos rel. er.      ";
                for(int method = 0; method < nb_methods; ++method)
                {
                    std::cout << " &  " << "\\scriptsize " << pos_rel_error_avg[type][d][method]*100.0 << "\\%";

                }
                std::cout << std::endl;
            }
            std::cout << std::endl << " Rot error ====" << std::endl;
            std::cout << "Duration \\ method";
            for(int i = 0; i < nb_methods; ++i) std::cout << "      " << i;
            std::cout << std::endl;
            for(int d = 0; d < durations.size(); ++d)
            {
                std::cout << durations[d] << "\\\\              ";
                for(int method = 0; method < nb_methods; ++method)
                {
                    std::cout << " &  " << rot_error_avg[type][d][method] << " $\\pm$ " << rot_error_std[type][d][method];
                }
                std::cout << std::endl;
            }
            std::cout << std::endl << " Vel error ====" << std::endl;
            std::cout << "Duration \\ method";
            for(int i = 0; i < nb_methods; ++i) std::cout << "      " << i;
            std::cout << std::endl;
            for(int d = 0; d < durations.size(); ++d)
            {
                std::cout << durations[d] << "\\\\              ";
                for(int method = 0; method < nb_methods; ++method)
                {
                    std::cout << " &  " << vel_error_avg[type][d][method] << " $\\pm$ " << vel_error_std[type][d][method];
                }
                std::cout << std::endl;
            }
            std::cout << std::endl << " Pos error ====" << std::endl;
            std::cout << "Duration \\ method";
            for(int i = 0; i < nb_methods; ++i) std::cout << "      " << i;
            std::cout << std::endl;
            for(int d = 0; d < durations.size(); ++d)
            {
                std::cout << durations[d] << "              ";
                for(int method = 0; method < nb_methods; ++method)
                {
                    std::cout << " &  " << pos_error_avg[type][d][method] << " $\\pm$ " << pos_error_std[type][d][method];
                }
                std::cout << std::endl;
            }
        }
    }

    if(experiment_type == 2)
    { // Metrics for the computation time  with realistic noise

        std::cout << "COMPUTATION TIME EXPERIMENT" << std::endl;
        std::cout.precision(4);

        std::vector<double> durations = {0.05, 0.1, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5};
        celib::ImuSimulatorOption sim_opt;
        sim_opt.acc_std = 0.04;
        sim_opt.gyr_std = 0.01;

        int nb_methods = 7;

        std::vector<std::vector<double> > time_average;

        celib::StopWatch stop_watch;

        std::cout << "===============================================" << std::endl;
        for(int d = 0; d < durations.size(); ++d)
        {

            time_average.push_back(std::vector<double>(nb_methods,0.0));
            std::cout << "Duration : " << durations[d] << " s" << std::endl;


            for(int i = 0; i < nb_monte_carlo; ++i)
            {
                std::cout << i << "." << std::flush;
                celib::ImuSimulator imu_sim(sim_opt);
                celib::RandomGenerator rand_gen;
                double start_t = rand_gen.randUniform(overlap,sim_opt.dataset_length - durations[d] - overlap);
                double end_t = start_t + durations[d];
                auto data = imu_sim.get(start_t-overlap, end_t+overlap);

                for(int method = 0; method < nb_methods; ++method)
                {
                    celib::PreintMeas preint;
                    double time;
                    stop_watch.reset();
                    if(method == 0)
                    {
                        stop_watch.start();
                        preint = VanillaPreintegration(data, start_t, end_t);
                        time = stop_watch.stop();
                    }
                    else
                    {
                        // Create a preintegration object
                        celib::PreintPrior prior;
                        std::vector<std::vector<double> > t;
                        std::vector<double> temp_t;
                        temp_t.push_back(end_t);
                        t.push_back(temp_t);
                        if(method == 1)
                        {
                            preint_opt.type = celib::LPM;
                            preint_opt.train_gpm = false;
                            preint_opt.quantum = -1;
                        }
                        else if(method == 2)
                        {
                            preint_opt.type = celib::GPM;
                            preint_opt.train_gpm = false;
                            preint_opt.quantum = -1;
                        }
                        else if(method == 3)
                        {
                            preint_opt.type = celib::GPM;
                            preint_opt.train_gpm = true;
                            preint_opt.quantum = -1;
                        }
                        else if(method == 4)
                        {
                            preint_opt.type = celib::UGPM;
                            preint_opt.train_gpm = false;
                            preint_opt.quantum = -1;
                        }
                        else if(method == 5)
                        {
                            preint_opt.type = celib::UGPM;
                            preint_opt.train_gpm = true;
                            preint_opt.quantum = -1;
                        }
                        else
                        {
                            preint_opt.type = celib::UGPM;
                            preint_opt.train_gpm = false;
                            preint_opt.quantum = 0.2;
                        }
                        stop_watch.start();
                        celib::ImuPreintegration imu_preint(data, start_t, t, preint_opt, prior);
                        preint = imu_preint.get(0,0);
                        time = stop_watch.stop();
                    }
                    time_average[d][method] += time;
                    auto error = imu_sim.testPreint(start_t, end_t, preint);
                }
            }
            std::cout << std::endl;

            for(int method = 0; method < nb_methods; ++method)
            {
                time_average[d][method] /= nb_monte_carlo;
            }
        }
        


        std::cout << std::endl << "RESULT computation time " << std::endl << "(Methods: 0 = PM, 1 = LPM, 2 = GPM, 3 = UGPM, 4 = GPM trained, 5 = UGPM trained, 6 = UGPM per chunk" << std::endl;
        std::cout << "Duration (ms)\\ method";
        for(int i = 0; i < nb_methods; ++i) std::cout << "      " << i;
        std::cout << std::endl;
        for(int d = 0; d < durations.size(); ++d)
        {
            std::cout << "\\scriptsize " << durations[d]*1000.0 << "            ";
            for(int method = 0; method < nb_methods; ++method)
            {
                std::cout << " &  \\scriptsize " << time_average[d][method];
            }
            std::cout << std::endl;
        }
    }


    if(experiment_type == 3)
    { // Metrics for the robustness to noise

        std::cout << "NOISE ROBUSTNESS EXPERIMENT" << std::endl;


        {
            celib::ImuSimulatorOption sim_opt;
            std::vector<double> noise_factor = {0.001, 0.33, 0.66, 1, 1.33, 1.66, 2, 2.33, 2.66, 3, 3.33, 3.66, 4, 4.33, 4.66, 5};
            double acc_std = 0.02;
            double gyr_std = 0.01;
            sim_opt.motion_type = "fast";
            double duration = 1;
            int nb_methods = 4;

            std::vector<std::vector<std::vector<double> > > rot_error_avg;
            std::vector<std::vector<std::vector<double> > > rot_error_std;
            std::vector<std::vector<std::vector<double> > > vel_error_avg;
            std::vector<std::vector<std::vector<double> > > vel_error_std;
            std::vector<std::vector<std::vector<double> > > pos_error_avg;
            std::vector<std::vector<std::vector<double> > > pos_error_std;
            for(int type = 0; type < 2; ++type)
            {
                rot_error_avg.push_back(std::vector<std::vector<double> >());
                rot_error_std.push_back(std::vector<std::vector<double> >());
                vel_error_avg.push_back(std::vector<std::vector<double> >());
                vel_error_std.push_back(std::vector<std::vector<double> >());
                pos_error_avg.push_back(std::vector<std::vector<double> >());
                pos_error_std.push_back(std::vector<std::vector<double> >());

                std::cout << "===============================================" << std::endl;
                std::cout << "Run results for " << sim_opt.motion_type << " motion" << std::endl;
                for(int d = 0; d < noise_factor.size(); ++d)
                {

                    if(type == 0)
                    {
                        sim_opt.acc_std = noise_factor[d] * acc_std;
                        sim_opt.gyr_std = 0.00001;
                    }
                    else
                    {
                        sim_opt.acc_std = 0.00001;
                        sim_opt.gyr_std = noise_factor[d] * gyr_std;
                    }

                    rot_error_avg[type].push_back(std::vector<double>(nb_methods,0.0));
                    rot_error_std[type].push_back(std::vector<double>(nb_methods,0.0));
                    vel_error_avg[type].push_back(std::vector<double>(nb_methods,0.0));
                    vel_error_std[type].push_back(std::vector<double>(nb_methods,0.0));
                    pos_error_avg[type].push_back(std::vector<double>(nb_methods,0.0));
                    pos_error_std[type].push_back(std::vector<double>(nb_methods,0.0));
                    std::cout << "Noise factor : " << noise_factor[d] << std::endl;


                    std::vector<std::vector<double> > rot_errors(nb_methods);
                    std::vector<std::vector<double> > vel_errors(nb_methods);
                    std::vector<std::vector<double> > pos_errors(nb_methods);
                    for(int i = 0; i < nb_monte_carlo; ++i)
                    {
                        std::cout << i << "." << std::flush;
                        celib::ImuSimulator imu_sim(sim_opt);
                        celib::RandomGenerator rand_gen;
                        double start_t = rand_gen.randUniform(overlap,sim_opt.dataset_length - duration - overlap);
                        double end_t = start_t + duration;
                        auto data = imu_sim.get(start_t-overlap, end_t+overlap);

                        for(int method = 0; method < nb_methods; ++method)
                        {
                            celib::PreintMeas preint;
                            if(method == 0)
                            {
                                preint = VanillaPreintegration(data, start_t, end_t);
                            }
                            else
                            {
                                // Create a preintegration object
                                celib::PreintPrior prior;
                                std::vector<std::vector<double> > t;
                                std::vector<double> temp_t;
                                temp_t.push_back(end_t);
                                t.push_back(temp_t);
                                if(method == 1)
                                {
                                    preint_opt.type = celib::LPM;
                                    preint_opt.train_gpm = false;
                                }
                                else if(method == 2)
                                {
                                    preint_opt.type = celib::GPM;
                                    preint_opt.train_gpm = true;
                                }
                                else
                                {
                                    preint_opt.type = celib::UGPM;
                                    preint_opt.train_gpm = true;
                                }
                                celib::ImuPreintegration imu_preint(data, start_t, t, preint_opt, prior);
                                preint = imu_preint.get(0,0);

                            }
                            auto error = imu_sim.testPreint(start_t, end_t, preint);
                            rot_errors[method].push_back(error[0]);
                            vel_errors[method].push_back(error[1]);
                            pos_errors[method].push_back(error[2]);
                        }
                    }
                    std::cout << std::endl;

                    for(int method = 0; method < nb_methods; ++method)
                    {
                        for(int i = 0; i < nb_monte_carlo; ++i)
                        {
                            rot_error_avg[type][d][method] += rot_errors[method][i];
                            vel_error_avg[type][d][method] += vel_errors[method][i];
                            pos_error_avg[type][d][method] += pos_errors[method][i];
                        }
                        rot_error_avg[type][d][method] /= nb_monte_carlo;
                        vel_error_avg[type][d][method] /= nb_monte_carlo;
                        pos_error_avg[type][d][method] /= nb_monte_carlo;
                        for(int i = 0; i < nb_monte_carlo; ++i)
                        {
                            rot_error_std[type][d][method] += std::pow(rot_errors[method][i] - rot_error_avg[type][d][method],2);
                            vel_error_std[type][d][method] += std::pow(vel_errors[method][i] - vel_error_avg[type][d][method],2);
                            pos_error_std[type][d][method] += std::pow(pos_errors[method][i] - pos_error_avg[type][d][method],2);
                        }
                        rot_error_std[type][d][method] = std::sqrt(rot_error_std[type][d][method]/nb_monte_carlo);
                        vel_error_std[type][d][method] = std::sqrt(vel_error_std[type][d][method]/nb_monte_carlo);
                        pos_error_std[type][d][method] = std::sqrt(pos_error_std[type][d][method]/nb_monte_carlo);

                    }
                }
            }
            std::cout << std::endl << "RESULTS " << std::endl << "(Methods: PM, LPM, GPM trained, UGPM trained" << std::endl;
            for(int type = 0; type < 2; ++type)
            {
                if(type == 0) std::cout << std::endl << std::endl << "Acc noise" << std::endl;
                if(type == 1) std::cout << std::endl << std::endl << "Gyr noise" << std::endl;
                std::cout << " Rot error ====" << std::endl;
                for(int i = 0; i < nb_methods; ++i) std::cout << "      " << i;
                std::cout << std::endl;
                for(int method = 0; method < nb_methods; ++method)
                {
                    for(int d = 0; d < noise_factor.size(); ++d)
                    {
                        if(type==0)
                        {
                                std::cout << " (" << acc_std*noise_factor[d] << "," << rot_error_avg[type][d][method] << ")";
                        }else
                        {
                                std::cout << " (" << gyr_std*noise_factor[d] << "," << rot_error_avg[type][d][method] << ")";
                        }
                    }
                    std::cout << std::endl;
                }
                std::cout << " Pos error ====" << std::endl;
                for(int i = 0; i < nb_methods; ++i) std::cout << "      " << i;
                std::cout << std::endl;
                for(int method = 0; method < nb_methods; ++method)
                {
                    for(int d = 0; d < noise_factor.size(); ++d)
                    {
                        if(type==0)
                        {
                                std::cout << " (" << acc_std*noise_factor[d] << "," << pos_error_avg[type][d][method] << ")";
                        }else
                        {
                                std::cout << " (" << gyr_std*noise_factor[d] << "," << pos_error_avg[type][d][method] << ")";
                        }
                    }
                    std::cout << std::endl;
                }
            }
        }
        {
            celib::ImuSimulatorOption sim_opt;
            std::vector<double> noise_factor = {0.0001, 0.33, 0.66, 1, 1.33, 1.66, 2, 2.33, 2.66, 3, 3.33, 3.66, 4, 4.33, 4.66, 5};
            double acc_std = 0.02;
            double gyr_std = 0.01;
            sim_opt.motion_type = "slow";
            double duration = 1;
            int nb_methods = 4;

            std::vector<std::vector<std::vector<double> > > rot_error_avg;
            std::vector<std::vector<std::vector<double> > > rot_error_std;
            std::vector<std::vector<std::vector<double> > > vel_error_avg;
            std::vector<std::vector<std::vector<double> > > vel_error_std;
            std::vector<std::vector<std::vector<double> > > pos_error_avg;
            std::vector<std::vector<std::vector<double> > > pos_error_std;
            for(int type = 0; type < 2; ++type)
            {
                rot_error_avg.push_back(std::vector<std::vector<double> >());
                rot_error_std.push_back(std::vector<std::vector<double> >());
                vel_error_avg.push_back(std::vector<std::vector<double> >());
                vel_error_std.push_back(std::vector<std::vector<double> >());
                pos_error_avg.push_back(std::vector<std::vector<double> >());
                pos_error_std.push_back(std::vector<std::vector<double> >());

                std::cout << "===============================================" << std::endl;
                std::cout << "Run results for " << sim_opt.motion_type << " motion" << std::endl;
                for(int d = 0; d < noise_factor.size(); ++d)
                {

                    if(type == 0)
                    {
                        sim_opt.acc_std = noise_factor[d] * acc_std;
                        sim_opt.gyr_std = 0.00001;
                    }
                    else
                    {
                        sim_opt.acc_std = 0.00001;
                        sim_opt.gyr_std = noise_factor[d] * gyr_std;
                    }

                    rot_error_avg[type].push_back(std::vector<double>(nb_methods,0.0));
                    rot_error_std[type].push_back(std::vector<double>(nb_methods,0.0));
                    vel_error_avg[type].push_back(std::vector<double>(nb_methods,0.0));
                    vel_error_std[type].push_back(std::vector<double>(nb_methods,0.0));
                    pos_error_avg[type].push_back(std::vector<double>(nb_methods,0.0));
                    pos_error_std[type].push_back(std::vector<double>(nb_methods,0.0));
                    std::cout << "Noise factor : " << noise_factor[d] << std::endl;


                    std::vector<std::vector<double> > rot_errors(nb_methods);
                    std::vector<std::vector<double> > vel_errors(nb_methods);
                    std::vector<std::vector<double> > pos_errors(nb_methods);
                    for(int i = 0; i < nb_monte_carlo; ++i)
                    {
                        std::cout << i << "." << std::flush;
                        celib::ImuSimulator imu_sim(sim_opt);
                        celib::RandomGenerator rand_gen;
                        double start_t = rand_gen.randUniform(overlap,sim_opt.dataset_length - duration - overlap);
                        double end_t = start_t + duration;
                        auto data = imu_sim.get(start_t-overlap, end_t+overlap);

                        for(int method = 0; method < nb_methods; ++method)
                        {
                            celib::PreintMeas preint;
                            if(method == 0)
                            {
                                preint = VanillaPreintegration(data, start_t, end_t);
                            }
                            else
                            {
                                // Create a preintegration object
                                celib::PreintPrior prior;
                                std::vector<std::vector<double> > t;
                                std::vector<double> temp_t;
                                temp_t.push_back(end_t);
                                t.push_back(temp_t);
                                if(method == 1)
                                {
                                    preint_opt.type = celib::LPM;
                                    preint_opt.train_gpm = false;
                                }
                                else if(method == 2)
                                {
                                    preint_opt.type = celib::GPM;
                                    preint_opt.train_gpm = true;
                                }
                                else
                                {
                                    preint_opt.type = celib::UGPM;
                                    preint_opt.train_gpm = true;
                                }
                                celib::ImuPreintegration imu_preint(data, start_t, t, preint_opt, prior);
                                preint = imu_preint.get(0,0);

                            }
                            auto error = imu_sim.testPreint(start_t, end_t, preint);
                            rot_errors[method].push_back(error[0]);
                            vel_errors[method].push_back(error[1]);
                            pos_errors[method].push_back(error[2]);
                        }
                    }
                    std::cout << std::endl;

                    for(int method = 0; method < nb_methods; ++method)
                    {
                        for(int i = 0; i < nb_monte_carlo; ++i)
                        {
                            rot_error_avg[type][d][method] += rot_errors[method][i];
                            vel_error_avg[type][d][method] += vel_errors[method][i];
                            pos_error_avg[type][d][method] += pos_errors[method][i];
                        }
                        rot_error_avg[type][d][method] /= nb_monte_carlo;
                        vel_error_avg[type][d][method] /= nb_monte_carlo;
                        pos_error_avg[type][d][method] /= nb_monte_carlo;
                        for(int i = 0; i < nb_monte_carlo; ++i)
                        {
                            rot_error_std[type][d][method] += std::pow(rot_errors[method][i] - rot_error_avg[type][d][method],2);
                            vel_error_std[type][d][method] += std::pow(vel_errors[method][i] - vel_error_avg[type][d][method],2);
                            pos_error_std[type][d][method] += std::pow(pos_errors[method][i] - pos_error_avg[type][d][method],2);
                        }
                        rot_error_std[type][d][method] = std::sqrt(rot_error_std[type][d][method]/nb_monte_carlo);
                        vel_error_std[type][d][method] = std::sqrt(vel_error_std[type][d][method]/nb_monte_carlo);
                        pos_error_std[type][d][method] = std::sqrt(pos_error_std[type][d][method]/nb_monte_carlo);

                    }
                }
            }
            std::cout << std::endl << "RESULTS " << std::endl << "(Methods: PM, LPM, GPM trained, UGPM trained" << std::endl;
            for(int type = 0; type < 2; ++type)
            {
                if(type == 0) std::cout << std::endl << std::endl << "Acc noise" << std::endl;
                if(type == 1) std::cout << std::endl << std::endl << "Gyr noise" << std::endl;
                std::cout << " Rot error ====" << std::endl;
                for(int i = 0; i < nb_methods; ++i) std::cout << "      " << i;
                std::cout << std::endl;
                for(int method = 0; method < nb_methods; ++method)
                {
                    for(int d = 0; d < noise_factor.size(); ++d)
                    {
                        if(type==0)
                        {
                                std::cout << " (" << acc_std*noise_factor[d] << "," << rot_error_avg[type][d][method] << ")";
                        }else
                        {
                                std::cout << " (" << gyr_std*noise_factor[d] << "," << rot_error_avg[type][d][method] << ")";
                        }
                    }
                    std::cout << std::endl;
                }
                std::cout << " Pos error ====" << std::endl;
                for(int i = 0; i < nb_methods; ++i) std::cout << "      " << i;
                std::cout << std::endl;
                for(int method = 0; method < nb_methods; ++method)
                {
                    for(int d = 0; d < noise_factor.size(); ++d)
                    {
                        if(type==0)
                        {
                                std::cout << " (" << acc_std*noise_factor[d] << "," << pos_error_avg[type][d][method] << ")";
                        }else
                        {
                                std::cout << " (" << gyr_std*noise_factor[d] << "," << pos_error_avg[type][d][method] << ")";
                        }
                    }
                    std::cout << std::endl;
                }
            }
        }
    }

    if(experiment_type == 4)
    { // Metrics for the robustness to noise

        std::cout << "BIAS CORRECTION EXPERIMENT" << std::endl;

        celib::ImuSimulatorOption sim_opt;
        std::vector<double> gyr_bias_norm = {0.01, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0};
        std::vector<double> acc_bias_norm = {0.01, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0};
        std::vector<double> dt_bias = {0.001, 0.005, 0.01, 0.02, 0.04, 0.06, 0.08 , 0.1};
        sim_opt.acc_std = 0.04;
        sim_opt.gyr_std = 0.01;

        double duration = 1.0;
        std::cout.precision(3);

        std::vector<std::vector<double> > gyr_rot_error_avg;
        std::vector<std::vector<double> > gyr_pos_error_avg;
        std::vector<std::vector<double> > acc_rot_error_avg;
        std::vector<std::vector<double> > acc_pos_error_avg;
        std::vector<std::vector<double> > dt_rot_error_avg;
        std::vector<std::vector<double> > dt_pos_error_avg;
        std::vector<std::vector<double> > gyr_rot_error_raw_avg;
        std::vector<std::vector<double> > gyr_pos_error_raw_avg;
        std::vector<std::vector<double> > acc_rot_error_raw_avg;
        std::vector<std::vector<double> > acc_pos_error_raw_avg;
        std::vector<std::vector<double> > dt_rot_error_raw_avg;
        std::vector<std::vector<double> > dt_pos_error_raw_avg;

        for(int type = 0; type < 2; ++type)
        {
            gyr_rot_error_avg.push_back(std::vector<double>());
            gyr_pos_error_avg.push_back(std::vector<double>());
            acc_rot_error_avg.push_back(std::vector<double>());
            acc_pos_error_avg.push_back(std::vector<double>());
            dt_rot_error_avg.push_back(std::vector<double>());
            dt_pos_error_avg.push_back(std::vector<double>());
            gyr_rot_error_raw_avg.push_back(std::vector<double>());
            gyr_pos_error_raw_avg.push_back(std::vector<double>());
            acc_rot_error_raw_avg.push_back(std::vector<double>());
            acc_pos_error_raw_avg.push_back(std::vector<double>());
            dt_rot_error_raw_avg.push_back(std::vector<double>());
            dt_pos_error_raw_avg.push_back(std::vector<double>());
            

            if(type == 0)
            {
                sim_opt.motion_type = "slow";
            }
            else
            {
                sim_opt.motion_type = "fast";
            }
            std::cout << "===============================================" << std::endl;
            std::cout << "Run results for " << sim_opt.motion_type << " motion" << std::endl;

            std::vector<std::vector<double> > gyr_rot_error(gyr_bias_norm.size());
            std::vector<std::vector<double> > gyr_pos_error(gyr_bias_norm.size());
            std::vector<std::vector<double> > acc_rot_error(acc_bias_norm.size());
            std::vector<std::vector<double> > acc_pos_error(acc_bias_norm.size());
            std::vector<std::vector<double> > dt_rot_error(acc_bias_norm.size());
            std::vector<std::vector<double> > dt_pos_error(acc_bias_norm.size());
            std::vector<std::vector<double> > gyr_rot_raw_error(gyr_bias_norm.size());
            std::vector<std::vector<double> > gyr_pos_raw_error(gyr_bias_norm.size());
            std::vector<std::vector<double> > acc_rot_raw_error(acc_bias_norm.size());
            std::vector<std::vector<double> > acc_pos_raw_error(acc_bias_norm.size());
            std::vector<std::vector<double> > dt_rot_raw_error(acc_bias_norm.size());
            std::vector<std::vector<double> > dt_pos_raw_error(acc_bias_norm.size());

            for(int i = 0; i < nb_monte_carlo; ++i)
            {
                std::cout << i << "." << std::flush;
                celib::ImuSimulator imu_sim(sim_opt);
                celib::RandomGenerator rand_gen;
                double start_t = rand_gen.randUniform(overlap,sim_opt.dataset_length - duration - overlap);
                double end_t = start_t + duration;
                auto data = imu_sim.get(start_t-overlap, end_t+overlap);
                auto data_save = data;

                double pos_dist = imu_sim.getTranslationDistance(start_t, end_t);
                double rot_dist = imu_sim.getOrientationDistance(start_t, end_t);

                celib::Vec3 unit_vec = celib::Vec3::Random();
                unit_vec = unit_vec / (unit_vec.norm());

                for(int ibw = 0; ibw < gyr_bias_norm.size(); ++ibw)
                {
                    data = data_save;
                    celib::Vec3 bias_vec = gyr_bias_norm[ibw] * unit_vec;
                    for(int idata = 0; idata < data.acc.size(); idata++)
                    {
                        data.gyr[idata].data[0] += bias_vec(0);
                        data.gyr[idata].data[1] += bias_vec(1);
                        data.gyr[idata].data[2] += bias_vec(2);
                    }

                    celib::PreintPrior prior;
                    std::vector<std::vector<double> > t;
                    std::vector<double> temp_t;
                    temp_t.push_back(end_t);
                    t.push_back(temp_t);
                    preint_opt.type = celib::UGPM;
                    preint_opt.train_gpm = true;
                    celib::ImuPreintegration imu_preint(data, start_t, t, preint_opt, prior);
                    celib::PreintMeas preint = imu_preint.get(0,0);
                    celib::PreintMeas preint_corrected;
                    preint_corrected.delta_R = preint.delta_R * celib::ExpMap(preint.d_delta_R_d_bw * (-bias_vec)); 
                    preint_corrected.delta_v = preint.delta_v + (preint.d_delta_v_d_bw * (-bias_vec)); 
                    preint_corrected.delta_p = preint.delta_p + (preint.d_delta_p_d_bw * (-bias_vec)); 

                    auto raw_error = imu_sim.testPreint(start_t, end_t, preint);
                    gyr_rot_raw_error[ibw].push_back(raw_error[0]/rot_dist);
                    gyr_pos_raw_error[ibw].push_back(raw_error[2]/pos_dist);
                    auto error = imu_sim.testPreint(start_t, end_t, preint_corrected);
                    gyr_rot_error[ibw].push_back(error[0]/rot_dist);
                    gyr_pos_error[ibw].push_back(error[2]/pos_dist);
                }


                for(int ibf = 0; ibf < acc_bias_norm.size(); ++ibf)
                {
                    data = data_save;
                    celib::Vec3 bias_vec = acc_bias_norm[ibf] * unit_vec;
                    for(int idata = 0; idata < data.acc.size(); idata++)
                    {
                        data.acc[idata].data[0] += bias_vec(0);
                        data.acc[idata].data[1] += bias_vec(1);
                        data.acc[idata].data[2] += bias_vec(2);
                    }

                    celib::PreintPrior prior;
                    std::vector<std::vector<double> > t;
                    std::vector<double> temp_t;
                    temp_t.push_back(end_t);
                    t.push_back(temp_t);
                    preint_opt.type = celib::UGPM;
                    preint_opt.train_gpm = true;
                    celib::ImuPreintegration imu_preint(data, start_t, t, preint_opt, prior);
                    celib::PreintMeas preint = imu_preint.get(0,0);
                    celib::PreintMeas preint_corrected;
                    preint_corrected.delta_v = preint.delta_v + (preint.d_delta_v_d_bf * (-bias_vec)); 
                    preint_corrected.delta_p = preint.delta_p + (preint.d_delta_p_d_bf * (-bias_vec)); 

                    auto raw_error = imu_sim.testPreint(start_t, end_t, preint);
                    acc_rot_raw_error[ibf].push_back(raw_error[0]/rot_dist);
                    acc_pos_raw_error[ibf].push_back(raw_error[2]/pos_dist);
                    auto error = imu_sim.testPreint(start_t, end_t, preint_corrected);
                    acc_rot_error[ibf].push_back(error[0]/rot_dist);
                    acc_pos_error[ibf].push_back(error[2]/pos_dist);
                }


                for(int idt = 0; idt < dt_bias.size(); ++idt)
                {
                    data = data_save;
                    for(int idata = 0; idata < data.acc.size(); idata++)
                    {
                        data.acc[idata].t += dt_bias[idt];
                        data.gyr[idata].t += dt_bias[idt];
                    }

                    celib::PreintPrior prior;
                    std::vector<std::vector<double> > t;
                    std::vector<double> temp_t;
                    temp_t.push_back(end_t);
                    t.push_back(temp_t);
                    preint_opt.type = celib::UGPM;
                    preint_opt.train_gpm = true;
                    celib::ImuPreintegration imu_preint(data, start_t, t, preint_opt, prior);
                    celib::PreintMeas preint = imu_preint.get(0,0);
                    celib::PreintMeas preint_corrected;
                    preint_corrected.delta_R = preint.delta_R * celib::ExpMap(preint.d_delta_R_d_t * dt_bias[idt]); 
                    preint_corrected.delta_v = preint.delta_v + (preint.d_delta_v_d_t * dt_bias[idt]); 
                    preint_corrected.delta_p = preint.delta_p + (preint.d_delta_p_d_t * dt_bias[idt]); 

                    auto raw_error = imu_sim.testPreint(start_t, end_t, preint);
                    dt_rot_raw_error[idt].push_back(raw_error[0]/rot_dist);
                    dt_pos_raw_error[idt].push_back(raw_error[2]/pos_dist);
                    auto error = imu_sim.testPreint(start_t, end_t, preint_corrected);
                    dt_rot_error[idt].push_back(error[0]/rot_dist);
                    dt_pos_error[idt].push_back(error[2]/pos_dist);
                }


            }
            std::cout << std::endl;
            for(int ibw = 0; ibw < gyr_bias_norm.size(); ++ibw)
            {
                gyr_rot_error_avg[type].push_back(0);
                gyr_pos_error_avg[type].push_back(0);
                gyr_rot_error_raw_avg[type].push_back(0);
                gyr_pos_error_raw_avg[type].push_back(0);
                for(int isample = 0; isample < nb_monte_carlo; ++isample)
                {
                    gyr_rot_error_avg[type][ibw] += gyr_rot_error[ibw][isample];
                    gyr_pos_error_avg[type][ibw] += gyr_pos_error[ibw][isample];
                    gyr_rot_error_raw_avg[type][ibw] += gyr_rot_raw_error[ibw][isample];
                    gyr_pos_error_raw_avg[type][ibw] += gyr_pos_raw_error[ibw][isample];
                }
                gyr_rot_error_avg[type][ibw] /= nb_monte_carlo;
                gyr_pos_error_avg[type][ibw] /= nb_monte_carlo;

                gyr_rot_error_raw_avg[type][ibw] /= nb_monte_carlo;
                gyr_pos_error_raw_avg[type][ibw] /= nb_monte_carlo;
            }
            for(int ibf = 0; ibf < acc_bias_norm.size(); ++ibf)
            {
                acc_rot_error_avg[type].push_back(0);
                acc_pos_error_avg[type].push_back(0);
                acc_rot_error_raw_avg[type].push_back(0);
                acc_pos_error_raw_avg[type].push_back(0);
                for(int isample = 0; isample < nb_monte_carlo; ++isample)
                {
                    acc_rot_error_avg[type][ibf] += acc_rot_error[ibf][isample];
                    acc_pos_error_avg[type][ibf] += acc_pos_error[ibf][isample];
                    acc_rot_error_raw_avg[type][ibf] += acc_rot_raw_error[ibf][isample];
                    acc_pos_error_raw_avg[type][ibf] += acc_pos_raw_error[ibf][isample];
                }
                acc_rot_error_avg[type][ibf] /= nb_monte_carlo;
                acc_pos_error_avg[type][ibf] /= nb_monte_carlo;
                acc_rot_error_raw_avg[type][ibf] /= nb_monte_carlo;
                acc_pos_error_raw_avg[type][ibf] /= nb_monte_carlo;

            }
            for(int idt = 0; idt < dt_bias.size(); ++idt)
            {
                dt_rot_error_avg[type].push_back(0);
                dt_pos_error_avg[type].push_back(0);
                dt_rot_error_raw_avg[type].push_back(0);
                dt_pos_error_raw_avg[type].push_back(0);
                for(int isample = 0; isample < nb_monte_carlo; ++isample)
                {
                    dt_rot_error_avg[type][idt] += dt_rot_error[idt][isample];
                    dt_pos_error_avg[type][idt] += dt_pos_error[idt][isample];
                    dt_rot_error_raw_avg[type][idt] += dt_rot_raw_error[idt][isample];
                    dt_pos_error_raw_avg[type][idt] += dt_pos_raw_error[idt][isample];
                }
                dt_rot_error_avg[type][idt] /= nb_monte_carlo;
                dt_pos_error_avg[type][idt] /= nb_monte_carlo;
                dt_rot_error_raw_avg[type][idt] /= nb_monte_carlo;
                dt_pos_error_raw_avg[type][idt] /= nb_monte_carlo;

            }
        }



        std::cout << "RESULTS" << std::endl;

        std::cout << std::endl;
        std::cout << ">>>> Gyr bias experiment" << std::endl;
        std::cout << "\\scriptsize Bias norm ";
        for(auto n:gyr_bias_norm)
        {
            std::cout << " & \\scriptsize " << n;
        }
        std::cout << std::endl << std::endl;
        for(int type = 0; type < 2; ++type)
        {
            if(type == 0) std::cout << "\\scriptsize Slow Rot er.";
            if(type == 1) std::cout << "\\scriptsize Fast Rot er.";
            for(int b = 0; b < gyr_bias_norm.size(); ++b)
            {
                std::cout << " & \\scriptsize " << 100*gyr_rot_error_raw_avg[type][b] << " & \\scriptsize " << 100*gyr_rot_error_avg[type][b];
            }
            std::cout << std::endl;
        }
        std::cout << std::endl;
        for(int type = 0; type < 2; ++type)
        {
            if(type == 0) std::cout << "\\scriptsize Slow Pos er.";
            if(type == 1) std::cout << "\\scriptsize Fast Pos er.";
            for(int b = 0; b < gyr_bias_norm.size(); ++b)
            {
                std::cout << " & \\scriptsize " << 100*gyr_pos_error_raw_avg[type][b] << " & \\scriptsize " << 100*gyr_pos_error_avg[type][b];
            }
            std::cout << std::endl;
        }
        std::cout << std::endl << std::endl;

        std::cout << ">>>> Acc bias experiment" << std::endl;
        std::cout << "\\scriptsize Bias norm ";
        for(auto n:acc_bias_norm)
        {
            std::cout << " & \\scriptsize " << n;
        }
        std::cout << std::endl << std::endl;
        for(int type = 0; type < 2; ++type)
        {
            if(type == 0) std::cout << "\\scriptsize Slow Pos er.";
            if(type == 1) std::cout << "\\scriptsize Fast Pos er.";
            for(int b = 0; b < acc_bias_norm.size(); ++b)
            {
                std::cout << " & \\scriptsize " << 100*acc_pos_error_raw_avg[type][b] << " & \\scriptsize " << 100*acc_pos_error_avg[type][b];
            }
            std::cout << std::endl;
        }
        std::cout << std::endl << std::endl;

        std::cout << ">>>> Timeshift experiment" << std::endl;
        std::cout << "\\scriptsize Timeshift ";
        for(auto n:dt_bias)
        {
            std::cout << " & \\scriptsize " << n;
        }
        std::cout << std::endl;
        for(int type = 0; type < 2; ++type)
        {
            if(type == 0) std::cout << "\\scriptsize Slow Rot er.";
            if(type == 1) std::cout << "\\scriptsize Fast Rot er.";
            for(int b = 0; b < dt_bias.size(); ++b)
            {
                std::cout << " & \\scriptsize " << 100*dt_rot_error_raw_avg[type][b] << " & \\scriptsize " << 100*dt_rot_error_avg[type][b];
            }
            std::cout << std::endl;
        }
        std::cout << std::endl;
        for(int type = 0; type < 2; ++type)
        {
            if(type == 0) std::cout << "\\scriptsize Slow Pos er.";
            if(type == 1) std::cout << "\\scriptsize Fast Pos er.";
            for(int b = 0; b < dt_bias.size(); ++b)
            {
                std::cout << " & \\scriptsize " << 100*dt_pos_error_raw_avg[type][b] << " & \\scriptsize " << 100*dt_pos_error_avg[type][b];
            }
            std::cout << std::endl;
        }
        std::cout << std::endl;
    }



    return 0;
}
