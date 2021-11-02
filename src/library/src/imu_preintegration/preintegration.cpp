/**
 *  Author: Cedric LE GENTIL 
 *
 *  Copyright 2021 Cedric LE GENTIL
 * 
 *  For any further question, recommendation or contribution
 *  le.gentil.cedric@gmail.com
 **/

#include "imu_preintegration/preintegration.h"
#include "common/math_utils.h"
#include "common/utils.h"
#include "common/types.h"
#include <stdexcept>
#include "libgp/include/rprop.h"
#include "libgp/include/cg.h"
#include "Eigen/Dense"
#include <ceres/gradient_checker.h>
#include <algorithm>


namespace celib
{

    // Constructor given IMU data and inference timestamps
    // (assumes increasing order in each vector of infer_t)
    ImuPreintegration::ImuPreintegration(celib::ImuData& imu_data,
                    const double start_t,
                    std::vector<std::vector<double> >& infer_t,
                    const PreintOption opt,
                    const PreintPrior prior,
                    const bool rot_only):
                        imu_data_(imu_data)
    {

        if(opt.quantum < 0)
        {
            opt_ = opt;
            prior_ = prior;
            start_t_ = start_t;

            int nb_infer_vec = infer_t.size();


            if(opt_.type == UGPM)
            {

                prepareUGPM(imu_data_, start_t, prior);
                preint_.resize(infer_t.size());
                for(int i = 0; i < infer_t.size(); ++i)
                {
                    preint_[i].reserve(infer_t[i].size());
                    for(int j = 0; j < infer_t[i].size(); ++j)
                    {
                        preint_[i].push_back(getUGPM(infer_t[i][j]));

                    }
                }


            }
            else
            {


                // Add the starting timestamp in the inference timestamps
                std::vector<double> temp_infer_t1;
                temp_infer_t1.push_back(start_t_);
                temp_infer_t1.push_back(start_t+kNumDtJacobianDelta);
                infer_t.push_back(temp_infer_t1);

                // Add the timestamps of the accelerometer data
                std::vector<double> temp_infer_t2;
                for(const auto& a : imu_data_.acc) temp_infer_t2.push_back(a.t);
                infer_t.push_back(temp_infer_t2);

                // Sort the timestamps
                celib::SortIndexTracker2<double> t(infer_t);

                // Check if the data has high enough frequency, otherwise add fake timestamps
                if(t.getSmallestGap() > (1.0/opt_.min_freq))
                {
                    std::vector<double> fake_time;
                    int nb_fake = std::floor((t.back() - t.get(0)) * opt_.min_freq);
                    double offset = t.get(0);
                    double quantum = (t.back() - t.get(0)) / nb_fake;
                    for(int i = 0; i < nb_fake; ++i) fake_time.push_back(offset + (i*quantum));
                    infer_t.push_back(fake_time);
                    t = celib::SortIndexTracker2<double>(infer_t);
                }

                start_index_ = t.getIndex(nb_infer_vec, 0);

                // Flag of timestamp interest to later prevent unnecessary computations
                std::vector<bool> interest_t;
                {
                    std::vector<std::vector<bool> > temp;
                    for(int i = 0; i < nb_infer_vec; ++i) temp.push_back(std::vector<bool>(infer_t[i].size(), true));
                    temp.push_back(std::vector<bool>(infer_t[nb_infer_vec].size(), true));
                    temp.push_back(std::vector<bool>(infer_t[nb_infer_vec+1].size(), true));
                    if( (nb_infer_vec+2) < infer_t.size()) temp.push_back(std::vector<bool>(infer_t[nb_infer_vec+2].size(), false));
                    interest_t = t.applySort(temp);
                }

                // Compute the rotational part of the preintegration 
                std::vector<PreintMeas> preint = rotPreintIterative(t, interest_t, opt_.type, prior_);


                // Demux the preintegrated mesurements
                for(int i = 0; i < nb_infer_vec; ++i)
                {
                    preint_.push_back(t.getVector(preint, i));

                }

                if(rot_only)
                {
                    return;
                }


                Mat3 delta_R_dt_start = t.get(nb_infer_vec, 1, preint).delta_R;


                // Reproject the accelerometer data (and apply the prior)
                std::vector<PreintMeas> acc_time_preint = t.getVector(preint, nb_infer_vec+1);
                std::vector<std::shared_ptr<Eigen::MatrixXd> > d_acc_d_bf;
                std::vector<std::shared_ptr<Eigen::MatrixXd> > d_acc_d_bw;
                std::vector<std::shared_ptr<Eigen::VectorXd> > d_acc_d_dt;
                reprojectAccData(acc_time_preint, prior_.acc_bias, delta_R_dt_start, d_acc_d_bf, d_acc_d_bw, d_acc_d_dt);

                
                while(infer_t.size() != nb_infer_vec)
                {
                    infer_t.pop_back();
                }



                // Compute the velocity and pos preintegrated measurements
                if(opt_.type == LPM)
                {
                    velPosPreintLPM(infer_t, d_acc_d_bf, d_acc_d_bw, d_acc_d_dt, preint_);
                }
                else if(opt_.type == GPM)
                {
                    velPosPreintGPM(infer_t, d_acc_d_bf, d_acc_d_bw, d_acc_d_dt, preint_);
                }
                
            }
        }
        else
        {
        
            // Get last inferrence timestamp
            std::vector<double> temp_t;
            for(auto t:infer_t)
            {
                if(t.size()>0) temp_t.push_back(t.back());
            }
            double last_t = infer_t[0].back();
            if(temp_t.size()>1)
            {
                last_t = *std::max_element(temp_t.begin(), temp_t.end());
            }


            // Get the overlap value (check diff between )
            std::vector<double> temp_overlap;
            if(imu_data.acc[0].t < start_t) temp_overlap.push_back(start_t-imu_data.acc[0].t);
            if(imu_data.gyr[0].t < start_t) temp_overlap.push_back(start_t-imu_data.gyr[0].t);
            if(imu_data.acc.back().t > last_t) temp_overlap.push_back(imu_data.acc.back().t - last_t);
            if(imu_data.gyr.back().t > last_t) temp_overlap.push_back(imu_data.gyr.back().t - last_t);
            double overlap = 0.1;
            if(temp_overlap.size() > 1)
            {
                overlap = *std::max_element(temp_overlap.begin(), temp_overlap.end());
            }
            else if(temp_overlap.size() == 1)
            {
                overlap = temp_overlap[0];
            }
            


            int nb_chuncks = (int) std::ceil((last_t - start_t)/opt.quantum);
            if(nb_chuncks == 0) nb_chuncks = 1;

            // Create pointers for the inference times
            std::vector<int> pointers(infer_t.size(),0);
            celib::PreintMeas prev_chunck_preint;
            preint_.resize(infer_t.size());
            for(int i = 0; i < nb_chuncks; ++i)
            {
                double chunck_start_t = start_t + (i*opt.quantum);
                double chunck_end_t = start_t + ((i+1)*opt.quantum);
                if(i == nb_chuncks-1) chunck_end_t = std::numeric_limits<double>::infinity();

                std::vector<std::vector<double> > temp_infer_t(infer_t.size());
                if(i != (nb_chuncks - 1))
                {
                    std::vector<double> temp_temp_t;
                    temp_temp_t.push_back(chunck_end_t);
                    temp_infer_t.push_back(temp_temp_t);
                }
                for(int j = 0; j < infer_t.size(); ++j)
                {
                    bool loop = true;
                    while(loop)
                    {
                        if(pointers[j] < (infer_t[j].size()))
                        {
                            if(infer_t[j][pointers[j]] < chunck_end_t)
                            {
                                temp_infer_t[j].push_back(infer_t[j][pointers[j]]);
                                pointers[j]++;
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
                auto temp_imu_data = imu_data.get(chunck_start_t-overlap, chunck_end_t+overlap);
                auto temp_opt = opt;
                temp_opt.quantum = -1;
                ImuPreintegration preint(
                        temp_imu_data,
                        chunck_start_t,
                        temp_infer_t,
                        temp_opt,
                        prior,
                        rot_only);

                if(i == 0)
                {
                    if(nb_chuncks > 1)
                    {
                        prev_chunck_preint = preint.get(infer_t.size(), 0);
                    }
                    for(int j = 0; j < infer_t.size(); ++j)
                    {
                        for(int h = 0; h < temp_infer_t[j].size(); ++h)
                        {
                            preint_[j].push_back(preint.get(j,h));
                        }
                    }
                }
                else
                {
                    Mat3 prev_pos_cov = prev_chunck_preint.cov.block<3,3>(6,6);
                    Mat3 prev_vel_cov = prev_chunck_preint.cov.block<3,3>(3,3);
                    Mat3 prev_rot_cov = prev_chunck_preint.cov.block<3,3>(0,0);

                    for(int j = 0; j < infer_t.size(); ++j)
                    {
                        for(int h = 0; h < temp_infer_t[j].size(); ++h)
                        {

                            auto temp_preint = preint.get(j,h);
                            

                            // Propagation of covariance
                            temp_preint.cov.block<3,3>(6,6) = prev_pos_cov + (temp_preint.dt*temp_preint.dt)*prev_vel_cov + UncertaintyRp(prev_chunck_preint.delta_R, prev_rot_cov, temp_preint.delta_p, temp_preint.cov.block<3,3>(6,6));

                            temp_preint.cov.block<3,3>(3,3) = prev_vel_cov + UncertaintyRp(prev_chunck_preint.delta_R, prev_rot_cov, temp_preint.delta_v, temp_preint.cov.block<3,3>(3,3));
                            temp_preint.cov.block<3,3>(0,0) = UncertaintyRR(prev_chunck_preint.delta_R, prev_rot_cov, temp_preint.delta_R, temp_preint.cov.block<3,3>(0,0));

                            // Propagation of acc Jacobians
                            temp_preint.d_delta_p_d_bf = prev_chunck_preint.d_delta_p_d_bf
                                    + (temp_preint.dt*prev_chunck_preint.d_delta_v_d_bf)
                                    + (prev_chunck_preint.delta_R*temp_preint.d_delta_p_d_bf);

                            temp_preint.d_delta_v_d_bf = prev_chunck_preint.d_delta_v_d_bf
                                    + (prev_chunck_preint.delta_R * temp_preint.d_delta_v_d_bf);


                            // Propagation of gyr Jacobians
                            temp_preint.d_delta_p_d_bw = prev_chunck_preint.d_delta_p_d_bw
                                    + (temp_preint.dt*prev_chunck_preint.d_delta_v_d_bw)
                                    + PropagateJacobianRp(prev_chunck_preint.delta_R, prev_chunck_preint.d_delta_R_d_bw, temp_preint.delta_p, temp_preint.d_delta_p_d_bw);

                            temp_preint.d_delta_v_d_bw = prev_chunck_preint.d_delta_v_d_bw
                                    + PropagateJacobianRp(prev_chunck_preint.delta_R, prev_chunck_preint.d_delta_R_d_bw, temp_preint.delta_v, temp_preint.d_delta_v_d_bw);
                            temp_preint.d_delta_R_d_bw = PropagateJacobianRR(temp_preint.delta_R.transpose(), prev_chunck_preint.d_delta_R_d_bw, temp_preint.delta_R, temp_preint.d_delta_R_d_bw);

                            // Propagation of time-shift Jacobians
                            temp_preint.d_delta_p_d_t = prev_chunck_preint.d_delta_p_d_t
                                    + (temp_preint.dt*prev_chunck_preint.d_delta_v_d_t)
                                    + PropagateJacobianRp(prev_chunck_preint.delta_R, prev_chunck_preint.d_delta_R_d_t, temp_preint.delta_p, temp_preint.d_delta_p_d_t);

                            temp_preint.d_delta_v_d_t = prev_chunck_preint.d_delta_v_d_t
                                    + PropagateJacobianRp(prev_chunck_preint.delta_R, prev_chunck_preint.d_delta_R_d_t, temp_preint.delta_v, temp_preint.d_delta_v_d_t);
                            temp_preint.d_delta_R_d_t = PropagateJacobianRR(temp_preint.delta_R.transpose(), prev_chunck_preint.d_delta_R_d_t, temp_preint.delta_R, temp_preint.d_delta_R_d_t);


                            // Chunck combination
                            temp_preint.delta_p = prev_chunck_preint.delta_p
                                    + prev_chunck_preint.delta_v*temp_preint.dt
                                    + prev_chunck_preint.delta_R*temp_preint.delta_p;
                            temp_preint.delta_v = prev_chunck_preint.delta_v
                                    + prev_chunck_preint.delta_R*temp_preint.delta_v;
                            temp_preint.delta_R = prev_chunck_preint.delta_R*temp_preint.delta_R;

                            temp_preint.dt = temp_infer_t[j][h] - start_t;
                            temp_preint.dt_sq_half = 0.5*temp_preint.dt* temp_preint.dt;
                            preint_[j].push_back(temp_preint);
                        }
                    }
                    if(i != (nb_chuncks-1))
                    {
                        auto temp_chunck_preint = preint.get(infer_t.size(), 0);

                        // Propagation of covariance
                        prev_chunck_preint.cov.block<3,3>(6,6) = prev_pos_cov + (opt.quantum*opt.quantum)*prev_vel_cov + UncertaintyRp(prev_chunck_preint.delta_R, prev_rot_cov, temp_chunck_preint.delta_p, temp_chunck_preint.cov.block<3,3>(6,6));
                        prev_chunck_preint.cov.block<3,3>(3,3) = prev_vel_cov + UncertaintyRp(prev_chunck_preint.delta_R, prev_rot_cov, temp_chunck_preint.delta_v, temp_chunck_preint.cov.block<3,3>(3,3));
                        prev_chunck_preint.cov.block<3,3>(0,0) = UncertaintyRR(prev_chunck_preint.delta_R, prev_rot_cov, temp_chunck_preint.delta_R, temp_chunck_preint.cov.block<3,3>(0,0));

                        // Propagation of acc Jacobians
                        prev_chunck_preint.d_delta_p_d_bf = prev_chunck_preint.d_delta_p_d_bf
                                + (opt.quantum*prev_chunck_preint.d_delta_v_d_bf)
                                + (prev_chunck_preint.delta_R * temp_chunck_preint.d_delta_p_d_bf);
                        prev_chunck_preint.d_delta_v_d_bf = prev_chunck_preint.d_delta_v_d_bf
                                + (prev_chunck_preint.delta_R * temp_chunck_preint.d_delta_v_d_bf);

                        // Propagation of gyr Jacobians
                        prev_chunck_preint.d_delta_p_d_bw = prev_chunck_preint.d_delta_p_d_bw
                                + (opt.quantum*prev_chunck_preint.d_delta_v_d_bw)
                                + PropagateJacobianRp(prev_chunck_preint.delta_R, prev_chunck_preint.d_delta_R_d_bw, temp_chunck_preint.delta_p, temp_chunck_preint.d_delta_p_d_bw);
                        prev_chunck_preint.d_delta_v_d_bw = prev_chunck_preint.d_delta_v_d_bw
                                + PropagateJacobianRp(prev_chunck_preint.delta_R, prev_chunck_preint.d_delta_R_d_bw, temp_chunck_preint.delta_v, temp_chunck_preint.d_delta_v_d_bw);
                        prev_chunck_preint.d_delta_R_d_bw = PropagateJacobianRR(temp_chunck_preint.delta_R.transpose(), prev_chunck_preint.d_delta_R_d_bw, temp_chunck_preint.delta_R, temp_chunck_preint.d_delta_R_d_bw);

                        // Propagation of time-shift Jacobians
                        prev_chunck_preint.d_delta_p_d_t = prev_chunck_preint.d_delta_p_d_t
                                + (opt.quantum*prev_chunck_preint.d_delta_v_d_t)
                                + PropagateJacobianRp(prev_chunck_preint.delta_R, prev_chunck_preint.d_delta_R_d_t, temp_chunck_preint.delta_p, temp_chunck_preint.d_delta_p_d_t);
                        prev_chunck_preint.d_delta_v_d_t = prev_chunck_preint.d_delta_v_d_t
                                + PropagateJacobianRp(prev_chunck_preint.delta_R, prev_chunck_preint.d_delta_R_d_t, temp_chunck_preint.delta_v, temp_chunck_preint.d_delta_v_d_t);
                        prev_chunck_preint.d_delta_R_d_t = PropagateJacobianRR(temp_chunck_preint.delta_R.transpose(), prev_chunck_preint.d_delta_R_d_t, temp_chunck_preint.delta_R, temp_chunck_preint.d_delta_R_d_t);
                            
                        // Chunk combination
                        prev_chunck_preint.delta_p = prev_chunck_preint.delta_p + prev_chunck_preint.delta_v*opt.quantum + prev_chunck_preint.delta_R*temp_chunck_preint.delta_p;
                        prev_chunck_preint.delta_v = prev_chunck_preint.delta_v + prev_chunck_preint.delta_R*temp_chunck_preint.delta_v;
                        prev_chunck_preint.delta_R = prev_chunck_preint.delta_R*temp_chunck_preint.delta_R;

                    }
                }
                
            }

        }


    }

    void ImuPreintegration::reprojectAccData(
                const std::vector<PreintMeas>& preint,
                const std::vector<double>& prior,
                const Mat3& delta_R_dt_start,
                std::vector<std::shared_ptr<MatX> >& d_acc_d_bf,
                std::vector<std::shared_ptr<MatX> >& d_acc_d_bw,
                std::vector<std::shared_ptr<VecX> >& d_acc_d_dt
                )
    {
        for(int i = 0; i < 3; ++i)
        {
            d_acc_d_bf.push_back(std::shared_ptr<MatX>(new MatX));
            d_acc_d_bw.push_back(std::shared_ptr<MatX>(new MatX));
            d_acc_d_dt.push_back(std::shared_ptr<VecX>(new VecX));

            d_acc_d_bf[i]->resize(imu_data_.acc.size(),3);
            d_acc_d_bw[i]->resize(imu_data_.acc.size(),3);
            d_acc_d_dt[i]->resize(imu_data_.acc.size());
        }

        for(int i = 0; i < imu_data_.acc.size(); ++i)
        {
            Eigen::Map<Vec3> temp_acc(&(imu_data_.acc[i].data[0]));
            Eigen::Map<const Vec3> bias_prior(&(prior[0]));
            temp_acc = temp_acc - bias_prior;


            d_acc_d_bf[0]->row(i) = preint[i].delta_R.row(0);
            d_acc_d_bf[1]->row(i) = preint[i].delta_R.row(1);
            d_acc_d_bf[2]->row(i) = preint[i].delta_R.row(2);



            Mat9_3 temp_d_R_d_bw = celib::JacobianExpMap(
                    Vec3::Zero() ) * preint[i].d_delta_R_d_bw;
            Row9 temp_1;
            temp_1 <<   preint[i].delta_R(0,0)*temp_acc(0),preint[i].delta_R(0,1)*temp_acc(0),preint[i].delta_R(0,2)*temp_acc(0),
                        preint[i].delta_R(0,0)*temp_acc(1),preint[i].delta_R(0,1)*temp_acc(1),preint[i].delta_R(0,2)*temp_acc(1),
                        preint[i].delta_R(0,0)*temp_acc(2),preint[i].delta_R(0,1)*temp_acc(2),preint[i].delta_R(0,2)*temp_acc(2);
            Row9 temp_2;
            temp_2 <<   preint[i].delta_R(1,0)*temp_acc(0),preint[i].delta_R(1,1)*temp_acc(0),preint[i].delta_R(1,2)*temp_acc(0),
                        preint[i].delta_R(1,0)*temp_acc(1),preint[i].delta_R(1,1)*temp_acc(1),preint[i].delta_R(1,2)*temp_acc(1),
                        preint[i].delta_R(1,0)*temp_acc(2),preint[i].delta_R(1,1)*temp_acc(2),preint[i].delta_R(1,2)*temp_acc(2);
            Row9 temp_3;
            temp_3 <<   preint[i].delta_R(2,0)*temp_acc(0),preint[i].delta_R(2,1)*temp_acc(0),preint[i].delta_R(2,2)*temp_acc(0),
                        preint[i].delta_R(2,0)*temp_acc(1),preint[i].delta_R(2,1)*temp_acc(1),preint[i].delta_R(2,2)*temp_acc(1),
                        preint[i].delta_R(2,0)*temp_acc(2),preint[i].delta_R(2,1)*temp_acc(2),preint[i].delta_R(2,2)*temp_acc(2);
            d_acc_d_bw[0]->row(i) = temp_1*temp_d_R_d_bw;
            d_acc_d_bw[1]->row(i) = temp_2*temp_d_R_d_bw;
            d_acc_d_bw[2]->row(i) = temp_3*temp_d_R_d_bw;

            temp_acc = preint[i].delta_R * temp_acc;
            Vec3 acc_rot_dt = delta_R_dt_start.transpose()*temp_acc;
            Vec3 d_acc_d_t = (acc_rot_dt - temp_acc)/kNumDtJacobianDelta;
            (*(d_acc_d_dt[0]))[i] = d_acc_d_t(0);
            (*(d_acc_d_dt[1]))[i] = d_acc_d_t(1);
            (*(d_acc_d_dt[2]))[i] = d_acc_d_t(2);
            


        }        
    }

    void ImuPreintegration::velPosPreintLPM(
                std::vector< std::vector<double> >& t,
                const std::vector<std::shared_ptr<MatX> >& d_acc_d_bf,
                const std::vector<std::shared_ptr<MatX> >& d_acc_d_bw,
                const std::vector<std::shared_ptr<VecX> >& d_acc_d_dt,
                std::vector<std::vector<PreintMeas> >& preint)
    {

        SortIndexTracker2<double> time(t);

        // Compute with the time shift (could be optimised somehow I guess)
        std::vector<ImuSample> temp_samples = imu_data_.acc;
        for(int i = 0; i < temp_samples.size(); ++i)
        {
            Eigen::Map<Vec3> acc(temp_samples[i].data);
            Vec3 temp_d_acc_d_dt;
            temp_d_acc_d_dt <<  d_acc_d_dt[0]->coeff(i), 
                                d_acc_d_dt[1]->coeff(i), 
                                d_acc_d_dt[2]->coeff(i);
            acc += kNumDtJacobianDelta*temp_d_acc_d_dt;
            temp_samples[i].t -= kNumDtJacobianDelta;
        }
        velPosPreintLPMPartial(time, temp_samples, preint);



        double var_prior = imu_data_.acc_var;
        int data_ptr = 0;

        // For each of the query points of the timeline
        int start_index = 0;
        while(time.get(start_index) < start_t_){
            start_index++;
            if(start_index == time.size()){
                throw "FullLPM: the start_time is not in the query domain";
            }
        }
        while(imu_data_.acc[data_ptr+1].t < start_t_){
            data_ptr++;
            if(data_ptr == (imu_data_.acc.size()-1)){
                throw "FullLPM: the start_time is not in the data domain";
            }
        }



        double data_ptr_save = data_ptr;
        double start_index_save = start_index;

        #pragma omp parallel for schedule(static)
        for(int axis = 0; axis < 3; ++axis)
        {
            int ptr = data_ptr;
            double alpha = (imu_data_.acc[ptr+1].data[axis] - imu_data_.acc[ptr].data[axis]) /
                (imu_data_.acc[ptr+1].t - imu_data_.acc[ptr].t);
            double beta = imu_data_.acc[ptr].data[axis] - alpha*imu_data_.acc[ptr].t;
            double t_0 = start_t_;
            double t_1 = imu_data_.acc[ptr+1].t;
            double d_0 = alpha*imu_data_.acc[ptr].t + beta;
            double d_1 = imu_data_.acc[ptr+1].data[axis];
            double d_v_backup = 0;
            double d_p_backup = 0;


            double temp_t_ratio = (start_t_ - imu_data_.acc[ptr].t) /
                ( imu_data_.acc[ptr+1].t - imu_data_.acc[ptr].t);
            Vec3 d_d_0_d_bw = d_acc_d_bw[axis]->row(ptr+1).transpose() * temp_t_ratio +
                d_acc_d_bw[axis]->row(ptr).transpose() * (1 - temp_t_ratio);
            Vec3 d_d_0_d_bf = d_acc_d_bf[axis]->row(ptr+1).transpose() * temp_t_ratio +
                d_acc_d_bf[axis]->row(ptr).transpose() * (1 - temp_t_ratio);

            Vec3 d_v_d_bf_backup = Vec3::Zero();
            Vec3 d_p_d_bf_backup = Vec3::Zero();
            Vec3 d_v_d_bw_backup = Vec3::Zero();
            Vec3 d_p_d_bw_backup = Vec3::Zero();


            for(int i = start_index; i < time.size(); ++i){
                // Move the data pointer so that the query time is in between two data points
                if( time.get(i) > imu_data_.acc[0].t ){
                    bool loop = true;
                    while(loop){
                        if( (time.get(i) >= imu_data_.acc[ptr].t)
                                && (time.get(i) <= imu_data_.acc[ptr+1].t) ){
                            loop = false;
                        }else{
                            if( ptr < (imu_data_.acc.size() - 2) ){

                                d_p_backup = d_p_backup +  d_v_backup*(t_1 - t_0)
                                    + ((t_0 - t_1)*(t_0 - t_1)*(2.0*d_0 + d_1)/6.0);
                                d_v_backup = d_v_backup + ((t_1 - t_0)*(d_0 + d_1)/2.0);

                                double dt = t_1 - t_0;
                                Vec3 d_d_1_d_bf = d_acc_d_bf[axis]->row(ptr + 1).transpose();
                                Vec3 d_d_1_d_bw = d_acc_d_bw[axis]->row(ptr + 1).transpose();
                                Vec3 temp_d_v_d_bf = dt*(d_d_0_d_bf + d_d_1_d_bf)/2.0;
                                Vec3 temp_d_v_d_bw = dt*(d_d_0_d_bw + d_d_1_d_bw)/2.0;
                                Vec3 temp_d_p_d_bf = dt*dt*(2.0*d_d_0_d_bf + d_d_1_d_bf)/6.0;
                                Vec3 temp_d_p_d_bw = dt*dt*(2.0*d_d_0_d_bw + d_d_1_d_bw)/6.0;

                                d_p_d_bf_backup = d_p_d_bf_backup + dt*d_v_d_bf_backup + temp_d_p_d_bf;
                                d_p_d_bw_backup = d_p_d_bw_backup + dt*d_v_d_bw_backup + temp_d_p_d_bw;
                                d_v_d_bf_backup = d_v_d_bf_backup + temp_d_v_d_bf;
                                d_v_d_bw_backup = d_v_d_bw_backup + temp_d_v_d_bw;


                                ptr++;
                                t_0 = imu_data_.acc[ptr].t;
                                t_1 = imu_data_.acc[ptr+1].t;
                                d_0 = imu_data_.acc[ptr].data[axis];
                                d_1 = imu_data_.acc[ptr+1].data[axis];

                                alpha = (d_1 - d_0) / (t_1 - t_0);
                                beta = d_0 - alpha*t_0;

                                d_d_0_d_bf = d_acc_d_bf[axis]->row(ptr).transpose();
                                d_d_0_d_bw = d_acc_d_bw[axis]->row(ptr).transpose();

                            }else{
                                loop=false;
                            }
                        }
                    }
                }

                // Integrate
                double time_temp = time.get(i);
                double temp_d_1 = alpha*time_temp + beta;
                double temp_d_v = d_v_backup +
                    ((time_temp - t_0)*(d_0 + temp_d_1 )/2.0);
                double temp_d_p = d_p_backup + d_v_backup*(time_temp - t_0) +
                    ((t_0 - time_temp)*(t_0 - time_temp)*(2.0*d_0 + temp_d_1)/6.0);
                double temp_d_v_var = (time_temp-start_t_)*var_prior;
                double temp_d_p_var = (time_temp-start_t_)*temp_d_v_var;
                std::pair<double, double> index = time.getIndexPair(i);

                preint[index.first][index.second].d_delta_v_d_t(axis) = (preint[index.first][index.second].delta_v(axis) - temp_d_v)/kNumDtJacobianDelta;
                preint[index.first][index.second].d_delta_p_d_t(axis) = (preint[index.first][index.second].delta_p(axis) - temp_d_p)/kNumDtJacobianDelta;
                preint[index.first][index.second].delta_v(axis) = temp_d_v;
                preint[index.first][index.second].delta_p(axis) = temp_d_p;
                preint[index.first][index.second].cov(3+axis, 3+axis) = temp_d_v_var;
                preint[index.first][index.second].cov(6+axis, 6+axis) = temp_d_p_var;

                // Jacobians for postintegration correction
                temp_t_ratio = (time_temp - imu_data_.acc[ptr].t) /
                    ( imu_data_.acc[ptr+1].t - imu_data_.acc[ptr].t);
                Vec3 d_d_1_d_bw = d_acc_d_bw[axis]->row(ptr+1).transpose() * temp_t_ratio +
                    d_acc_d_bw[axis]->row(ptr).transpose() * (1 - temp_t_ratio);
                Vec3 d_d_1_d_bf = d_acc_d_bf[axis]->row(ptr+1).transpose() * temp_t_ratio +
                    d_acc_d_bf[axis]->row(ptr).transpose() * (1 - temp_t_ratio);

                double dt = time_temp - t_0;
                Vec3 temp_d_v_d_bf = dt*(d_d_0_d_bf + d_d_1_d_bf)/2.0;
                Vec3 temp_d_v_d_bw = dt*(d_d_0_d_bw + d_d_1_d_bw)/2.0;
                Vec3 temp_d_p_d_bf = dt*dt*(2.0*d_d_0_d_bf + d_d_1_d_bf)/6.0;
                Vec3 temp_d_p_d_bw = dt*dt*(2.0*d_d_0_d_bw + d_d_1_d_bw)/6.0;

                preint[index.first][index.second].d_delta_v_d_bf.row(axis) = d_v_d_bf_backup + temp_d_v_d_bf;
                preint[index.first][index.second].d_delta_v_d_bw.row(axis) = d_v_d_bw_backup + temp_d_v_d_bw;
                preint[index.first][index.second].d_delta_p_d_bf.row(axis) = d_p_d_bf_backup + dt*d_v_d_bf_backup + temp_d_p_d_bf;
                preint[index.first][index.second].d_delta_p_d_bw.row(axis) = d_p_d_bw_backup + dt*d_v_d_bw_backup + temp_d_p_d_bw;


            }
        }

    }


    void ImuPreintegration::velPosPreintLPMPartial(
                const SortIndexTracker2<double>& time,
                std::vector<ImuSample>& acc_data,
                std::vector<std::vector<PreintMeas> >& preint)
    {
        int data_ptr = 0;

        // For each of the query points of the timeline
        int start_index = 0;
        while(time.get(start_index) < start_t_){
            start_index++;
            if(start_index == time.size()){
                throw std::range_error("LPM Partial: the start_time is not in the query domain");
            }
        }
        while(acc_data[data_ptr+1].t < start_t_){
            data_ptr++;
            if(data_ptr == (acc_data.size()-1)){
                throw std::range_error("LPM Partial: the start_time is not in the data domain");
            }
        }
    


        #pragma omp parallel for schedule(static)
        for(int axis = 0; axis < 3; ++axis)
        {
            int ptr = data_ptr;
            double alpha = (acc_data[ptr+1].data[axis] - acc_data[ptr].data[axis]) /
                (acc_data[ptr+1].t - acc_data[ptr].t);
            double beta = acc_data[ptr].data[axis] - alpha*acc_data[ptr].t;
            double t_0 = start_t_;
            double t_1 = acc_data[ptr+1].t;
            double d_0 = alpha*acc_data[ptr].t + beta;
            double d_1 = acc_data[ptr+1].data[axis];
            double d_v_backup = 0;
            double d_p_backup = 0;

            for(int i = start_index; i < time.size(); ++i){
                // Move the data pointer so that the query time is in between two data points
                if( time.get(i) > acc_data[0].t ){
                    bool loop = true;
                    while(loop){
                        if( (time.get(i) >= acc_data[ptr].t)
                                && (time.get(i) <= acc_data[ptr+1].t) ){
                            loop = false;
                        }else{
                            if( ptr < (acc_data.size() - 1) ){

                                d_p_backup = d_p_backup +  d_v_backup*(t_1 - t_0)
                                    + ((t_0 - t_1)*(t_0 - t_1)*(2.0*d_0 + d_1)/6.0);
                                d_v_backup = d_v_backup + ((t_1 - t_0)*(d_0 + d_1)/2.0);

                                ptr++;
                                t_0 = acc_data[ptr].t;
                                t_1 = acc_data[ptr+1].t;
                                d_0 = acc_data[ptr].data[axis];
                                d_1 = acc_data[ptr+1].data[axis];

                                alpha = (d_1 - d_0) / (t_1 - t_0);
                                beta = d_0 - alpha*t_0;


                            }else{
                                loop=false;
                            }
                        }
                    }
                }

                // Integrate
                double time_temp = time.get(i);
                double temp_d_1 = alpha*time_temp + beta;
                double temp_d_v = d_v_backup +
                    ((time_temp - t_0)*(d_0 + temp_d_1 )/2.0);
                double temp_d_p = d_p_backup + d_v_backup*(time_temp - t_0) +
                    ((t_0 - time_temp)*(t_0 - time_temp)*(2.0*d_0 + temp_d_1)/6.0);
                std::pair<double, double> index = time.getIndexPair(i);
                preint[index.first][index.second].delta_v(axis) = temp_d_v;
                preint[index.first][index.second].delta_p(axis) = temp_d_p;

            }
        }
    }



    void ImuPreintegration::velPosPreintGPM(
                const std::vector<std::vector<double> >& t,
                std::vector<std::shared_ptr<MatX> >& d_acc_d_bf,
                std::vector<std::shared_ptr<MatX> >& d_acc_d_bw,
                std::vector<std::shared_ptr<VecX> >& d_acc_d_dt,
                std::vector<std::vector<PreintMeas> >& preint)
    {
        
        #pragma omp parallel for schedule(static)
        for(int axis = 0; axis < 3; ++axis)
        {
            GPMGP gp(imu_data_.acc, axis, imu_data_.acc_var, opt_.train_gpm, d_acc_d_bf[axis], d_acc_d_bw[axis], d_acc_d_dt[axis]);

            int nb_infer = 0;
            for(const auto& a: t) nb_infer += a.size();

            Eigen::VectorXd t_vect(nb_infer);
            int counter = 0;
            for(int i = 0; i < t.size(); ++i)
            {
                for( int j = 0; j < t[i].size(); ++j)
                {
                    t_vect(counter) = t[i][j];
                    counter++;
                }
            }

            Eigen::VectorXd d_v;
            Eigen::VectorXd var_d_v;
            Eigen::VectorXd d_p;
            Eigen::VectorXd var_d_p;
            Eigen::MatrixXd d_v_d_bf;
            Eigen::MatrixXd d_p_d_bf;
            Eigen::MatrixXd d_v_d_bw;
            Eigen::MatrixXd d_p_d_bw;
            Eigen::VectorXd d_v_d_dt;
            Eigen::VectorXd d_p_d_dt;
            gp.integralAndVar(start_t_, t_vect, d_v, var_d_v,
                    d_v_d_bf, d_v_d_bw, d_v_d_dt);
            gp.integral2AndVar(start_t_, t_vect, d_p, var_d_p,
                    d_p_d_bf, d_p_d_bw, d_p_d_dt);


            counter = 0;
            for(int i = 0; i < t.size(); ++i)
            {
                for(int j = 0; j < t[i].size(); ++j)
                {
                    preint[i][j].delta_v(axis) = d_v(counter);
                    preint[i][j].cov(3+axis, 3+axis) = var_d_v(counter);

                    preint[i][j].delta_p(axis) = d_p(counter);
                    preint[i][j].cov(6+axis, 6+axis) = var_d_p(counter);

                    preint[i][j].d_delta_v_d_bf.row(axis) = d_v_d_bf.row(counter);
                    preint[i][j].d_delta_v_d_bw.row(axis) = d_v_d_bw.row(counter);
                    preint[i][j].d_delta_v_d_t(axis) = d_v_d_dt(counter);

                    preint[i][j].d_delta_p_d_bf.row(axis) = d_p_d_bf.row(counter);
                    preint[i][j].d_delta_p_d_bw.row(axis) = d_p_d_bw.row(counter);
                    preint[i][j].d_delta_p_d_t(axis) = d_p_d_dt(counter);

                    counter++;
                }
            }
        }

    }
    
    // Get the preintegrated measurement as per indexed in the given inference times of the constructor
    PreintMeas ImuPreintegration::get(const int index_1, const int index_2)
    {
        if ((index_1 >= 0) && 
            (index_2 >= 0) &&
            (index_1 < preint_.size()) &&
            (index_2 < preint_[index_1].size()))
        {
            return preint_[index_1][index_2];
        }
        else
        {
            throw std::range_error("Trying to get precomputed preintegrated measurements (wrong index query or no precomputed measurements)");
        }

    }



    
    // Compute the preintegration for the rotational part
    std::vector<PreintMeas> ImuPreintegration::rotPreintIterative(
                const SortIndexTracker2<double>& t,
                const std::vector<bool>& interest_t,
                const PreintType type,
                const PreintPrior prior)
    {
        std::vector<PreintMeas> output(t.size());

        // Interpolate the samples (and get the variance)
        std::vector<std::vector<std::pair<double, double> > > inter_data(3);
        std::vector<std::vector<std::pair<double, double> > > inter_data_dt(3);
        std::vector<ImuSample> temp_data = imu_data_.gyr;
        for(auto& td : temp_data) td.t -= kNumDtJacobianDelta;

        #pragma omp parallel for schedule(static)
        for(int i = 0; i < 3; ++i)
        {
            if(type == LPM)
            {
                inter_data[i] = linearInterpolation(imu_data_.gyr, i, imu_data_.gyr_var, t);
                inter_data_dt[i] = linearInterpolation(temp_data, i, imu_data_.gyr_var, t);
            }
            else if(type == GPM)
            {
                GpInterpolator gp_inter(imu_data_.gyr, i, imu_data_.gyr_var, opt_.train_gpm);
                inter_data[i] = gp_inter.get(t);
                inter_data_dt[i] = gp_inter.getShifted(t, kNumDtJacobianDelta);
            }
        }


        rotPreintLoop(inter_data, t, prior_, output);

        std::vector<Mat3> d_R_dt(output.size());
        std::vector<std::vector<Mat3> > d_R_d_bw(3, std::vector<Mat3>(output.size()));
        
        rotPreintLoop(inter_data_dt, t, prior_, d_R_dt);
        for(int j = 0; j < t.size(); ++j)
        {
            if(interest_t[j])
            {
                output[j].d_delta_R_d_t = celib::LogMap(output[j].delta_R.transpose()*(d_R_dt[j]))/kNumDtJacobianDelta;
            }
        }
        // Could also parallelise with the previous loop if really seeking performance
        #pragma omp parallel for schedule(static)
        for(int i = 0; i < 3; ++i)
        {
            auto prior_temp = prior_;
            prior_temp.gyr_bias[i] -= kNumGyrBiasJacobianDelta;
            rotPreintLoop(inter_data, t, prior_temp, d_R_d_bw[i]);
            for(int j = 0; j < t.size(); ++j)
            {
                if(interest_t[j])
                {
                    output[j].d_delta_R_d_bw.col(i) = celib::LogMap(output[j].delta_R.transpose()*(d_R_d_bw[i][j]))/kNumGyrBiasJacobianDelta;
                }
            }
        }
        
        return output;
    }


    // Linear interpolation
    std::vector<std::pair<double, double> > ImuPreintegration::linearInterpolation(
                const std::vector<ImuSample>& samples,
                const int axis,
                const double var,
                const SortIndexTracker2<double>& infer_t)
    {
        // Initialise the output data structure
        std::vector<std::pair< double, double> >output;
        output.resize(infer_t.size());



        if( samples.size() <2 ){
            throw std::range_error("InterpolateLinear: this function need at least 2 data points to interpolate");
        }

        
        int data_ptr = 0;
        double alpha = (samples[1].data[axis] - samples[0].data[axis]) / (samples[1].t - samples[0].t);
        double beta = samples[0].data[axis] - alpha*samples[0].t;
        // For each timeline
        for(int i = 0; i < infer_t.size(); ++i){
            
            // Move the data pointer so that the query time is in between two data points
            if( infer_t.get(i) > samples[0].t ){
                bool loop = true;
                while(loop){
                    if(infer_t.get(i) <= samples[data_ptr+1].t)
                    {
                        loop = false;
                    }
                    else
                    {
                        if( data_ptr < (samples.size() - 1) )
                        {
                            data_ptr++;
                            alpha = (samples[data_ptr+1].data[axis] - samples[data_ptr].data[axis]) / (samples[data_ptr+1].t - samples[data_ptr].t);
                            beta = samples[data_ptr].data[axis] - alpha*samples[data_ptr].t;
                        }
                        else
                        {
                            loop=false;
                        }
                    }
                }
            }
            // Interpolate
            output[i].first = alpha*infer_t.get(i) + beta;
            output[i].second = var;
        }
        return output;

    }

    void ImuPreintegration::rotPreintLoop(
                        const std::vector<std::vector<std::pair<double, double> > >& inter_data,
                        const SortIndexTracker2<double>& t,
                        const PreintPrior prior,
                        std::vector<PreintMeas>& output)
    {
        if(output.size() != inter_data.front().size()) output.resize(inter_data.front().size());

        Mat3 rot_mat = Mat3::Identity();
        Mat9 cov = Mat9::Zero();

        output[0].delta_R = rot_mat;
        output[0].cov = cov;
        output[0].dt = t.get(0) - start_t_;
        output[0].dt_sq_half = 0.5*(output[0].dt)*(output[0].dt);

        
        for(int i = 0; i < (t.size() - 1); ++i)
        {
            // Prepare some variables for convenience sake
            double dt = t.get(i+1)-t.get(i);
            
            double gyr_x_dt = (inter_data[0][i].first - prior.gyr_bias[0]) * dt;
            double gyr_y_dt = (inter_data[1][i].first - prior.gyr_bias[1]) * dt;
            double gyr_z_dt = (inter_data[2][i].first - prior.gyr_bias[2]) * dt;

            double gyr_norm = std::sqrt( (gyr_x_dt * gyr_x_dt)
                    + (gyr_y_dt * gyr_y_dt) + (gyr_z_dt * gyr_z_dt) );
            

            Mat3 e_R = Mat3::Identity();
            Mat3 j_r = Mat3::Identity();

            // If the angular velocity norm is non-null
            if(gyr_norm > 0.0000000001){

                // Turn the angular velocity into skew-simmetric matrix
                Mat3 gyr_skew_mat;
                gyr_skew_mat <<     0, -gyr_z_dt,  gyr_y_dt,
                             gyr_z_dt,         0, -gyr_x_dt,
                            -gyr_y_dt,  gyr_x_dt,         0;

                
                double s_gyr_norm = std::sin(gyr_norm);
                double gyr_norm_sq = gyr_norm * gyr_norm;
                double scalar_2 = (1 - std::cos(gyr_norm)) / gyr_norm_sq;
                Mat3 skew_mat_sq = gyr_skew_mat * gyr_skew_mat;

                e_R = e_R + ( (s_gyr_norm / gyr_norm ) * gyr_skew_mat )
                            + ( scalar_2 * skew_mat_sq);

                j_r = j_r - (scalar_2 * gyr_skew_mat)
                            + ( ( (gyr_norm - s_gyr_norm)/ (gyr_norm_sq * gyr_norm) ) * skew_mat_sq);

            }


            if( (i+1) > start_index_ ){

                Mat3 A = e_R.transpose();
                Mat3 B = (j_r*dt);
                Mat3 imu_cov;
                imu_cov <<  inter_data[0][i].second, 0.0, 0.0,
                            0.0, inter_data[1][i].second, 0.0,
                            0.0, 0.0, inter_data[2][i].second;
                cov.block<3,3>(0,0) = A*cov.block<3,3>(0,0)*A.transpose()
                    + B*imu_cov*B.transpose();
            }

            rot_mat = rot_mat * e_R;
            output[i+1].delta_R = rot_mat;
            output[i+1].cov = cov;
            output[i+1].dt = t.get(i+1) - start_t_;
            output[i+1].dt_sq_half = output[i+1].dt*output[i+1].dt*0.5;

            

            // Reproject the preintegrated measurements from in the starting frame
            if( (i+1) == start_index_)
            {
                for(int j = 0; j < (i+1); ++j)
                {                        
                    output[j].delta_R = rot_mat.transpose()*output[j].delta_R;
                }
                rot_mat = Mat3::Identity();
                output[start_index_].delta_R = rot_mat;
            }
        }
    }

    void ImuPreintegration::rotPreintLoop(
                        const std::vector<std::vector<std::pair<double, double> > >& inter_data,
                        const SortIndexTracker2<double>& t,
                        const PreintPrior prior,
                        std::vector<Mat3>& output)
    {
        if(output.size() != inter_data.front().size()) output.resize(inter_data.front().size());

        Mat3 rot_mat = Mat3::Identity();
        Mat9 cov = Mat9::Zero();

        output[0] = rot_mat;

        
        for(int i = 0; i < (t.size() - 1); ++i)
        {
            // Prepare some variables for convenience sake
            double dt = t.get(i+1)-t.get(i);
            
            Vec3 gyr_dt;
            gyr_dt << (inter_data[0][i].first - prior.gyr_bias[0]),
                    (inter_data[1][i].first - prior.gyr_bias[1]),
                    (inter_data[2][i].first - prior.gyr_bias[2]); 
            gyr_dt = gyr_dt*dt;
            

            Mat3 e_R = ExpMap(gyr_dt);

            rot_mat = rot_mat * e_R;
            output[i+1] = rot_mat;

            // Reproject the preintegrated measurements from in the starting frame
            if( (i+1) == start_index_)
            {
                for(int j = 0; j < (i+1); ++j)
                {                        
                    output[j] = rot_mat.transpose()*output[j];
                }
                rot_mat = Mat3::Identity();
                output[start_index_] = rot_mat;
            }
        }
    }


    void ImuPreintegration::prepareUGPM(
                celib::ImuData& imu_data, 
                const double start_time,
                const PreintPrior bias_prior)
    {
        // Create the rotation integrator
        rot_integrator_ = new RotIntegrator(imu_data, start_time, bias_prior);

        // Query the rotation at the accelerometer timestamps to project them
        std::vector<PreintMeas> acc_time_preint;
        int nb_acc = imu_data.acc.size();
        acc_time_preint.resize(nb_acc);
        #pragma omp parallel for schedule(static)
        for(int i = 0; i < nb_acc; ++i)
        {
            Mat3 rot;
            Mat3 var_rot;
            Vec3 d_rot_d_t;
            Mat3 d_rot_d_bw;
            rot_integrator_->get(
                imu_data.acc[i].t, rot, var_rot, d_rot_d_t, d_rot_d_bw);
            PreintMeas temp_preint;
            temp_preint.delta_R = rot;
            temp_preint.cov.block<3,3>(0,0) = var_rot;
            temp_preint.d_delta_R_d_t = d_rot_d_t;
            temp_preint.d_delta_R_d_bw = d_rot_d_bw;
            acc_time_preint[i] = temp_preint;
        }
        Mat3 delta_R_dt_start = rot_integrator_->get(start_time+kNumDtJacobianDelta);
        std::vector<std::shared_ptr<MatX> > d_acc_d_bf;
        std::vector<std::shared_ptr<MatX> > d_acc_d_bw;
        std::vector<std::shared_ptr<VecX> > d_acc_d_dt;
        reprojectAccData(acc_time_preint, prior_.acc_bias, delta_R_dt_start, d_acc_d_bf, d_acc_d_bw, d_acc_d_dt);

        for(int axis = 0; axis < 3; ++axis)
        {
            vel_ugpm_.push_back(GPMGP(imu_data_.acc, axis, imu_data_.acc_var, opt_.train_gpm, d_acc_d_bf[axis], d_acc_d_bw[axis], d_acc_d_dt[axis]));

        }

    }


    PreintMeas ImuPreintegration::getUGPM(const double t)
    {
        if(rot_integrator_ && (vel_ugpm_.size() == 3))
        {
            PreintMeas output;

            // Rot part
            Mat3 rot;
            Mat3 rot_var;
            Vec3 d_rot_d_t;
            Mat3 d_rot_d_w;
            rot_integrator_->get(t,rot,rot_var,d_rot_d_t,d_rot_d_w);
            output.delta_R = rot;
            output.cov = Mat9::Zero();
            output.cov.block<3,3>(0,0) = rot_var;
            output.d_delta_R_d_t = d_rot_d_t;
            output.d_delta_R_d_bw = d_rot_d_w;
            output.dt = t - start_t_;
            output.dt_sq_half = 0.5*output.dt*output.dt;


            // Vel and pos part
            Eigen::VectorXd t_vect(1);
            t_vect[0] = t;
            #pragma omp parallel for schedule(static)
            for(int axis = 0; axis < 3; ++axis)
            {
                Eigen::VectorXd d_v;
                Eigen::VectorXd var_d_v;
                Eigen::VectorXd d_p;
                Eigen::VectorXd var_d_p;
                Eigen::MatrixXd d_v_d_bf;
                Eigen::MatrixXd d_p_d_bf;
                Eigen::MatrixXd d_v_d_bw;
                Eigen::MatrixXd d_p_d_bw;
                Eigen::VectorXd d_v_d_dt;
                Eigen::VectorXd d_p_d_dt;
                vel_ugpm_[axis].integralAndVar(start_t_, t_vect, d_v, var_d_v,
                        d_v_d_bf, d_v_d_bw, d_v_d_dt);
                vel_ugpm_[axis].integral2AndVar(start_t_, t_vect, d_p, var_d_p,
                        d_p_d_bf, d_p_d_bw, d_p_d_dt);


                output.delta_v(axis) = d_v(0);
                output.cov(3+axis, 3+axis) = var_d_v(0);

                output.delta_p(axis) = d_p(0);
                output.cov(6+axis, 6+axis) = var_d_p(0);

                output.d_delta_v_d_bf.row(axis) = d_v_d_bf.row(0);
                output.d_delta_v_d_bw.row(axis) = d_v_d_bw.row(0);
                output.d_delta_v_d_t(axis) = d_v_d_dt(0);

                output.d_delta_p_d_bf.row(axis) = d_p_d_bf.row(0);
                output.d_delta_p_d_bw.row(axis) = d_p_d_bw.row(0);
                output.d_delta_p_d_t(axis) = d_p_d_dt(0);

            }
            return output;
        }
        else
        {
            throw "In ImuPreintegration::getUGPM, rot_integrator_ or vel_upgm_ not initialised";
        }
    }




    GpInterpolator::GpInterpolator(const std::vector<ImuSample>& samples,
                        const int axis,
                        const double var,
                        const bool train):
                        gp_(1,"CovSum( CovSEiso, CovNoise)")
    {
        // Compute the mean of the data (as per zero mean kernel)
        double mean = 0;
        double sig_var = 0;
        for(const auto& d : samples){
            mean += d.data[axis];
            sig_var += d.data[axis]*d.data[axis];
        }
        mean /= samples.size();
        sig_var = (sig_var + (mean*samples.size()) )/(samples.size()-1);
        mean_ = mean;



        // Prior on the covariance parameter (0-1 for Matern and 2 for noise)
        double prior_param[3];
        prior_param[0] = std::log(5.0*(samples.back().t - samples.front().t)/(samples.size()-1));
        prior_param[1] = std::log(std::sqrt(std::max(sig_var-var,var)));
        prior_param[2] = std::log(std::sqrt(var));


        Eigen::VectorXd h_param(3);
        h_param << prior_param[0], prior_param[1], prior_param[2];
        gp_.covf().set_loghyper(h_param);

        // Give the data to the GP
        for(int i = 0; i < samples.size(); ++i)
        {
            gp_.add_pattern(&(samples[i].t), samples[i].data[axis]-mean);
        }

        if(train)
        {
            libgp::RProp rprop;
            rprop.init();
            rprop.maximize(&gp_,10,0);
        }
    }

    // Infer a particular timestamp
    std::vector<std::pair<double, double> > GpInterpolator::get(const SortIndexTracker2<double>& t)
    {
        std::vector<std::pair<double, double> > output(t.size());
        for(int i = 0; i < t.size(); ++i)
        {                
            double temp[1];
            temp[0] = t.get(i);
            output[i] = std::make_pair(gp_.f(temp)+mean_, gp_.var(temp));
        }
        return output;
    }

    // Infer a particular timestamp + dt 
    std::vector<std::pair<double, double> > GpInterpolator::getShifted(const SortIndexTracker2<double>& t, const double dt)
    {
        std::vector<std::pair<double, double> > output(t.size());
        for(int i = 0; i < t.size(); ++i)
        {                
            double temp[1];
            temp[0] = t.get(i) + dt;
            output[i] = std::make_pair(gp_.f(temp)+mean_, gp_.var(temp));
        }
        return output;
    }
    


    GPMGP::GPMGP(const std::vector<ImuSample>& samples,
                const int axis,
                const double var,
                const bool train)
    {
        // Compute the mean of the data (as per zero mean kernel)
        double mean = 0;
        double sig_var = 0;
        for(const auto& d : samples){
            mean += d.data[axis];
            sig_var += d.data[axis]*d.data[axis];
        }
        sig_var = (sig_var + mean )/(samples.size()-1);
        mean /= samples.size();
        mean_ = mean;

        // Number of training samples
        int nb_tr = samples.size();


        // Prior on the covariance parameter (0-1 for Matern and 2 for noise)
        double prior_param[3];
        prior_param[0] = std::log(5.0*(samples.back().t - samples.front().t)/(samples.size()-1));
        prior_param[1] = std::log(std::sqrt(std::max(sig_var-var,var)));
        prior_param[2] = std::log(std::sqrt(var));

        for(int i = 0; i < samples.size(); ++i)
        {   
            data_time_.push_back(samples[i].t);
        }

        if(train)
        {
            libgp::GaussianProcess gp(1,"CovSum( CovSEiso, CovNoise)");
            Eigen::VectorXd h_param(3);
            h_param << prior_param[0], prior_param[1], prior_param[2];
            gp.covf().set_loghyper(h_param);

            // Give the data to the GP
            for(int i = 0; i < samples.size(); ++i)
            {   
                gp.add_pattern(&(samples[i].t), samples[i].data[axis]-mean);
            }

            libgp::RProp rprop;
            rprop.init();
            rprop.maximize(&gp,10,0);
            // Just to force the alpha computation
            double rubbish_x[1];
            rubbish_x[0] = 0; 
            double rubbish_y = gp.f(rubbish_x);

            hyp_sq_.resize(3);
            Eigen::Vector3d hyper = gp.get_loghyper();
            hyp_sq_[0] = exp(2.0*hyper(0));
            hyp_sq_[1] = exp(2.0*hyper(1));
            hyp_sq_[2] = exp(2.0*hyper(2));

            // Get alpha and L from libgp to prevent recomputation
            alpha_vec_.resize(nb_tr);
            alpha_vec_ = gp.getAlpha();

            L_.resize(nb_tr, nb_tr);
            L_ = gp.getL();
        }
        else
        {
            hyp_sq_.resize(3);
            hyp_sq_[0] = exp(2.0*prior_param[0]);
            hyp_sq_[1] = exp(2.0*prior_param[1]);
            hyp_sq_[2] = exp(2.0*prior_param[2]);

            double l2  = hyp_sq_[0];
            double sf2 = hyp_sq_[1];

            // Computing the Cholesky and alpha vectors
            Eigen::Map<Eigen::VectorXd> x(&(data_time_[0]), data_time_.size());
            L_.resize(nb_tr, nb_tr);
            L_ = seKernel(x,x,l2,sf2) + hyp_sq_[2]*Eigen::MatrixXd::Identity(nb_tr,nb_tr);
            Eigen::LLT<Eigen::MatrixXd> lltOfA(L_);
            L_ = lltOfA.matrixL();



            alpha_vec_.resize(nb_tr);
            for(int i = 0; i < samples.size(); ++i)
            {   
                alpha_vec_(i) = samples[i].data[axis] - mean_;
            }
            L_.triangularView<Eigen::Lower>().solveInPlace(alpha_vec_);
            L_.triangularView<Eigen::Lower>().transpose().solveInPlace(alpha_vec_);
        }
        
    }

    GPMGP::GPMGP(
                const std::vector<ImuSample>& samples,
                const int axis,
                const double var,
                const bool train,
                std::shared_ptr<MatX> data_d_bf,
                std::shared_ptr<MatX> data_d_bw,
                std::shared_ptr<VecX> data_d_dt):
                GPMGP(samples, axis, var, train)
    {
        data_d_bf_ = data_d_bf;
        data_d_bw_ = data_d_bw;
        data_d_dt_ = data_d_dt;
    }



    void GPMGP::integralAndVar(const double a, const Eigen::VectorXd& b,
            Eigen::VectorXd& integral_out, Eigen::VectorXd& var_integral_out,
            Eigen::MatrixXd& d_bf, Eigen::MatrixXd& d_bw, Eigen::VectorXd& d_dt)
    {
        int nb_tr = data_time_.size();
        int nb_infer = b.size();
        integral_out.resize(nb_infer);
        var_integral_out.resize(nb_infer);
        d_bf.resize(nb_infer,3);
        d_bw.resize(nb_infer,3);
        d_dt.resize(nb_infer);
        double l2  = hyp_sq_[0];
        double sf2 = hyp_sq_[1];
        double lik = hyp_sq_[2];
        Eigen::MatrixXd ks;
        Eigen::MatrixXd ksdt;
        Eigen::Map<Eigen::VectorXd> x(&(data_time_[0]), data_time_.size());
        ks = seKernelIntegral(a, b, x, l2, sf2);
        ksdt = seKernelIntegralDt(a, b, x, l2, sf2);
        integral_out = ks*alpha_vec_ + (b.array()-a).matrix()*mean_;
        d_bf = ks * L_.triangularView<Eigen::Lower>().transpose().solve(L_.triangularView<Eigen::Lower>().solve(*data_d_bf_));
        d_bw = ks * L_.triangularView<Eigen::Lower>().transpose().solve(L_.triangularView<Eigen::Lower>().solve(*data_d_bw_));
        d_dt = ksdt*alpha_vec_ + (ks*L_.triangularView<Eigen::Lower>().transpose().solve(L_.triangularView<Eigen::Lower>().solve(*data_d_dt_)));
       
        Eigen::MatrixXd temp(nb_tr, nb_infer);
        temp = L_.triangularView<Eigen::Lower>().transpose().solve(L_.triangularView<Eigen::Lower>().solve(ks.transpose()));
        var_integral_out = (temp.transpose()*temp).diagonal() *lik;

        for(int i = 0; i < nb_infer; ++i)
        {
            if( var_integral_out(i) < 0){
                var_integral_out(i) = ((b(i) - a)*lik);
            }
        }
    }




    void GPMGP::integral2AndVar(const double a, const Eigen::VectorXd& b,
            Eigen::VectorXd& integral_out, Eigen::VectorXd& var_integral_out,
            Eigen::MatrixXd& d_bf, Eigen::MatrixXd& d_bw, Eigen::VectorXd& d_dt)
    {
        int nb_tr = data_time_.size();
        int nb_infer = b.size();
        integral_out.resize(nb_infer);
        var_integral_out.resize(nb_infer);
        d_bf.resize(nb_infer,3);
        d_bw.resize(nb_infer,3);
        d_dt.resize(nb_infer);
        double l2  = hyp_sq_[0];
        double sf2 = hyp_sq_[1];
        double lik = hyp_sq_[2];
        Eigen::MatrixXd ks;
        Eigen::MatrixXd ksdt;
        Eigen::Map<Eigen::VectorXd> x(&(data_time_[0]), data_time_.size());
        ks = seKernelIntegral2(a, b, x, l2, sf2);
        ksdt = seKernelIntegral2Dt(a, b, x, l2, sf2);
        integral_out = ks*alpha_vec_ + 0.5*((b.array()-a)*(b.array()-a)).matrix()*mean_;
        d_bf = ks * L_.triangularView<Eigen::Lower>().transpose().solve(L_.triangularView<Eigen::Lower>().solve(*data_d_bf_));
        d_bw = ks * L_.triangularView<Eigen::Lower>().transpose().solve(L_.triangularView<Eigen::Lower>().solve(*data_d_bw_));
        d_dt = ksdt*alpha_vec_ + (ks*L_.triangularView<Eigen::Lower>().transpose().solve(L_.triangularView<Eigen::Lower>().solve(*data_d_dt_)));
       
        Eigen::MatrixXd temp(nb_tr, nb_infer);
        temp = L_.triangularView<Eigen::Lower>().transpose().solve(L_.triangularView<Eigen::Lower>().solve(ks.transpose()));
        var_integral_out = (temp.transpose()*temp).diagonal() *lik;

        for(int i = 0; i < nb_infer; ++i)
        {
            if( var_integral_out(i) < 0){
                var_integral_out(i) = 0.5*(b(i) - a)*(b(i) - a)*lik;
            }
        }
    }







    RotIntegrator::RotIntegrator(
                celib::ImuData& imu_data, 
                const double start_time,
                const PreintPrior bias_prior
                ):
                    start_t_(start_time),
                    nb_data_(imu_data.gyr.size())
    {

        // Fill the private structures and other variables
        ang_vel_.resize(3, nb_data_);
        data_time_.resize(nb_data_);
        std::vector<double> t_vect(nb_data_);
        std::vector<double> t_vect_dt(nb_data_);
        for(int i = 0; i < nb_data_; ++i)
        {
            ang_vel_(0,i) = imu_data.gyr[i].data[0] - bias_prior.gyr_bias[0];
            ang_vel_(1,i) = imu_data.gyr[i].data[1] - bias_prior.gyr_bias[1];
            ang_vel_(2,i) = imu_data.gyr[i].data[2] - bias_prior.gyr_bias[2];
            data_time_(i) = imu_data.gyr[i].t;
            t_vect[i] = imu_data.gyr[i].t;
            t_vect_dt[i] = imu_data.gyr[i].t + kNumDtJacobianDelta;
        }


        // Get prior from LPM preintegration
        celib::PreintOption preint_opt;
        preint_opt.min_freq = 1000;
        preint_opt.type = LPM;
        celib::PreintPrior prior;
        std::vector<std::vector<double> > t;
        t.push_back(t_vect);
        t.push_back(t_vect_dt);
        celib::ImuPreintegration preint(imu_data, start_t_, t, preint_opt, prior);
        MatX d_r_d_t;
        d_r_d_t.resize(3, nb_data_);
        mean_ = std::vector<double>(3,0.0);
        sf2_ = std::vector<double>(3,0.0);
        Vec3 r0 = LogMap(preint.get(0,0).delta_R);
        for(int i = 0; i < nb_data_; ++i)
        {
            if(i==0)
            {
                d_r_d_t.col(i) = JacobianRighthandExpMap(LogMap(preint.get(0,i).delta_R)).inverse() * ang_vel_.col(i);
            }
            else
            {
                Vec3 r1 = LogMap(preint.get(0,i).delta_R);
                Vec3 r1_dt = LogMap(preint.get(1,i).delta_R);
                double cos_r0r1 = r0.dot(r1) / (r0.norm()*r1.norm());
                double cos_r0r1_dt = r0.dot(r1_dt) / (r0.norm()*r1_dt.norm());
                if((cos_r0r1 < 0) && !((r0.norm()<0.1*M_PI)||((r1.norm()<0.1*M_PI))) )
                {
                    double new_norm = r1.norm()-(2.0*M_PI);
                    r1 = (r1/(r1.norm()))*new_norm;
                    
                }
                if((cos_r0r1_dt < 0) && !((r0.norm()<0.1*M_PI)||((r1_dt.norm()<0.1*M_PI))) )
                {
                    double new_norm = r1_dt.norm()-(2.0*M_PI);
                    r1_dt = (r1_dt/(r1_dt.norm()))*new_norm;
                }
                d_r_d_t.col(i) = ( r1_dt - r1 )/kNumDtJacobianDelta;
                r0 = r1;
            }
            
            mean_[0] += d_r_d_t(0,i);
            mean_[1] += d_r_d_t(1,i);
            mean_[2] += d_r_d_t(2,i);
        }
        for(int i = 0; i < 3; ++i)
        {
            mean_[i] /= nb_data_;
        }
        for(int i = 0; i < nb_data_; ++i){
            sf2_[0] += std::pow(d_r_d_t(0,i) - mean_[0],2);
            sf2_[1] += std::pow(d_r_d_t(1,i) - mean_[1],2);
            sf2_[2] += std::pow(d_r_d_t(2,i) - mean_[2],2);
        }
        for(int i = 0; i < 3; ++i)
        {
            sf2_[i] = (sf2_[i] - mean_[i]) / (nb_data_ - 1);
            if(sf2_[i] < 0) sf2_[i] = imu_data.gyr_var;
        }
        Eigen::Map<Vec3> mean_eigen(&(mean_[0]));
        d_r_d_t.colwise() -= mean_eigen;


        // Compute the hyper parameters
        lik_ = imu_data.gyr_var;
        double l = 5.0*(imu_data.gyr.back().t - imu_data.gyr.front().t)/(imu_data.gyr.size()-1);
        l2_ = l*l;


        // Compute Cholesky and store
        Eigen::MatrixXd K_int_scaleless = seKernelIntegral(start_t_, data_time_, data_time_, l2_, 1.0);
        Eigen::MatrixXd K_scaleless = seKernel(data_time_, data_time_, l2_, 1.0);
        std::vector<Eigen::MatrixXd> K_int;
        K_int.resize(3);
        std::vector<Eigen::MatrixXd> K(3);
        std::vector<Eigen::MatrixXd> L(3);
        K_inv_.resize(3);
        KK_inv_.resize(3);
        K_int_K_inv_.resize(3);
        #pragma omp parallel for schedule(static)
        for(int i=0; i < 3; ++i)
        {
            K_int[i].resize(nb_data_,nb_data_);
            K[i].resize(nb_data_,nb_data_);
            K_int[i] = sf2_[i]*K_int_scaleless;
            K[i] = sf2_[i]*K_scaleless;
            L[i].resize(nb_data_, nb_data_);
            MatX temp_to_inv(nb_data_, nb_data_);
            temp_to_inv = K[i] + lik_*Eigen::MatrixXd::Identity(nb_data_,nb_data_);
            L[i] = temp_to_inv;
            Eigen::LLT<Eigen::MatrixXd> lltOfA(L[i]);
            L[i] = lltOfA.matrixL();

            K_inv_[i].resize(nb_data_, nb_data_);
            K_inv_[i] = L[i].triangularView<Eigen::Lower>().transpose().solve(L[i].triangularView<Eigen::Lower>().solve(Eigen::MatrixXd::Identity(nb_data_, nb_data_)));
            KK_inv_[i].resize(nb_data_, nb_data_);
            KK_inv_[i] = K[i]*K_inv_[i];
            K_int_K_inv_[i].resize(nb_data_, nb_data_);
            K_int_K_inv_[i] = K_int[i]*K_inv_[i];
        }


        // Declaration of optimisation problem and state variable (with initialisation)
        ceres::Problem optimisation;
        Eigen::MatrixXd state;
        state.resize(nb_data_, 3);
        state = d_r_d_t.transpose();

        

        // Create the cost function and add to optimisation problem`
        IntegrationCostFunction* cost_fun = new IntegrationCostFunction(&ang_vel_, &KK_inv_, &K_int_K_inv_, &t_vect, start_t_, &mean_);
        //cost_fun->CheckGradient();
        optimisation.AddResidualBlock(cost_fun, nullptr, state.data());

        // Solve the optimisation problem
        ceres::Solver::Options solver_opt;
        solver_opt.minimizer_progress_to_stdout = false;
        solver_opt.num_threads = 4;
        solver_opt.max_num_iterations = 50;

        solver_opt.linear_solver_type = ceres::DENSE_NORMAL_CHOLESKY;
        ceres::Solver::Summary summary;
        ceres::Solve(solver_opt, &optimisation, &summary);
        state_.resize(nb_data_,3);
        state_ = state;

        alpha_.resize(3);
        #pragma omp parallel for schedule(static)
        for(int i = 0; i < 3; ++i)
        {
            alpha_[i] = K_inv_[i] * state_.col(i);
        }


        d_state_bw_.resize(3);
        for(auto& d:d_state_bw_) d.resize(nb_data_,3);
        solver_opt.max_num_iterations = 1;
        for(int i = 0; i < 3; ++i)
        {
            Vec3 offset;
            if( i == 0 )
            {
                offset << kNumGyrBiasJacobianDelta, 0.0, 0.0;
            }
            else if( i == 1 )
            {
                offset << -kNumGyrBiasJacobianDelta, kNumGyrBiasJacobianDelta, 0.0;
            }
            else
            {
                offset << 0.0, -kNumGyrBiasJacobianDelta, kNumGyrBiasJacobianDelta;
            }
            ang_vel_.colwise() += offset;
            // Declaration of optimisation problem and state variable (with initialisation)
            ceres::Problem optimisation_bw;
            MatX state_bw(nb_data_,3);
            state_bw = state_;

            // Create the cost function and add to optimisation problem
            IntegrationCostFunction* cost_fun_bw = new IntegrationCostFunction(&ang_vel_, &KK_inv_, &K_int_K_inv_, &t_vect, start_t_, &mean_);
            optimisation_bw.AddResidualBlock(cost_fun_bw, nullptr, state_bw.data());

            ceres::Solver::Summary summary_bw;
            ceres::Solve(solver_opt, &optimisation_bw, &summary_bw);
            
            MatX d_temp(nb_data_,3);
            d_temp = (state_bw - state_)/kNumGyrBiasJacobianDelta;
            for(int j = 0; j < 3; ++j)
            {
                d_state_bw_[j].col(i) = d_temp.col(j);
            }
        }


    }


    void RotIntegrator::get(const double t, Mat3& rot, Mat3& var, Vec3& d_rot_d_t, Mat3& d_rot_d_bw)
    {

        Vec3 r;
        var = Mat3::Zero();
        VecX t_vec(1);
        t_vec[0] = t;

        #pragma omp parallel for schedule(static)
        for(int i = 0; i < 3; ++i)
        {
            MatX ks = seKernelIntegral(start_t_, t_vec, data_time_, l2_, sf2_[i] );
            MatX ks_dt = seKernelIntegralDt(start_t_, t_vec, data_time_, l2_, sf2_[i] );
            r[i] =  (ks*alpha_[i] )(0,0) + (t-start_t_)*mean_[i];
            d_rot_d_t[i] = (ks_dt*alpha_[i] )(0,0);

            double kss = (seKernelIntegral(start_t_, t_vec, t_vec, l2_, sf2_[i] ))(0,0);

            Eigen::MatrixXd temp(1,nb_data_);
            temp = ks*K_inv_[i];
            var(i,i) = (temp.squaredNorm()) *lik_;

            d_rot_d_bw.row(i) = temp*d_state_bw_[i];

        }
        rot = ExpMap(r);
        d_rot_d_bw = JacobianRighthandExpMap(r)*d_rot_d_bw;
        d_rot_d_t = JacobianRighthandExpMap(r)*d_rot_d_t;
    }

    Mat3 RotIntegrator::get(const double t)
    {

        Vec3 r;
        VecX t_vec(1);
        t_vec[0] = t;

        #pragma omp parallel for schedule(static)
        for(int i = 0; i < 3; ++i)
        {
            MatX ks = seKernelIntegral(start_t_, t_vec, data_time_, l2_, sf2_[i] );
            r[i] =  (ks*alpha_[i] )(0,0) + (t-start_t_)*mean_[i];
        }

        return ExpMap(r);
    }


    // Combined cost function for SO3 integration
    bool IntegrationCostFunction::Evaluate(double const* const* parameters, double* residuals, double** jacobians) const
    {
        Eigen::Map<const MatX> s(&(parameters[0][0]), nb_data_,3);
        Eigen::Map<MatX> r(residuals, nb_data_, 3);

        Eigen::Map<MatX> r_norm(&(residuals[nb_data_*3]), nb_data_, 3);

        MatX d_rot_d_t;
        d_rot_d_t.resize(nb_data_, 3);
        MatX rot;
        rot.resize(nb_data_, 3);

        for(int i = 0; i < 3; ++i)
        {
            d_rot_d_t.col(i) = KK_inv_->at(i)*s.col(i);
            rot.col(i) = K_int_K_inv_->at(i)*s.col(i);
        
        }


        MatX temp;
        temp.resize(nb_data_,3);
        Eigen::Map<Vec3> mean(&(mean_->at(0)));
        #pragma omp parallel for schedule(static)
        for(int i = 0; i < nb_data_; ++i)
        {
            Vec3 rot_vec = rot.row(i).transpose() + (t_->at(i) - start_t_)*mean;
            Vec3 d_rot_vec = d_rot_d_t.row(i).transpose() + mean;
            temp.row(i) = ( JacobianRighthandExpMap<double>(rot_vec)*(d_rot_vec) ).transpose();

            if(jacobians != NULL)
            {
                if(jacobians[0] != NULL)
                {
                    Mat3_6 d_res_d_rdr = JacobianRes(rot_vec, d_rot_vec);
                    MatX d_r_d_s(6, 3*nb_data_);
                    d_r_d_s.setZero();
                    d_r_d_s.block(0,0,1,nb_data_) = K_int_K_inv_->at(0).row(i);
                    d_r_d_s.block(1,nb_data_,1,nb_data_) = K_int_K_inv_->at(1).row(i);
                    d_r_d_s.block(2,2*nb_data_,1,nb_data_) = K_int_K_inv_->at(2).row(i);
                    d_r_d_s.block(3,0,1,nb_data_) = KK_inv_->at(0).row(i);
                    d_r_d_s.block(4,nb_data_,1,nb_data_) = KK_inv_->at(1).row(i);
                    d_r_d_s.block(5,2*nb_data_,1,nb_data_) = KK_inv_->at(2).row(i);
                    MatX temp_jacobian(3, nb_data_);
                    temp_jacobian = d_res_d_rdr*d_r_d_s;

                    Eigen::Map<Eigen::Matrix<double,1,Eigen::Dynamic> > j_s_0(&(jacobians[0][i*3*nb_data_]),1,3*nb_data_);
                    j_s_0 = temp_jacobian.row(0);
                    Eigen::Map<Eigen::Matrix<double,1,Eigen::Dynamic> > j_s_1(&(jacobians[0][(nb_data_+i)*3*nb_data_]),1,3*nb_data_);
                    j_s_1 = temp_jacobian.row(1);
                    Eigen::Map<Eigen::Matrix<double,1,Eigen::Dynamic> > j_s_2(&(jacobians[0][(2*nb_data_+i)*3*nb_data_]),1,3*nb_data_);
                    j_s_2 = temp_jacobian.row(2);


                    Eigen::Map<Eigen::Matrix<double,1,Eigen::Dynamic> > j2_s_0(&(jacobians[0][(3*nb_data_+i)*3*nb_data_]),1,3*nb_data_);
                    j2_s_0.setZero();
                    j2_s_0.block(0,0,1,nb_data_) = KK_inv_->at(0).row(i);
                    j2_s_0(i) -= 1.0;
                    Eigen::Map<Eigen::Matrix<double,1,Eigen::Dynamic> > j2_s_1(&(jacobians[0][(4*nb_data_+i)*3*nb_data_]),1,3*nb_data_);
                    j2_s_1.setZero();
                    j2_s_1.block(0,nb_data_,1,nb_data_) = KK_inv_->at(1).row(i);
                    j2_s_1(nb_data_+i) -= 1.0;
                    Eigen::Map<Eigen::Matrix<double,1,Eigen::Dynamic> > j2_s_2(&(jacobians[0][(5*nb_data_+i)*3*nb_data_]),1,3*nb_data_);
                    j2_s_2.setZero();
                    j2_s_2.block(0,2*nb_data_,1,nb_data_) = KK_inv_->at(2).row(i);
                    j2_s_2(2*nb_data_+i) -= 1.0;
                }
            }
        }

        r = temp - ang_vel_->transpose();
        r_norm = d_rot_d_t - s;

        return true;
    }


    bool IntegrationCostFunction::CheckGradient(bool show_report)
    {

        std::cout << "Optimisation jacobian check:" << std::endl;

        std::vector<const ceres::LocalParameterization*> local_parameterizations;
        local_parameterizations.push_back(nullptr);

        std::vector<double*> parameter_blocks;
        double state[3*nb_data_];
        parameter_blocks.push_back(&state[0]);

        // Map the ceres block to eigen format and put random values on them
        Eigen::Map<MatX> eigen_state(parameter_blocks[0], nb_data_, 3);
        eigen_state = MatX::Random(nb_data_,3);


        ceres::NumericDiffOptions numeric_diff_options;
        ceres::GradientChecker gradient_checker(this, &local_parameterizations, numeric_diff_options);
        ceres::GradientChecker::ProbeResults results;

        bool output;
        if(!gradient_checker.Probe(parameter_blocks.data(), 1e-6, &results)){
            if(show_report){
                std::cout<< results.error_log<<std::endl;
            }
            std::cout << "Optimisation jacobian is above threshold: to be investigated" << std::endl;
            output = false;
        }else{
            std::cout << "Optimisation jacobian seems good" << std::endl;
            output = true;
        }

        return output;
    }
}