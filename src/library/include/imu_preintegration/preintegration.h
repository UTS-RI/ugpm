/**
 *  Author: Cedric LE GENTIL 
 *
 *  Copyright 2021 Cedric LE GENTIL
 *
 *  For any further question, recommendation or contribution
 *  le.gentil.cedric@gmail.com
 **/

#ifndef PREINTEGRATION_H
#define PREINTEGRATION_H

#include "common/types.h"
#include "common/internal_utils.h"
#include <iostream>
#include "libgp/include/gp.h"
#include <ceres/ceres.h>
#include <memory>

namespace celib
{


    const double kNumDtJacobianDelta = 0.001;
    const double kNumAccBiasJacobianDelta = 0.01;
    const double kNumGyrBiasJacobianDelta = 0.01;


    enum PreintType {LPM, GPM, UGPM};

    struct PreintOption
    {
        double min_freq = 1000;
        PreintType type = GPM;
        bool train_gpm = false;
        double quantum = -1;
    };

    struct PreintPrior
    {
        std::vector<double> acc_bias = {0,0,0};
        std::vector<double> gyr_bias = {0,0,0};
    };




    // Class to perform nice integration over SO3
    class RotIntegrator{

        public:
            RotIntegrator(
                celib::ImuData& imu_data, 
                const double start_time,
                const PreintPrior bias_prior
            );


            // Method for inference
            void get(const double t, Mat3& rot, Mat3& var, Vec3& d_rot_d_t, Mat3& d_rot_d_q);
            Mat3 get(const double t);

        private:
            Eigen::MatrixXd ang_vel_;
            double l2_;
            double lik_;
            std::vector<double> sf2_;
            std::vector<double> mean_;
            Eigen::VectorXd data_time_;
            double start_t_;
            int nb_data_;
            Eigen::MatrixXd state_;
            std::vector<Eigen::MatrixXd> d_state_bw_;

            //std::vector<Eigen::MatrixXd> L_;
            std::vector<Eigen::VectorXd> alpha_;
            std::vector<Eigen::MatrixXd> KK_inv_;
            std::vector<Eigen::MatrixXd> K_int_K_inv_;
            std::vector<Eigen::MatrixXd> K_inv_;
    };




    // Class for GP inference of signal's integrals
    class GPMGP
    {
        public:
            GPMGP(
                    const std::vector<ImuSample>& samples,
                    const int axis,
                    const double var,
                    const bool train);
            
            GPMGP(
                    const std::vector<ImuSample>& samples,
                    const int axis,
                    const double var,
                    const bool train,
                    std::shared_ptr<MatX> data_d_bf,
                    std::shared_ptr<MatX> data_d_bw,
                    std::shared_ptr<VecX> data_d_dt);
            
            void integralAndVar(const double a, const Eigen::VectorXd& b,
                    Eigen::VectorXd& integral_out, Eigen::VectorXd& var_integral_out, 
                    Eigen::MatrixXd& d_bf, Eigen::MatrixXd& d_bw, Eigen::VectorXd& d_dt);
            void integral2AndVar(const double a, const Eigen::VectorXd& b,
                    Eigen::VectorXd& integral_out, Eigen::VectorXd& var_integral_out, 
                    Eigen::MatrixXd& d_bf, Eigen::MatrixXd& d_bw, Eigen::VectorXd& d_dt);
        
        private:
            double mean_;

            Eigen::VectorXd alpha_vec_;
            Eigen::MatrixXd L_;


            std::vector<double> hyp_sq_;
            std::vector<double> data_time_;
            std::shared_ptr<Eigen::MatrixXd> data_d_bf_;
            std::shared_ptr<Eigen::MatrixXd> data_d_bw_;
            std::shared_ptr<Eigen::VectorXd> data_d_dt_;

    };


    // Class to compute preintegrated measurements and potentially store the pre-computed meausrements
    class ImuPreintegration
    {

        public:
            // Constructor given IMU data and inference timestamps
            ImuPreintegration(celib::ImuData& imu_data,
                    const double start_t,
                    std::vector<std::vector<double> >& infer_t,
                    const PreintOption opt,
                    const PreintPrior prior,
                    const bool rot_only = false);

            ~ImuPreintegration(){
                if(rot_integrator_) delete rot_integrator_;
            }

            // Compute the preintegrated measurement for a given timestamp
            PreintMeas get(const double t);
            // Get the preintegrated measurement as per indexed in the given inference times of the constructor
            PreintMeas get(const int index_1, const int index_2);

            // Accessor for prior
            PreintPrior getPrior() { return prior_;}

        private:
            // Store a copy of the imu_data used
            celib::ImuData imu_data_;

            double start_t_;
            int start_index_;
            PreintOption opt_;
            PreintPrior prior_;

            RotIntegrator* rot_integrator_ = nullptr;
            std::vector<GPMGP> vel_ugpm_;


            // Store the pre-computed preintegrated measurements if contructor with time-stamps
            std::vector<std::vector<PreintMeas> > preint_;


            // Reproject the accelerometer data in imu_data_ based on the given rotational preintegrated measurements
            void reprojectAccData(
                        const std::vector<PreintMeas>& preint,
                        const std::vector<double>& prior,
                        const Mat3& delta_R_dt_start,
                        std::vector<std::shared_ptr<MatX> >& d_acc_d_bf,
                        std::vector<std::shared_ptr<MatX> >& d_acc_d_bw,
                        std::vector<std::shared_ptr<VecX> >& d_acc_d_dt);

            // Compute the preintegration for the rotational part
            std::vector<PreintMeas> rotPreintIterative(
                        const SortIndexTracker2<double>& t,
                        const std::vector<bool>& interest_t,
                        const PreintType type,
                        const PreintPrior prior);
            void rotPreintLoop(
                        const std::vector<std::vector<std::pair<double, double> > >& inter_data,
                        const SortIndexTracker2<double>& t,
                        const PreintPrior prior,
                        std::vector<PreintMeas>& output);
            void rotPreintLoop(
                        const std::vector<std::vector<std::pair<double, double> > >& inter_data,
                        const SortIndexTracker2<double>& t,
                        const PreintPrior prior,
                        std::vector<Mat3>& output);




            // Compute the preintegration for the velocity and position part
            void velPosPreintLPM(
                        std::vector<std::vector<double> >& t,
                        const std::vector<std::shared_ptr<MatX> >& d_acc_d_bf,
                        const std::vector<std::shared_ptr<MatX> >& d_acc_d_bw,
                        const std::vector<std::shared_ptr<VecX> >& d_acc_d_dt,
                        std::vector<std::vector<PreintMeas> >& preint);
            void velPosPreintLPMPartial(
                        const SortIndexTracker2<double>& time,
                        std::vector<ImuSample>& acc_data,
                        std::vector<std::vector<PreintMeas> >& preint);
            void velPosPreintGPM(
                        const std::vector<std::vector<double> >& t,
                        std::vector<std::shared_ptr<MatX> >& d_acc_d_bf,
                        std::vector<std::shared_ptr<MatX> >& d_acc_d_bw,
                        std::vector<std::shared_ptr<VecX> >& d_acc_d_dt,
                        std::vector<std::vector<PreintMeas> >& preint);

            // Interpolation methods
            std::vector<std::pair<double, double> > linearInterpolation(
                        const std::vector<ImuSample>& samples,
                        const int axis,
                        const double var,
                        const SortIndexTracker2<double>& infer_t);
            std::vector<std::pair<double, double> > gpInterpolation(
                        const std::vector<ImuSample>& samples,
                        const int axis,
                        const double var,
                        const SortIndexTracker2<double>& infer_t);


            
            void prepareUGPM(
                        celib::ImuData& imu_data, 
                        const double start_time,
                        const PreintPrior bias_prior);

            PreintMeas getUGPM(
                        const double t);
    };



    // Class to perform GP interpolation for the preintegration
    class GpInterpolator
    {            
        public:
            GpInterpolator(const std::vector<ImuSample>& samples,
                        const int axis,
                        const double var,
                        const bool train);

            std::vector<std::pair<double, double> > get(const SortIndexTracker2<double>& t);

            std::vector<std::pair<double, double> > getShifted(const SortIndexTracker2<double>& t, const double dt);

        private:
            libgp::GaussianProcess gp_;
            double mean_;
            
    };




    class IntegrationCostFunction: public ceres::CostFunction
    {


        public:
            IntegrationCostFunction(
                Eigen::MatrixXd* ang_vel,
                std::vector<MatX>* KK_inv,
                std::vector<MatX>* K_int_K_inv,
                std::vector<double>* t,
                double start_t,
                std::vector<double>* mean): 
                    ang_vel_(ang_vel),
                    KK_inv_(KK_inv),
                    K_int_K_inv_(K_int_K_inv),
                    t_(t),
                    start_t_(start_t),
                    mean_(mean)
                    {
                        nb_data_ = KK_inv_->at(0).cols();
                        set_num_residuals(6*nb_data_);
                        std::vector<int>* block_sizes = mutable_parameter_block_sizes();
                        block_sizes->push_back(3*nb_data_);
                    };


            // Inherited from ceres::CostFunction, compute the residuals and jacobians
            virtual bool Evaluate(double const* const* parameters,
                                    double* residuals,
                                    double** jacobians) const;
            
            
            bool CheckGradient(bool show_report = true);



        private:
            Eigen::MatrixXd* ang_vel_;
            std::vector<MatX>* KK_inv_;
            std::vector<MatX>* K_int_K_inv_;
            std::vector<double>* t_;
            double start_t_;
            std::vector<double>* mean_;
            int nb_data_;


    };


    inline celib::Mat3_6 JacobianRes(celib::Vec3 rot_vec, celib::Vec3 d_r)
    {
        celib::Mat3_6 output;

        double r0_sq = rot_vec(0)*rot_vec(0);
        double r1_sq = rot_vec(1)*rot_vec(1);
        double r2_sq = rot_vec(2)*rot_vec(2);
        double temp_r = (r0_sq + r1_sq + r2_sq);
        double norm_r = std::sqrt(temp_r);
        celib::Vec3 r = rot_vec;

        //if(norm_r >= 2.0*M_PI)
        //{
        //    celib::Vec3 unit_vec;
        //    unit_vec = rot_vec / norm_r;
        //    norm_r = std::fmod(norm_r,2.0*M_PI);
        //    r = unit_vec*norm_r;
        //    r0_sq = r(0)*r(0);
        //    r1_sq = r(1)*r(1);
        //    r2_sq = r(2)*r(2);
        //    temp_r = (r0_sq + r1_sq + r2_sq);
        //}

        if(norm_r > kExpNormTolerance)
        {

            double r0_cu = r(0)*r(0)*r(0);
            double r1_cu = r(1)*r(1)*r(1);
            double r2_cu = r(2)*r(2)*r(2);
            double norm_r_2 = std::pow(temp_r, 2);
            double norm_r_3 = std::pow(temp_r, 1.5);
            double norm_r_5 = std::pow(temp_r, 2.5);
            double s_r = std::sin(norm_r);
            double c_r = std::cos(norm_r);

            output(0,0) = d_r(1)*((r(0)*r(2)*s_r)/norm_r_3 - (r(1)*(s_r - norm_r))/norm_r_3 + (2.0*r(0)*r(2)*(c_r - 1.0))/norm_r_2 + (3.0*r0_sq*r(1)*(s_r - norm_r))/norm_r_5 + (r(0)*r(1)*(r(0)/norm_r - (r(0)*c_r)/norm_r))/norm_r_3) - d_r(2)*((r(2)*(s_r - norm_r))/norm_r_3 + (r(0)*r(1)*s_r)/norm_r_3 + (2.0*r(0)*r(1)*(c_r - 1.0))/norm_r_2 - (3.0*r0_sq*r(2)*(s_r - norm_r))/norm_r_5 - (r(0)*r(2)*(r(0)/norm_r - (r(0)*c_r)/norm_r))/norm_r_3) - d_r(0)*((r1_sq*(r(0)/norm_r - (r(0)*c_r)/norm_r))/norm_r_3 + (r2_sq*(r(0)/norm_r - (r(0)*c_r)/norm_r))/norm_r_3 + (3.0*r(0)*r1_sq*(s_r - norm_r))/norm_r_5 + (3.0*r(0)*r2_sq*(s_r - norm_r))/norm_r_5);

            output(0,1) = d_r(2)*((c_r - 1.0)/temp_r - (r1_sq*s_r)/norm_r_3 - (2.0*r1_sq*(c_r - 1.0))/norm_r_2 + (r(0)*r(2)*(r(1)/norm_r - (r(1)*c_r)/norm_r))/norm_r_3 + (3.0*r(0)*r(1)*r(2)*(s_r - norm_r))/norm_r_5) - d_r(0)*((3.0*r1_cu*(s_r - norm_r))/norm_r_5 + (r1_sq*(r(1)/norm_r - (r(1)*c_r)/norm_r))/norm_r_3 + (r2_sq*(r(1)/norm_r - (r(1)*c_r)/norm_r))/norm_r_3 - (2.0*r(1)*(s_r - norm_r))/norm_r_3 + (3.0*r(1)*r2_sq*(s_r - norm_r))/norm_r_5) + d_r(1)*((r(1)*r(2)*s_r)/norm_r_3 - (r(0)*(s_r - norm_r))/norm_r_3 + (2.0*r(1)*r(2)*(c_r - 1.0))/norm_r_2 + (3.0*r(0)*r1_sq*(s_r - norm_r))/norm_r_5 + (r(0)*r(1)*(r(1)/norm_r - (r(1)*c_r)/norm_r))/norm_r_3);

            output(0,2) = d_r(1)*((r2_sq*s_r)/norm_r_3 - (c_r - 1.0)/temp_r + (2.0*r2_sq*(c_r - 1.0))/norm_r_2 + (r(0)*r(1)*(r(2)/norm_r - (r(2)*c_r)/norm_r))/norm_r_3 + (3.0*r(0)*r(1)*r(2)*(s_r - norm_r))/norm_r_5) - d_r(0)*((3.0*r2_cu*(s_r - norm_r))/norm_r_5 + (r1_sq*(r(2)/norm_r - (r(2)*c_r)/norm_r))/norm_r_3 + (r2_sq*(r(2)/norm_r - (r(2)*c_r)/norm_r))/norm_r_3 - (2.0*r(2)*(s_r - norm_r))/norm_r_3 + (3.0*r1_sq*r(2)*(s_r - norm_r))/norm_r_5) - d_r(2)*((r(0)*(s_r - norm_r))/norm_r_3 + (r(1)*r(2)*s_r)/norm_r_3 + (2.0*r(1)*r(2)*(c_r - 1.0))/norm_r_2 - (3.0*r(0)*r2_sq*(s_r - norm_r))/norm_r_5 - (r(0)*r(2)*(r(2)/norm_r - (r(2)*c_r)/norm_r))/norm_r_3);

            output(0,3) = (r1_sq*(s_r - norm_r))/norm_r_3 + (r2_sq*(s_r - norm_r))/norm_r_3 + 1.0;

            output(0,4) = - (r(2)*(c_r - 1.0))/temp_r - (r(0)*r(1)*(s_r - norm_r))/norm_r_3;

            output(0,5) = (r(1)*(c_r - 1.0))/temp_r - (r(0)*r(2)*(s_r - norm_r))/norm_r_3;

            

            output(1,0) = d_r(2)*((r0_sq*s_r)/norm_r_3 - (c_r - 1.0)/temp_r + (2.0*r0_sq*(c_r - 1.0))/norm_r_2 + (r(1)*r(2)*(r(0)/norm_r - (r(0)*c_r)/norm_r))/norm_r_3 + (3.0*r(0)*r(1)*r(2)*(s_r - norm_r))/norm_r_5) - d_r(1)*((3.0*r0_cu*(s_r - norm_r))/norm_r_5 + (r0_sq*(r(0)/norm_r - (r(0)*c_r)/norm_r))/norm_r_3 + (r2_sq*(r(0)/norm_r - (r(0)*c_r)/norm_r))/norm_r_3 - (2.0*r(0)*(s_r - norm_r))/norm_r_3 + (3.0*r(0)*r2_sq*(s_r - norm_r))/norm_r_5) - d_r(0)*((r(1)*(s_r - norm_r))/norm_r_3 + (r(0)*r(2)*s_r)/norm_r_3 + (2.0*r(0)*r(2)*(c_r - 1.0))/norm_r_2 - (3.0*r0_sq*r(1)*(s_r - norm_r))/norm_r_5 - (r(0)*r(1)*(r(0)/norm_r - (r(0)*c_r)/norm_r))/norm_r_3);

            output(1,1) = d_r(2)*((r(0)*r(1)*s_r)/norm_r_3 - (r(2)*(s_r - norm_r))/norm_r_3 + (2.0*r(0)*r(1)*(c_r - 1.0))/norm_r_2 + (3.0*r1_sq*r(2)*(s_r - norm_r))/norm_r_5 + (r(1)*r(2)*(r(1)/norm_r - (r(1)*c_r)/norm_r))/norm_r_3) - d_r(0)*((r(0)*(s_r - norm_r))/norm_r_3 + (r(1)*r(2)*s_r)/norm_r_3 + (2.0*r(1)*r(2)*(c_r - 1.0))/norm_r_2 - (3.0*r(0)*r1_sq*(s_r - norm_r))/norm_r_5 - (r(0)*r(1)*(r(1)/norm_r - (r(1)*c_r)/norm_r))/norm_r_3) - d_r(1)*((r0_sq*(r(1)/norm_r - (r(1)*c_r)/norm_r))/norm_r_3 + (r2_sq*(r(1)/norm_r - (r(1)*c_r)/norm_r))/norm_r_3 + (3.0*r0_sq*r(1)*(s_r - norm_r))/norm_r_5 + (3.0*r(1)*r2_sq*(s_r - norm_r))/norm_r_5);

            output(1,2) = d_r(0)*((c_r - 1.0)/temp_r - (r2_sq*s_r)/norm_r_3 - (2.0*r2_sq*(c_r - 1.0))/norm_r_2 + (r(0)*r(1)*(r(2)/norm_r - (r(2)*c_r)/norm_r))/norm_r_3 + (3.0*r(0)*r(1)*r(2)*(s_r - norm_r))/norm_r_5) - d_r(1)*((3.0*r2_cu*(s_r - norm_r))/norm_r_5 + (r0_sq*(r(2)/norm_r - (r(2)*c_r)/norm_r))/norm_r_3 + (r2_sq*(r(2)/norm_r - (r(2)*c_r)/norm_r))/norm_r_3 - (2.0*r(2)*(s_r - norm_r))/norm_r_3 + (3.0*r0_sq*r(2)*(s_r - norm_r))/norm_r_5) + d_r(2)*((r(0)*r(2)*s_r)/norm_r_3 - (r(1)*(s_r - norm_r))/norm_r_3 + (2.0*r(0)*r(2)*(c_r - 1.0))/norm_r_2 + (3.0*r(1)*r2_sq*(s_r - norm_r))/norm_r_5 + (r(1)*r(2)*(r(2)/norm_r - (r(2)*c_r)/norm_r))/norm_r_3);

            output(1,3) = (r(2)*(c_r - 1.0))/temp_r - (r(0)*r(1)*(s_r - norm_r))/norm_r_3;

            output(1,4) = (r0_sq*(s_r - norm_r))/norm_r_3 + (r2_sq*(s_r - norm_r))/norm_r_3 + 1.0;

            output(1,5) = - (r(0)*(c_r - 1.0))/temp_r - (r(1)*r(2)*(s_r - norm_r))/norm_r_3;




            output(2,0) = d_r(1)*((c_r - 1.0)/temp_r - (r0_sq*s_r)/norm_r_3 - (2.0*r0_sq*(c_r - 1.0))/norm_r_2 + (r(1)*r(2)*(r(0)/norm_r - (r(0)*c_r)/norm_r))/norm_r_3 + (3.0*r(0)*r(1)*r(2)*(s_r - norm_r))/norm_r_5) - d_r(2)*((3.0*r0_cu*(s_r - norm_r))/norm_r_5 + (r0_sq*(r(0)/norm_r - (r(0)*c_r)/norm_r))/norm_r_3 + (r1_sq*(r(0)/norm_r - (r(0)*c_r)/norm_r))/norm_r_3 - (2.0*r(0)*(s_r - norm_r))/norm_r_3 + (3.0*r(0)*r1_sq*(s_r - norm_r))/norm_r_5) + d_r(0)*((r(0)*r(1)*s_r)/norm_r_3 - (r(2)*(s_r - norm_r))/norm_r_3 + (2.0*r(0)*r(1)*(c_r - 1.0))/norm_r_2 + (3.0*r0_sq*r(2)*(s_r - norm_r))/norm_r_5 + (r(0)*r(2)*(r(0)/norm_r - (r(0)*c_r)/norm_r))/norm_r_3);

            output(2,1) = d_r(0)*((r1_sq*s_r)/norm_r_3 - (c_r - 1.0)/temp_r + (2.0*r1_sq*(c_r - 1.0))/norm_r_2 + (r(0)*r(2)*(r(1)/norm_r - (r(1)*c_r)/norm_r))/norm_r_3 + (3.0*r(0)*r(1)*r(2)*(s_r - norm_r))/norm_r_5) - d_r(2)*((3.0*r1_cu*(s_r - norm_r))/norm_r_5 + (r0_sq*(r(1)/norm_r - (r(1)*c_r)/norm_r))/norm_r_3 + (r1_sq*(r(1)/norm_r - (r(1)*c_r)/norm_r))/norm_r_3 - (2.0*r(1)*(s_r - norm_r))/norm_r_3 + (3.0*r0_sq*r(1)*(s_r - norm_r))/norm_r_5) - d_r(1)*((r(2)*(s_r - norm_r))/norm_r_3 + (r(0)*r(1)*s_r)/norm_r_3 + (2.0*r(0)*r(1)*(c_r - 1.0))/norm_r_2 - (3.0*r1_sq*r(2)*(s_r - norm_r))/norm_r_5 - (r(1)*r(2)*(r(1)/norm_r - (r(1)*c_r)/norm_r))/norm_r_3);

            output(2,2) = d_r(0)*((r(1)*r(2)*s_r)/norm_r_3 - (r(0)*(s_r - norm_r))/norm_r_3 + (2.0*r(1)*r(2)*(c_r - 1.0))/norm_r_2 + (3.0*r(0)*r2_sq*(s_r - norm_r))/norm_r_5 + (r(0)*r(2)*(r(2)/norm_r - (r(2)*c_r)/norm_r))/norm_r_3) - d_r(1)*((r(1)*(s_r - norm_r))/norm_r_3 + (r(0)*r(2)*s_r)/norm_r_3 + (2.0*r(0)*r(2)*(c_r - 1.0))/norm_r_2 - (3.0*r(1)*r2_sq*(s_r - norm_r))/norm_r_5 - (r(1)*r(2)*(r(2)/norm_r - (r(2)*c_r)/norm_r))/norm_r_3) - d_r(2)*((r0_sq*(r(2)/norm_r - (r(2)*c_r)/norm_r))/norm_r_3 + (r1_sq*(r(2)/norm_r - (r(2)*c_r)/norm_r))/norm_r_3 + (3.0*r0_sq*r(2)*(s_r - norm_r))/norm_r_5 + (3.0*r1_sq*r(2)*(s_r - norm_r))/norm_r_5);

            output(2,3) = - (r(1)*(c_r - 1.0))/temp_r - (r(0)*r(2)*(s_r - norm_r))/norm_r_3;

            output(2,4) = (r(0)*(c_r - 1.0))/temp_r - (r(1)*r(2)*(s_r - norm_r))/norm_r_3;

            output(2,5) = (r0_sq*(s_r - norm_r))/norm_r_3 + (r1_sq*(s_r - norm_r))/norm_r_3 + 1.0;
        }
        else
        {
            output.block<3,3>(0,0) = 0.5*ToSkewSymMat(d_r);
            output.block<3,3>(0,3) = Mat3::Identity();
        }

        return output;
    }

}





#endif