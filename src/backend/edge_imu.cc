#include "backend/vertex_pose.h"
#include "backend/vertex_speedbias.h"
#include "backend/edge_imu.h"

#include <iostream>

namespace myslam {
namespace backend {
using Sophus::SO3d;

Vec3 EdgeImu::gravity_ = Vec3(0, 0, 9.8);

void EdgeImu::ComputeResidual() 
{
    // Quaterniond (w x y z) and position for frame i
    VecX param_0 = verticies_[0]->Parameters();
    Qd Qi(param_0[6], param_0[3], param_0[4], param_0[5]);
    Vec3 Pi = param_0.head<3>(); 

    // vel, acc_bias and gyr_bias for frame i
    VecX param_1 = verticies_[1]->Parameters();
    Vec3 Vi =  param_1.head<3>();
    Vec3 Bai = param_1.segment(3, 3);
    Vec3 Bgi = param_1.tail<3>();

    VecX param_2 = verticies_[2]->Parameters();
    Qd Qj(param_2[6], param_2[3], param_2[4], param_2[5]);
    Vec3 Pj = param_2.head<3>();

    VecX param_3 = verticies_[3]->Parameters();
    Vec3 Vj = param_3.head<3>();
    Vec3 Baj = param_3.segment(3, 3);
    Vec3 Bgj = param_3.tail<3>();

    residual_ = pre_integration_->evaluate(Pi, Qi, Vi, Bai, Bgi,
                                           Pj, Qj, Vj, Baj, Bgj);
    // Mat1515 sqrt_info  = Eigen::LLT< Mat1515 >(pre_integration_->covariance.inverse()).matrixL().transpose();
    SetInformation(pre_integration_->covariance.inverse());
}

void EdgeImu::ComputeJacobians() 
{
    VecX param_0 = verticies_[0]->Parameters();
    Qd Qi(param_0[6], param_0[3], param_0[4], param_0[5]);
    Vec3 Pi = param_0.head<3>();

    VecX param_1 = verticies_[1]->Parameters();
    Vec3 Vi  = param_1.head<3>();
    Vec3 Bai = param_1.segment(3, 3);
    Vec3 Bgi = param_1.tail<3>();

    VecX param_2 = verticies_[2]->Parameters();
    Qd Qj(param_2[6], param_2[3], param_2[4], param_2[5]);
    Vec3 Pj = param_2.head<3>();

    VecX param_3 = verticies_[3]->Parameters();
    Vec3 Vj  = param_3.head<3>();
    Vec3 Baj = param_3.segment(3, 3);
    Vec3 Bgj = param_3.tail<3>();

    double sum_dt = pre_integration_->sum_dt;
    Eigen::Matrix3d dp_dba = pre_integration_->jacobian.template block<3, 3>(O_P, O_BA);
    Eigen::Matrix3d dp_dbg = pre_integration_->jacobian.template block<3, 3>(O_P, O_BG);

    Eigen::Matrix3d dq_dbg = pre_integration_->jacobian.template block<3, 3>(O_R, O_BG);

    Eigen::Matrix3d dv_dba = pre_integration_->jacobian.template block<3, 3>(O_V, O_BA);
    Eigen::Matrix3d dv_dbg = pre_integration_->jacobian.template block<3, 3>(O_V, O_BG);

    if (pre_integration_->jacobian.maxCoeff() > 1e8 || pre_integration_->jacobian.minCoeff() < -1e8)
    {
        // ROS_WARN("numerical unstable in preintegration");
    }

    // 残差对P,Q的jacobians, row:residual col:state
    // if (jacobians[0])
    {
        Eigen::Matrix<double, 15, 6, Eigen::RowMajor> jacobian_pose_i;
        jacobian_pose_i.setZero();

        // J(0, 0) P增量误差/P(w,bk)误差
        jacobian_pose_i.block<3, 3>(O_P, O_P) = -Qi.inverse().toRotationMatrix();
        // J(0, 3) P增量误差/Q（w,bk)误差
        jacobian_pose_i.block<3, 3>(O_P, O_R)
            = Utility::skewSymmetric(Qi.inverse() * (0.5 * G * sum_dt * sum_dt + Pj - Pi - Vi * sum_dt));

        // J(3, 3) Q增量误差/Q（w,bk)误差
        // for initial window: no difference between frame i and j.
        Eigen::Quaterniond corrected_delta_q 
            = pre_integration_->delta_q * Utility::deltaQ(dq_dbg * (Bgi - pre_integration_->linearized_bg));
        jacobian_pose_i.block<3, 3>(O_R, O_R) 
            = -( Utility::Qleft(Qj.inverse() * Qi) * Utility::Qright(corrected_delta_q) )
                    .bottomRightCorner<3, 3>();
        // J(6, 3) V增量误差/Q（w,bk)误差
        jacobian_pose_i.block<3, 3>(O_V, O_R) = Utility::skewSymmetric(Qi.inverse() * (G * sum_dt + Vj - Vi));
        // jacobian_pose_i = sqrt_info * jacobian_pose_i;

        if (jacobian_pose_i.maxCoeff() > 1e8 || jacobian_pose_i.minCoeff() < -1e8)
        {
            // ROS_WARN("numerical unstable in preintegration");
        }
        jacobians_[0] = jacobian_pose_i;
    }

    // if (jacobians[1])
    {
        Eigen::Matrix<double, 15, 9, Eigen::RowMajor> jacobian_speedbias_i;
        jacobian_speedbias_i.setZero();
        jacobian_speedbias_i.block<3, 3>(O_P, O_V - O_V) = -Qi.inverse().toRotationMatrix() * sum_dt;
        jacobian_speedbias_i.block<3, 3>(O_P, O_BA - O_V) = -dp_dba;
        jacobian_speedbias_i.block<3, 3>(O_P, O_BG - O_V) = -dp_dbg;

        //Eigen::Quaterniond corrected_delta_q = pre_integration->delta_q * Utility::deltaQ(dq_dbg * (Bgi - pre_integration->linearized_bg));
        //jacobian_speedbias_i.block<3, 3>(O_R, O_BG - O_V) = -Utility::Qleft(Qj.inverse() * Qi * corrected_delta_q).bottomRightCorner<3, 3>() * dq_dbg;
        // Q增量误差/ba（bk?,bk)误差
        jacobian_speedbias_i.block<3, 3>(O_R, O_BG - O_V)
            = -Utility::Qleft(Qj.inverse() * Qi * pre_integration_->delta_q)
                    .bottomRightCorner<3, 3>() * dq_dbg;
        jacobian_speedbias_i.block<3, 3>(O_V, O_V - O_V) = -Qi.inverse().toRotationMatrix();
        jacobian_speedbias_i.block<3, 3>(O_V, O_BA - O_V) = -dv_dba;
        jacobian_speedbias_i.block<3, 3>(O_V, O_BG - O_V) = -dv_dbg;

        jacobian_speedbias_i.block<3, 3>(O_BA, O_BA - O_V) = -Eigen::Matrix3d::Identity();
        jacobian_speedbias_i.block<3, 3>(O_BG, O_BG - O_V) = -Eigen::Matrix3d::Identity();

        // jacobian_speedbias_i = sqrt_info * jacobian_speedbias_i;
        jacobians_[1] = jacobian_speedbias_i;
    }
    // if (jacobians[2])
    {
        Eigen::Matrix<double, 15, 6, Eigen::RowMajor> jacobian_pose_j;
        jacobian_pose_j.setZero();
        jacobian_pose_j.block<3, 3>(O_P, O_P) = Qi.inverse().toRotationMatrix();

        Eigen::Quaterniond corrected_delta_q = pre_integration_->delta_q * Utility::deltaQ(dq_dbg * (Bgi - pre_integration_->linearized_bg));
        jacobian_pose_j.block<3, 3>(O_R, O_R) = Utility::Qleft(corrected_delta_q.inverse() * Qi.inverse() * Qj).bottomRightCorner<3, 3>();
        // jacobian_pose_j = sqrt_info * jacobian_pose_j;
        jacobians_[2] = jacobian_pose_j;
    }
    // if (jacobians[3])
    {
        Eigen::Matrix<double, 15, 9, Eigen::RowMajor> jacobian_speedbias_j;
        jacobian_speedbias_j.setZero();

        jacobian_speedbias_j.block<3, 3>(O_V, O_V - O_V) = Qi.inverse().toRotationMatrix();

        jacobian_speedbias_j.block<3, 3>(O_BA, O_BA - O_V) = Eigen::Matrix3d::Identity();

        jacobian_speedbias_j.block<3, 3>(O_BG, O_BG - O_V) = Eigen::Matrix3d::Identity();

        // jacobian_speedbias_j = sqrt_info * jacobian_speedbias_j;
        jacobians_[3] = jacobian_speedbias_j;
    }
}

}
}