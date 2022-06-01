#pragma once
#include <eigen3/Eigen/Dense>
#include <iostream>
#include <map>

#include "../factor/integration_base.h"
#include "../utility/utility.h"
#include "../feature_manager.h"

using namespace Eigen;
using namespace std;

class ImageFrame
{
public:
    ImageFrame(){};
    ImageFrame(const map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>> &_points, double _t) : t{_t}, is_key_frame{false}
    {
        points = _points;
    };
    // points :: int:feat id, int: camera id, max:{x,y,z,u,v,vx,vy}
    map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>> points;
    double t;
    Matrix3d R; // 从某帧的IMU参考系(bk)->起始帧的相机参考系(c0)的旋转
    Vector3d T; // 从某帧的相机参考系(ck)->起始帧的相机参考系(c0)的旋转
    IntegrationBase *pre_integration;
    bool is_key_frame;
};

bool VisualIMUAlignment(map<double, ImageFrame> &all_image_frame, Vector3d *Bgs, Vector3d &g, VectorXd &x);