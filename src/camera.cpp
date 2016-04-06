/**
 This file is part of Defocus.
 
 Copyright(C) 2016 Christoph Heindl
 All rights reserved.
 
 This software may be modified and distributed under the terms
 of the BSD license.See the LICENSE file for details.
 */

#include <defocus/camera.h>
#include <Eigen/Dense>
#include <iostream>

namespace defocus {
    
    Eigen::Vector3d pixelToRetina(double x, double y, const Eigen::Matrix3d &kInverse) {
        Eigen::Vector3d p(x, y, 1.0);
        return kInverse * p;
    }
    
    Eigen::Vector3d PinholeCamera::pixelToRetina(double x, double y, const Eigen::Matrix3d & kInverse)
    {
        Eigen::Vector3d p(x, y, 1.0);
        return kInverse * p;
    }

    Eigen::Matrix3Xd PinholeCamera::reconstruct(Eigen::Ref<const Eigen::Matrix2Xd> f, const Eigen::VectorXd &depths, const Eigen::MatrixXd &k)
    {

        Eigen::Matrix3Xd points(3, f.cols());
        Eigen::Matrix3d kinv = k.inverse();

        for (Eigen::DenseIndex i = 0; i < f.cols(); ++i) {
            points.col(i) = pixelToRetina(f(0, i), f(1, i), kinv) * depths(i);
        }

        return points;
    }
}