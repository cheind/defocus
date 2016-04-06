/**
 This file is part of Defocus.
 
 Copyright(C) 2016 Christoph Heindl
 All rights reserved.
 
 This software may be modified and distributed under the terms
 of the BSD license.See the LICENSE file for details.
 */

#ifndef DEFOCUS_CAMERA_H
#define DEFOCUS_CAMERA_H

#include <opencv2/core/core.hpp>
#include <Eigen/Core>

namespace defocus {

    class PinholeCamera {
    public:
        static Eigen::Vector3d pixelToRetina(double x, double y, const Eigen::Matrix3d &kInverse);
        static Eigen::Matrix3Xd reconstruct(Eigen::Ref<const Eigen::Matrix2Xd> f, const Eigen::VectorXd &depths, const Eigen::MatrixXd &k);
    };
}

#endif