/**
 This file is part of Defocus.
 
 Copyright(C) 2016 Christoph Heindl
 All rights reserved.
 
 This software may be modified and distributed under the terms
 of the BSD license.See the LICENSE file for details.
 */

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <random>
namespace defocus {
    namespace testhelper {
        
        inline Eigen::Matrix3Xd uniformRandomPointsInBox(const Eigen::Vector3d &minCorner,
                                                         const Eigen::Vector3d &maxCorner,
                                                         Eigen::DenseIndex count)
        {
            Eigen::AlignedBox3d box(minCorner, maxCorner);
            Eigen::Matrix3Xd points(3, count);
            for (int i = 0; i < count; ++i) {
                points.col(i) = box.sample();
            }
            return points;
        }
        
        inline Eigen::Matrix<double,3,4> uniformRandomPose()
        {
            std::random_device rd;
            std::default_random_engine e(rd());
            std::uniform_real_distribution<double> distTranslation(-3.0, 3.0);
            std::uniform_real_distribution<double> distRotation(-0.005, 0.005);
            
            Eigen::AffineCompact3d t;
            t = Eigen::Translation3d(distTranslation(e), distTranslation(e), distTranslation(e)) *
                Eigen::AngleAxisd(distRotation(e), Eigen::Vector3d(1,0,0)) *
                Eigen::AngleAxisd(distRotation(e), Eigen::Vector3d(0,1,0)) *
                Eigen::AngleAxisd(distRotation(e), Eigen::Vector3d(0,0,1));
            
            return t.matrix();
        }
    }
}