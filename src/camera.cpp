/**
 This file is part of Defocus.
 
 Copyright(C) 2016 Christoph Heindl
 All rights reserved.
 
 This software may be modified and distributed under the terms
 of the BSD license.See the LICENSE file for details.
 */

#include <defocus/camera.h>
#include <Eigen/Dense>
#include <Eigen/Geometry>

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

    Eigen::Matrix3Xd PinholeCamera::reconstructPoints(Eigen::Ref<const Eigen::Matrix2Xd> f, const Eigen::VectorXd &depths, const Eigen::Matrix3d &k)
    {

        Eigen::Matrix3Xd points(3, f.cols());
        Eigen::Matrix3d kinv = k.inverse();

        for (Eigen::DenseIndex i = 0; i < f.cols(); ++i) {
            points.col(i) = pixelToRetina(f(0, i), f(1, i), kinv) * depths(i);
        }

        return points;
    }
    
    Eigen::Matrix3Xd PinholeCamera::perspectiveProject(Eigen::Ref<const Eigen::Matrix3Xd> points, const Eigen::Matrix<double, 3, 4>  &pose, const Eigen::Matrix3d &k)
    {
        Eigen::AffineCompact3d t(pose);
        Eigen::Matrix<double, 3, 4> p = k * t.inverse(Eigen::Isometry).matrix();

        return p * points.colwise().homogeneous();
    }
    
    Eigen::Matrix<double, 3, 4> PinholeCamera::smallMotionCameraParameterVectorToMatrix(Eigen::Ref<const Eigen::Matrix<double, 6, 1> > v)
    {
        Eigen::Matrix<double, 3, 4> m;
        m.setIdentity();
        
        m(0, 1) = -v(2);
        m(1, 0) = v(2);
        m(0, 2) = v(1);
        m(2, 0) = -v(1);
        m(1, 2) = -v(0);
        m(2, 1) = v(0);
        
        m(0, 3) = v(3);
        m(1, 3) = v(4);
        m(2, 3) = v(5);
        
        return m;
    }
    
    Eigen::Matrix<double, 6, 1> PinholeCamera::smallMotionCameraMatrixToParameterVector(Eigen::Ref<const Eigen::Matrix<double, 3, 4> > m)
    {
        Eigen::Matrix<double, 6, 1> v;
        
        v(0) = m(2, 1);
        v(1) = m(0, 2);
        v(2) = m(1, 0);
        
        v(3) = m(0, 3);
        v(4) = m(1, 3);
        v(5) = m(2, 3);
        
        return v;
    }
}