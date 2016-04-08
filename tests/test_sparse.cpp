/**
 This file is part of Defocus.
 
 Copyright(C) 2016 Christoph Heindl
 All rights reserved.
 
 This software may be modified and distributed under the terms
 of the BSD license.See the LICENSE file for details.
*/

#define CATCH_CONFIG_MAIN
#include "catch.hpp"

#include <defocus/camera.h>
#include <defocus/sparse.h>

#include "test_helpers.h"

TEST_CASE("test_bundleadjustment")
{
    const double foc = 530.0;
    const int width = 640;
    const int height = 480;
    const Eigen::DenseIndex nObs = 10;
    const Eigen::DenseIndex nFrames = 50;
    
    Eigen::Matrix3d k;
    k <<
    foc, 0.0, 0.5 *(width - 1),
    0.0, foc, 0.5 *(height - 1),
    0.0, 0.0, 1.0;
    
    // Generate random 3D points
    Eigen::Matrix3Xd points = defocus::testhelper::uniformRandomPointsInBox(Eigen::Vector3d(-500, -500, 300), Eigen::Vector3d(500, 500, 1500), nObs);
    
    // Generate random motions
    typedef Eigen::Matrix<double, 3, 4> Pose;
    std::vector<Pose> poses(nFrames);
    Eigen::Matrix<double, 6, Eigen::Dynamic> cameraParameters(6, nFrames);
    
    poses[0] = Pose::Identity();
    cameraParameters.col(0).setZero();
    for (Eigen::DenseIndex i = 1; i < nFrames; ++i) {
        poses[i] = defocus::testhelper::uniformRandomPose();
        cameraParameters.col(i) = defocus::PinholeCamera::smallMotionCameraMatrixToParameterVector(poses[i]);
    }
    
    // Project points with respect to camera motion
    Eigen::MatrixXd features(2 * nFrames, nObs);
    for (Eigen::DenseIndex i = 0; i < nFrames; ++i) {
        Eigen::Matrix2Xd proj = defocus::PinholeCamera::perspectiveProject(points, poses[i], k).colwise().hnormalized();
        features.row(i*2+0) = proj.row(0);
        features.row(i*2+1) = proj.row(1);
    }

    double reprojerror2 = 0.0;

    for (Eigen::DenseIndex f = 0; f < nFrames; ++f) {
        auto observed = features.block(f * 2, 0, 2, nObs);
        auto m = defocus::PinholeCamera::smallMotionCameraParameterVectorToMatrix(cameraParameters.col(f));
        Eigen::Matrix2Xd proj = defocus::PinholeCamera::perspectiveProject(points, m, k).colwise().hnormalized();       

        reprojerror2 += (observed - proj).colwise().squaredNorm().sum();
    }

    std::cout << reprojerror2 << std::endl;


    defocus::SparseSmallMotionBundleAdjustment ba;
    ba.setCameraMatrix(k);
    ba.setFeatures(features);
    ba.setInitialCameraParameters(cameraParameters);
    ba.setInitialDepths(points.row(2));
    ba.solve();

    std::cout << points.row(2) << std::endl;
    std::cout << ba.depths() << std::endl;
    
    
    
}