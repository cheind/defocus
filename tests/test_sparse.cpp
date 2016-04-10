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

struct SmallMotionEuclideanReprojectionErrorF {
    SmallMotionEuclideanReprojectionErrorF(const double *no, const double *np)
    : _no(no), _np(np)
    {}
    
    template <typename T>
    bool operator()(
                    const T* const camera,
                    const T* const depth,
                    T* residuals) const
    {
        T p[3] = {T(_np[0]) / depth[0], T(_np[1]) / depth[0], T(1.0) / depth[0] };
        
        T rpt[3] = {
            p[0] - camera[2] * p[1] + camera[1] * p[2] + camera[3],
            p[0] * camera[2] + p[1] - camera[0] * p[2] + camera[4],
            -p[0] * camera[1] + p[1] * camera[0] + p[2] + camera[5]
        };
        
        residuals[0] = rpt[0] / rpt[2] - T(_no[0]) ;
        residuals[1] = rpt[1] / rpt[2] - T(_no[1]) ;
        
        return true;
    }
    
    const double *_no;
    const double *_np;
};


TEST_CASE("test_bundleadjustment")
{
    const double foc = 530.0;
    const int width = 640;
    const int height = 480;
    const Eigen::DenseIndex nObs = 50;
    const Eigen::DenseIndex nFrames = 10;
    
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

    // Solve bunde adjustment starting with ideal initial solution.
    defocus::SparseSmallMotionBundleAdjustment ba;
    ba.setCameraMatrix(k);
    ba.setFeatures(features);
    ba.setInitialCameraParameters(cameraParameters);
    ba.setInitialDepths(points.row(2));
    ba.solve();
    
    const double factor = points(2, 0) / ba.depths()(0);
    Eigen::Matrix3Xd recPoints = defocus::PinholeCamera::reconstructPoints(features.block(0,0,2,nObs), (ba.depths() * factor), k);
    
    REQUIRE((recPoints - points).colwise().norm().maxCoeff() < 20.0);
}