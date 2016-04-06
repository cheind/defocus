/**
 This file is part of Defocus.
 
 Copyright(C) 2016 Christoph Heindl
 All rights reserved.
 
 This software may be modified and distributed under the terms
 of the BSD license.See the LICENSE file for details.
 */

#ifndef DEFOCUS_SPARSE_H
#define DEFOCUS_SPARSE_H

#include <Eigen/Core>
#include <vector>

namespace defocus {

    class SparseSmallMotionBundleAdjustment {
    public:
        void setFeatures(const Eigen::MatrixXd &features);
        void setInitialDepths(const Eigen::VectorXd &depths);
        void setInitialCameraParameters(const Eigen::Matrix<double, 6, Eigen::Dynamic> &params);
        void setCameraMatrix(const Eigen::Matrix3d &k);

        double solve();

        const Eigen::VectorXd &depths() const;
        const Eigen::Matrix<double, 6, Eigen::Dynamic> &cameraParameters() const;
    private:
        Eigen::MatrixXd _features;
        Eigen::VectorXd _depths;
        Eigen::Matrix<double, 6, Eigen::Dynamic> _cameras;
        Eigen::Matrix3d _intr;
    };

    class InitialConditions {
    public:
        static Eigen::VectorXd uniformRandomDepths(double minDepth, double maxDepth, Eigen::DenseIndex nObservations);
        static Eigen::VectorXd constantDepths(double depth, Eigen::DenseIndex nObservations);
        static Eigen::Matrix<double, 6, Eigen::Dynamic> identityCameraParameters(Eigen::DenseIndex nCameras);
    };
}

#endif