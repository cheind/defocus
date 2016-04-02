/**
 This file is part of Defocus.
 
 Copyright(C) 2016 Christoph Heindl
 All rights reserved.
 
 This software may be modified and distributed under the terms
 of the BSD license.See the LICENSE file for details.
 */

#ifndef DEFOCUS_SFM_H
#define DEFOCUS_SFM_H

#include <Eigen/Core>
#include <vector>

namespace defocus {
    
    class SmallMotionBundleAdjustment {
    public:
        
        /** Create bundle adjustment problem */
        SmallMotionBundleAdjustment(Eigen::DenseIndex nCameras, Eigen::DenseIndex nObservationsPerCamera);
        
        /** Add an observation in the retina plane (x, y, 1) by a specific camera. */
        void setObservation(Eigen::DenseIndex cameraIdx, Eigen::DenseIndex observationIdx, const Eigen::Vector3d &point);
        
        /** Run bundle adjustment */
        double run(bool debug, Eigen::DenseIndex refCameraIdx);
        
        /** Get reconstructed points */
        Eigen::Matrix<double, 3, Eigen::Dynamic> pointsInReferenceCamera() const;
        
    private:
        Eigen::Matrix<double, 3, Eigen::Dynamic> _points;
        Eigen::Matrix<double, 6, Eigen::Dynamic> _cameras;
        Eigen::DenseIndex _nCameras, _nObservations, _refCamera;
        std::vector<double> _idepths;
    };
}

#endif