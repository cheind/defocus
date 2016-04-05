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
    
    
    double solveSmallMotionBundleAdjustment(const Eigen::Matrix<double, 3, Eigen::Dynamic> &retinaPoints,
                                          Eigen::Matrix<double, 6, Eigen::Dynamic> &cameraParameters,
                                          Eigen::Matrix<double, 3, Eigen::Dynamic> &reconstructedPoints,
                                          Eigen::DenseIndex nCameras,
                                          Eigen::DenseIndex nObservationsPerCamera);
}

#endif