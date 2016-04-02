/**
 This file is part of Defocus.
 
 Copyright(C) 2016 Christoph Heindl
 All rights reserved.
 
 This software may be modified and distributed under the terms
 of the BSD license.See the LICENSE file for details.
 */

#include <defocus/camera.h>


namespace defocus {
    
    Eigen::Vector3d pixelToRetina(double x, double y, const Eigen::Matrix3d &kInverse) {
        Eigen::Vector3d p(x, y, 1.0);
        return kInverse * p;
    }
}