/**
 This file is part of Defocus.
 
 Copyright(C) 2016 Christoph Heindl
 All rights reserved.
 
 This software may be modified and distributed under the terms
 of the BSD license.See the LICENSE file for details.
 */

#ifndef DEFOCUS_IO_H
#define DEFOCUS_IO_H

#include <Eigen/Core>

namespace defocus {
    
    void writePointsAndColorsAsPLY(const char *path, Eigen::Ref<const Eigen::MatrixXd> points, Eigen::Ref<Eigen::MatrixXd> colors);    
}

#endif