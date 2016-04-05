/**
 This file is part of Defocus.
 
 Copyright(C) 2016 Christoph Heindl
 All rights reserved.
 
 This software may be modified and distributed under the terms
 of the BSD license.See the LICENSE file for details.
 */

#include <defocus/io.h>
#include <fstream>


namespace defocus {
    
    void writePointsAndColorsAsPLY(const char *path, Eigen::Ref<const Eigen::MatrixXd> points, Eigen::Ref<Eigen::MatrixXd> colors)
    {
        std::ofstream ofs(path);

        ofs
            << "ply" << std::endl
            << "format ascii 1.0" << std::endl
            << "element vertex " << points.cols() << std::endl
            << "property float x" << std::endl
            << "property float y" << std::endl
            << "property float z" << std::endl
            << "property uchar red" << std::endl
            << "property uchar green" << std::endl
            << "property uchar blue" << std::endl
            << "end_header" << std::endl;


        for (Eigen::DenseIndex i = 0; i < points.cols(); ++i) {

            Eigen::Vector3d x = points.col(i);
            Eigen::Vector3d c = colors.col(i);

            ofs << x(0) << " " << x(1) << " " << x(2) << " " << (int)c(0) << " " << (int)c(1) << " " << (int)c(2) << std::endl;
        }

        ofs.close();
    }
    
}