/**
 This file is part of Defocus.
 
 Copyright(C) 2016 Christoph Heindl
 All rights reserved.
 
 This software may be modified and distributed under the terms
 of the BSD license.See the LICENSE file for details.
 */

#ifndef DEFOCUS_DENSE_H
#define DEFOCUS_DENSE_H

#include <Eigen/Core>
#include <opencv2/core/core.hpp>

namespace defocus {

    class DenseDepthPropagation {
    public:
        DenseDepthPropagation();
        void setSparsePoints(const Eigen::Matrix3Xd &points);
        void setCameraMatrix(const Eigen::Matrix3d &k);
        void setColorImage(const cv::Mat &image);
        void setImageResolution(int width, int height);
        void setMultiScaleLevels(int levels);

        void solve();

        cv::Mat denseDepthMap() const;
    private:
        
        int scaleFactorForLevel(int level) const;
        Eigen::VectorXd solveForLevel(int level, const Eigen::VectorXd guess) const;
        cv::Mat_<double> sparseDepthMapForScaleFactor(int scale) const;
        
        cv::Mat_<double> solutionVectorToImage(int width, int height, const Eigen::VectorXd &v) const;
        Eigen::VectorXd solutionImageToVector(const cv::Mat_<double> &image) const;
        
        Eigen::Matrix3Xd _points;
        Eigen::Matrix3d _intr;
        int _width, _height, _levels;
        cv::Mat _depths, _colors;
    };
}

#endif