/**
 This file is part of Defocus.
 
 Copyright(C) 2016 Christoph Heindl
 All rights reserved.
 
 This software may be modified and distributed under the terms
 of the BSD license.See the LICENSE file for details.
 */

#ifndef DEFOCUS_FEATURES_H
#define DEFOCUS_FEATURES_H

#include <opencv2/core/core.hpp>
#include <Eigen/Core>

namespace defocus {
    
    /** Detect good features to track. */
    void findFeatureInImage(cv::Mat &gray, std::vector<cv::Point2f> &corners);
    
    
    /** Track features between two frames. */
    void trackFeatures(const cv::Mat &refGray,
                       const std::vector<cv::Point2f> &refLocs,
                       const cv::Mat &targetGray,
                       std::vector<cv::Point2f> &targetLocs,
                       std::vector<uchar> &status,
                       double maxError = 5);
    
    /** Remove features without a valid status */
    std::vector<cv::Point2f> eliminateInvalidFeatures(const std::vector<cv::Point2f> &locs, const std::vector<uchar> &status);
    
    
    template<class T>
    void removeByStatus(std::vector<T> &v, const std::vector<uchar> &status) {
        size_t i, k;
        for( i = k = 0; i < v.size(); i++) {
            if( !status[i] )
                continue;
            
            std::swap(v[k++], v[i]);
        }
        v.resize(k);
    }
}

#endif