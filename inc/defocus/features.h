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

    class KLT;

    class SmallMotionTracker {
    public:
        typedef std::vector<cv::Point2f> CVFeatureLocations;
        typedef std::pair<CVFeatureLocations, std::vector<uchar> > CVFrameResult;

        SmallMotionTracker();
        ~SmallMotionTracker();
        void setMaxError(double error);
        void initializeFromReferenceFrame(const cv::Mat &bgr);
        CVFrameResult addFrame(const cv::Mat &bgr);

        Eigen::MatrixXd trackedFeaturesPerFrame() const;

    private:
        typedef std::vector<CVFeatureLocations> CVFeatureLocationsPerFrame;

        CVFeatureLocationsPerFrame _features;
        std::vector<uchar> _status;
        double _maxError;
        cv::Mat _refGray;
    };

    

}

#endif