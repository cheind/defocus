/**
 This file is part of Defocus.
 
 Copyright(C) 2016 Christoph Heindl
 All rights reserved.
 
 This software may be modified and distributed under the terms
 of the BSD license.See the LICENSE file for details.
 */

#include <defocus/features.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/video/video.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <iostream>


namespace defocus {
       
    class KLT {
    public:
        
        KLT(const cv::Mat &img, const std::vector<cv::Point2f> &initial, std::vector<uchar> &initialStatus, double maxError)
        : _next(initial), _nextStatus(initialStatus)
        {
            if (_nextStatus.size() != initial.size()) {
                _nextStatus.clear();
                _nextStatus.resize(initial.size(), 1);
            }
            img.copyTo(_nextGray);
            _maxError = maxError;
        }
        
        void update(const cv::Mat &img) {
            
            std::swap(_prev, _next);
            std::swap(_prevGray, _nextGray);
            std::swap(_prevStatus, _nextStatus);
            
            img.copyTo(_nextGray);
            cv::TermCriteria term(cv::TermCriteria::COUNT+cv::TermCriteria::EPS, 50, 0.001);
            cv::calcOpticalFlowPyrLK(_prevGray, _nextGray, _prev, _next, _nextStatus, _err, cv::Size(21, 21), 5, term);
            
            for (size_t i = 0; i < _nextStatus.size(); ++i) {
                _nextStatus[i] &= _prevStatus[i];
                _nextStatus[i] &= (_err[i] < _maxError);
            }
        }
        
        std::vector<cv::Point2f> &location() {
            return _next;
        }
        
        std::vector<uchar> &status() {
            return _nextStatus;
        }
        
        
    private:
        std::vector<cv::Point2f> _prev, _next;
        cv::Mat _prevGray, _nextGray;
        std::vector<uchar> _prevStatus, _nextStatus;
        std::vector<float> _err;
        double _maxError;
    };

    
    
    
    SmallMotionTracker::SmallMotionTracker()
        :_maxError(5.0)
    {
    }

    SmallMotionTracker::~SmallMotionTracker()
    {
    }

    void SmallMotionTracker::setMaxError(double error)
    {
        _maxError = error;
    }

    void SmallMotionTracker::initializeFromReferenceFrame(const cv::Mat & bgr)
    {
        cv::cvtColor(bgr, _refGray, CV_BGR2GRAY);

        _features.resize(1);
        
        /*
        double qualityLevel = 0.01;
        double minDistance = 5;
        int blockSize = 5;
        bool useHarrisDetector = false;
        double k = 0.04;
        int maxCorners = 4000;
        
        cv::goodFeaturesToTrack(_refGray,
                                _features[0],
                                maxCorners,
                                qualityLevel,
                                minDistance,
                                cv::Mat(),
                                blockSize,
                                useHarrisDetector,
                                k);
        
        cv::TermCriteria termcrit(CV_TERMCRIT_ITER|CV_TERMCRIT_EPS,20,0.03);
        cv::cornerSubPix(_refGray, _features[0], cv::Size(10,10), cv::Size(-1,-1), termcrit);
*/
        std::vector<cv::KeyPoint> keys;
        cv::FAST(_refGray, keys, 25, true);

        _features[0].clear();
        for (size_t i = 0; i < keys.size(); ++i) {
            _features[0].push_back(keys[i].pt);
        }
        
        cv::TermCriteria termcrit(CV_TERMCRIT_ITER|CV_TERMCRIT_EPS, 20, 0.03);
        cv::cornerSubPix(_refGray, _features[0], cv::Size(10,10), cv::Size(-1,-1), termcrit);
        
        
        _status.clear();
        _status.resize(_features[0].size(), 1);
    }

    SmallMotionTracker::CVFrameResult SmallMotionTracker::addFrame(const cv::Mat & bgr)
    {
        cv::Mat gray;
        cv::cvtColor(bgr, gray, CV_BGR2GRAY);
        
        KLT klt(_refGray, _features[0], _status, _maxError);
        klt.update(gray);

        _features.push_back(klt.location());
        _status = klt.status();

        return std::make_pair(klt.location(), klt.status());
    }

    template<class T>
    void removeByStatus(std::vector<T> &v, const std::vector<uchar> &status) {
        size_t i, k;
        for (i = k = 0; i < v.size(); i++) {
            if (!status[i])
                continue;

            std::swap(v[k++], v[i]);
        }
        v.resize(k);
    }

    Eigen::MatrixXd SmallMotionTracker::trackedFeaturesPerFrame() const
    {
        eigen_assert(_tracker != 0);
        eigen_assert(_features.size() > 0);

        CVFeatureLocationsPerFrame fcopy = _features;

        for (size_t c = 0; c < fcopy.size(); ++c) {
            removeByStatus(fcopy[c], _status);
        }

        Eigen::DenseIndex nFrames = fcopy.size();
        Eigen::DenseIndex nObs = fcopy[0].size();

        Eigen::MatrixXd features(nFrames * 2, nObs);

        for (Eigen::DenseIndex f = 0; f < nFrames; ++f) {   
            const CVFeatureLocations &l = fcopy[f];
            for (Eigen::DenseIndex o = 0; o < nObs; ++o) {
                features(f*2 + 0, o) = l[o].x;
                features(f*2 + 1, o) = l[o].y;
            }
        }

        return features;
    }

}