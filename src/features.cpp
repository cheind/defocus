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
        
        KLT(const cv::Mat &img, const std::vector<cv::Point2f> &initial, double maxError)
        :_next(initial)
        {
            img.copyTo(_nextGray);
            _nextStatus.resize(initial.size(), 1);
            _maxError = maxError;
        }
        
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
            cv::calcOpticalFlowPyrLK(_prevGray, _nextGray, _prev, _next, _nextStatus, _err, cv::Size(21, 21), 3, term, 0, 0.001);
            
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
        :_maxError(5.0), _tracker(0)
    {
    }

    SmallMotionTracker::~SmallMotionTracker()
    {
        if (_tracker)
            delete _tracker;
    }

    void SmallMotionTracker::setMaxError(double error)
    {
        _maxError = error;
    }

    void SmallMotionTracker::initializeFromReferenceFrame(const cv::Mat & bgr)
    {
        cv::cvtColor(bgr, _refGray, CV_BGR2GRAY);

        _features.resize(1);

        std::vector<cv::KeyPoint> keys;
        cv::FAST(_refGray, keys, 35, true);

        _features[0].clear();
        for (size_t i = 0; i < keys.size(); ++i) {
            _features[0].push_back(keys[i].pt);
        }
        
        //cv::TermCriteria termcrit(CV_TERMCRIT_ITER|CV_TERMCRIT_EPS, 20, 0.03);
        //cv::cornerSubPix(_refGray, _features[0], cv::Size(10,10), cv::Size(-1,-1), termcrit);
        
        _status.clear();
        _status.resize(_features[0].size(), 1);

        if (_tracker)
            delete _tracker;

        _tracker = new KLT(_refGray, _features[0], _status, _maxError);
    }

    SmallMotionTracker::CVFrameResult SmallMotionTracker::addFrame(const cv::Mat & bgr)
    {
        cv::Mat gray;
        cv::cvtColor(bgr, gray, CV_BGR2GRAY);

        _tracker->update(gray);
        _features.push_back(_tracker->location());      

        return std::make_pair(_tracker->location(), _tracker->status());
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

        std::vector<uchar> status = _tracker->status();
        std::cout << "before " << fcopy[0].size();
        for (size_t c = 0; c < fcopy.size(); ++c) {
            removeByStatus(fcopy[c], status);
        }
        std::cout << "after " << fcopy[0].size();

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