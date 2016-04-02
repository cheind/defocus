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


namespace defocus {
    
    void findFeatureInImage(cv::Mat &gray, std::vector<cv::Point2f> &corners) {
        double qualityLevel = 0.01;
        double minDistance = 5;
        int blockSize = 5;
        bool useHarrisDetector = false;
        double k = 0.04;
        int maxCorners = 4000;
        
        cv::goodFeaturesToTrack(gray,
                                corners,
                                maxCorners,
                                qualityLevel,
                                minDistance,
                                cv::Mat(),
                                blockSize,
                                useHarrisDetector,
                                k);
        
        cv::TermCriteria termcrit(CV_TERMCRIT_ITER|CV_TERMCRIT_EPS, 20, 0.03);
        cv::cornerSubPix(gray, corners, cv::Size(10,10), cv::Size(-1,-1), termcrit);
    }
    
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
            cv::calcOpticalFlowPyrLK(_prevGray, _nextGray, _prev, _next, _nextStatus, _err, cv::Size(11, 11), 5, term);
            
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

    
    void trackFeatures(const cv::Mat &refGray,
                       const std::vector<cv::Point2f> &refLocs,
                       const cv::Mat &targetGray,
                       std::vector<cv::Point2f> &targetLocs,
                       std::vector<uchar> &status,
                       double maxError)
    {
        KLT klt(refGray, refLocs, status, maxError);
        klt.update(targetGray);
        targetLocs = klt.location();
        status = klt.status();
    }
    
    std::vector<cv::Point2f> eliminateInvalidFeatures(const std::vector<cv::Point2f> &locs, const std::vector<uchar> &status)
    {
        std::vector<cv::Point2f> r;
        for (size_t i = 0; i < locs.size(); ++i) {
            if (status[i])
                r.push_back(locs[i]);
        }
        return r;
    }

    
    
    
}