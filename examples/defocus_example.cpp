/**
    This file is part of Defocus.

    Copyright(C) 2016 Christoph Heindl
    All rights reserved.

    This software may be modified and distributed under the terms
    of the BSD license.See the LICENSE file for details.
*/

#include <defocus/features.h>
#include <defocus/camera.h>
#include <defocus/sparse.h>
#include <defocus/io.h>
#include <defocus/dense.h>

#include <opencv2/opencv.hpp>



int main(int argc, char **argv) {

    //google::InitGoogleLogging(argv[0]);

    if (argc != 2) {
        std::cerr << argv[0] << " videofile" << std::endl;
        return -1;
    }

    cv::VideoCapture vc;
    if (!vc.open(argv[1])) {
        std::cerr << "Failed to open video" << std::endl;
        return -1;
    }

    // Taken from http://yf.io/p/tiny/
    // Use 8point_defocus.exe "stream\stone6_still_%04d.png"
    
    Eigen::Matrix3d k;
    k <<
    1781.0, 0.0, 960.0,
    0.0, 1781.0, 540.0,
    0.0, 0.0, 1.0;


    // Detect trackable features in reference frame
    cv::Mat ref;
    vc >> ref;
    
    defocus::SmallMotionTracker tracker;
    tracker.setMaxError(5.0);
    tracker.initializeFromReferenceFrame(ref);
    
    cv::Mat f;
    while (vc.grab()) {
        vc.retrieve(f);

        defocus::SmallMotionTracker::CVFrameResult r = tracker.addFrame(f);
        
        for (size_t i = 0; i < r.second.size(); ++i) {
            if (r.second[i]) {
                cv::circle(f, r.first[i], 2, cv::Scalar(0, 255, 0));
            }
        }

        cv::Mat tmp;
        cv::resize(f, tmp, cv::Size(), 0.5, 0.5);
        cv::imshow("track", tmp);
        cv::waitKey(10);
    }

    Eigen::MatrixXd features = tracker.trackedFeaturesPerFrame();

    
    Eigen::Matrix<double, 6, Eigen::Dynamic> cameras;
    Eigen::VectorXd depths;

    cameras = defocus::InitialConditions::identityCameraParameters(features.rows() / 2);
    depths = defocus::InitialConditions::uniformRandomDepths(0.5, 2.0, features.cols());

    defocus::SparseSmallMotionBundleAdjustment ba;
    ba.setCameraMatrix(k);
    ba.setFeatures(features);
    ba.setInitialCameraParameters(cameras);
    ba.setInitialDepths(depths);
    ba.solve();

    cameras = ba.cameraParameters();
    depths = ba.depths();
    
    Eigen::Matrix3Xd points3d = defocus::PinholeCamera::reconstructPoints(features.topRows(2), depths, k);

    Eigen::MatrixXd colors(3, points3d.cols());
    cv::Mat depthmap(ref.size(), CV_64FC1);
    depthmap.setTo(0);

    for (Eigen::DenseIndex i = 0; i < points3d.cols(); ++i) {
    
        cv::Point2f f((float)features(0, i), (float)features(1, i));
        cv::Vec3b c = ref.at<cv::Vec3b>(f);
        colors(0, i) = c(2);
        colors(1, i) = c(1);
        colors(2, i) = c(0);
            
        depthmap.at<double>(f) = points3d.col(i).z();
    }
    defocus::writePointsAndColorsAsPLY("sparse.ply", points3d, colors);
    
    defocus::DenseDepthPropagation dense;
    dense.setCameraMatrix(k);
    dense.setColorImage(ref);
    dense.setImageResolution(ref.cols, ref.rows);
    dense.setMultiScaleLevels(3);
    dense.setSparsePoints(points3d);
    dense.solve();
    
    double minv, maxv;
    cv::minMaxLoc(dense.denseDepthMap(), &minv, &maxv);
    
    cv::Mat tmp;
    dense.denseDepthMap().convertTo(tmp, CV_8U, 255.0 / (maxv - minv), -minv * 255.0 / (maxv - minv));
    cv::resize(tmp, tmp, cv::Size(), 0.5, 0.5);
    cv::imshow("dense", tmp);
    cv::waitKey();

    
}
