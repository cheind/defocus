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
#include <random>
#include <iostream>

#include <opencv2/opencv.hpp>
#include <fstream>
#include <random>

#include <Eigen/Sparse>
#include <Eigen/Dense>


Eigen::MatrixXd computeFeatureDistanceMatrix(const Eigen::MatrixXd &f)
{
    Eigen::MatrixXd d(f.cols(), f.rows() / 2);

    // For each feature
    for (Eigen::DenseIndex i = 0; i < f.cols(); ++i) {
        // In each frame
        for (Eigen::DenseIndex j = 0; j < f.rows() / 2; ++j) {
            d(i, j) = (f.block<2, 1>(0, i) - f.block<2, 1>(j * 2, i)).norm();
        }
    }

    return d;
}

void writeFeatureDistanceMatrixToFile(const char *path, const Eigen::MatrixXd &m)
{
    std::ofstream ofs(path);
    ofs << m.format(Eigen::IOFormat(Eigen::StreamPrecision, Eigen::DontAlignCols, " ", "\n"));
}

cv::Mat generateFeatureMaxMovementImage(cv::Mat &ref, const Eigen::MatrixXd &f, const Eigen::MatrixXd &m)
{
    cv::Mat tmp = ref.clone();

    Eigen::VectorXd maxD = m.rowwise().mean();

    for (Eigen::DenseIndex i = 0; i < maxD.rows(); ++i) {
        cv::Point2d p(f(0, i), f(1, i));
        cv::circle(tmp, p, (int)maxD(i), cv::Scalar(0, 255, 0));
    }

    return tmp;
}


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
    tracker.setMaxError(10.0);
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
    Eigen::MatrixXd dists = computeFeatureDistanceMatrix(features);

    writeFeatureDistanceMatrixToFile("featuredists.csv", dists);
    cv::Mat featureMovementImage = generateFeatureMaxMovementImage(ref, features.topRows(2), dists);
    cv::imwrite("featuremovement.png", featureMovementImage);

}
