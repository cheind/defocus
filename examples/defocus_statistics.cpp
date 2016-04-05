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

typedef std::vector<cv::Point2f> OpenCVFeatures;
typedef std::vector< OpenCVFeatures > OpenCVFeaturesPerFrame;

Eigen::MatrixXd computeFeatureDistanceMatrix(const OpenCVFeaturesPerFrame &f)
{
    Eigen::MatrixXd d(f[0].size(), f.size());

    // For each feature
    for (size_t i = 0; i < f[0].size(); ++i) {
        cv::Vec2f ref = f[0][i];

        // In each frame
        for (size_t j = 0; j < f.size(); ++j) {
            cv::Vec2f t = f[j][i];
            d(i, j) = sqrt((ref - t).dot((ref - t)));
        }
    }

    return d;
}

void writeFeatureDistanceMatrixToFile(const char *path, const Eigen::MatrixXd &m)
{
    std::ofstream ofs(path);
    ofs << m.format(Eigen::IOFormat(Eigen::StreamPrecision, Eigen::DontAlignCols, " ", "\n"));
}

cv::Mat generateFeatureMaxMovementImage(cv::Mat &ref, const OpenCVFeatures &f, const Eigen::MatrixXd &m)
{
    cv::Mat tmp = ref.clone();

    Eigen::VectorXd maxD = m.rowwise().mean();

    for (Eigen::DenseIndex i = 0; i < maxD.rows(); ++i) {
        std::cout << (int)maxD(i) << std::endl;
        cv::circle(tmp, f[i], (int)maxD(i), cv::Scalar(0, 255, 0));
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
    
    Eigen::Matrix3d invk = k.inverse();

    // Detect trackable features in reference frame
    cv::Mat ref, refF, refGray, refLab, gray;
    vc >> ref;
    
    cv::cvtColor(ref, refGray, CV_BGR2GRAY);
    OpenCVFeaturesPerFrame featuresPerFrame;
    
    std::vector<cv::Point2f> corners;
    defocus::findFeatureInImage(refGray, corners);
    std::vector<uchar> status(corners.size(), 1);
    featuresPerFrame.push_back(corners);

    cv::Mat f;
    while (vc.grab()) {
        vc.retrieve(f);
        cv::cvtColor(f, gray, CV_BGR2GRAY);

        std::vector<cv::Point2f> loc;

        defocus::trackFeatures(refGray, corners, gray, loc, status, 5);
        featuresPerFrame.push_back(loc);
        
        for (size_t i = 0; i < status.size(); ++i) {
            if (status[i]) {
                cv::circle(f, loc[i], 2, cv::Scalar(0, 255, 0));
            }
        }
        
        cv::Mat tmp;
        cv::resize(f, tmp, cv::Size(), 0.5, 0.5);
        cv::imshow("track", tmp);
        cv::waitKey(10);
    }

    for (Eigen::DenseIndex c = 0; c < featuresPerFrame.size(); ++c) {
        defocus::removeByStatus(featuresPerFrame[c], status);
    }    

    Eigen::MatrixXd dists = computeFeatureDistanceMatrix(featuresPerFrame);

    writeFeatureDistanceMatrixToFile("featuredists.csv", dists);
    cv::Mat featureMovementImage = generateFeatureMaxMovementImage(ref, featuresPerFrame[0], dists);
    cv::imwrite("featuremovement.png", featureMovementImage);

}
