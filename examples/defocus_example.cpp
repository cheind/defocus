/**
    This file is part of Defocus.

    Copyright(C) 2016 Christoph Heindl
    All rights reserved.

    This software may be modified and distributed under the terms
    of the BSD license.See the LICENSE file for details.
*/

#include <defocus/features.h>
#include <defocus/camera.h>
#include <defocus/sfm.h>
#include <random>
#include <iostream>

#include <opencv2/opencv.hpp>
#include <fstream>
#include <random>

#include <Eigen/Sparse>
#include <Eigen/Dense>



void writePly(const char *path, const Eigen::MatrixXd &points, const Eigen::MatrixXd &colors) {
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

void dense(cv::Mat &depths, cv::Mat &colors) {
    // Variational with energy: (dout - dsparse)^2 + lambda * |nabla(dout)|^2
    // First term is only defined for sparse pixel positions.
    // leads to linear system of equations: Ax = b with one row per pixel: 
    //  dout(x,y) + lambda * laplacian(dout(x,y)) = dsparse(x,y)
    //  dout(x,y) + lambda * (-4*dout(x,y) + dout(x,y-1) + dout(x+1,y) + dout(x,y+1) + dout(x-1,y)) = dsparse(x,y)


    typedef Eigen::Triplet<double> T;
    std::vector<T> triplets;

    std::cout << __LINE__ << std::endl;

    int rows = depths.rows;
    int cols = depths.cols;

    Eigen::MatrixXd rhs(rows * cols, 1);
    rhs.setZero();

    std::cout << __LINE__ << std::endl;

    const double lambda = 0.2;

    int idx = 0;
    for (int y = 0; y < depths.rows; ++y) {
        for (int x = 0; x < depths.cols; ++x, ++idx) {

            double d = depths.at<double>(y, x);
            
            double c = 0.0;

            if (d > 0.0) {
                rhs(idx, 0) = d;
                c += 1.0;
            }
            

            if (y > 0) {
                // North neighbor 
                c -= lambda;
                triplets.push_back(T(idx, (y - 1)*cols + x, lambda));
            }

            if (x > 0) {
                // West neighbor 
                c -= lambda;
                triplets.push_back(T(idx, (y)*cols + (x-1), lambda));
            }

            if (y < (rows - 1)) {
                // South neighbor 
                c -= lambda;
                triplets.push_back(T(idx, (y+1)*cols + x, lambda));
            }

            if (x < (cols - 1)) {
                // East neighbor 
                c -= lambda;
                triplets.push_back(T(idx, (y)*cols + (x+1), lambda));
            }

            // Center
            triplets.push_back(T(idx, idx, c));
        }
    }

    std::cout << __LINE__ << std::endl;

    Eigen::SparseMatrix<double> A(rows*cols, rows*cols);
    A.setFromTriplets(triplets.begin(), triplets.end());

    std::cout << __LINE__ << std::endl;

    Eigen::SparseLU< Eigen::SparseMatrix<double> > solver;
    solver.analyzePattern(A);
    solver.factorize(A);

    Eigen::MatrixXd result(rows*cols, 1);
    result = solver.solve(rhs);

    idx = 0;
    for (int y = 0; y < depths.rows; ++y) {
        for (int x = 0; x < depths.cols; ++x, ++idx) {
            depths.at<double>(y, x) = result(idx, 0);
        }
    }

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
    cv::Mat ref, refGray, gray;
    vc >> ref;
    cv::cvtColor(ref, refGray, CV_BGR2GRAY);
    
    typedef std::vector<cv::Point2f> OpenCVFeatures;

    
    std::vector< OpenCVFeatures > featuresPerFrame;
    
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
    
    const Eigen::DenseIndex nFrames = featuresPerFrame.size();
    const Eigen::DenseIndex nObs = featuresPerFrame[0].size();
    
    Eigen::Matrix<double, 3, Eigen::Dynamic> retinaPoints(3, nFrames * nObs);
    
    for (Eigen::DenseIndex c = 0; c < nFrames; ++c) {
    
        const OpenCVFeatures &f = featuresPerFrame[c];
    
        for (Eigen::DenseIndex o = 0; o < nObs; ++o) {
            retinaPoints.col(c * nObs + o) = defocus::pixelToRetina(f[o].x, f[o].y, invk);
        }
    }
    
    Eigen::Matrix<double, 6, Eigen::Dynamic> cameras;
    Eigen::Matrix<double, 3, Eigen::Dynamic> points3d;
    double err = defocus::solveSmallMotionBundleAdjustment(retinaPoints, cameras, points3d, nFrames, nObs);
    
    
    Eigen::MatrixXd colors(3, points3d.cols());
    cv::Mat depths(ref.size(), CV_64FC1);
    depths.setTo(0);

    for (Eigen::DenseIndex i = 0; i < nObs; ++i) {
    
        cv::Point2f f = featuresPerFrame[0][i];
        cv::Vec3b c = ref.at<cv::Vec3b>(f);
        colors(0, i) = c(2);
        colors(1, i) = c(1);
        colors(2, i) = c(0);
            
        depths.at<double>(f) = points3d.col(i).z();
    }
    writePly("points.ply", points3d, colors);

    dense(depths, ref);    

    double minv, maxv;
    cv::minMaxLoc(depths, &minv, &maxv);

    cv::Mat tmp;
    depths.convertTo(tmp, CV_8U, 255.0 / (maxv - minv), -minv * 255.0 / (maxv - minv));
    cv::resize(tmp, tmp, cv::Size(), 0.5, 0.5);
    cv::imshow("dense", tmp);
    cv::waitKey();

}
