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


const int neighbors3x3[] = {
    0, -1, // N
    1, -1, // NE
    1,  0, // E
    1,  1, // SE
    0,  1, // S
   -1,  1, // SW
   -1,  0, // W
   -1, -1, // NW
};

void getNeighbors(int x, int y, int width, int height, const int *neighbors, int n, int *validneighbors, int *nvalid) {
    *nvalid = 0;
    for (int i = 0; i < n; ++i) {
        int nx = neighbors[i*2+0];
        int ny = neighbors[i*2+1];
        
        int tx = x + nx;
        int ty = y + ny;
        
        bool avail = (tx >= 0 && ty >= 0 && tx < width && ty < height);
        if (avail) {
            validneighbors[*nvalid*2+0] = tx;
            validneighbors[*nvalid*2+1] = ty;
            (*nvalid) += 1;
        }
    }
}

void computeColorWeights(int x, int y, const int *neighbors, int nn, const cv::Mat &colorsLab, double strength, double *weights) {
    cv::Vec3d labRef = colorsLab.at<cv::Vec3b>(y, x);

    for (int i = 0; i < nn; ++i) {
        int nx = neighbors[i*2+0];
        int ny = neighbors[i*2+1];
        
        cv::Vec3d labN = colorsLab.at<cv::Vec3b>(ny, nx);
        cv::Vec3d diff = labRef - labN;
        
        
        weights[i] = std::max<double>(std::exp(- std::sqrt(diff.dot(diff)) / strength), 0.001);
    }
}

inline int toIdx(int x, int y, int cols) {
    return y * cols + x;
}

void dense(cv::Mat &depths, cv::Mat &colors) {

    typedef Eigen::Triplet<double> T;
    std::vector<T> triplets;

    int rows = depths.rows;
    int cols = depths.cols;

    Eigen::MatrixXd rhs(rows * cols, 1);
    rhs.setZero();

    const double colorSimilarityStrength = 10;
    
    int neighbors[8*2];
    double colorWeights[8];
    

    int idx = 0;
    for (int y = 0; y < depths.rows; ++y) {
        for (int x = 0; x < depths.cols; ++x, ++idx) {

            double d = depths.at<double>(y, x);

            if (d > 0.0) {
                triplets.push_back(T(idx, idx, 1.0));
                rhs(idx, 0) = d;
            } else {
                int nn = 0;
                getNeighbors(x, y, depths.cols, depths.rows, neighbors3x3, 8, neighbors, &nn);
                computeColorWeights(x, y, neighbors, nn, colors, colorSimilarityStrength, colorWeights);
                
                double sumweights = 0.0;
                for (int i = 0; i < nn; ++i) {
                    sumweights += colorWeights[i];
                }

                triplets.push_back(T(idx, idx, 1.0));
                
                for (int i = 0; i < nn; ++i) {
                    int nidx = toIdx(neighbors[i*2+0], neighbors[i*2+1], depths.cols);
                    
                    if (y == 0 && x == 0)
                        std::cout << colorWeights[i] << std::endl;
                    

                    triplets.push_back(T(idx, nidx, - (1.0 / sumweights) * colorWeights[i]));
                }
            }
            
        }
    }

    Eigen::SparseMatrix<double> A(rows*cols, rows*cols);
    A.setFromTriplets(triplets.begin(), triplets.end());

    Eigen::SparseLU< Eigen::SparseMatrix<double> > solver;
    solver.compute(A);

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
}
