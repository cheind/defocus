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
    
    /*
    Eigen::Matrix3d k;
    k <<
        1781.0, 0.0, 960.0,
        0.0, 1781.0, 540.0,
        0.0, 0.0, 1.0;

     */
    Eigen::Matrix3d k;
    k <<
    1781.0 / 2, 0.0, 960.0 / 2,
    0.0, 1781.0 / 2, 540.0 / 2,
    0.0, 0.0, 1.0;
    
    Eigen::Matrix3d invk = k.inverse();


    // Detect trackable features in reference frame
    cv::Mat ref, refF, refGray, refLab, gray;
    vc >> ref;
    
    cv::resize(ref, ref, cv::Size(), 0.5, 0.5);
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
        cv::resize(f, f, cv::Size(), 0.5, 0.5);
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
