/**
 This file is part of Defocus.
 
 Copyright(C) 2016 Christoph Heindl
 All rights reserved.
 
 This software may be modified and distributed under the terms
 of the BSD license.See the LICENSE file for details.
 */

#include <defocus/dense.h>
#include <defocus/camera.h>
#include <opencv2/opencv.hpp>

#include <Eigen/Sparse>
#include <Eigen/Geometry>


namespace defocus {
    
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
    
    void computeColorWeights(int x, int y, const int *neighbors, int nn, const cv::Mat &colors, double strength, double *weights) {
        cv::Vec3d ref = colors.at<cv::Vec3b>(y, x);
        
        for (int i = 0; i < nn; ++i) {
            int nx = neighbors[i*2+0];
            int ny = neighbors[i*2+1];
            
            cv::Vec3d cur = colors.at<cv::Vec3b>(ny, nx);
            cv::Vec3d diff = ref - cur;
            
            
            weights[i] = std::max<double>(std::exp(- std::sqrt(diff.dot(diff)) / strength), 0.001);
        }
    }
    
    inline int toIdx(int x, int y, int cols) {
        return y * cols + x;
    }

    
    DenseDepthPropagation::DenseDepthPropagation()
    :_width(0), _height(0), _levels(1)
    {}
    
    void DenseDepthPropagation::setCameraMatrix(const Eigen::Matrix3d &k) {
        _intr = k;
    }
    
    void DenseDepthPropagation::setImageResolution(int width, int height) {
        _width = width;
        _height = height;
    }
    
    void DenseDepthPropagation::setMultiScaleLevels(int levels) {
        _levels = levels;
    }
   
    
    void DenseDepthPropagation::setSparsePoints(const Eigen::Matrix3Xd &points) {
        _points = points;
    }
    
    void DenseDepthPropagation::setColorImage(const cv::Mat &image) {
        _colors = image.clone();
    }
    
    cv::Mat DenseDepthPropagation::denseDepthMap() const {
        return _depths;
    }
    
    void DenseDepthPropagation::solve()
    {
        eigen_assert(_width > 0);
        eigen_assert(_height > 0);
        eigen_assert(_levels > 0);
        
        // Build pyramid
        
        Eigen::VectorXd solution;
        Eigen::VectorXd prevSolution;
        for (int l = _levels-1; l >= 0; --l) {
            std::cout << "level " << l << std::endl;
            int scale = scaleFactorForLevel(l);
            
            solution = solveForLevel(l, prevSolution);
        
            
            // Pyr up solution for next level guess
            cv::Mat_<double> curDepths = solutionVectorToImage(_width / scale, _height / scale, solution);
            cv::Mat_<double> nextDepths;
            cv::resize(curDepths, nextDepths, cv::Size(), 2.0, 2.0, CV_INTER_NN);
            prevSolution = solutionImageToVector(nextDepths);
            
        }
        
        _depths = solutionVectorToImage(_width, _height, solution);
        
    }
    
    int DenseDepthPropagation::scaleFactorForLevel(int level) const
    {
        return static_cast<int>(std::pow(2.0, level));
    }
    
    cv::Mat_<double> DenseDepthPropagation::solutionVectorToImage(int width, int height, const Eigen::VectorXd &v) const {
        cv::Mat_<double> img(height, width);
        
        int idx = 0;
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x, ++idx) {
                img.at<double>(y, x) = v(idx);
            }
        }
        
        return img;
    }
    
    
    Eigen::VectorXd DenseDepthPropagation::solutionImageToVector(const cv::Mat_<double> &image) const
    {
        Eigen::VectorXd v(image.rows * image.cols);
        
        int idx = 0;
        for (int y = 0; y < image.rows; ++y) {
            for (int x = 0; x < image.cols; ++x, ++idx) {
                v(idx) = image.at<double>(y, x);
            }
        }
        
        return v;
    }
    
    Eigen::VectorXd DenseDepthPropagation::solveForLevel(int level, const Eigen::VectorXd guess) const
    {
        int scale = scaleFactorForLevel(level);
        
        cv::Mat_<double> leveldepths = sparseDepthMapForScaleFactor(scale);
        cv::Mat levelcolors;
        cv::resize(_colors, levelcolors, cv::Size(), 1.0 / scale, 1.0 / scale, CV_INTER_AREA);
        
        typedef Eigen::Triplet<double> T;
        std::vector<T> triplets;
        
        int rows = leveldepths.rows;
        int cols = leveldepths.cols;
        
        Eigen::VectorXd rhs(rows * cols, 1);
        rhs.setZero();
        
        const double colorSimilarityStrength = 5;
        
        int neighbors[8*2];
        Eigen::Matrix<double, 8, 1> colorWeights;
        
        int idx = 0;
        for (int y = 0; y < rows; ++y) {
            for (int x = 0; x < cols; ++x, ++idx) {
                
                double d = leveldepths(y, x);
                
                if (d > 0.0) {
                    triplets.push_back(T(idx, idx, 1.0));
                    rhs(idx, 0) = d;
                } else {
                    int nn = 0;
                    getNeighbors(x, y, cols, rows, neighbors3x3, 8, neighbors, &nn);
                    computeColorWeights(x, y, neighbors, nn, levelcolors, colorSimilarityStrength, colorWeights.data());
                    
                    double sumweights = colorWeights.head(nn).sum();
                    
                    triplets.push_back(T(idx, idx, 1.0));
                    
                    for (int i = 0; i < nn; ++i) {
                        int nidx = toIdx(neighbors[i*2+0], neighbors[i*2+1], cols);
                        triplets.push_back(T(idx, nidx, - (1.0 / sumweights) * colorWeights[i]));
                    }
                }
                
            }
        }
        
        Eigen::SparseMatrix<double> A(rows*cols, rows*cols);
        A.setFromTriplets(triplets.begin(), triplets.end());
        
        Eigen::VectorXd result(rows * cols);
        
        if (level == (_levels - 1)) {
            // Solve direct
            Eigen::SparseLU< Eigen::SparseMatrix<double> > solver;
            solver.compute(A);
            result = solver.solve(rhs);
        } else {
            Eigen::BiCGSTAB< Eigen::SparseMatrix<double>, Eigen::IncompleteLUT<double>> solver;
            solver.preconditioner().setDroptol(.001);
            solver.compute(A);
            solver.setMaxIterations(10);
            result = solver.solveWithGuess(rhs, guess);
        }
        
        return result;
        
    }
    
    cv::Mat_<double> DenseDepthPropagation::sparseDepthMapForScaleFactor(int scale) const {
        
        
        Eigen::Matrix3d k = _intr;
        k.topRows(2).array() /= scale;

        Eigen::Matrix2Xd features = PinholeCamera::perspectiveProject(_points, Eigen::Matrix<double,3,4>::Identity(), k).colwise().hnormalized();
        
        cv::Mat_<double> depths(_height / scale, _width / scale);
        depths.setTo(0);
        
        for (Eigen::DenseIndex i = 0; i < features.cols(); ++i) {
            cv::Point pixel(cvRound(features(0,i)), cvRound(features(1,i)));
            depths(pixel) = _points.col(i).z();
        }
        
        return depths;
    }
    
    
}