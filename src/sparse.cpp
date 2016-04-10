/**
 This file is part of Defocus.
 
 Copyright(C) 2016 Christoph Heindl
 All rights reserved.
 
 This software may be modified and distributed under the terms
 of the BSD license.See the LICENSE file for details.
 */

#include <defocus/sparse.h>
#include <defocus/camera.h>
#include <random>

#ifdef _WIN32
#define GOOGLE_GLOG_DLL_DECL
#pragma warning(push)
#pragma warning(disable : 4251 4355)
#include <ceres/ceres.h>
#include <glog/logging.h>
#pragma warning(pop)
#else
#include <ceres/ceres.h>
#include <glog/logging.h>
#endif


namespace defocus {
    
    struct SmallMotionEuclideanReprojectionError {
        SmallMotionEuclideanReprojectionError(const double *no, const double *np)
        : _no(no), _np(np)
        {}
        
        template <typename T>
        bool operator()(
                        const T* const camera,
                        const T* const depth,
                        T* residuals) const
        {
            T p[3] = {T(_np[0]) * depth[0], T(_np[1]) * depth[0], depth[0] };
            
            T rpt[3] = {
                p[0] - camera[2] * p[1] + camera[1] * p[2] + camera[3],
                p[0] * camera[2] + p[1] - camera[0] * p[2] + camera[4],
                -p[0] * camera[1] + p[1] * camera[0] + p[2] + camera[5]
            };
            
            residuals[0] = rpt[0] / rpt[2] - T(_no[0]) ;
            residuals[1] = rpt[1] / rpt[2] - T(_no[1]) ;
            
            return true;
        }
        
        static ceres::CostFunction* Create(const double *obs, const double *point)
        {
            return (new ceres::AutoDiffCostFunction<SmallMotionEuclideanReprojectionError, 2, 6, 1>(new SmallMotionEuclideanReprojectionError(obs, point)));
        }
        
        const double *_no;
        const double *_np;
    };
    
    
    void SparseSmallMotionBundleAdjustment::setFeatures(const Eigen::MatrixXd & features)
    {
        _features = features;
    }

    void SparseSmallMotionBundleAdjustment::setInitialDepths(const Eigen::VectorXd & depths)
    {
        _depths = depths;
    }

    void SparseSmallMotionBundleAdjustment::setInitialCameraParameters(const Eigen::Matrix<double, 6, Eigen::Dynamic>& params)
    {
        _cameras = params;
    }

    void SparseSmallMotionBundleAdjustment::setCameraMatrix(const Eigen::Matrix3d & k)
    {
        _intr = k;
    }

    double SparseSmallMotionBundleAdjustment::solve()
    {
        Eigen::DenseIndex nFrames = _features.rows() / 2;
        Eigen::DenseIndex nObs = _features.cols();

        eigen_assert(nFrames > 2);
        eigen_assert(_cameras.cols() == nFrames);
        eigen_assert(_depths.cols() == nObs);
        
        Eigen::Matrix3d kinv = _intr.inverse();
        Eigen::Matrix3Xd retinaPoints(3, nFrames * nObs);

        for (Eigen::DenseIndex c = 0; c < nFrames; ++c) {
            for (Eigen::DenseIndex o = 0; o < nObs; ++o) {
                retinaPoints.col(c * nObs + o) = PinholeCamera::pixelToRetina(_features(c * 2 + 0, o), _features(c * 2 + 1, o), kinv);
            }
        }
        
        // Setup NLLS
        ceres::Problem problem;

        // For each camera (except the reference camera)
        for (Eigen::DenseIndex c = 1; c < nFrames; ++c) {

            // For each observation
            for (Eigen::DenseIndex o = 0; o < nObs; ++o) {

                // Add a residual block
                Eigen::DenseIndex idx = c * nObs + o;
                Eigen::DenseIndex idxInRef = o;

                ceres::CostFunction *cost_function =
                    SmallMotionEuclideanReprojectionError::Create(retinaPoints.col(idx).data(), retinaPoints.col(idxInRef).data());

                problem.AddResidualBlock(cost_function, NULL, _cameras.col(c).data(), _depths.data() + o);
            }
        }

        // Solve

        ceres::Solver::Options options;
        options.linear_solver_type = ceres::DENSE_SCHUR;
        options.max_num_iterations = 20;
        options.minimizer_progress_to_stdout = true;

        ceres::Solver::Summary summary;
        ceres::Solve(options, &problem, &summary);
        std::cout << summary.FullReport() << std::endl;

        return summary.final_cost;
    }

    const Eigen::VectorXd & SparseSmallMotionBundleAdjustment::depths() const
    {
        return _depths;
    }

    const Eigen::Matrix<double, 6, Eigen::Dynamic>& SparseSmallMotionBundleAdjustment::cameraParameters() const
    {
        return _cameras;
    }

    Eigen::VectorXd InitialConditions::uniformRandomDepths(double minDepth, double maxDepth, Eigen::DenseIndex nObservations)
    {
        Eigen::VectorXd depths(nObservations);

        std::random_device rd;
        std::default_random_engine gen(rd());
        std::uniform_real_distribution<double> dist(minDepth, maxDepth);

        for (Eigen::DenseIndex i = 0; i < nObservations; ++i)
            depths(i) = dist(gen);
        
        return depths;
    }

    Eigen::VectorXd InitialConditions::constantDepths(double depth, Eigen::DenseIndex nObservations)
    {
        return Eigen::VectorXd::Constant(nObservations, depth);
    }

    Eigen::Matrix<double, 6, Eigen::Dynamic> InitialConditions::identityCameraParameters(Eigen::DenseIndex nCameras)
    {
        Eigen::Matrix<double, 6, Eigen::Dynamic> cameraParameters(6, nCameras);
        cameraParameters.setZero();
        return cameraParameters;
    }

}