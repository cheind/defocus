/**
 This file is part of Defocus.
 
 Copyright(C) 2016 Christoph Heindl
 All rights reserved.
 
 This software may be modified and distributed under the terms
 of the BSD license.See the LICENSE file for details.
 */

#include <defocus/sfm.h>
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
    
    struct ReprojectionError {
        ReprojectionError(const double *no, const double *np)
        : _no(no), _np(np)
        {}
        
        template <typename T>
        bool operator()(
                        const T* const camera,
                        const T* const depth,
                        T* residuals) const
        {
            T p[3] = {T(_np[0]) / depth[0], T(_np[1]) / depth[0], T(1.0) / depth[0] };
            
            T rpt[3] = {
                p[0] - camera[2] * p[1] + camera[1] * p[2] + camera[3],
                p[0] * camera[2] + p[1] - camera[0] * p[2] + camera[4],
                -p[0] * camera[1] + p[1] * camera[0] + p[2] + camera[5]
            };
            
            residuals[0] = T(_no[0]) - rpt[0] / rpt[2] ;
            residuals[1] = T(_no[1]) - rpt[1] / rpt[2] ;
            
            return true;
        }
        
        static ceres::CostFunction* Create(const double *obs, const double *point)
        {
            return (new ceres::AutoDiffCostFunction<ReprojectionError, 2, 6, 1>(new ReprojectionError(obs, point)));
        }
        
        const double *_no;
        const double *_np;
    };

    
    SmallMotionBundleAdjustment::SmallMotionBundleAdjustment(Eigen::DenseIndex nCameras, Eigen::DenseIndex nObservationsPerCamera)
    {
        _points.resize(3, nCameras * nObservationsPerCamera);
        _cameras.resize(6, nCameras);
        _idepths.resize(nObservationsPerCamera);
        
        _nCameras = nCameras;
        _nObservations = nObservationsPerCamera;
        _refCamera = 0;
    }
    
    void SmallMotionBundleAdjustment::setObservation(Eigen::DenseIndex cameraIdx, Eigen::DenseIndex observationIdx, const Eigen::Vector3d &point)
    {
        Eigen::DenseIndex col = cameraIdx * _nObservations + observationIdx;
        _points.col(col) = point;
    }
    
    double SmallMotionBundleAdjustment::run(bool debug, Eigen::DenseIndex refCameraIdx) {
        _refCamera = refCameraIdx;
        
        // Setup cameras with identical poses
        _cameras.setZero();
        
        // Setup initial inverse depths
        std::default_random_engine gen;
        std::uniform_real_distribution<double> dist(0.1, 1.5);
        std::generate(_idepths.begin(), _idepths.end(), [&] () { return dist(gen); });
        
        // Setup NLLS
        
        ceres::Problem problem;
        
        // For each camera (except the reference camera)
        for (Eigen::DenseIndex c = 0; c < _nCameras; ++c) {
            if (c == _refCamera)
                continue;
            
            // For each observation
            for (Eigen::DenseIndex o = 0; o < _nObservations; ++o) {
                
                // Add a residual block
                Eigen::DenseIndex idx = c * _nObservations + o;
                Eigen::DenseIndex idxInRef = _refCamera * _nObservations + o;
                
                ceres::CostFunction *cost_function = ReprojectionError::Create(_points.col(idx).data(), _points.col(idxInRef).data());
                ceres::LossFunction *loss_function = new ceres::HuberLoss(1.0);
                
                problem.AddResidualBlock(cost_function,
                                         loss_function,
                                         _cameras.col(c).data(),
                                         &_idepths.at(o));
                
            }
        }
        
        // Solve
        
        ceres::Solver::Options options;
        options.use_nonmonotonic_steps = true;
        options.preconditioner_type = ceres::SCHUR_JACOBI;
        options.linear_solver_type = ceres::ITERATIVE_SCHUR;
        options.use_inner_iterations = true;
        options.max_num_iterations = 100;
        options.minimizer_progress_to_stdout = debug;
        
        ceres::Solver::Summary summary;
        ceres::Solve(options, &problem, &summary);
        if (debug) {
            std::cout << summary.FullReport() << std::endl;
        }
        
        // Make sure the reference camera is at identity.
        _cameras.col(_refCamera).setZero();
        
        return summary.final_cost;
        
    }
    
    Eigen::Matrix<double, 3, Eigen::Dynamic> SmallMotionBundleAdjustment::pointsInReferenceCamera() const {
        Eigen::Matrix<double, 3, Eigen::Dynamic> points(3, _nObservations);
        
        for (Eigen::DenseIndex i = 0; i < _nObservations; ++i) {
            Eigen::DenseIndex idx = _refCamera * _nObservations + i;
            points.col(i) = _points.col(idx) * (1.0 / _idepths[i]);
        }
        
        return points;
        
    }
    
    
    
}