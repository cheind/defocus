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


class LensBlur {
public:
    LensBlur(cv::Mat colors, cv::Mat depths)
    :_colors(colors), _depths(depths)
    {}
    
    
    cv::Mat refocus(int selX, int selY) {
        
        cv::Mat_<cv::Vec3d> img(_colors.size());
        img.setTo(cv::Scalar(0,0,0));
        
        std::cout << _colors.size() << std::endl;
        std::cout << _depths.size() << std::endl;
        
        const double dfocused = _depths(selY, selX);
        
        for (int y = 0; y < img.rows; ++y) {
            for (int x = 0; x < img.cols; ++x) {
                
                const double d = _depths(y, x);
                double coc = defocus::ThinLensCamera::circleOfConfusion(1, 200.0, dfocused, d);
                
                
                coc = std::min<double>(20.0, std::max<double>(coc, 0.0));
                
                distributeUniformly(y, x, coc, _colors(y,x), img);
                
            }
        }
        
        std::cout << "done" << std::endl;
        
        cv::Mat tmp;
        img.convertTo(tmp, CV_8U);
        
        return tmp;
    }
private:
    
    void distributeUniformly(int yCenter, int xCenter, double coc, const cv::Vec3d &color, cv::Mat_<cv::Vec3d> &dst) {
        
        double r = coc * 0.5;
        if (fabs(r) < 0.5) {
            dst(yCenter, xCenter) = color;
            return;
        }
        
       
        
        double r2 = r * r;
        double a = M_PI * r * r;
        cv::Vec3d ud = color / a;
        
        const int miny = (int)std::floor(std::max<double>(yCenter - r, 0));
        const int maxy = (int)std::ceil(std::min<double>(yCenter + r, dst.rows - 1));
        
        const int minx = (int)std::floor(std::max<double>(xCenter - r, 0));
        const int maxx = (int)std::ceil(std::min<double>(xCenter + r, dst.cols - 1));
        
        for (int y = miny; y <= maxy; ++y) {
            cv::Vec3d *row = dst.ptr<cv::Vec3d>(y);
            for (int x = minx; x <= maxx; ++x) {
                if (((y-yCenter)*(y-yCenter) + (x-xCenter)*(x-xCenter)) < r2) {
                    row[x] += ud;
                }
            }
        }
    }
    
    cv::Mat_<cv::Vec3b> _colors;
    cv::Mat_<double> _depths;
};

void onMouse(int event,int x,int y, int flags,void* param)
{
    if (event == cv::EVENT_LBUTTONDOWN) {
        cv::Mat refocused = static_cast<LensBlur*>(param)->refocus(x, y);
        cv::imshow("Lens Blur", refocused);
    }
}


int main(int argc, char **argv) {

    if (argc != 3) {
        std::cerr << argv[0] << " depths.png colors.png" << std::endl;
        return -1;
    }
    
    cv::Mat depths = cv::imread(argv[1], CV_LOAD_IMAGE_ANYDEPTH);
    cv::Mat colors = cv::imread(argv[2], CV_LOAD_IMAGE_COLOR);
    
    cv::resize(colors, colors, cv::Size(), 0.5, 0.5);
    
    cv::Mat depthsD;
    depths.convertTo(depthsD, CV_64F);
    
    LensBlur lensBlur(colors, depthsD);
    
    cv::namedWindow("Lens Blur");
    cv::setMouseCallback("Lens Blur", onMouse, &lensBlur);
    cv::imshow("Lens Blur", colors);
    cv::waitKey();
    
}
