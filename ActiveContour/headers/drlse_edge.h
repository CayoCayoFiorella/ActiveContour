//
//  drlse_edge.h
//  DRLSE
//
//  Created by Di Yang on 13/02/12.
//  Copyright (c) 2012 The Australian National University. All rights reserved.
//

#ifndef DRLSE_drlse_edge_h
#define DRLSE_drlse_edge_h
#include <string.h>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <fstream> 
#endif

#define CV_LSE_SHR 1
#define CV_LSE_EXP 0

CV_EXPORTS_W void cvDirac(const cv::Mat& src, cv::Mat& dst, double sigma);

CV_EXPORTS_W void cvCalS(const cv::Mat& src, cv::Mat& dst);

CV_EXPORTS_W void cvCurvature(const cv::Mat& src_x, const cv::Mat& src_y, cv::Mat& dst);

CV_EXPORTS_W void cvDistReg(const cv::Mat& src, cv::Mat& dst);

CV_EXPORTS_W void cvDrlse_edge(cv::Mat& srcphi, 
                          cv::Mat& srcgrad,
                          cv::Mat& dstarr,
                          double lambda, 
                          double mu, 
                          double alfa, 
                          double epsilon, 
                          int timestep,
                          int iter);

cv::Point*
cvDRLSE(const cv::Mat& image,
        const cv::Mat& mask,
        int *length,
        double lambda,
        double alfa,
        double epsilon,
        int timestep,
        int ITER_ext,
        int ITER_int,
        int flag);

