//
//  gvfc.h
//  gvfc
//
//  Created by Di Yang on 14/03/12.
//  Copyright (c) 2012 The Australian National University. All rights reserved.
//

#ifndef snaketest2_snaketest2_h
#define snaketest2_snaketest2_h
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#endif

#define CV_MAX_ELEM 1
#define CV_MIN_ELEM 0
#define CV_PT_RM -100
#define CV_WITH_HUN 0
#define CV_WITHOUT_HUN 1
#define CV_REINITIAL 1
#define CV_NREINITIAL 0
#define CV_GVF 3
#define CV_IMG 2
#define CV_GRD 1

CV_EXPORTS_W float cvFindOpElem(const cv::Mat& srcarr, int flag);
CV_EXPORTS_W void cvGVF(const cv::Mat& srcarr, cv::Mat& dstarr_u, cv::Mat& dstarr_v, double mu, int ITER, int flag);

cv::Point*
cvSnakeImageGVF(const cv::Mat* src,
                cv::Point* points,
                int *length, 
                float alpha,
                float beta, 
                float gamma, 
                float kappa, 
                int ITER_ext, 
                int ITER_int,
                int calcInitial,
                int alg);

static cv::Point* cvSnakeInterp(cv::Point* points, int* _length, int dmin, int dmax, int flag);

