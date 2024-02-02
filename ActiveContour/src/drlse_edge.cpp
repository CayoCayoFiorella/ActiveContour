//
//  drlse_edge.cpp
//  DRLSE_P/Users/diyang/Desktop/Research_src/DRLSE/DRLSE/drlse_edge.cpp
//
//  Created by Di Yang on 1/03/12.
//  Copyright (c) 2012 The Australian National University. All rights reserved.
//

#include <iostream>
#include <math.h>
#include "../headers/drlse_edge.h"
#include "../headers/common.h"
#include "opencv2/core.hpp"

#define SMALLNUM 1e-10f
#define PI 3.1416f

CV_EXPORTS_W void cvDirac(const cv::Mat& src, cv::Mat& dst, double sigma)
{
    cv::Mat srcMat = src.clone();  // Clonar para asegurar que src sea modificable
    dst.create(srcMat.size(), CV_32FC1);

    if (srcMat.type() != CV_32FC1)
        CV_Error(cv::Error::StsUnsupportedFormat, "Only-32bit, 1-channel input images are supported");

    if (dst.type() != CV_32FC1)
        CV_Error(cv::Error::StsUnsupportedFormat, "Only-32bit, 1-channel output images are supported");

    if (srcMat.size() != dst.size())
        CV_Error(cv::Error::StsUnmatchedSizes, "The input images must have the same size");

    int iStep_src = srcMat.step / sizeof(float);
    float* fPtr_src = srcMat.ptr<float>();
    int iStep_dst = dst.step / sizeof(float);
    float* fPtr_dst = dst.ptr<float>();

    float temp1 = 0.0f, temp2 = 0.0f;
    float flag = 0.0f;

    for (int j = 0; j < srcMat.rows; j++) {
        for (int i = 0; i < srcMat.cols; i++) {
            temp1 = fPtr_src[i + iStep_src * j];
            temp2 = (1.0f / (2.0f * sigma)) * (1.0f + std::cos(PI * temp1 / sigma));
            
            if (static_cast<int>(temp1 * 10000) <= static_cast<int>(sigma * 10000) &&
                static_cast<int>(temp1 * 10000) >= static_cast<int>(-sigma * 10000)) {
                flag = 1.0f;
            } else {
                flag = 0.0f;
            }

            fPtr_dst[i + iStep_dst * j] = temp2 * flag;
        }
    }
}

CV_EXPORTS_W void cvCalS(const cv::Mat& src, cv::Mat& dst)
{
    cv::Mat srcMat = src.clone();  // Clonar para asegurar que src sea modificable
    dst.create(srcMat.size(), CV_32FC1);

    cv::Mat src_dx, src_dy;

    if (srcMat.type() != CV_32FC1)
        CV_Error(cv::Error::StsUnsupportedFormat, "Only-32bit, 1-channel input images are supported");

    if (dst.type() != CV_32FC1)
        CV_Error(cv::Error::StsUnsupportedFormat, "Only-32bit, 1-channel input images are supported");

    if (srcMat.size() != dst.size())
        CV_Error(cv::Error::StsUnmatchedSizes, "The input images must have the same size");

    src_dx = cv::Mat(srcMat.size(), CV_32FC1);
    src_dy = cv::Mat(srcMat.size(), CV_32FC1);
    src_dx.setTo(0);
    src_dy.setTo(0);

    int iStep = dst.step / sizeof(float);
    float* fPtr = dst.ptr<float>();

    cv::Sobel(srcMat, src_dx, CV_32F, 1, 0, 1);
    cv::Sobel(srcMat, src_dy, CV_32F, 0, 1, 1);
    src_dx = src_dx.mul(src_dx) * 0.0625f;  // rescale gradient
    src_dy = src_dy.mul(src_dy) * 0.0625f;  // rescale gradient
    cv::add(src_dx, src_dy, dst);

    for (int j = 0; j < srcMat.rows; j++) {
        for (int i = 0; i < srcMat.cols; i++) {
            fPtr[i + iStep * j] = std::sqrt(fPtr[i + iStep * j]) + 1e-10f;
        }
    }
}

CV_EXPORTS_W void cvCurvature(const cv::Mat& src_x, const cv::Mat& src_y, cv::Mat& dst)
{
    cv::Mat srcMat_x = src_x.clone();  // Clonar para asegurar que src_x sea modificable
    cv::Mat srcMat_y = src_y.clone();  // Clonar para asegurar que src_y sea modificable
    dst.create(srcMat_x.size(), CV_32FC1);

    if (srcMat_x.type() != CV_32FC1 || srcMat_y.type() != CV_32FC1 || dst.type() != CV_32FC1)
        CV_Error(cv::Error::StsUnsupportedFormat, "Only-32bit, 1-channel input images are supported");

    if (srcMat_x.size() != srcMat_y.size())
        CV_Error(cv::Error::StsUnmatchedSizes, "The input images must have the same size");

    cv::Mat Nxx, Nyy, ones;
    Nxx = cv::Mat(srcMat_x.size(), CV_32FC1);
    Nyy = cv::Mat(srcMat_x.size(), CV_32FC1);
    ones = cv::Mat(srcMat_x.size(), CV_32FC1);
    ones.setTo(1.0f);

    cv::Sobel(srcMat_x, Nxx, CV_32F, 1, 0, 1);
    cv::Sobel(srcMat_y, Nyy, CV_32F, 0, 1, 1);
    Nxx = Nxx.mul(ones) * 0.25f;
    Nyy = Nyy.mul(ones) * 0.25f;
    cv::add(Nxx, Nyy, dst);
}

CV_EXPORTS_W void cvDistReg(const cv::Mat& src, cv::Mat& dst)
{
    cv::Mat srcMat = src.clone();  // Clonar para asegurar que src sea modificable
    dst.create(srcMat.size(), CV_32FC1);

    if (srcMat.type() != CV_32FC1 || dst.type() != CV_32FC1)
        CV_Error(cv::Error::StsUnsupportedFormat, "Only-32bit, 1-channel input images are supported");

    if (srcMat.size() != dst.size())
        CV_Error(cv::Error::StsUnmatchedSizes, "The input images must have the same size");

    cv::Mat src_dx, src_dy, s, ps, dps_x, dps_y, del, ones;
    src_dx = cv::Mat(srcMat.size(), CV_32FC1);
    src_dy = cv::Mat(srcMat.size(), CV_32FC1);
    s = cv::Mat(srcMat.size(), CV_32FC1);
    ps = cv::Mat(srcMat.size(), CV_32FC1);
    dps_x = cv::Mat(srcMat.size(), CV_32FC1);
    dps_y = cv::Mat(srcMat.size(), CV_32FC1);
    del = cv::Mat(srcMat.size(), CV_32FC1);
    ones = cv::Mat(srcMat.size(), CV_32FC1);
    ones.setTo(1.0f);

    cv::Sobel(srcMat, src_dx, CV_32F, 1, 0, 1);
    cv::Sobel(srcMat, src_dy, CV_32F, 0, 1, 1);
    src_dx = src_dx.mul(ones) * 0.25f;
    src_dy = src_dy.mul(ones) * 0.25f;

    int iStep_s = s.step / sizeof(float);
    float* fPtr_s = s.ptr<float>();

    for (int j = 0; j < srcMat.rows; j++) {
        for (int i = 0; i < srcMat.cols; i++) {
            float temp_s = srcMat.at<float>(j, i);
            float flag_s1 = (temp_s >= 0 && temp_s <= 1.0f) ? 1.0f : 0.0f;
            float flag_s2 = (temp_s > 1.0f) ? 1.0f : 0.0f;

            fPtr_s[i + iStep_s * j] = flag_s1 * std::sin(2 * PI * temp_s) / 2 / PI + flag_s2 * (temp_s - 1.0f);
        }
    }

    cv::multiply(ps, src_dx, dps_x);
    cv::multiply(ps, src_dy, dps_y);
    cv::subtract(dps_x, src_dx, dps_x);
    cv::subtract(dps_y, src_dy, dps_y);
    cvCurvature(dps_x, dps_y, dst);

    cv::Laplacian(srcMat, del, CV_32F, 1);
    del = del.mul(ones) * 0.2f;
    cv::add(dst, del, dst);
}

CV_EXPORTS_W void cvDrlse_edge(cv::Mat& srcphi, 
                          cv::Mat& srcgrad,
                          cv::Mat& dstarr,
                          double lambda, 
                          double mu, 
                          double alfa, 
                          double epsilon, 
                          int timestep,
                          int iter)
{   
    //CV_FUNCNAME( "cvDrlse_edge" );

    cv::Mat phi = srcphi.clone(); // Clonar para asegurar que srcphi sea modificable
    cv::Mat grad = srcgrad.clone(); // Clonar para asegurar que srcgrad sea modificable
    cv::Mat dst = dstarr.clone(); // Clonar para asegurar que dstarr sea modificable

    if (phi.type() != CV_32FC1)
        CV_Error(cv::Error::StsUnsupportedFormat, "Only-32bit, 1-channel input images are supported");

    if (grad.type() != CV_32FC1)
        CV_Error(cv::Error::StsUnsupportedFormat, "Only-32bit, 1-channel input images are supported");

    if (phi.size() != grad.size())
        CV_Error(cv::Error::StsUnmatchedSizes, "The input images must have the same size");

    cv::Mat gradx, grady, phi_0, phix, phiy;
    cv::Mat Nx, Ny, s, curvature, distRegTerm;
    cv::Mat diracPhi, areaTerm, edgeTerm;
    cv::Mat temp1, temp2, temp3, ones;
    gradx = cv::Mat(phi.size(), CV_32FC1);
    grady = cv::Mat(phi.size(), CV_32FC1);
    phi_0 = cv::Mat(phi.size(), CV_32FC1);
    phix = cv::Mat(phi.size(), CV_32FC1);
    phiy = cv::Mat(phi.size(), CV_32FC1);
    Nx = cv::Mat(phi.size(), CV_32FC1);
    Ny = cv::Mat(phi.size(), CV_32FC1);
    s = cv::Mat(phi.size(), CV_32FC1);
    curvature = cv::Mat(phi.size(), CV_32FC1);
    distRegTerm = cv::Mat(phi.size(), CV_32FC1);
    diracPhi = cv::Mat(phi.size(), CV_32FC1);
    areaTerm = cv::Mat(phi.size(), CV_32FC1);
    edgeTerm = cv::Mat(phi.size(), CV_32FC1);
    temp1 = cv::Mat(phi.size(), CV_32FC1);
    temp2 = cv::Mat(phi.size(), CV_32FC1);
    temp3 = cv::Mat(phi.size(), CV_32FC1);
    ones = cv::Mat(phi.size(), CV_32FC1);
    ones.setTo(1.0f);

    cv::Sobel(grad, gradx, CV_32F, 1, 0, 1);
    cv::Sobel(grad, grady, CV_32F, 0, 1, 1);
    gradx = gradx.mul(ones) * 0.25f;
    grady = grady.mul(ones) * 0.25f;
    phi.copyTo(dst);
    
    for(int i = 0; i < iter; i++) {
        cvNeumannBoundCond(dst, dst);
        cv::Sobel(dst, phix, CV_32F, 1, 0, 1);
        cv::Sobel(dst, phiy, CV_32F, 0, 1, 1);
        cvCalS(dst, s);
        cv::divide(phix, s, Nx, 0.25f);
        cv::divide(phiy, s, Ny, 0.25f);
        cvCurvature(Nx, Ny, curvature);
        cvDistReg(dst, distRegTerm);
        cvDirac(dst, diracPhi, epsilon);        //Compute diracPhi;
        cv::multiply(diracPhi, grad, areaTerm);        //Compute areaTerm
        
        cv::multiply(gradx, Nx, gradx);                //------------------//
        cv::multiply(grady, Ny, grady);                // Computing        //
        cv::add(gradx, grady, temp1);             //                  //
        cv::multiply(diracPhi, temp1, temp2);          // edgeTerm         //
        cv::multiply(areaTerm, curvature, temp3);      //                  //
        cv::add(temp2, temp3, edgeTerm);          //------------------//
        
        cv::multiply(distRegTerm, ones, distRegTerm, mu);              //  distRegTerm = mu     * distRegTerm
        cv::multiply(edgeTerm,    ones, edgeTerm,    lambda);          //  edgeTerm    = lambda * edgeTerm
        cv::multiply(areaTerm,    ones, areaTerm,    alfa);            //  areaTerm    = alfa   * areaTerm
        cv::add(distRegTerm, edgeTerm, temp1);
        cv::add(temp1, areaTerm, temp2);                          //  (distRegTerm + edgeTerm + areaTerm)
        cv::multiply(temp2, ones, temp2, double(timestep));            //  timestep * (distRegTerm + edgeTerm + areaTerm)
        cv::add(dst, temp2, dst);                                 //  phi = phi + timestep * (distRegTerm + edgeTerm + areaTerm)
    }

} 

cv::Point* cvDRLSE(const cv::Mat& image,
                   const cv::Mat& mask,
                   int* length,
                   double lambda,
                   double alfa,
                   double epsilon,
                   int timestep,
                   int ITER_ext,
                   int ITER_int,
                   int flag)
{
    cv::Mat msk, img, marker, levelset, ones, Ix, Iy, phi, f, g;
    cv::Size size;
    int comp_count = 0, iStep;
    float* fPtr;
    cv::RNG rng; // Necesario para generar colores aleatorios para los contornos
    std::vector<std::vector<cv::Point>> contours;
    cv::Point pt = cv::Point(0, 0), *point = NULL;
    double mu = 0.2 / double(timestep);
    char c;

    msk = mask.clone(); // Necesario clonar la máscara para evitar cambiar la original
    img = image.clone(); // Necesario clonar la imagen para evitar cambiar la original
    cv::GaussianBlur(img, img, cv::Size(5, 5), 1.5);
    size = img.size();
    levelset = cv::Mat(size, CV_8U);
    ones = cv::Mat(size, CV_32F, cv::Scalar(1.0));
    Ix = cv::Mat(size, CV_32F);
    Iy = cv::Mat(size, CV_32F);
    phi = cv::Mat(size, CV_32F);
    f = cv::Mat(size, CV_32F);
    g = cv::Mat(size, CV_32F);
    marker = cv::Mat(size, CV_32S, cv::Scalar(0));

    cv::Sobel(img, Ix, CV_32F, 1, 0, 1);
    cv::Sobel(img, Iy, CV_32F, 0, 1, 1);
    cv::multiply(Ix, Ix, Ix, 0.25 * 0.25);
    cv::multiply(Iy, Iy, Iy, 0.25 * 0.25);
    cv::add(Ix, Iy, f);
    cv::add(f, ones, f);
    cv::divide(ones, f, g, 1.0);

    // Generar colores aleatorios para los contornos
    std::vector<cv::Scalar> colors;
    for (int i = 0; i < 100; i++) {
        colors.push_back(cv::Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255)));
    }

    cv::findContours(msk, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    marker.setTo(cv::Scalar(0));
    for (size_t i = 0; i < contours.size(); i++) {
        cv::drawContours(marker, contours, static_cast<int>(i), cv::Scalar(255), cv::FILLED, 8);
    }

    iStep = phi.step / sizeof(fPtr[0]);
    fPtr = phi.ptr<float>();

    for (int j = 0; j < size.height; j++)
        for (int i = 0; i < size.width; i++) {
            int idx = marker.at<int>(j, i);
            if (idx > 0)
                fPtr[i + iStep * j] = (flag == CV_LSE_SHR) ? -2.0 : 2.0;
            else
                fPtr[i + iStep * j] = (flag == CV_LSE_SHR) ? 2.0 : -2.0;
        }

    for (int i = 0; i < ITER_ext; i++) {
        cvDrlse_edge(phi, g, phi, lambda, mu, alfa, epsilon, timestep, ITER_int);
        // loadBar(i + 1, ITER_ext, 50); // Esto debe ser implementado si lo necesitas
    }
    cvDrlse_edge(phi, g, phi, lambda, mu, 0.0, epsilon, timestep, ITER_int);
    msk.setTo(cv::Scalar(0));
    if (flag == CV_LSE_SHR)
        cv::threshold(phi, msk, 0.0, 255, cv::THRESH_BINARY_INV);
    else
        cv::threshold(phi, msk, 0.0, 255, cv::THRESH_BINARY);

    contours.clear();
    cv::findContours(msk, contours, cv::RETR_CCOMP, cv::CHAIN_APPROX_SIMPLE);

    if (contours.empty()) return nullptr;
    *length = static_cast<int>(contours.size());
    point = new cv::Point[*length];

    for (int i = 0; i < *length; i++) {
        if (i < colors.size()) {
            cv::drawContours(img, contours, i, colors[i], 2, 8);
        }

        point[i] = contours[i][0];
    }

    cv::imshow("Contours", img);
    cv::waitKey(0);

    return point;
}
