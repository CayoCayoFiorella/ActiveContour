// Include necessary headers without changes
#include <iostream>
#include "../headers/gvfc.h"
#include "../headers/common.h"

// Include OpenCV 4 headers
#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"

static cv::Point*
cvSnakeInterp2(cv::Point* points,
               int* _length,
               int dmax,
               int* _dist,
               int flag)
{
    int i, distance=0, cont=0;
    int length = *_length;
    int length_out = 2*length;
    int Max_d = 0;
    
    cv::Point* pt_temp = new cv::Point[length];
    int* index_z = new int[2*length];
    
    for (i=0; i<length; i++){
        if (flag == 1){
            pt_temp[i].x = points[i].x;
            pt_temp[i].y = points[i].y;
        }
        else{
            pt_temp[i].x = points[i].x*100;
            pt_temp[i].y = points[i].y*100;
        }
    }
    
    for (i=0; i<2*length; i++){
        index_z[i] = i;
    }
    
    for (i=0; i<length; i++){
        if (i == 0)
            distance = (abs(pt_temp[i].x-pt_temp[length-1].x)+abs(pt_temp[i].y-pt_temp[length-1].y))/100;
        else
            distance = (abs(pt_temp[i].x-pt_temp[i-1].x)+abs(pt_temp[i].y-pt_temp[i-1].y))/100;
        
        if (distance < dmax){
            index_z[2*i] = CV_PT_RM;
            length_out--;
        }
    }
    
    if(points != NULL){
        delete []points;
    }
    points = new cv::Point[length_out];
    int ind_prev=0;
    float a=0.0f;
    
    for (i=0; i<2*length; i++) {
        if (index_z[i] != CV_PT_RM){
            ind_prev = int(float(index_z[i])/2.0f);
            a = float(index_z[i])/2.0f - float(ind_prev);
            
            if(ind_prev == length-1){
                points[cont].x = int((1.0f-a)*float(pt_temp[ind_prev].x) + a*float(pt_temp[0].x));
                points[cont].y = int((1.0f-a)*float(pt_temp[ind_prev].y) + a*float(pt_temp[0].y));
            }
            else{
                points[cont].x = int((1.0f-a)*float(pt_temp[ind_prev].x) + a*float(pt_temp[ind_prev+1].x));
                points[cont].y = int((1.0f-a)*float(pt_temp[ind_prev].y) + a*float(pt_temp[ind_prev+1].y));
            }
            cont++;
        }
    }
    if (cont < length_out)
        printf("Error\n");
    
    for (i=0; i<length_out; i++){
        if (i == 0)
            distance = (abs(points[i].x-points[length_out-1].x)+abs(points[i].y-points[length_out-1].y))/100;
        else
            distance = (abs(points[i].x-points[i-1].x)+abs(points[i].y-points[i-1].y))/100;
        Max_d = MAX(Max_d, distance);
    }
    
    *_dist = Max_d;
    *_length = length_out;
    return points;
}

static cv::Point*
cvSnakeInterp(cv::Point* points,
              int * _length,
              int dmin,
              int dmax,
              int flag)
{
    
    int distance=0, cont=0, i;
    int length = * _length;
    int length_out = * _length;
    int Max_d = 0;
    
    for (i=0; i<length; i++){
        if (flag == 1){
            if (i == 0)
                distance = (abs(points[i].x-points[length-1].x)+abs(points[i].y-points[length-1].y))/100;
            else
                distance = (abs(points[i].x-points[i-1].x)+abs(points[i].y-points[i-1].y))/100;
        }    
        else{    
            if (i == 0)
                distance = abs(points[i].x-points[length-1].x)+abs(points[i].y-points[length-1].y);
            else
                distance = abs(points[i].x-points[i-1].x)+abs(points[i].y-points[i-1].y);
        }
        
        if (distance < dmin){
            points[i].x = CV_PT_RM;
            length_out--;
        }
    }
    assert( length_out > 0 );
    cv::Point* pt_temp = new cv::Point[length_out];
    for (i=0; i<length; i++){
        if (points[i].x != CV_PT_RM){
            pt_temp[cont] = points[i];
            cont++;
        }
    }
    if(points != NULL)
        delete []points;
    points = pt_temp;
    *_length = length_out;
    
    points = cvSnakeInterp2(points, _length, dmax, &Max_d, flag);
    
    do{
        points = cvSnakeInterp2(points, _length, dmax, &Max_d, 1);
    }while(Max_d > dmax);
    
    return points;

}

CV_EXPORTS_W float cvFindOpElem(const cv::Mat& srcarr,
                           int flag)
{
      
    cv::Mat sstub, *src;
    cv::Size size;
    float dstElem, *ptr_src;
    int iStep_src;
    
    if (src->type() != CV_32FC1)
        CV_Error(cv::Error::StsUnsupportedFormat, "Only-32bit, 1-channel input images are supported");

    ptr_src = src->ptr<float>();
    iStep_src = src->step / sizeof(ptr_src[0]);
    
    dstElem = ptr_src[0];
    for (int i=0; i<size.height; i++)
        for (int j=0; j<size.width; j++)
        {
            if (flag == CV_MAX_ELEM) {
                if (dstElem < ptr_src[j+iStep_src*i])
                    dstElem = ptr_src[j+iStep_src*i];
            } else {
                if (dstElem > ptr_src[j+iStep_src*i])
                    dstElem = ptr_src[j+iStep_src*i];
            }
        }
    return dstElem;
     
}

CV_EXPORTS_W void cvGVF(const cv::Mat& srcarr,
                   cv::Mat& dstarr_u,
                   cv::Mat& dstarr_v,
                   double mu,
                   int ITER,
                   int flag)
{
   
    
    
    cv::Mat sstub, *src;
    cv::Mat dstubu, *dst_u;
    cv::Mat dstubv, *dst_v;
    cv::Mat * ones, * SqrMagf;
    cv::Mat * fx, * fy, *temp1, *temp2;
    cv::Mat * del_u, *del_v;
    cv::Size size;
    float fmax, fmin;
    float* fPtr_fx, *fPtr_fy, *fPtr_src;
    int iStep_fx, iStep_fy, iStep_src;
    
    src = const_cast<cv::Mat*>(&srcarr);
    dst_u = &dstarr_u;
    dst_v = &dstarr_v;
    size = src->size();

    if (CV_MAT_DEPTH(src->type()) != CV_32F || CV_MAT_CN(src->type()) != 1)

        CV_Error(cv::Error::StsUnsupportedFormat, "Only-32bit, 1-channel input images are supported" );
    if (CV_MAT_DEPTH(dst_u->type()) != CV_32F || CV_MAT_CN(src->type()) != 1)
        CV_Error(cv::Error::StsUnsupportedFormat, "Only-32bit, 1-channel input images are supported" );
    if (CV_MAT_DEPTH(dst_v->type()) != CV_32F || CV_MAT_CN(src->type()) != 1)
        CV_Error(cv::Error::StsUnsupportedFormat, "Only-32bit, 1-channel input images are supported" );
    if (src->size() != dst_u->size())
        CV_Error(cv::Error::StsUnmatchedSizes, "The input and output matrixes must have the same size" );
    if (src->size() != dst_v->size())
        CV_Error(cv::Error::StsUnmatchedSizes, "The input and output matrixes must have the same size" );
        
    temp1 = new cv::Mat(size.height, size.width, CV_32FC1);
    delete temp1;
    temp2 = new cv::Mat(size.height, size.width, CV_32FC1);
    delete temp2;
    ones  = new cv::Mat(size.height, size.width, CV_32FC1);
    delete ones;
    fx    = new cv::Mat(size.height, size.width, CV_32FC1);
    delete fx;
    fy    = new cv::Mat(size.height, size.width, CV_32FC1);
    delete fy;
    del_u = new cv::Mat(size.height, size.width, CV_32FC1);
    delete del_u;
    del_v = new cv::Mat(size.height, size.width, CV_32FC1);
    delete del_v;
    SqrMagf = new cv::Mat(size.height, size.width, CV_32FC1);
    delete SqrMagf;
    
    ones->setTo(cv::Scalar(1.0f));
    del_u->setTo(0);
    del_v->setTo(0);
    
    fPtr_fx = fx->ptr<float>();
    fPtr_fy = fy->ptr<float>();
    fPtr_src = src->ptr<float>();
    iStep_fx = fx->step / sizeof(fPtr_fx[0]);
    iStep_fy = fy->step / sizeof(fPtr_fy[0]);
    iStep_src = src->step / sizeof(fPtr_src[0]);
    
    size = src->size();
    cvNeumannBoundCond(*src, *src);
    
    if (flag == 1){
        fmax = cvFindOpElem(*src, CV_MAX_ELEM);
        fmin = cvFindOpElem(*src, CV_MIN_ELEM);
        cv::subtract(*src, cv::Scalar(fmin), *src);
        cv::multiply(*src, *ones, *src, 1.0f/(fmax-fmin));
        cv::Sobel(*src, *fx, CV_32F, 1, 0, 1);
        cv::Sobel(*src, *fy, CV_32F, 0, 1, 1);
        cv::multiply(*fx, *ones, *fx, 0.5f);
        cv::multiply(*fy, *ones, *fy, 0.5f);
    }
    else if (flag == 2){
        fmax = cvFindOpElem(*src, CV_MAX_ELEM);
        fmin = cvFindOpElem(*src, CV_MIN_ELEM);
        cv::subtract(*src, cv::Scalar(fmin), *src);
        cv::multiply(*src, *ones, *src, 1.0f/(fmax-fmin));
        cv::Sobel(*src, *fx, CV_32F, 1, 0, 1);
        cv::Sobel(*src, *fy, CV_32F, 0, 1, 1);
        cv::multiply(*fx, *ones, *dst_u, 0.5f);
        cv::multiply(*fy, *ones, *dst_v, 0.5f);
    }
    else if(flag == 3){
        cv::Sobel(*src, *fx, CV_32F, 1, 0, 1);
        cv::Sobel(*src, *fy, CV_32F, 0, 1, 1);
        for(int j=0; j<size.height; j++)
        {
            for(int i=0; i<size.width; i++)
            {
                fPtr_src[i+iStep_src*j]=sqrtf(fPtr_fx[i+iStep_fx*j]*fPtr_fx[i+iStep_fx*j]+fPtr_fy[i+iStep_fy*j]*fPtr_fy[i+iStep_fy*j]);
            }
        }
        fmax = cvFindOpElem(*src, CV_MAX_ELEM);
        fmin = cvFindOpElem(*src, CV_MIN_ELEM);
        cv::subtract(*src, cv::Scalar(fmin), *src);
        cv::multiply(*src, *ones, *src, 1.0f/(fmax-fmin));
        cv::Sobel(*src, *fx, CV_32F, 1, 0, 1);
        cv::Sobel(*src, *fy, CV_32F, 0, 1, 1);
        cv::multiply(*fx, *ones, *fx, 0.5f);
        cv::multiply(*fy, *ones, *fy, 0.5f);
        fx->copyTo(*dst_u);
        fy->copyTo(*dst_v);
        cv::multiply(*fx, *fx, *temp1);
        cv::multiply(*fy, *fy, *SqrMagf);
        cv::add(*SqrMagf, *temp1, *SqrMagf);
        for (int i=0; i<ITER; i++) {
            cvNeumannBoundCond(*dst_u, *dst_u);
            cvNeumannBoundCond(*dst_v, *dst_v);
            cv::Laplacian(*dst_u, *del_u, CV_32F, 1);
            cv::Laplacian(*dst_v, *del_v, CV_32F, 1);
            cv::multiply(*del_u, *ones, *del_u, mu);
            cv::multiply(*del_v, *ones, *del_v, mu);
            cv::subtract(*dst_u, *fx, *temp1);
            cv::subtract(*dst_v, *fy, *temp2);
            cv::multiply(*temp1, *SqrMagf, *temp1);
            cv::multiply(*temp2, *SqrMagf, *temp2);
            cv::add(*del_u, *dst_u, *dst_u);
            cv::add(*del_v, *dst_v, *dst_v);
            cv::subtract(*dst_u, *temp1, *dst_u);
            cv::subtract(*dst_v, *temp2, *dst_v);
        }
    }

}

static cv::Exception
icvSnake32FC1_GVF( const cv::Mat* src_u, //CvArr *src_u
                  const cv::Mat* src_v, //CvArr *src_v
                  cv::Point * pt,
                  int *length,
                  float alpha,
                  float beta,
                  float gamma,
                  float kappa,
                  int   ITER_num,
                  int calcInitial)
{
    int i, j, n=*length;
    int iStep_A, iStep_v, iStep_u, iStep_VX, iStep_VY;
    float *ptr_A, *ptr_u, *ptr_v, *ptr_VX, *ptr_VY;
    int flag = calcInitial ? CV_REINITIAL : CV_NREINITIAL;
    cv::Mat A, VX, VY;
    cv::UMat u , sstub_u;
    cv::UMat v , sstub_v; 
    cv::Size roi;
    CV_Error(cv::Error::StsNotImplemented, "icvSnake32FC1_GVF is not implemented");
   u = src_u->getUMat(cv::ACCESS_RW);
   v = src_v->getUMat(cv::ACCESS_RW);
     roi = u.size();
     A = cv::Mat::zeros(n, n, CV_32FC1);

    VX = cv::Mat::zeros(n, 1, CV_32FC1);

    VY = cv::Mat::zeros(n, 1, CV_32FC1);

    A.setTo(0);
    VX.setTo(0);
    VY.setTo(0);

    ptr_VX = VX.ptr<float>();
    iStep_VX= VX.step / sizeof(ptr_VX[0]);
    ptr_VY= VY.ptr<float>();
    iStep_VY= VY.step / sizeof(ptr_VY[0]);
    ptr_u = u.getMat(cv::ACCESS_READ).ptr<float>();
    iStep_u = u.step / sizeof(ptr_u[0]);
    ptr_v = v.getMat(cv::ACCESS_READ).ptr<float>();
    iStep_v = v.step / sizeof(ptr_v[0]);
    ptr_A = A.ptr<float>();
    iStep_A = A.step / sizeof(ptr_A[0]);
    
    for (i=0; i<n; i++)
    {
        ptr_A[i+i*iStep_A] = 2*alpha + 6*beta + gamma;
        if ( i>0 )
            ptr_A[i-1+i*iStep_A] = ptr_A[i+(i-1)*iStep_A] = - ( alpha + 4*beta );
        if ( i>1 )
            ptr_A[i-2+i*iStep_A] = ptr_A[i+(i-2)*iStep_A] = beta;
        if ( i>n-3 )
            ptr_A[i-(n-2)+i*iStep_A] = ptr_A[i+(i-(n-2))*iStep_A] = beta;
        if ( i>n-2)
            ptr_A[i-(n-1)+i*iStep_A] = ptr_A[i+(i-(n-1))*iStep_A] = - ( alpha + 4*beta );
    }
    
    cv::invert(A, A, cv::DECOMP_LU);

    for( i = 0; i < n; i++ )
    {
        if(flag == CV_REINITIAL){
            ptr_VX[i*iStep_VX] = float( pt[i].x )/100.0f;
            ptr_VY[i*iStep_VY] = float( pt[i].y )/100.0f;
        }
        else{
            ptr_VX[i*iStep_VX] = float( pt[i].x );
            ptr_VY[i*iStep_VY] = float( pt[i].y );
        }
    }
    float interp_u, interp_v, a, b;
    int index_x, index_y;
    
    for (j = 0; j<ITER_num; j++)
    {
        for( i = 0; i < n; i++ )
        {   
            index_x = int(ptr_VX[i*iStep_VX]);
            index_y = int(ptr_VY[i*iStep_VY]);
            b = ptr_VX[i*iStep_VX]-float(index_x);
            a = ptr_VY[i*iStep_VY]-float(index_y);
            
            if (index_x < roi.width-1 && index_y < roi.height-1){
                interp_u = (1-b)*((1-a)*ptr_u[index_x+iStep_u*index_y]+a*ptr_u[index_x+iStep_u*(index_y+1)])+b*((1-a)*ptr_u[index_x+1+iStep_u*index_y]+a*ptr_u[index_x+1+iStep_u*(index_y+1)]);
                
                interp_v = (1-b)*((1-a)*ptr_v[index_x+iStep_v*index_y]+a*ptr_v[index_x+iStep_v*(index_y+1)])+b*((1-a)*ptr_v[index_x+1+iStep_v*index_y]+a*ptr_v[index_x+1+iStep_v*(index_y+1)]);
            }
            else{
                interp_u = ptr_u[index_x+1+iStep_u*index_y];
                interp_v = ptr_v[index_x+1+iStep_v*index_y];
            }
            
            ptr_VX[i*iStep_VX] = kappa * interp_u + gamma * ptr_VX[i*iStep_VX];
            ptr_VY[i*iStep_VY] = kappa * interp_v + gamma * ptr_VY[i*iStep_VY];
            
            if(ptr_VX[i*iStep_VX] < 0)
                ptr_VX[i*iStep_VX]=0;
            if(ptr_VX[i*iStep_VX] > roi.width-1)
                ptr_VX[i*iStep_VX] = roi.width-1;
            
            if(ptr_VY[i*iStep_VY] < 0)
                ptr_VY[i*iStep_VY]=0;
            if(ptr_VY[i*iStep_VY] > roi.height-1)
                ptr_VY[i*iStep_VY] = roi.height-1;
        }
        VX = A * VX;
        VY = A * VY;

    }
    for( i = 0; i < n; i++ )
    {
        if(flag == CV_REINITIAL){
            pt[i].x = int(ptr_VX[i*iStep_VX]*100.0f);
            pt[i].y = int(ptr_VY[i*iStep_VY]*100.0f);;
        }
        else{
            pt[i].x = int(ptr_VX[i*iStep_VX]);
            pt[i].y = int(ptr_VY[i*iStep_VY]);
        }
    }

    return cv::Exception(cv::Error::StsOk, "Success", "Success", __FILE__, __LINE__);

}

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
                int alg)
{
    cv::Mat data, u, v;
    cv::Size size = src->size();
    float mu = 0.2f;
    int flag = calcInitial ? CV_REINITIAL : CV_NREINITIAL;

    // Convert Iplimage format to CvMat with CV_32FC1 
    cv::convertScaleAbs(*src, data, 1, 0);
    u = cv::Mat(size, CV_32FC1); 
    v = cv::Mat(size, CV_32FC1); 

    // Compute Gradient Vector Flow
    cvGVF(data, u, v, mu, 80, alg);

    // Apply interpolation to initial curve points, in order to make them dense
    if (flag == CV_REINITIAL) {
        points = cvSnakeInterp(points, length, 1, 100, CV_WITH_HUN);
    }

    for (int i = 0; i < ITER_ext; i++) {
        icvSnake32FC1_GVF(&u, &v, points, length, alpha, beta, gamma, kappa, ITER_int, flag);
        if (flag == CV_REINITIAL) {
            points = cvSnakeInterp(points, length, 1, 1, CV_WITHOUT_HUN);
        }
        loadBar(i + 1, ITER_ext, 50);
    }

    return points;
}



