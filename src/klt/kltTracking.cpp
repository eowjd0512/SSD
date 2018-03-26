#include <iostream>
#include "cv.hpp"
#include "klt.h"
#include "convolve.hpp"
#include <vector>
#include <string>
#include "eigen3/Eigen/Dense"

using namespace std;
using namespace cv;
namespace klt{

    static float _interpolate(
    float x, 
    float y, 
    Mat img)
    {
    int xt = (int) x;  /* coordinates of top-left corner */
    int yt = (int) y;
    float ax = x - xt;
    float ay = y - yt;
    float *ptr = (float*)img.data + (img.cols*yt) + xt;

    #ifndef _DNDEBUG
    if (xt<0 || yt<0 || xt>=img.cols-1 || yt>=img.rows-1) {
        fprintf(stderr, "(xt,yt)=(%d,%d)  imgsize=(%d,%d)\n"
                "(x,y)=(%f,%f)  (ax,ay)=(%f,%f)\n",
                xt, yt, img.cols, img.rows, x, y, ax, ay);
        fflush(stderr);
    }
    #endif

    assert (xt >= 0 && yt >= 0 && xt <= img.cols - 2 && yt <= img.rows - 2);

    return ( (1-ax) * (1-ay) * *ptr +
            ax   * (1-ay) * *(ptr+1) +
            (1-ax) *   ay   * *(ptr+(img.cols)) +
            ax   *   ay   * *(ptr+(img.cols)+1) );
    }
    void _computeIntensityDifferenceLightingInsensitive(
    Mat img1,   /* images */
    Mat img2,
    float x1, float y1,     /* center of window in 1st img */
    float x2, float y2,     /* center of window in 2nd img */
    int width, int height,  /* size of window */
    Mat imgdiff)   /* output */
    {
    int hw = width/2, hh = height/2;
    float g1, g2, sum1_squared = 0, sum2_squared = 0;
    int i, j;
    
    float sum1 = 0, sum2 = 0;
    float mean1, mean2,alpha,belta;
    /* Compute values */
    for (j = -hh ; j <= hh ; j++)
        for (i = -hw ; i <= hw ; i++)  {
        g1 = _interpolate(x1+i, y1+j, img1);
        g2 = _interpolate(x2+i, y2+j, img2);
        sum1 += g1;    sum2 += g2;
        sum1_squared += g1*g1;
        sum2_squared += g2*g2;
    }
    mean1=sum1_squared/(width*height);
    mean2=sum2_squared/(width*height);
    alpha = (float) sqrt(mean1/mean2);
    mean1=sum1/(width*height);
    mean2=sum2/(width*height);
    belta = mean1-alpha*mean2;
    int k=0;
    float* imgdiffdata = (float*)imgdiff.data;
    for (j = -hh ; j <= hh ; j++)
        for (i = -hw ; i <= hw ; i++)  {
        g1 = _interpolate(x1+i, y1+j, img1);
        g2 = _interpolate(x2+i, y2+j, img2);
        imgdiffdata[k] = g1- g2*alpha-belta;
        k++;
        } 
    }
    void _computeGradientSumLightingInsensitive(
    Mat gradx1,  /* gradient images */
    Mat grady1,
    Mat gradx2,
    Mat grady2,
    Mat img1,   /* images */
    Mat img2,
    
    float x1, float y1,      /* center of window in 1st img */
    float x2, float y2,      /* center of window in 2nd img */
    int width, int height,   /* size of window */
    Mat gradx,      /* output */
    Mat grady)      /*   " */
    {
    int hw = width/2, hh = height/2;
    float g1, g2, sum1_squared = 0, sum2_squared = 0;
    int i, j;
    float* gradxdata = (float*)gradx.data;
    float* gradydata = (float*)grady.data;
    /* Compute values */
    float sum1 = 0, sum2 = 0;
    float mean1, mean2, alpha;
    for (j = -hh ; j <= hh ; j++)
        for (i = -hw ; i <= hw ; i++)  {
        g1 = _interpolate(x1+i, y1+j, img1);
        g2 = _interpolate(x2+i, y2+j, img2);
        sum1_squared += g1;    sum2_squared += g2;
        }
    mean1 = sum1_squared/(width*height);
    mean2 = sum2_squared/(width*height);
    alpha = (float) sqrt(mean1/mean2);
    int k=0;
    /* Compute values */
    for (j = -hh ; j <= hh ; j++)
        for (i = -hw ; i <= hw ; i++)  {
        g1 = _interpolate(x1+i, y1+j, gradx1);
        g2 = _interpolate(x2+i, y2+j, gradx2);
        gradxdata[k] = g1 + g2*alpha;
        g1 = _interpolate(x1+i, y1+j, grady1);
        g2 = _interpolate(x2+i, y2+j, grady2);
        gradydata[k] = g1+ g2*alpha;
        k++;
        }  
    }
    static void _computeIntensityDifference(
    Mat img1,   /* images */
    Mat img2,
    float x1, float y1,     /* center of window in 1st img */
    float x2, float y2,     /* center of window in 2nd img */
    int width, int height,  /* size of window */
    Mat &imgdiff)   /* output */
    {
    int hw = width/2, hh = height/2;
    float g1, g2;
    int i, j;
    int k=0;
    float* diffdata = (float*)imgdiff.data;
    /* Compute values */
    for (j = -hh ; j <= hh ; j++)
        for (i = -hw ; i <= hw ; i++)  {
        g1 = _interpolate(x1+i, y1+j, img1);
        g2 = _interpolate(x2+i, y2+j, img2);
        diffdata[k] = g1 - g2;
        k++;
        }
    }
    static void _computeGradientSum(
    Mat gradx1,  /* gradient images */
    Mat grady1,
    Mat gradx2,
    Mat grady2,
    float x1, float y1,      /* center of window in 1st img */
    float x2, float y2,      /* center of window in 2nd img */
    int width, int height,   /* size of window */
    Mat &gradx,      /* output */
    Mat &grady)      /*   " */
    {
    int hw = width/2, hh = height/2;
    float g1, g2;
    int i, j;
    int k=0;
    float* gradxdata = (float*)gradx.data;
    float* gradydata = (float*)grady.data;
    /* Compute values */
    for (j = -hh ; j <= hh ; j++)
        for (i = -hw ; i <= hw ; i++)  {
        g1 = _interpolate(x1+i, y1+j, gradx1);
        g2 = _interpolate(x2+i, y2+j, gradx2);
        //gradx.at<float>(j+hh,i+hw) = g1 + g2;
        gradxdata[k] = g1+g2;
        g1 = _interpolate(x1+i, y1+j, grady1);
        g2 = _interpolate(x2+i, y2+j, grady2);
        //grady.at<float>(j+hh,i+hw) = g1 + g2;
        gradydata[k] = g1+g2;
        k++;
        }
    } 
    static void _compute2by2GradientMatrix(Mat gradx,Mat grady,
    int width,   /* size of window */
    int height,
    float *gxx,  /* return values */
    float *gxy, 
    float *gyy) 

    {
        float gx, gy;
        int i,j;
        /* Compute values */
        *gxx = 0.0;  *gxy = 0.0;  *gyy = 0.0;
        float* gradxdata = (float*)gradx.data;
        float* gradydata = (float*)grady.data;
        for (i = 0 ; i < width * height ; i++)  {
                gx = gradxdata[i];
                gy = gradydata[i];
                *gxx += gx*gx;
                *gxy += gx*gy;
                *gyy += gy*gy;
        
        }
        //cout<<"gx: "<<gx<<endl;
        //cout<<"gy: "<<gy<<endl;
        
    }
    static void _compute2by1ErrorVector(Mat imgdiff,Mat gradx,Mat grady,
    int width,   /* size of window */
    int height,
    float step_factor, /* 2.0 comes from equations, 1.0 seems to avoid overshooting */
    float *ex,   /* return values */
    float *ey)
    {
        float diff;
        int i,j;

        /* Compute values */
        *ex = 0;  *ey = 0;
        float* diffdata = (float*)imgdiff.data;
        float* gradxdata = (float*)gradx.data;
        float* gradydata = (float*)grady.data;
        for (i = 0 ; i < width * height ; i++)  {
            diff = diffdata[i];
            *ex += diff * gradxdata[i];
            *ey += diff * gradydata[i];
        }
        *ex *= step_factor;
        *ey *= step_factor;
        *ex *= 2;
        *ey *= 2;
    }

    static int _solveEquation(
    float gxx, float gxy, float gyy,
    float ex, float ey,
    float small,
    float *dx, float *dy)
    {
        float det = gxx*gyy - gxy*gxy;
        //cout<<"gxx: "<<gxx<<endl;
        //cout<<"gyy: "<<gyy<<endl;
        //cout<<"gxy: "<<gxy<<endl;
        //cout<<"det: "<<det<<endl;
            
        if (det < small)  return KLT_SMALL_DET;

        *dx = (gyy*ex - gxy*ey)/det;
        *dy = (gxx*ey - gxy*ex)/det;
        return KLT_TRACKED;
    }
    double _sumAbsFloatWindow(
    Mat fw,
    int width,
    int height)
    {
        Scalar sum_;
        int w;
        Mat absfw = abs(fw);
        sum_ = sum(absfw);
        
        return sum_[0];
    }
      
    static bool _outOfBounds(
    float x,
    float y,
    int ncols,
    int nrows,
    int borderx,
    int bordery)
    {
    return (x < borderx || x > ncols-1-borderx ||
            y < bordery || y > nrows-1-bordery );
    }

    vector<Mat> _KLTCreatePyramid(Mat floatimg1, int subsampling, int MaxPyLevel){
        vector<Mat> pyr;
        Mat pyramid[MaxPyLevel];
        pyramid[0]=floatimg1;
        pyr.push_back(pyramid[0]);
        if(subsampling ==2){ 
            for(int i=0; i<MaxPyLevel-1;i++){
                pyrDown(pyramid[i],pyramid[i+1]);
                pyr.push_back(pyramid[i+1]);
            }
        }
        else if(subsampling ==4){
            for(int i=0; i<MaxPyLevel-1;i++){
                pyrDown(pyramid[i],pyramid[i+1]);
                pyrDown(pyramid[i+1],pyramid[i+1]);
                pyr.push_back(pyramid[i+1]);
            }
        }else if(subsampling ==8){
            for(int i=0; i<MaxPyLevel-1;i++){
                pyrDown(pyramid[i],pyramid[i+1]);
                pyrDown(pyramid[i+1],pyramid[i+1]);
                pyrDown(pyramid[i+1],pyramid[i+1]);
                pyr.push_back(pyramid[i+1]);
            }
        }
        
        return pyr;
    }
    vector<Mat> _KLTComputeGradients(vector<Mat> pyramid1, int x, int y,int ksize, float sigma){
        vector<Mat> grad;
        convolution conv;
        for (int i=0; i<pyramid1.size();i++){
            Mat grad_ = Mat::zeros(pyramid1[i].rows,pyramid1[i].cols,CV_32FC1);

            GaussianBlur(pyramid1[i],pyramid1[i],Size(ksize,ksize),sigma);
            //conv._KLTComputeSmoothedImage(pyramid1[i], sigma, pyramid1[i]);
		
            //if (fabs(sigma - conv.sigma_last) > 0.05)
                //conv._computeKernels(sigma, &conv.gauss_kernel, &conv.gaussderiv_kernel);
            //cout<<"3"<<endl;
            //if(x==1 && y==0){
            //conv._convolveSeparate(pyramid1[i], conv.gaussderiv_kernel, conv.gauss_kernel, grad_);
            //}else if(x==0, y==1)
            //conv._convolveSeparate(pyramid1[i], conv.gauss_kernel, conv.gaussderiv_kernel, grad_);

            //cout<<"4"<<endl;
            Sobel(pyramid1[i], grad_,-1,x,y,3);
            grad_ = grad_/9;
            //convertScaleAbs( grad_, grad_);
            //imwrite("/home/jun/SSD_SLAM/debug/grad.jpg", grad_);
            grad.push_back(grad_);
        }
        return grad;
    }
    int KLTtracker::_trackFeature(float x1, float y1, float *x2, float *y2,
                      Mat img1,Mat gradx1,Mat grady1,Mat img2,Mat gradx2,Mat grady2){
        int width = this->tracker.window_width;          /* size of window */
        int height = this->tracker.window_height;
        float step_factor= this->tracker.step_factor; /* 2.0 comes from equations, 1.0 seems to avoid overshooting */
        int max_iterations= this->tracker.max_iterations;
        float small= this->tracker.min_determinant;         /* determinant threshold for declaring KLT_SMALL_DET */
        float th= this->tracker.min_displacement;            /* displacement threshold for stopping               */
        float max_residue= this->tracker.max_residue;   /* residue threshold for declaring KLT_LARGE_RESIDUE */
        int lighting_insensitive= this->tracker.lighting_insensitive;  /* whether to normalize for gain and bias */
        
        Mat imgdiff, gradx, grady;
        float gxx, gxy, gyy, ex, ey, dx, dy;
        int iteration = 0;
        int status;
        int hw = width/2;
        int hh = height/2;
        int nc = img1.cols;
        int nr = img1.rows;
        float one_plus_eps = 1.001f;   /* To prevent rounding errors */
        /* Allocate memory for windows */
        imgdiff = Mat::zeros(height,width,CV_32FC1);
        gradx   = Mat::zeros(height,width,CV_32FC1);
        grady   = Mat::zeros(height,width,CV_32FC1);

        /* Iteratively update the window position */
        do  {

            /* If out of bounds, exit loop */
            if (  x1-hw < 0.0f || nc-( x1+hw) < one_plus_eps ||
                *x2-hw < 0.0f || nc-(*x2+hw) < one_plus_eps ||
                y1-hh < 0.0f || nr-( y1+hh) < one_plus_eps ||
                *y2-hh < 0.0f || nr-(*y2+hh) < one_plus_eps) {
            status = KLT_OOB;
            break;
            }
            
            /* Compute gradient and difference windows */
            if (lighting_insensitive) {
                
            _computeIntensityDifferenceLightingInsensitive(img1, img2, x1, y1, *x2, *y2, 
                                        width, height, imgdiff);
            _computeGradientSumLightingInsensitive(gradx1, grady1, gradx2, grady2, 
                    img1, img2, x1, y1, *x2, *y2, width, height, gradx, grady);
            } else {
            _computeIntensityDifference(img1, img2, x1, y1, *x2, *y2, 
                                        width, height, imgdiff);
            _computeGradientSum(gradx1, grady1, gradx2, grady2, 
                    x1, y1, *x2, *y2, width, height, gradx, grady);
            }
                

            /* Use these windows to construct matrices */
            _compute2by2GradientMatrix(gradx, grady, width, height, 
                                    &gxx, &gxy, &gyy);
            _compute2by1ErrorVector(imgdiff, gradx, grady, width, height, step_factor,
                                    &ex, &ey);
                        
            /* Using matrices, solve equation for new displacement */
            status = _solveEquation(gxx, gxy, gyy, ex, ey, small, &dx, &dy);
            if (status == KLT_SMALL_DET)  break;

            *x2 += dx;
            *y2 += dy;
            iteration++;

        }  while ((fabs(dx)>=th || fabs(dy)>=th) && iteration < max_iterations);

        /* Check whether window is out of bounds */
        if (*x2-hw < 0.0f || nc-(*x2+hw) < one_plus_eps || 
            *y2-hh < 0.0f || nr-(*y2+hh) < one_plus_eps)
            status = KLT_OOB;

        /* Check whether residue is too large */
        if (status == KLT_TRACKED)  {
            if (lighting_insensitive)
            _computeIntensityDifferenceLightingInsensitive(img1, img2, x1, y1, *x2, *y2, 
                                        width, height, imgdiff);
            else
            _computeIntensityDifference(img1, img2, x1, y1, *x2, *y2, 
                                        width, height, imgdiff);
            if (_sumAbsFloatWindow(imgdiff, width, height)/(width*height) > max_residue) 
            status = KLT_LARGE_RESIDUE;
        }

        /* Free memory */
        //free(imgdiff);  free(gradx);  free(grady);

        /* Return appropriate value */
        if (status == KLT_SMALL_DET)  return KLT_SMALL_DET;
        else if (status == KLT_OOB)  return KLT_OOB;
        else if (status == KLT_LARGE_RESIDUE)  return KLT_LARGE_RESIDUE;
        else if (iteration >= max_iterations)  return KLT_MAX_ITERATIONS;
        else  return KLT_TRACKED;

    }
    void KLTtracker::trackFeatures(Mat prevImg, Mat currImg, vector<kltFeature> prevfl, vector<kltFeature> &currfl){
        Mat tmpimg, floatimg1, floatimg2;
        currfl=prevfl;
        //int MaxPyLevel = 3;
        int MaxPyLevel = this->tracker.nPyramidLevels;
        vector<Mat> pyramid1, pyramid1_gradx, pyramid1_grady,
            pyramid2, pyramid2_gradx, pyramid2_grady;
        float subsampling = (float) this->tracker.subsampling;
        //float subsampling =2.0;
        float xloc, yloc, xlocout, ylocout;
        int val;
        int indx, r;
        bool floatimg1_created = false;
        int i;
        float sigma = (this->tracker.smooth_sigma_fact * max(this->tracker.window_width, this->tracker.window_height));
        int ksize = this->tracker.window_width;
        /*
        if (KLT_verbose >= 1)  {
            fprintf(stderr,  "(KLT) Tracking %d features in a %d by %d image...  ",
                KLTCountRemainingFeatures(featurelist), ncols, nrows);
            fflush(stderr);
        }*/

        /* Check window size (and correct if necessary) 
        if (this->tracker.window_width % 2 != 1) {
            this->tracker.window_width = this->tracker.window_width+1;
        }
        if (this->tracker.window_height % 2 != 1) {
            this->tracker.window_height = this->tracker.window_height+1;
        }
        if (this->tracker.window_width < 3) {
            this->tracker.window_width = 3;
        }
        if (this->tracker.window_height < 3) {
            this->tracker.window_height = 3;
        }*/
        /* Create temporary image */
        //tmpimg = _KLTCreateFloatImage(ncols, nrows);

        /* Process first image by converting to float, smoothing, computing */
        /* pyramid, and computing gradient pyramids */
        if (this->tracker.sequentialMode && !this->tracker.pyramid_last.empty())  {
            pyramid1 = this->tracker.pyramid_last;
            pyramid1_gradx = this->tracker.pyramid_last_gradx;
            pyramid1_grady = this->tracker.pyramid_last_grady;
            if (pyramid1[0].cols != prevImg.cols || pyramid1[0].rows != prevImg.rows)
                cerr<<"(KLTTrackFeatures) Size of incoming image is different from size of previous image"<<endl; 
            assert(!pyramid1_gradx.empty());
            assert(!pyramid1_grady.empty());
        } else  {
            //floatimg1_created = TRUE;
            prevImg.convertTo(floatimg1, CV_32FC1);
            
            GaussianBlur(floatimg1,floatimg1,Size(ksize,ksize),sigma);
            //_KLTComputeSmoothedImage(floatimg1, sigma, floatimg1);
		
            pyramid1 = _KLTCreatePyramid(floatimg1, (int) subsampling, MaxPyLevel);
            //_KLTComputePyramid(floatimg1, pyramid1, tc->pyramid_sigma_fact);
            pyramid1_gradx = _KLTComputeGradients(pyramid1, 1,0,ksize,this->tracker.grad_sigma);
            pyramid1_grady = _KLTComputeGradients(pyramid1, 0,1,ksize,this->tracker.grad_sigma);
        }
        /* Do the same thing with second image */
        currImg.convertTo(floatimg2, CV_32FC1);
        
        GaussianBlur(floatimg2,floatimg2,Size(ksize,ksize),sigma);
        //_KLTComputeSmoothedImage(floatimg2, sigma, floatimg2);
		
        pyramid2 = _KLTCreatePyramid(floatimg2, (int) subsampling, MaxPyLevel);
        //_KLTComputePyramid(floatimg1, pyramid1, tc->pyramid_sigma_fact);
        pyramid2_gradx = _KLTComputeGradients(pyramid2, 1,0,ksize,this->tracker.grad_sigma);
        pyramid2_grady = _KLTComputeGradients(pyramid2, 0,1,ksize,this->tracker.grad_sigma);
        
        /* Write internal images */
        if (1)  {
            //char fname[80];
            string s[5] = {"a","b","c","d","e"};
            for (i = 0 ; i < MaxPyLevel ; i++)  {
                //sprintf(fname, "kltimg_tf_i%d.pgm", i);
                imwrite("/home/jun/SSD_SLAM/debug/pyr1"+ s[i] +".jpg", pyramid1[i]);
                imwrite("/home/jun/SSD_SLAM/debug/pyr1_gradx"+ s[i] +".jpg",  pyramid1_gradx[i]);
                imwrite("/home/jun/SSD_SLAM/debug/pyr1_grady"+ s[i] +".jpg",  pyramid1_grady[i]);
                imwrite("/home/jun/SSD_SLAM/debug/pyr2"+ s[i] +".jpg", pyramid2[i]);
                imwrite("/home/jun/SSD_SLAM/debug/pyr2_gradx"+ s[i] +".jpg",  pyramid2_gradx[i]);
                imwrite("/home/jun/SSD_SLAM/debug/pyr2_grady"+ s[i] +".jpg",  pyramid2_grady[i]);
                
            }
        }

        /* For each feature, do ... */
        for (indx = 0 ; indx < this->nfeatures ; indx++)  {

            /* Only track features that are not lost */
            if (prevfl[indx].val >= 0)  {

                xloc = prevfl[indx].pt.x;
                yloc = prevfl[indx].pt.y;

                /* Transform location to coarsest resolution */
                for (r = MaxPyLevel - 1 ; r >= 0 ; r--)  {
                    xloc /= subsampling;  yloc /= subsampling;
                }
                xlocout = xloc;  ylocout = yloc;

                /* Beginning with coarsest resolution, do ... */
                for (r = MaxPyLevel - 1 ; r >= 0 ; r--)  {

                    /* Track feature at current resolution */
                    xloc *= subsampling;  yloc *= subsampling;
                    xlocout *= subsampling;  ylocout *= subsampling;

                    val = _trackFeature(xloc, yloc, 
                        &xlocout, &ylocout,
                        pyramid1[r], 
                        pyramid1_gradx[r], pyramid1_grady[r], 
                        pyramid2[r], 
                        pyramid2_gradx[r], pyramid2_grady[r]);

                    if (val==KLT_SMALL_DET || val==KLT_OOB)
                        break;
                }
                //cout<<"val: "<<val<<endl;
                /* Record feature */
                if (val == KLT_OOB) {
                    currfl[indx].pt.x   = -1.0;
                    currfl[indx].pt.y   = -1.0;
                    currfl[indx].val = KLT_OOB;
                    currfl[indx].status = -1;
                    if( !currfl[indx].aff_img.empty() ) currfl[indx].aff_img.release();
                    if( !currfl[indx].aff_img_gradx.empty() ) currfl[indx].aff_img_gradx.release();
                    if( !currfl[indx].aff_img_grady.empty() ) currfl[indx].aff_img_grady.release();
                } 
                else if (_outOfBounds(xlocout, ylocout, currImg.cols, currImg.rows, this->tracker.borderx, this->tracker.bordery))  {
                    currfl[indx].pt.x   = -1.0;
                    currfl[indx].pt.y   = -1.0;
                    currfl[indx].val = KLT_OOB;
                    currfl[indx].status = -1;
                    if( !currfl[indx].aff_img.empty() ) currfl[indx].aff_img.release();
                    if( !currfl[indx].aff_img_gradx.empty() ) currfl[indx].aff_img_gradx.release();
                    if( !currfl[indx].aff_img_grady.empty() ) currfl[indx].aff_img_grady.release();
                }
                 else if (val == KLT_SMALL_DET)  {
                    currfl[indx].pt.x   = -1.0;
                    currfl[indx].pt.y   = -1.0;
                    currfl[indx].val = KLT_SMALL_DET;
                    currfl[indx].status = -1;
                    if( !currfl[indx].aff_img.empty() ) currfl[indx].aff_img.release();
                    if( !currfl[indx].aff_img_gradx.empty() ) currfl[indx].aff_img_gradx.release();
                    if( !currfl[indx].aff_img_grady.empty() ) currfl[indx].aff_img_grady.release();
                }
                 else if (val == KLT_LARGE_RESIDUE)  {
                    currfl[indx].pt.x   = -1.0;
                    currfl[indx].pt.y   = -1.0;
                    currfl[indx].val = KLT_LARGE_RESIDUE;
                    currfl[indx].status = -1;
                    if( !currfl[indx].aff_img.empty() ) currfl[indx].aff_img.release();
                    if( !currfl[indx].aff_img_gradx.empty() ) currfl[indx].aff_img_gradx.release();
                    if( !currfl[indx].aff_img_grady.empty() ) currfl[indx].aff_img_grady.release();
                }
                 else if (val == KLT_MAX_ITERATIONS)  {
                    currfl[indx].pt.x = xlocout;
                    currfl[indx].pt.y  = ylocout;
                    currfl[indx].val  = KLT_TRACKED;
                    currfl[indx].status = 0;
                    if( !currfl[indx].aff_img.empty() ) currfl[indx].aff_img.release();
                    if( !currfl[indx].aff_img_gradx.empty() ) currfl[indx].aff_img_gradx.release();
                    if( !currfl[indx].aff_img_grady.empty() ) currfl[indx].aff_img_grady.release();
                } else  {
                    currfl[indx].pt.x = xlocout;
                    currfl[indx].pt.y  = ylocout;
                    currfl[indx].val  = KLT_TRACKED;
                    currfl[indx].status = 1;
                    if (this->tracker.affineConsistencyCheck >= 0 && val == KLT_TRACKED)  { /*for affine mapping*/
                        int border = 2; /* add border for interpolation */

    #ifdef DEBUG_AFFINE_MAPPING	  
                        glob_index = indx;
    #endif
/*
                        if(!featurelist->feature[indx]->aff_img){
                            // save image and gradient for each feature at finest resolution after first successful track 
                            featurelist->feature[indx]->aff_img = _KLTCreateFloatImage((tc->affine_window_width+border), (tc->affine_window_height+border));
                            featurelist->feature[indx]->aff_img_gradx = _KLTCreateFloatImage((tc->affine_window_width+border), (tc->affine_window_height+border));
                            featurelist->feature[indx]->aff_img_grady = _KLTCreateFloatImage((tc->affine_window_width+border), (tc->affine_window_height+border));
                            _am_getSubFloatImage(pyramid1->img[0],xloc,yloc,featurelist->feature[indx]->aff_img);
                            _am_getSubFloatImage(pyramid1_gradx->img[0],xloc,yloc,featurelist->feature[indx]->aff_img_gradx);
                            _am_getSubFloatImage(pyramid1_grady->img[0],xloc,yloc,featurelist->feature[indx]->aff_img_grady);
                            featurelist->feature[indx]->aff_x = xloc - (int) xloc + (tc->affine_window_width+border)/2;
                            featurelist->feature[indx]->aff_y = yloc - (int) yloc + (tc->affine_window_height+border)/2;;
                        }else{
                            // affine tracking 
                            val = _am_trackFeatureAffine(featurelist->feature[indx]->aff_x, featurelist->feature[indx]->aff_y,
                                &xlocout, &ylocout,
                                featurelist->feature[indx]->aff_img, 
                                featurelist->feature[indx]->aff_img_gradx, 
                                featurelist->feature[indx]->aff_img_grady,
                                pyramid2->img[0], 
                                pyramid2_gradx->img[0], pyramid2_grady->img[0],
                                tc->affine_window_width, tc->affine_window_height,
                                tc->step_factor,
                                tc->affine_max_iterations,
                                tc->min_determinant,
                                tc->min_displacement,
                                tc->affine_min_displacement,
                                tc->affine_max_residue, 
                                tc->lighting_insensitive,
                                tc->affineConsistencyCheck,
                                tc->affine_max_displacement_differ,
                                &featurelist->feature[indx]->aff_Axx,
                                &featurelist->feature[indx]->aff_Ayx,
                                &featurelist->feature[indx]->aff_Axy,
                                &featurelist->feature[indx]->aff_Ayy 
                                );
                            featurelist->feature[indx]->val = val;
                            if(val != KLT_TRACKED){
                                featurelist->feature[indx]->x   = -1.0;
                                featurelist->feature[indx]->y   = -1.0;
                                featurelist->feature[indx]->aff_x = -1.0;
                                featurelist->feature[indx]->aff_y = -1.0;
                                // free image and gradient for lost feature //
                                _KLTFreeFloatImage(featurelist->feature[indx]->aff_img);
                                _KLTFreeFloatImage(featurelist->feature[indx]->aff_img_gradx);
                                _KLTFreeFloatImage(featurelist->feature[indx]->aff_img_grady);
                                featurelist->feature[indx]->aff_img = NULL;
                                featurelist->feature[indx]->aff_img_gradx = NULL;
                                featurelist->feature[indx]->aff_img_grady = NULL;
                            }else{
                                //featurelist->feature[indx]->x = xlocout;
                                //featurelist->feature[indx]->y = ylocout;
                            }
                        }*/
                    }

                }
            }
        }

        if (this->tracker.sequentialMode)  {
            this->tracker.pyramid_last = pyramid2;
            this->tracker.pyramid_last_gradx = pyramid2_gradx;
            this->tracker.pyramid_last_grady = pyramid2_grady;
        } else  {
            pyramid2.clear();
            pyramid2_gradx.clear();
            pyramid2_grady.clear();
        }

        /* Free memory */
        pyramid1.clear();
        pyramid1_gradx.clear();
        pyramid1_grady.clear();

        //if (KLT_verbose >= 1)  {
        //    fprintf(stderr,  "\n\t%d features successfully tracked.\n",
        //        KLTCountRemainingFeatures(featurelist));
        //    if (tc->writeInternalImages)
        //        fprintf(stderr,  "\tWrote images to 'kltimg_tf*.pgm'.\n");
         //   fflush(stderr);
        //}
        
    }

    void KLTtracker::JRCtrackFeatures(Mat prevImg, Mat currImg, vector<kltFeature> prevfl, vector<kltFeature> &currfl){
        Mat tmpimg, floatimg1, floatimg2;
        currfl=prevfl;
        //int MaxPyLevel = 3;
        int MaxPyLevel = this->tracker.nPyramidLevels;
        vector<Mat> pyramid1, pyramid1_gradx, pyramid1_grady,
            pyramid2, pyramid2_gradx, pyramid2_grady;
        float subsampling = (float) this->tracker.subsampling;
        //float subsampling =2.0;
        float xloc, yloc, xlocout, ylocout;
        int val;
        int indx, r;
        bool floatimg1_created = false;
        int i;
        float sigma = (this->tracker.smooth_sigma_fact * max(this->tracker.window_width, this->tracker.window_height));
        int ksize = this->tracker.window_width;
        int nfeature = this->nfeatures;
        /* Process first image by converting to float, smoothing, computing */
        /* pyramid, and computing gradient pyramids */
        if (this->tracker.sequentialMode && !this->tracker.pyramid_last.empty())  {
            pyramid1 = this->tracker.pyramid_last;
            pyramid1_gradx = this->tracker.pyramid_last_gradx;
            pyramid1_grady = this->tracker.pyramid_last_grady;
            if (pyramid1[0].cols != prevImg.cols || pyramid1[0].rows != prevImg.rows)
                cerr<<"(KLTTrackFeatures) Size of incoming image is different from size of previous image"<<endl; 
            assert(!pyramid1_gradx.empty());
            assert(!pyramid1_grady.empty());
        } else  {
            //floatimg1_created = TRUE;
            prevImg.convertTo(floatimg1, CV_32FC1);
            
            GaussianBlur(floatimg1,floatimg1,Size(ksize,ksize),sigma);
            //_KLTComputeSmoothedImage(floatimg1, sigma, floatimg1);
		
            pyramid1 = _KLTCreatePyramid(floatimg1, (int) subsampling, MaxPyLevel);
            //_KLTComputePyramid(floatimg1, pyramid1, tc->pyramid_sigma_fact);
            pyramid1_gradx = _KLTComputeGradients(pyramid1, 1,0,ksize,this->tracker.grad_sigma);
            pyramid1_grady = _KLTComputeGradients(pyramid1, 0,1,ksize,this->tracker.grad_sigma);
        }
        /* Do the same thing with second image */
        currImg.convertTo(floatimg2, CV_32FC1);
        
        GaussianBlur(floatimg2,floatimg2,Size(ksize,ksize),sigma);
        //_KLTComputeSmoothedImage(floatimg2, sigma, floatimg2);
		
        pyramid2 = _KLTCreatePyramid(floatimg2, (int) subsampling, MaxPyLevel);
        //_KLTComputePyramid(floatimg1, pyramid1, tc->pyramid_sigma_fact);
        pyramid2_gradx = _KLTComputeGradients(pyramid2, 1,0,ksize,this->tracker.grad_sigma);
        pyramid2_grady = _KLTComputeGradients(pyramid2, 0,1,ksize,this->tracker.grad_sigma);
        
        /* Write internal images */
        if (0)  {
            //char fname[80];
            string s[5] = {"a","b","c","d","e"};
            for (i = 0 ; i < MaxPyLevel ; i++)  {
                //sprintf(fname, "kltimg_tf_i%d.pgm", i);
                imwrite("/home/jun/SSD_SLAM/debug/pyr1"+ s[i] +".jpg", pyramid1[i]);
                imwrite("/home/jun/SSD_SLAM/debug/pyr1_gradx"+ s[i] +".jpg",  pyramid1_gradx[i]);
                imwrite("/home/jun/SSD_SLAM/debug/pyr1_grady"+ s[i] +".jpg",  pyramid1_grady[i]);
                imwrite("/home/jun/SSD_SLAM/debug/pyr2"+ s[i] +".jpg", pyramid2[i]);
                imwrite("/home/jun/SSD_SLAM/debug/pyr2_gradx"+ s[i] +".jpg",  pyramid2_gradx[i]);
                imwrite("/home/jun/SSD_SLAM/debug/pyr2_grady"+ s[i] +".jpg",  pyramid2_grady[i]);
                
            }
        }
        JRC::JointRadiometicCalib jrc;
        bool JRCtrackingMode = jrc.trackingMode;
        Eigen::MatrixXf Uinv_all = Eigen::MatrixXf::Zero(nfeature*8,nfeature*8);
        Eigen::MatrixXf w_all = Eigen::MatrixXf::Zero(nfeature*8,nfeature*4);
        Eigen::VectorXf v_all = Eigen::MatrixXf::Zero(nfeature*8,nfeature*1);
        //Eigen::MatrixXf z_all = Eigen::MatrixXf::Zero(nfeature*8,nfeature*8);
        Eigen::MatrixXf lamda_all = Eigen::MatrixXf::Zero(5,4);
        Eigen::VectorXf m_all = Eigen::MatrixXf::Zero(5,1);
        int numOfTrackFeature=0;
        for (indx = 0 ; indx < nfeature ; indx++)  {
            
            /* Only track features that are not lost */
            if (prevfl[indx].val >= 0)  {
                if (prevfl[indx].used == false){
                        //initialization
                    jrc.initialization(prevfl[indx]);
                    prevfl[indx].used = true;
                }
                // 1.  create  U, w, lamda, v, m for all features
                jrc.constructMatrix(prevfl[indx],0, ksize, prevfl[indx].pt.x, prevfl[indx].pt.y, pyramid2[0], pyramid1[0] ,pyramid2_gradx[0], pyramid1_gradx[0],pyramid2_grady[0], pyramid1_grady[0],JRCtrackingMode);
                //for constructing all Matrix
                jrc.constructAllMatrix(prevfl[indx], numOfTrackFeature,Uinv_all,w_all,v_all,lamda_all,m_all);
                numOfTrackFeature++;
            }
    
        }
        // 2. calculate K by all featurs
        float K=0; //K is exposure time difference between two images.
        jrc.blockAllMatrix(numOfTrackFeature,Uinv_all,w_all,v_all,lamda_all,m_all,JRCtrackingMode);
        K= jrc.get_K(prevfl[indx],Uinv_all,w_all,v_all,lamda_all,m_all,JRCtrackingMode);
        
        // 3.  calculate each dx, dy with respect to each feature     
        /* For each feature, do ... */
        for (indx = 0 ; indx < nfeature ; indx++)  {

            /* Only track features that are not lost */
            if (prevfl[indx].val >= 0)  {

                xloc = prevfl[indx].pt.x;
                yloc = prevfl[indx].pt.y;

                /* Transform location to coarsest resolution */
                for (r = MaxPyLevel - 1 ; r >= 0 ; r--)  {
                    xloc /= subsampling;  yloc /= subsampling;
                }
                xlocout = xloc;  ylocout = yloc;

                /* Beginning with coarsest resolution, do ... */
                for (r = MaxPyLevel - 1 ; r >= 0 ; r--)  {

                    /* Track feature at current resolution */
                    xloc *= subsampling;  yloc *= subsampling;
                    xlocout *= subsampling;  ylocout *= subsampling;

                    val = _trackFeature(xloc, yloc, 
                        &xlocout, &ylocout,
                        pyramid1[r], 
                        pyramid1_gradx[r], pyramid1_grady[r], 
                        pyramid2[r], 
                        pyramid2_gradx[r], pyramid2_grady[r]);

                    if (val==KLT_SMALL_DET || val==KLT_OOB)
                        break;
                }
                //cout<<"val: "<<val<<endl;
                /* Record feature */
                if (val == KLT_OOB) {
                    currfl[indx].pt.x   = -1.0;
                    currfl[indx].pt.y   = -1.0;
                    currfl[indx].val = KLT_OOB;
                    currfl[indx].status = -1;
                    if( !currfl[indx].aff_img.empty() ) currfl[indx].aff_img.release();
                    if( !currfl[indx].aff_img_gradx.empty() ) currfl[indx].aff_img_gradx.release();
                    if( !currfl[indx].aff_img_grady.empty() ) currfl[indx].aff_img_grady.release();
                } 
                else if (_outOfBounds(xlocout, ylocout, currImg.cols, currImg.rows, this->tracker.borderx, this->tracker.bordery))  {
                    currfl[indx].pt.x   = -1.0;
                    currfl[indx].pt.y   = -1.0;
                    currfl[indx].val = KLT_OOB;
                    currfl[indx].status = -1;
                    if( !currfl[indx].aff_img.empty() ) currfl[indx].aff_img.release();
                    if( !currfl[indx].aff_img_gradx.empty() ) currfl[indx].aff_img_gradx.release();
                    if( !currfl[indx].aff_img_grady.empty() ) currfl[indx].aff_img_grady.release();
                }
                 else if (val == KLT_SMALL_DET)  {
                    currfl[indx].pt.x   = -1.0;
                    currfl[indx].pt.y   = -1.0;
                    currfl[indx].val = KLT_SMALL_DET;
                    currfl[indx].status = -1;
                    if( !currfl[indx].aff_img.empty() ) currfl[indx].aff_img.release();
                    if( !currfl[indx].aff_img_gradx.empty() ) currfl[indx].aff_img_gradx.release();
                    if( !currfl[indx].aff_img_grady.empty() ) currfl[indx].aff_img_grady.release();
                }
                 else if (val == KLT_LARGE_RESIDUE)  {
                    currfl[indx].pt.x   = -1.0;
                    currfl[indx].pt.y   = -1.0;
                    currfl[indx].val = KLT_LARGE_RESIDUE;
                    currfl[indx].status = -1;
                    if( !currfl[indx].aff_img.empty() ) currfl[indx].aff_img.release();
                    if( !currfl[indx].aff_img_gradx.empty() ) currfl[indx].aff_img_gradx.release();
                    if( !currfl[indx].aff_img_grady.empty() ) currfl[indx].aff_img_grady.release();
                }
                 else if (val == KLT_MAX_ITERATIONS)  {
                    currfl[indx].pt.x = xlocout;
                    currfl[indx].pt.y  = ylocout;
                    currfl[indx].val  = KLT_TRACKED;
                    currfl[indx].status = 0;
                    if( !currfl[indx].aff_img.empty() ) currfl[indx].aff_img.release();
                    if( !currfl[indx].aff_img_gradx.empty() ) currfl[indx].aff_img_gradx.release();
                    if( !currfl[indx].aff_img_grady.empty() ) currfl[indx].aff_img_grady.release();
                } else  {
                    currfl[indx].pt.x = xlocout;
                    currfl[indx].pt.y  = ylocout;
                    currfl[indx].val  = KLT_TRACKED;
                    currfl[indx].status = 1;
                }
            }
        }

        if (this->tracker.sequentialMode)  {
            this->tracker.pyramid_last = pyramid2;
            this->tracker.pyramid_last_gradx = pyramid2_gradx;
            this->tracker.pyramid_last_grady = pyramid2_grady;
        } else  {
            pyramid2.clear();
            pyramid2_gradx.clear();
            pyramid2_grady.clear();
        }

        /* Free memory */
        pyramid1.clear();
        pyramid1_gradx.clear();
        pyramid1_grady.clear();
    }

    int KLTtracker::_JRCtrackFeature(float x1, float y1, float *x2, float *y2,
                      Mat img1,Mat gradx1,Mat grady1,Mat img2,Mat gradx2,Mat grady2){
        int width = this->tracker.window_width;          /* size of window */
        int height = this->tracker.window_height;
        float step_factor= this->tracker.step_factor; /* 2.0 comes from equations, 1.0 seems to avoid overshooting */
        int max_iterations= this->tracker.max_iterations;
        float small= this->tracker.min_determinant;         /* determinant threshold for declaring KLT_SMALL_DET */
        float th= this->tracker.min_displacement;            /* displacement threshold for stopping               */
        float max_residue= this->tracker.max_residue;   /* residue threshold for declaring KLT_LARGE_RESIDUE */
        int lighting_insensitive= this->tracker.lighting_insensitive;  /* whether to normalize for gain and bias */
        
        Mat imgdiff, gradx, grady;
        float gxx, gxy, gyy, ex, ey, dx, dy;
        int iteration = 0;
        int status;
        int hw = width/2;
        int hh = height/2;
        int nc = img1.cols;
        int nr = img1.rows;
        float one_plus_eps = 1.001f;   /* To prevent rounding errors */
        /* Allocate memory for windows */
        imgdiff = Mat::zeros(height,width,CV_32FC1);
        gradx   = Mat::zeros(height,width,CV_32FC1);
        grady   = Mat::zeros(height,width,CV_32FC1);

        /* Iteratively update the window position */
        do  {

            /* If out of bounds, exit loop */
            if (  x1-hw < 0.0f || nc-( x1+hw) < one_plus_eps ||
                *x2-hw < 0.0f || nc-(*x2+hw) < one_plus_eps ||
                y1-hh < 0.0f || nr-( y1+hh) < one_plus_eps ||
                *y2-hh < 0.0f || nr-(*y2+hh) < one_plus_eps) {
            status = KLT_OOB;
            break;
            }
            
            /* Compute gradient and difference windows */
            if (lighting_insensitive) {
                
            _computeIntensityDifferenceLightingInsensitive(img1, img2, x1, y1, *x2, *y2, 
                                        width, height, imgdiff);
            _computeGradientSumLightingInsensitive(gradx1, grady1, gradx2, grady2, 
                    img1, img2, x1, y1, *x2, *y2, width, height, gradx, grady);
            } else {
            _computeIntensityDifference(img1, img2, x1, y1, *x2, *y2, 
                                        width, height, imgdiff);
            _computeGradientSum(gradx1, grady1, gradx2, grady2, 
                    x1, y1, *x2, *y2, width, height, gradx, grady);
            }
                

            /* Use these windows to construct matrices */
            _compute2by2GradientMatrix(gradx, grady, width, height, 
                                    &gxx, &gxy, &gyy);
            _compute2by1ErrorVector(imgdiff, gradx, grady, width, height, step_factor,
                                    &ex, &ey);
                        
            /* Using matrices, solve equation for new displacement */
            status = _solveEquation(gxx, gxy, gyy, ex, ey, small, &dx, &dy);
            if (status == KLT_SMALL_DET)  break;

            *x2 += dx;
            *y2 += dy;
            iteration++;

        }  while ((fabs(dx)>=th || fabs(dy)>=th) && iteration < max_iterations);

        /* Check whether window is out of bounds */
        if (*x2-hw < 0.0f || nc-(*x2+hw) < one_plus_eps || 
            *y2-hh < 0.0f || nr-(*y2+hh) < one_plus_eps)
            status = KLT_OOB;

        /* Check whether residue is too large */
        if (status == KLT_TRACKED)  {
            if (lighting_insensitive)
            _computeIntensityDifferenceLightingInsensitive(img1, img2, x1, y1, *x2, *y2, 
                                        width, height, imgdiff);
            else
            _computeIntensityDifference(img1, img2, x1, y1, *x2, *y2, 
                                        width, height, imgdiff);
            if (_sumAbsFloatWindow(imgdiff, width, height)/(width*height) > max_residue) 
            status = KLT_LARGE_RESIDUE;
        }

        /* Free memory */
        //free(imgdiff);  free(gradx);  free(grady);

        /* Return appropriate value */
        if (status == KLT_SMALL_DET)  return KLT_SMALL_DET;
        else if (status == KLT_OOB)  return KLT_OOB;
        else if (status == KLT_LARGE_RESIDUE)  return KLT_LARGE_RESIDUE;
        else if (iteration >= max_iterations)  return KLT_MAX_ITERATIONS;
        else  return KLT_TRACKED;

    }
}