#include "cv.hpp"
#include <iostream>
//#include <Eigen/Dense>
#include "JRC.hpp"
#include <iostream>
#include <math.h>
//#include "eigen3/Eigen/Dense"
using namespace cv;
using namespace std;
namespace JRC{

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
void JointRadiometicCalib::JRCtrackFeatures(Mat prevImg, Mat currImg, vector<kltFeature> prevfl, vector<kltFeature> &currfl){
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
    int max_iterations= this->tracker.max_iterations;
    float small= this->tracker.min_determinant;         /* determinant threshold for declaring KLT_SMALL_DET */
    float th= this->tracker.min_displacement;            /* displacement threshold for stopping               */
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
    //JRC::JointRadiometicCalib jrc;
    bool JRCtrackingMode = this->trackingMode;
    Eigen::MatrixXd Uinv_all = Eigen::MatrixXd::Zero(nfeature*8,nfeature*8);
    Eigen::MatrixXd w_all = Eigen::MatrixXd::Zero(nfeature*8,nfeature*4);
    Eigen::VectorXd v_all = Eigen::MatrixXd::Zero(nfeature*8,nfeature*1);
    //Eigen::MatrixXd z_all = Eigen::MatrixXd::Zero(nfeature*8,nfeature*8);
    Eigen::MatrixXd lamda_all = Eigen::MatrixXd::Zero(4,4);
    Eigen::VectorXd m_all = Eigen::MatrixXd::Zero(4,1);
    int numOfTrackFeature=0;
    static unsigned int knownRFfirst=0;
    if (JRCtrackingMode == 1){
        knownRFfirst++;
    }
    if (knownRFfirst ==1) cal_g_and_g_();
    for (indx = 0 ; indx < nfeature ; indx++)  {
        
        /* Only track features that are not lost */
        if (prevfl[indx].val >= 0)  {
            if (prevfl[indx].used == false){
                    //initialization
                initialization(prevfl[indx]);
                prevfl[indx].used = true;
            }
            cout<<"0"<<endl;
            // 1.  create  U, w, lamda, v, m for all features
            constructMatrix(prevfl[indx],0, ksize, prevfl[indx].pt.x, prevfl[indx].pt.y, pyramid2[0], pyramid1[0] ,pyramid2_gradx[0], pyramid1_gradx[0],pyramid2_grady[0], pyramid1_grady[0],JRCtrackingMode);
            
            //for constructing all Matrix
            constructAllMatrix(prevfl[indx], numOfTrackFeature,Uinv_all,w_all,v_all,lamda_all,m_all);
            numOfTrackFeature++;
        }

    }
    // 2. calculate K by all featurs
    float K=0; //K is exposure time difference between two images.
    blockAllMatrix(numOfTrackFeature,Uinv_all,w_all,v_all,lamda_all,m_all,JRCtrackingMode);
    K= get_K(prevfl[indx],Uinv_all,w_all,v_all,lamda_all,m_all,JRCtrackingMode);
    
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
                val = _JRCtrackFeature(prevfl[indx], r, xloc, yloc, 
                    &xlocout, &ylocout,K, ksize, ksize, max_iterations, small,th,
                    pyramid2[r], 
                    pyramid2_gradx[r], pyramid2_grady[r],
                    pyramid1[r], 
                    pyramid1_gradx[r], pyramid1_grady[r], JRCtrackingMode);
                /*val = _trackFeature(xloc, yloc, 
                    &xlocout, &ylocout,
                    pyramid1[r], 
                    pyramid1_gradx[r], pyramid1_grady[r], 
                    pyramid2[r], 
                    pyramid2_gradx[r], pyramid2_grady[r]);*/

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
    if (JRCtrackingMode == 1)
        updateByKalmanFilter();

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


int JointRadiometicCalib::_JRCtrackFeature(kltFeature f, int pylevel, float x1, float y1, float *x2, float *y2,float K,
                int width, int height, int max_iterations, float small, float th,
                    Mat J_origin,Mat J_gradx,Mat J_grady,Mat I_origin,Mat I_gradx,Mat I_grady, bool trackingMode){
    //Mat imgdiff, gradx, grady;
    float gxx, gxy, gyy, ex, ey, dx, dy;
    int iteration = 0;
    int status;
    int hw = width/2;
    int hh = height/2;
    int nc = J_origin.cols;
    int nr = J_origin.rows;
    float one_plus_eps = 1.001f;   /* To prevent rounding errors */
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
        if(pylevel!=0)
            constructMatrix(f,pylevel,width,x1,y1,J_origin,I_origin,J_gradx,I_gradx,J_grady,I_grady,trackingMode);
                    
        /* Using matrices, solve equation for new displacement */
        status = solveEquation(f, pylevel, K,&dx, &dy, small,trackingMode);
        if (status == KLT_SMALL_DET)  break;
        
        *x2 += dx;
        *y2 += dy;
        iteration++;

    }  while ((fabs(dx)>=th || fabs(dy)>=th) && iteration < max_iterations);

    /* Check whether window is out of bounds */
    if (*x2-hw < 0.0f || nc-(*x2+hw) < one_plus_eps || 
        *y2-hh < 0.0f || nr-(*y2+hh) < one_plus_eps)
        status = KLT_OOB;

    /* Check whether residue is too large 
    if (status == KLT_TRACKED)  {
        _computeIntensityDifference(img1, img2, x1, y1, *x2, *y2, 
                                    width, height, imgdiff);
        if (_sumAbsFloatWindow(imgdiff, width, height)/(width*height) > max_residue) 
        status = KLT_LARGE_RESIDUE;
    }*/

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