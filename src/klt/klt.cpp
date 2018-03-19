#include <iostream>
#include "cv.hpp"
#include "klt.h"

using namespace cv;

namespace klt{
    typedef enum {SELECTING_ALL, REPLACING_SOME} selectionMode;
    /*********************************************************************
     * _sortPointList
     */
    #define SWAP3(list, i, j)               \
    {   int *pi, *pj, tmp;            \
        pi=list+3*(i); pj=list+3*(j);      \
                                            \
        tmp=*pi;    \
        *pi++=*pj;  \
        *pj++=tmp;  \
                    \
        tmp=*pi;    \
        *pi++=*pj;  \
        *pj++=tmp;  \
                    \
        tmp=*pi;    \
        *pi=*pj;    \
        *pj=tmp;    \
    }

    void _quicksort(int *pointlist, int n){
    unsigned int i, j, ln, rn;

    while (n > 1)
    {
        SWAP3(pointlist, 0, n/2);
        for (i = 0, j = n; ; )
        {
        do
            --j;
        while (pointlist[3*j+2] < pointlist[2]);
        do
            ++i;
        while (i < j && pointlist[3*i+2] > pointlist[2]);
        if (i >= j)
            break;
        SWAP3(pointlist, i, j);
        }
        SWAP3(pointlist, j, 0);
        ln = j;
        rn = n - ++j;
        if (ln < rn)
        {
        _quicksort(pointlist, ln);
        pointlist += 3*j;
        n = rn;
        }
        else
        {
        _quicksort(pointlist + 3*j, rn);
        n = ln;
        }
    }
    }
    #undef SWAP3

    static void _sortPointList(
    int *pointlist,
    int npoints)
    {
    #ifdef KLT_USE_QSORT
    qsort(pointlist, npoints, 3*sizeof(int), _comparePoints);
    #else
    _quicksort(pointlist, npoints);
    #endif
    }
    int KLTCountRemainingFeatures(int nfeatures, vector<kltFeature> fl){
        int count = 0;
        for (int i = 0 ; i < nfeatures ; i++)
            if (fl[i].val >= 0)
            count++;

        return count;
    }
    static float _minEigenvalue(float gxx, float gxy, float gyy)
    {
    return (float) ((gxx + gyy - sqrt((gxx - gyy)*(gxx - gyy) + 4*gxy*gxy))/2.0f);
    }
    void KLTtracker::_enforceMinimumDistance(int *pointlist, int npoints,vector<kltFeature> &fl,bool overwriteAllFeatures){
        
    }
    void KLTtracker::_KLTSelectGoodFeatures(Mat Img, vector<kltFeature> &fl, selectionMode mode){
        Mat floatimg, gradx, grady;
        int window_hw, window_hh;
        int *pointlist;
        int npoints = 0;
        bool overwriteAllFeatures = (mode == SELECTING_ALL) ? true : false;
        bool floatimages_created = false;
        window_hw = this->tracker.window_width/2; 
        window_hh = this->tracker.window_height/2;
        int nrows = floatimg.rows;
        int ncols = floatimg.cols;
        /* Create pointlist, which is a simplified version of a featurelist, */
         /* for speed.  Contains only integer locations and values. */
        //vector<int> pointlist;
        pointlist = (int *) malloc(ncols * nrows * 3 * sizeof(int));

        if (mode == REPLACING_SOME && this->tracker.sequentialMode && !this->currPyr.empty())  {
            floatimg = this->currPyr[0];
            gradx = this->currGradx[0];
            grady = this->currGrady[0];
            assert(!gradx.empty());
            assert(!grady.empty());
        } else  {
            floatimages_created = TRUE;
            //floatimg = _KLTCreateFloatImage(ncols, nrows);
            //gradx    = _KLTCreateFloatImage(ncols, nrows);
            //grady    = _KLTCreateFloatImage(ncols, nrows);
            if (this->tracker.smoothBeforeSelecting)  {
            //_KLT_FloatImage tmpimg;
            //tmpimg = _KLTCreateFloatImage(ncols, nrows);
            //_KLTToFloatImage(img, ncols, nrows, tmpimg);
            int ksize = this->tracker.window_width;
            float sigma = (this->tracker.smooth_sigma_fact * max(this->tracker.window_width, this->tracker.window_height));
            GaussianBlur(Img,floatimg,Size(ksize,ksize),sigma);
            //_KLTComputeSmoothedImage(tmpimg, _KLTComputeSmoothSigma(tc), floatimg);
            //_KLTFreeFloatImage(tmpimg);
            } else
                //KLTToFloatImage(img, ncols, nrows, floatimg);
                floatimg = Img;
            
            /* Compute gradient of image in x and y direction */
            //_KLTComputeGradients(floatimg, tc->grad_sigma, gradx, grady);
            Sobel(floatimg, gradx,-1,1,0,3);
            Sobel(floatimg, grady,-1,0,1,3);

            {
            float gx, gy;
            float gxx, gxy, gyy;
            int xx, yy;
            int *ptr;
            float val;
            unsigned int limit = 1;
            int borderx = this->tracker.borderx;	/* Must not touch cols */
            int bordery = this->tracker.bordery;	/* lost by convolution */
            int x, y;
            int i;
            
            if (borderx < window_hw)  borderx = window_hw;
            if (bordery < window_hh)  bordery = window_hh;

            /* Find largest value of an int */
            limit = 2147483647;
                
            /* For most of the pixels in the image, do ... */
            ptr = pointlist;
            
            for (y = bordery ; y < nrows - bordery ; y += this->tracker.nSkippedPixels + 1)
            for (x = borderx ; x < ncols - borderx ; x += this->tracker.nSkippedPixels + 1)  {

                /* Sum the gradients in the surrounding window */
                gxx = 0;  gxy = 0;  gyy = 0;
                for (yy = y-window_hh ; yy <= y+window_hh ; yy++)
                for (xx = x-window_hw ; xx <= x+window_hw ; xx++)  {
                    gx = *(gradx.data + ncols*yy+xx);
                    gy = *(grady.data + ncols*yy+xx);
                    gxx += gx * gx;
                    gxy += gx * gy;
                    gyy += gy * gy;
                }

                /* Store the trackability of the pixel as the minimum
                of the two eigenvalues */
                *ptr++ = x;
                *ptr++ = y;
                val = _minEigenvalue(gxx, gxy, gyy);
                if (val > limit)  {
                std::cout<<"(_KLTSelectGoodFeatures) minimum eigenvalueisgreater than the capacity of an int; setting "<<std::endl;
                val = (float) limit;
                }
                *ptr++ = (int) val;
                npoints++;
            }
        }
          /* Sort the features  */
        _sortPointList(pointlist, npoints);
        /* Enforce minimum distance between features */
        _enforceMinimumDistance(pointlist,npoints,fl,overwriteAllFeatures);
        /* Free memory */
        free(pointlist);
        }
    }
    void KLTtracker::selectGoodFeatures(Mat Img, vector<kltFeature> &fl){
        fprintf(stderr,  "(KLT) Selecting the %d best features ",this->nfeatures);
        fflush(stderr);
        _KLTSelectGoodFeatures(Img, fl, SELECTING_ALL);
    }
    void KLTtracker::trackFeatures(Mat currImg, Mat prevImg, vector<kltFeature> currfl, vector<kltFeature> &prevfl){
  
    }
    void KLTtracker::replaceLostFeatures(Mat Img, vector<kltFeature> &fl){
        int nLostFeatures = this->nfeatures - KLTCountRemainingFeatures(this->nfeatures, fl);

        fprintf(stderr,  "(KLT) Attempting to replace %d features ", nLostFeatures);
        fflush(stderr);
        /* If there are any lost features, replace them */
        if (nLostFeatures > 0)
            _KLTSelectGoodFeatures(Img, fl, REPLACING_SOME);
    }


}