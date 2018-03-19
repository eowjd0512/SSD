#include <iostream>
#include "cv.hpp"
#include "klt.h"
#include <vector>
using namespace std;
using namespace cv;

namespace klt{
    vector<Mat> _KLTCreatePyramid(Mat floatimg1, int subsampling, int MaxPyLevel){
        vector<Mat> pyr;
        Mat pyramid[MaxPyLevel];
        pyramid[0]=floatimg1;
        pyr.push_back(pyramid[0]);
        for(int i=0; i<MaxPyLevel-1;i++){
            pyrDown(pyramid[i],pyramid[i+1]);
            pyr.push_back(pyramid[i+1]);
        }
        return pyr;
    }
    vector<Mat> _KLTComputeGradients(vector<Mat> pyramid1, int x, int y){
        vector<Mat> grad;
        for (int i=0; i<pyramid1.size();i++){
            Mat grad_;
            Sobel(pyramid1[i], grad_,-1,x,y,3);
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
        
    }
    void KLTtracker::trackFeatures(Mat prevImg, Mat currImg, vector<kltFeature> prevfl, vector<kltFeature> &currfl){
        Mat tmpimg, floatimg1, floatimg2;
        int MaxPyLevel = this->tracker.nPyramidLevels;
        vector<Mat> pyramid1, pyramid1_gradx, pyramid1_grady,
            pyramid2, pyramid2_gradx, pyramid2_grady;
        //float subsampling = (float) this->tracker.subsampling;
        float subsampling =2.0;
        float xloc, yloc, xlocout, ylocout;
        int val;
        int indx, r;
        bool floatimg1_created = false;
        int i;
        /*
        if (KLT_verbose >= 1)  {
            fprintf(stderr,  "(KLT) Tracking %d features in a %d by %d image...  ",
                KLTCountRemainingFeatures(featurelist), ncols, nrows);
            fflush(stderr);
        }*/

        /* Check window size (and correct if necessary) */
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
        }

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
            Mat floatimg1;
            int ksize = this->tracker.window_width;
            float sigma = (this->tracker.smooth_sigma_fact * max(this->tracker.window_width, this->tracker.window_height));
            GaussianBlur(prevImg,floatimg1,Size(ksize,ksize),sigma);

            pyramid1 = _KLTCreatePyramid(floatimg1, (int) subsampling, MaxPyLevel);
            //_KLTComputePyramid(floatimg1, pyramid1, tc->pyramid_sigma_fact);
            pyramid1_gradx = _KLTComputeGradients(pyramid1, 1,0);
            pyramid1_grady = _KLTComputeGradients(pyramid1, 0,1);
        }

        /* Do the same thing with second image */
        Mat floatimg2;
        int ksize = this->tracker.window_width;
        float sigma = (this->tracker.smooth_sigma_fact * max(this->tracker.window_width, this->tracker.window_height));
        GaussianBlur(currImg,floatimg1,Size(ksize,ksize),sigma);
        pyramid2 = _KLTCreatePyramid(floatimg2, (int) subsampling, MaxPyLevel);
        //_KLTComputePyramid(floatimg1, pyramid1, tc->pyramid_sigma_fact);
        pyramid2_gradx = _KLTComputeGradients(pyramid2, 1,0);
        pyramid2_grady = _KLTComputeGradients(pyramid2, 0,1);

        /* Write internal images */
        /*if (tc->writeInternalImages)  {
            char fname[80];
            for (i = 0 ; i < tc->nPyramidLevels ; i++)  {
                sprintf(fname, "kltimg_tf_i%d.pgm", i);
                _KLTWriteFloatImageToPGM(pyramid1->img[i], fname);
                sprintf(fname, "kltimg_tf_i%d_gx.pgm", i);
                _KLTWriteFloatImageToPGM(pyramid1_gradx->img[i], fname);
                sprintf(fname, "kltimg_tf_i%d_gy.pgm", i);
                _KLTWriteFloatImageToPGM(pyramid1_grady->img[i], fname);
                sprintf(fname, "kltimg_tf_j%d.pgm", i);
                _KLTWriteFloatImageToPGM(pyramid2->img[i], fname);
                sprintf(fname, "kltimg_tf_j%d_gx.pgm", i);
                _KLTWriteFloatImageToPGM(pyramid2_gradx->img[i], fname);
                sprintf(fname, "kltimg_tf_j%d_gy.pgm", i);
                _KLTWriteFloatImageToPGM(pyramid2_grady->img[i], fname);
            }
        }*/

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

                /* Record feature */
                if (val == KLT_OOB) {
                    featurelist->feature[indx]->x   = -1.0;
                    featurelist->feature[indx]->y   = -1.0;
                    featurelist->feature[indx]->val = KLT_OOB;
                    if( featurelist->feature[indx]->aff_img ) _KLTFreeFloatImage(featurelist->feature[indx]->aff_img);
                    if( featurelist->feature[indx]->aff_img_gradx ) _KLTFreeFloatImage(featurelist->feature[indx]->aff_img_gradx);
                    if( featurelist->feature[indx]->aff_img_grady ) _KLTFreeFloatImage(featurelist->feature[indx]->aff_img_grady);
                    featurelist->feature[indx]->aff_img = NULL;
                    featurelist->feature[indx]->aff_img_gradx = NULL;
                    featurelist->feature[indx]->aff_img_grady = NULL;

                } else if (_outOfBounds(xlocout, ylocout, ncols, nrows, tc->borderx, tc->bordery))  {
                    featurelist->feature[indx]->x   = -1.0;
                    featurelist->feature[indx]->y   = -1.0;
                    featurelist->feature[indx]->val = KLT_OOB;
                    if( featurelist->feature[indx]->aff_img ) _KLTFreeFloatImage(featurelist->feature[indx]->aff_img);
                    if( featurelist->feature[indx]->aff_img_gradx ) _KLTFreeFloatImage(featurelist->feature[indx]->aff_img_gradx);
                    if( featurelist->feature[indx]->aff_img_grady ) _KLTFreeFloatImage(featurelist->feature[indx]->aff_img_grady);
                    featurelist->feature[indx]->aff_img = NULL;
                    featurelist->feature[indx]->aff_img_gradx = NULL;
                    featurelist->feature[indx]->aff_img_grady = NULL;
                } else if (val == KLT_SMALL_DET)  {
                    featurelist->feature[indx]->x   = -1.0;
                    featurelist->feature[indx]->y   = -1.0;
                    featurelist->feature[indx]->val = KLT_SMALL_DET;
                    if( featurelist->feature[indx]->aff_img ) _KLTFreeFloatImage(featurelist->feature[indx]->aff_img);
                    if( featurelist->feature[indx]->aff_img_gradx ) _KLTFreeFloatImage(featurelist->feature[indx]->aff_img_gradx);
                    if( featurelist->feature[indx]->aff_img_grady ) _KLTFreeFloatImage(featurelist->feature[indx]->aff_img_grady);
                    featurelist->feature[indx]->aff_img = NULL;
                    featurelist->feature[indx]->aff_img_gradx = NULL;
                    featurelist->feature[indx]->aff_img_grady = NULL;
                } else if (val == KLT_LARGE_RESIDUE)  {
                    featurelist->feature[indx]->x   = -1.0;
                    featurelist->feature[indx]->y   = -1.0;
                    featurelist->feature[indx]->val = KLT_LARGE_RESIDUE;
                    if( featurelist->feature[indx]->aff_img ) _KLTFreeFloatImage(featurelist->feature[indx]->aff_img);
                    if( featurelist->feature[indx]->aff_img_gradx ) _KLTFreeFloatImage(featurelist->feature[indx]->aff_img_gradx);
                    if( featurelist->feature[indx]->aff_img_grady ) _KLTFreeFloatImage(featurelist->feature[indx]->aff_img_grady);
                    featurelist->feature[indx]->aff_img = NULL;
                    featurelist->feature[indx]->aff_img_gradx = NULL;
                    featurelist->feature[indx]->aff_img_grady = NULL;
                } else if (val == KLT_MAX_ITERATIONS)  {
                    featurelist->feature[indx]->x   = -1.0;
                    featurelist->feature[indx]->y   = -1.0;
                    featurelist->feature[indx]->val = KLT_MAX_ITERATIONS;
                    if( featurelist->feature[indx]->aff_img ) _KLTFreeFloatImage(featurelist->feature[indx]->aff_img);
                    if( featurelist->feature[indx]->aff_img_gradx ) _KLTFreeFloatImage(featurelist->feature[indx]->aff_img_gradx);
                    if( featurelist->feature[indx]->aff_img_grady ) _KLTFreeFloatImage(featurelist->feature[indx]->aff_img_grady);
                    featurelist->feature[indx]->aff_img = NULL;
                    featurelist->feature[indx]->aff_img_gradx = NULL;
                    featurelist->feature[indx]->aff_img_grady = NULL;
                } else  {
                    featurelist->feature[indx]->x = xlocout;
                    featurelist->feature[indx]->y = ylocout;
                    featurelist->feature[indx]->val = KLT_TRACKED;
                    if (tc->affineConsistencyCheck >= 0 && val == KLT_TRACKED)  { /*for affine mapping*/
                        int border = 2; /* add border for interpolation */

    #ifdef DEBUG_AFFINE_MAPPING	  
                        glob_index = indx;
    #endif

                        if(!featurelist->feature[indx]->aff_img){
                            /* save image and gradient for each feature at finest resolution after first successful track */
                            featurelist->feature[indx]->aff_img = _KLTCreateFloatImage((tc->affine_window_width+border), (tc->affine_window_height+border));
                            featurelist->feature[indx]->aff_img_gradx = _KLTCreateFloatImage((tc->affine_window_width+border), (tc->affine_window_height+border));
                            featurelist->feature[indx]->aff_img_grady = _KLTCreateFloatImage((tc->affine_window_width+border), (tc->affine_window_height+border));
                            _am_getSubFloatImage(pyramid1->img[0],xloc,yloc,featurelist->feature[indx]->aff_img);
                            _am_getSubFloatImage(pyramid1_gradx->img[0],xloc,yloc,featurelist->feature[indx]->aff_img_gradx);
                            _am_getSubFloatImage(pyramid1_grady->img[0],xloc,yloc,featurelist->feature[indx]->aff_img_grady);
                            featurelist->feature[indx]->aff_x = xloc - (int) xloc + (tc->affine_window_width+border)/2;
                            featurelist->feature[indx]->aff_y = yloc - (int) yloc + (tc->affine_window_height+border)/2;;
                        }else{
                            /* affine tracking */
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
                                /* free image and gradient for lost feature */
                                _KLTFreeFloatImage(featurelist->feature[indx]->aff_img);
                                _KLTFreeFloatImage(featurelist->feature[indx]->aff_img_gradx);
                                _KLTFreeFloatImage(featurelist->feature[indx]->aff_img_grady);
                                featurelist->feature[indx]->aff_img = NULL;
                                featurelist->feature[indx]->aff_img_gradx = NULL;
                                featurelist->feature[indx]->aff_img_grady = NULL;
                            }else{
                                /*featurelist->feature[indx]->x = xlocout;*/
                                /*featurelist->feature[indx]->y = ylocout;*/
                            }
                        }
                    }

                }
            }
        }

        if (tc->sequentialMode)  {
            tc->pyramid_last = pyramid2;
            tc->pyramid_last_gradx = pyramid2_gradx;
            tc->pyramid_last_grady = pyramid2_grady;
        } else  {
            _KLTFreePyramid(pyramid2);
            _KLTFreePyramid(pyramid2_gradx);
            _KLTFreePyramid(pyramid2_grady);
        }

        /* Free memory */
        _KLTFreeFloatImage(tmpimg);
        if (floatimg1_created)  _KLTFreeFloatImage(floatimg1);
        _KLTFreeFloatImage(floatimg2);
        _KLTFreePyramid(pyramid1);
        _KLTFreePyramid(pyramid1_gradx);
        _KLTFreePyramid(pyramid1_grady);

        if (KLT_verbose >= 1)  {
            fprintf(stderr,  "\n\t%d features successfully tracked.\n",
                KLTCountRemainingFeatures(featurelist));
            if (tc->writeInternalImages)
                fprintf(stderr,  "\tWrote images to 'kltimg_tf*.pgm'.\n");
            fflush(stderr);
        }
        
    }
}