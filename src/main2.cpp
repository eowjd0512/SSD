// A simple program that computes the square root of a number
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <iostream>
#include "SSDConfig.h"
#include "cv.hpp"
#include "pnmio.hpp"
#include "klt.h" //here, all of implementation are included

#include <vector>

using namespace cv;
using namespace std;
RNG rng(12345);
/// Function header
int main (int argc, char *argv[])
{
  VideoCapture cap(1);
  double fps = cap.get(CV_CAP_PROP_FPS);
  KLT_TrackingContext tc;
  KLT_FeatureList flt;
  KLT_FeatureTable ft;
  int nFeatures = 150, nFrames = 1000;
  int ncols, nrows;
  tc = KLTCreateTrackingContext();
  flt = KLTCreateFeatureList(nFeatures);

  //creates a feature table, given the number of frames and the number of features to store. Although in this example the number of frames is the same as the total number of actual images, this does not have to be the case if all the features do not need to be stored.
  ft = KLTCreateFeatureTable(nFrames, nFeatures);
  tc->sequentialMode = TRUE;
  tc->writeInternalImages = FALSE;
  tc->affineConsistencyCheck = -1;  /* set this to 2 to turn on affine consistency check */
 
  int i=0;
  Mat src, prev_src;
  unsigned char *img, *prev_img;    
  for(int k=0;k<1000;k++){
     
      cap>>src;
      Mat draw;
      src.copyTo(draw);
      /// Copy the source image
      Mat src_gray;
      cvtColor(src,src_gray,CV_BGR2GRAY);

      img = readMat(src_gray, NULL, &ncols, &nrows);
      prev_img = readMat(prev_src, NULL, &ncols, &nrows);
      if (i==0){
        KLTSelectGoodFeatures(tc, img, ncols, nrows, flt);
        KLTStoreFeatureList(flt, ft, 0);
        //draw features
        //KLTWriteFeatureListToPPM(fl, img1, ncols, nrows, "feat0.ppm");


      }else{

        KLTTrackFeatures(tc, img, prev_img, ncols, nrows, flt);
        KLTReplaceLostFeatures(tc, img, ncols, nrows, flt);
        KLTStoreFeatureList(flt, ft, i);
        //draw features
        //KLTWriteFeatureListToPPM(fl, img1, ncols, nrows, "feat0.ppm");

      }
      /// Draw corners detected
      int r = 4;
      for( int j = 0; j < (flt->nFeatures); j++ ){ 
        Point p = Point(int(flt->feature[j]->x+0.5),int(flt->feature[j]->y+0.5));
        circle( draw, p, r, Scalar(rng.uniform(0,255), rng.uniform(0,255),rng.uniform(0,255)), -1, 8, 0 ); }

      /// Show what you got
      //namedWindow( source_window, CV_WINDOW_AUTOSIZE );
      imshow( "KLT track", draw );
      //imshow("frame",frame);
      i++;
      prev_src = src_gray;
      char ch = waitKey(fps/1000);
      if(ch == 27) break;       // 27 == ESC key
      if(ch == 32)                // 32 == SPACE key
      {
      while((ch = waitKey(10)) != 32 && ch != 27);
      if(ch == 27) break;
      }
  }


  //writes a feature table to a file, in a manner similar to that of KLTWriteFeatureList(), which was described in Chapter 2.
  //KLTWriteFeatureTable(ft, "features.txt", "%5.1f");
  //KLTWriteFeatureTable(ft, "features.ft", NULL);

  KLTFreeFeatureTable(ft);
  KLTFreeFeatureList(flt);
  KLTFreeTrackingContext(tc);

  return 0;
}
