// A simple program that computes the square root of a number
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <iostream>
#include "SSDConfig.h"
#include "cv.hpp"
#include "klt/klt.h"
//#include "klt/kltFeature.h"
//#include "pnmio.hpp"
//#include "klt.h" //here, all of implementation are included

#include <vector>

using namespace cv;
using namespace std;
using namespace klt;
/// Function header
#define KLT

#ifdef KLT
bool initialization = false;
void DrawTrackingPoints(vector<kltFeature> featurelist, Mat &image){
  /// Draw corners detected
    for(int i = 0; i < featurelist.size(); i++){ 
      if (featurelist[i].status == 0){
        int x= cvRound(featurelist[i].pt.x);
        int y= cvRound(featurelist[i].pt.y);
        circle( image, Point(x,y), 3, Scalar(0,0,255),2 );
      }else if  (featurelist[i].status == 1) {
        int x= cvRound(featurelist[i].pt.x);
        int y= cvRound(featurelist[i].pt.y);
        circle( image, Point(x,y), 3, Scalar(255,0,0),2 );
      }
    }

}
int main (int argc, char *argv[])
{
  VideoCapture cap(0);
  if(!cap.isOpened()){
    cout<<"Cannot open cap"<<endl;
    return 0;
  }
  Size size = Size((int)cap.get(CV_CAP_PROP_FRAME_WIDTH),(int)cap.get(CV_CAP_PROP_FRAME_HEIGHT));
  double fps = cap.get(CV_CAP_PROP_FPS);
  Mat currImg, prevImg;
  Mat frame,dstImg;
  namedWindow("distImage");
  double qualityLevel = 0.01;
  double minDistance = 10;
  int blockSize = 3;
  bool useHarrisDetector = false;
  double k = 0.04;
  int maxCorners = 500;

  KLTtracker kltTracker(maxCorners);
  kltTracker.tracker.sequentialMode = true;
 
  //TermCriteria criteria=TermCriteria(TermCriteria::COUNT+TermCriteria::EPS,10,0.01);
  Size winSize(11,11);

  //vector<Point2f> prevPoints;
  //vector<Point2f> currPoints;
  //vector<Point2f> boundPoints;
  
  int delay= 1000/fps;

  int nframe=0;
  //KLT klt = new KLT();
  
  vector<kltFeature> prevFeaturelist;
  vector<kltFeature> currFeaturelist;
  prevFeaturelist.reserve(maxCorners);
  currFeaturelist.reserve(maxCorners);
  for(;;){
    
    cap>>frame;
    if(frame.empty()) break;
    frame.copyTo(dstImg);
    /// Copy the source image

    cvtColor(dstImg,currImg,CV_BGR2GRAY);
    //GaussianBlur(currImage,currImage,Size(5,5),0.5);
    
    //feature detection
    if(initialization){
      prevFeaturelist.clear();
      currFeaturelist.clear();
      kltTracker.selectGoodFeatures(currImg,currFeaturelist);
      DrawTrackingPoints(currFeaturelist,dstImg);
      initialization = false;
    }

    if(!initialization &&prevFeaturelist.size()>0){
      kltTracker.trackFeatures(prevImg,currImg,prevFeaturelist,currFeaturelist);
      //kltTracker.replaceLostFeatures(currImg, currFeaturelist);
      //cout<<currFeaturelist[0].val<<endl;
      DrawTrackingPoints(currFeaturelist,dstImg);
    }
    /// Show what you got 
    //namedWindow( source_window, CV_WINDOW_AUTOSIZE );
    imshow( "KLT dstImage", dstImg );
    //imshow("frame",frame);
    currImg.copyTo(prevImg);
    prevFeaturelist.clear();
    prevFeaturelist = currFeaturelist;
    int ch = waitKey(delay);
    if(ch == 27) break;       // 27 == ESC key
    if (ch ==32) initialization = true;
    
  }
  return 0;
}
#endif


#ifdef opencv

struct feature{
  Point2f pt;
  int val;
};
bool initialization = false;
void DrawTrackingPoints(vector<Point2f> &points, Mat &image){
  /// Draw corners detected
    for(int i = 0; i < points.size(); i++){ 
      int x= cvRound(points[i].x);
      int y= cvRound(points[i].y);
      circle( image, Point(x,y), 3, Scalar(255,0,0),2 );}

}
int main (int argc, char *argv[])
{
  VideoCapture cap(0);
  if(!cap.isOpened()){
    cout<<"Cannot open cap"<<endl;
    return 0;
  }
  Size size = Size((int)cap.get(CV_CAP_PROP_FRAME_WIDTH),(int)cap.get(CV_CAP_PROP_FRAME_HEIGHT));
  double fps = cap.get(CV_CAP_PROP_FPS);
  Mat currImage, prevImage;
  Mat frame,dstImage;
  namedWindow("distImage");
  double qualityLevel = 0.01;
  double minDistance = 10;
  int blockSize = 3;
  bool useHarrisDetector = false;
  double k = 0.04;
  int maxCorners = 500;

  TermCriteria criteria=TermCriteria(TermCriteria::COUNT+TermCriteria::EPS,10,0.01);
  Size winSize(11,11);

  vector<Point2f> prevPoints;
  vector<Point2f> currPoints;
  vector<Point2f> boundPoints;

  int delay= 1000/fps;

  int nframe=0;
  //KLT klt = new KLT();
  for(;;){
    cap>>frame;
    if(frame.empty()) break;
    frame.copyTo(dstImage);
    /// Copy the source image

    cvtColor(dstImage,currImage,CV_BGR2GRAY);
    GaussianBlur(currImage,currImage,Size(5,5),0.5);

    //feature detection
    if(initialization){
      goodFeaturesToTrack( prevImage,prevPoints, maxCorners, qualityLevel, minDistance, Mat(), blockSize, useHarrisDetector, k );
      cornerSubPix(prevImage,prevPoints,winSize,Size(-1,-1),criteria);
      DrawTrackingPoints(prevPoints,dstImage);
      initialization = false;
    }
    if(prevPoints.size()>0){
      vector<Mat> prevPyr, currPyr;
      Mat status,err;
      buildOpticalFlowPyramid(prevImage,prevPyr,winSize,3,true);
      buildOpticalFlowPyramid(currImage,currPyr,winSize,3,true);
      //currPoints = prevPoints; //for OPTFLOW_USE_INITIAL_FLOW
      calcOpticalFlowPyrLK(prevPyr,currPyr,prevPoints,currPoints,status,err,winSize);

      //delete invalid correspondinig points
      /*for(int i=0;i<prevPoints.size();i++){
        if(!status.at<uchar>(i)){
          prevPoints.erase(prevPoints.begin()+i);
          currPoints.erase(currPoints.begin()+i);
        }
      }*/
      //cornerSubPix(currImage,currPoints,winSize,Size(-1,-1),criteria);
      DrawTrackingPoints(currPoints,dstImage);

      prevPoints = currPoints;
    }
    /// Show what you got 
    //namedWindow( source_window, CV_WINDOW_AUTOSIZE );
    imshow( "KLT dstImage", dstImage );
    //imshow("frame",frame);
    currImage.copyTo(prevImage);

    int ch = waitKey(delay);
    if(ch == 27) break;       // 27 == ESC key
    if (ch ==32) initialization = true;
    
  }
  return 0;
}

#endif