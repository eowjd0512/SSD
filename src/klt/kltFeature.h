#pragma once
#include <iostream>
#include "cv.hpp"
#include "convolve.hpp"
#include "eigen3/Eigen/Dense"
#include <vector>
//#include "../JointRadiometicCalib/JRC.hpp"
using namespace cv;
using namespace std;
namespace klt{
class kltFeature{
    public:

    Point2f pt;
    int val;	
    /* for affine mapping */
    Mat aff_img; 
    Mat aff_img_gradx;
    Mat aff_img_grady;
    float aff_x;
    float aff_y;
    float aff_Axx;
    float aff_Ayx;
    float aff_Axy;
    float aff_Ayy;
    int status;
    bool used;
    //for JRC

    //vector for multi pyramid
 
    //for unknown Response Function
    vector<Eigen::MatrixXf> U;
    vector<Eigen::MatrixXf> w;
    vector<Eigen::VectorXf> v;
    vector<Eigen::VectorXf> z;
    vector<Eigen::MatrixXf> lamda;
    vector<Eigen::VectorXf> m;
    
    kltFeature(){ used = false;}
    ~kltFeature(){}
};

};