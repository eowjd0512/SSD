#include <iostream>
#include "cv.hpp"
#include "convolve.hpp"
using namespace cv;

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

    //for JRC
    float a;
    float b;
    float beta;
    //for unknown Response Function
    float r[3];
    float p[3];
    float q[3];
    float d;
    
    kltFeature(){}
    ~kltFeature(){}
};

};