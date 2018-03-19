#include <iostream>
#include "cv.hpp"

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

    kltFeature(){}
    ~kltFeature(){}
};

};