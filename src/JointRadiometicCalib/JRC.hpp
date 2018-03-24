#include "cv.hpp"
#include <iostream>
#include <Eigen/Dense>
using namespace cv;

namespace JRC{

class JRC{
public:
//for unknown and known Response Funtion
float a;
float b;
float beta;
float c[3];
float g0[10000];
float h[3]10000];
float g0_[10000];
float h_[3]][10000];
//float log_inv_RF[];
//float grad_log_inv_RF[];
//for unknown Response Function
float r[3];
float p[3];
float q[3];
float d;

JRC(){}
~JRC(){}
void get_log_inv_RF(float RF[]);
void get_differential(float log_inv_RF[], float x);
void get_a(Mat J_gradx, Mat I_gradx, float grad_log_inv_RF[], Mat J_origin, Mat I_origin){}
void get_b(Mat J_gradY, Mat I_gradY, float grad_log_inv_RF[], Mat J_origin, Mat I_origin){}
void get_beta(Mat J_origin, Mat I_origin, float log_inv_RF[]){}


void constructMatrix();
void solveEquation();
void knownRF();
void unknownRF();
void updateRFusingKF();
void decomposition();

_computeIntensityDifference(img1, img2, x1, y1, *x2, *y2, 
                                        width, height, imgdiff);

_computeGradientSum(gradx1, grady1, gradx2, grady2, 
                    x1, y1, *x2, *y2, width, height, gradx, grady);
            
                

/* Use these windows to construct matrices */
_compute2by2GradientMatrix(gradx, grady, width, height, 
                                    &gxx, &gxy, &gyy);
_compute2by1ErrorVector(imgdiff, gradx, grady, width, height, step_factor,
                                    &ex, &ey);
                        
            /* Using matrices, solve equation for new displacement */
_solveEquation(gxx, gxy, gyy, ex, ey, small, &dx, &dy);

};

}