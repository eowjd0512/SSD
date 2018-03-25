#include "cv.hpp"
#include <iostream>
//#include <Eigen3>
#include <iostream>
#include <string>
using namespace std;
using namespace cv;

namespace JRC{

class JointRadiometicCalib{
public:
int M; //order
//for unknown and known Response Funtion

float c[3];
float K;
float B[1024];
float g0[1024]; //log-inverse of f
float h[3][1024];
float g0_[1024]; //derivative for g0
float h_[3][1024]; //derivative for h
//float log_inv_RF[];
//float grad_log_inv_RF[];

bool trackingMode; //0 : knownRF, 1: unknownRF
JointRadiometicCalib(){setRFs(); trackingMode = 1;}
~JointRadiometicCalib(){}

void setRFs();
void setRFderivatives();
float getRF_Value(int idx);
float getRF_DerivValue(int idx);
float get_a(int x,int y,float g0_[], Mat J_origin, Mat I_origin,Mat J_gradx, Mat I_gradx);
float get_b(int x,int y,float g0_[], Mat J_origin, Mat I_origin,Mat J_gradY, Mat I_gradY);
float get_beta(int x, int y, Mat J_origin, Mat I_origin);
float get_r_k(int x, int y, float h[], Mat J_origin, Mat I_origin);
float get_p_k(int x, int y, float h_[], Mat J_origin, Mat I_origin,Mat J_gradx, Mat I_gradx);
float get_q_k(int x, int y, float h_[], Mat J_origin, Mat I_origin,Mat J_grady, Mat I_grady);
float get_d(int x, int y, float g0[],Mat J_origin, Mat I_origin);

void constructMatrix();
void solveEquation();
void decomposition();
int trackingKnownRF(float x1, float y1, float *x2, float *y2,Mat img1,Mat gradx1,Mat grady1,Mat img2,Mat gradx2,Mat grady2);
int trackingUnknownRF(float x1, float y1, float *x2, float *y2, Mat img1,Mat gradx1,Mat grady1,Mat img2,Mat gradx2,Mat grady2);
void updateRF_usingKalmanFilter();

/*
_computeIntensityDifference(img1, img2, x1, y1, *x2, *y2, 
                                        width, height, imgdiff);

_computeGradientSum(gradx1, grady1, gradx2, grady2, 
                    x1, y1, *x2, *y2, width, height, gradx, grady);
            

_compute2by2GradientMatrix(gradx, grady, width, height, 
                                    &gxx, &gxy, &gyy);
_compute2by1ErrorVector(imgdiff, gradx, grady, width, height, step_factor,

_solveEquation(gxx, gxy, gyy, ex, ey, small, &dx, &dy);
*/
};

}