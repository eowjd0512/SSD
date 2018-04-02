#pragma once
#include "cv.hpp"
#include <iostream>
#include "eigen3/Eigen/Dense"
#include <iostream>
#include <string>
#include "../klt/klt.h"

using namespace std;
using namespace cv;
using namespace klt;
namespace JRC{

    #define KLT_TRACKED           0
    #define KLT_NOT_FOUND        -1
    #define KLT_SMALL_DET        -2
    #define KLT_MAX_ITERATIONS   -3
    #define KLT_OOB              -4
    #define KLT_LARGE_RESIDUE    -5

class JointRadiometicCalib: public klt::KLTtracker{
public:
int M; //order
//for unknown and known Response Funtion

float c[3];
float K;
float B[1024];
float g[1024];
float g_[1024];
float g0[1024]; //log-inverse of f
float h[3][1024];
float g0_[1024]; //derivative for g0
float h_[3][1024]; //derivative for h
//float log_inv_RF[];
//float grad_log_inv_RF[];

//for kalman filter estimation
Eigen::MatrixXd D;
Eigen::MatrixXd b;
Eigen::MatrixXd u_;
Eigen::MatrixXd R;
Eigen::MatrixXd P_new;
Eigen::VectorXd c_new;
Eigen::VectorXd c_prev;


bool trackingMode; //0 : knownRF, 1: unknownRF
JointRadiometicCalib(int nfeatures){
    setRFs(); setRFderivatives(); trackingMode = 1; this->M=3;
    this->tracker.sequentialMode = true;
    this->tracker.writeInternalImages = false;
    this->tracker.affineConsistencyCheck = -1;  /* set this to 2 to turn on affine consistency check */
    this->nfeatures = nfeatures;
    this->c_new = Eigen::VectorXd::Zero(4);
    this->c_prev = Eigen::VectorXd::Zero(4);
    this->P_new = Eigen::MatrixXd::Identity(4,4);
    
}
~JointRadiometicCalib(){}
void cal_g_and_g_();
int getIdxForRF(float x);
void setRFs();
void setRFs(int a);
void setRFderivatives();
float getRF_Value(int idx);
float getRF_DerivValue(int idx);

Eigen::MatrixXd get_U(int window_size, int x, int y, Mat a,Mat b);
Eigen::MatrixXd get_U(int window_size, int x, int y, Mat aM,Mat bM,Mat pM[],Mat qM[]);
Eigen::MatrixXd get_w(int window_size, int x, int y, Mat aM, Mat bM);
Eigen::MatrixXd get_w(int window_size, int x, int y, Mat aM, Mat bM,Mat rM[],Mat pM[],Mat qM[]);
Eigen::VectorXd get_v(int window_size, int x, int y, Mat aM, Mat bM, Mat betaM);
Eigen::VectorXd get_v(int window_size, int x, int y, Mat aM, Mat bM, Mat dM, Mat pM[], Mat qM[]);
Eigen::VectorXd get_z(bool JRCtrackingMode);
Eigen::MatrixXd get_lamda(int window_size,int x, int y);
Eigen::MatrixXd get_lamda(int window_size,int x, int y, Mat rM[]);
Eigen::VectorXd get_m(int window_size, int x, int y, Mat betaM);
Eigen::VectorXd get_m(int window_size, int x, int y, Mat dM, Mat rM[]);

void blockAllMatrix(int numOfTrackFeature,Eigen::MatrixXd &Uinv_all,Eigen::MatrixXd &w_all,Eigen::VectorXd &v_all,Eigen::MatrixXd &lamda_all,Eigen::VectorXd &m_all, bool JRCtrackingMode);
float get_K(Eigen::MatrixXd Uinv_all,Eigen::MatrixXd w_all,Eigen::VectorXd v_all,Eigen::MatrixXd lamda_all,Eigen::VectorXd m_all,bool JRCtrackingMode);
void initialization(kltFeature &f);
void constructMatrix(kltFeature &f,float g0[],float g0_[],float h[][1024],float h_[][1024], int pylevel,int window_size, int x, int y, Mat J_origin, Mat I_origin,Mat J_gradx, Mat I_gradx,Mat J_grady, Mat I_grady, bool JRCtrackingMode);
void constructAllMatrix(kltFeature &f, int numOfTrackFeature, Eigen::MatrixXd &Uinv_all,Eigen::MatrixXd &w_all,Eigen::VectorXd &v_all,Eigen::MatrixXd &lamda_all,Eigen::VectorXd &m_all);
int solveEquation(kltFeature f, int r,float K, float *dx, float *dy,float small,bool trackingMode);

//int trackingKnownRF(float x1, float y1, float *x2, float *y2,Mat img1,Mat gradx1,Mat grady1,Mat img2,Mat gradx2,Mat grady2);
//int trackingUnknownRF(float x1, float y1, float *x2, float *y2, Mat img1,Mat gradx1,Mat grady1,Mat img2,Mat gradx2,Mat grady2);
void updateByKalmanFilter();
int JRCtrackFeatures(Mat prevImg, Mat currImg, vector<kltFeature> prevfl, vector<kltFeature> &currfl);
int _JRCtrackFeature(kltFeature f, int pylevel, float x1, float y1, float *x2, float *y2,float K,
                    int width, int height, int max_iterations, float small, float th,
                      Mat J_origin,Mat J_gradx,Mat J_grady,Mat I_origin,Mat I_gradx,Mat I_grady, bool trackingMode);


vector<double> polyRegression(vector<double> x_,vector<double> y_, int n);

//unused
float get_a(int x,int y,float g0_[], Mat J_origin, Mat I_origin,Mat J_gradx, Mat I_gradx,bool JRCtrackingMode);
float get_b(int x,int y,float g0_[], Mat J_origin, Mat I_origin,Mat J_gradY, Mat I_gradY,bool JRCtrackingMode);
float get_beta(int x, int y, Mat J_origin, Mat I_origin);
float get_r_k(int x, int y, float h[], Mat J_origin, Mat I_origin);
float get_p_k(int x, int y, float h_[], Mat J_origin, Mat I_origin,Mat J_gradx, Mat I_gradx);
float get_q_k(int x, int y, float h_[], Mat J_origin, Mat I_origin,Mat J_grady, Mat I_grady);
float get_d(int x, int y, float g0[],Mat J_origin, Mat I_origin);


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