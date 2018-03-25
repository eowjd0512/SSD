#include "cv.hpp"
#include <iostream>
//#include <Eigen/Dense>
#include "JRC.hpp"
#include <iostream>
#include <string>
#include <fstream>
#include <math.h>
using namespace cv;
using namespace std;
namespace JRC{
float get_numerical_derivative(float RF[], int x){
    float dx = (RF[x+1]-RF[x]);
}
void JointRadiometicCalib::setRFderivatives(){
    for(int i=0;i<1023;i++)
        this->g0_[i]=get_numerical_derivative(this->g0,i);
    this->g0_[1023] = this->g0_[1022];
    for(int i=0;i<1023;i++)
        this->h_[0][i]=get_numerical_derivative(this->h[0],i);
    this->h_[0][1023] = this->h_[0][1022];
    for(int i=0;i<1023;i++)
        this->h_[1][i]=get_numerical_derivative(this->h[1],i);
    this->h_[1][1023] = this->h_[1][1022];
    for(int i=0;i<1023;i++)
        this->h_[2][i]=get_numerical_derivative(this->h[2],i);
    this->h_[2][1023] = this->h_[2][1022];
    
}
void JointRadiometicCalib::setRFs(){
fstream infile;
//infile.open(path);
infile.open("/home/jun/SSD_SLAM/src/JointRadiometicCalib/invemor.txt");
string title;
string temp;
string number = "";
infile >> title>>temp;
int i=0;
cout<< "title: "<<title<<", temp: "<< temp<< endl;
int k=0;

while(k<5){
    if (i==1024){
        infile >> title >> temp;
        cout<< "title: "<<title<<", temp: "<< temp<< endl;
        i=0;
        k++;
    }
    
    infile >> number;
    
    if(k==0){
        this->B[i] = stof(number)*255.0;
    }
    if(k==1){
        this->g0[i] = log(stof(number)*255.0);
    }else if(k>1){
        this->h[k-2][i]=log(stof(number)*255.0);
    }
    i++;
}
//cout<< "i: "<<i<<endl;
//for(int i=0;i<1024;i++) if(i==514)cout<<this->g0[i]<<" ";
}
int getIdxForRF(float x){
    int idx= int(x/255.0*1024);
    return idx;
}
float JointRadiometicCalib::get_a(int x, int y, float g0_[], Mat J_origin, Mat I_origin,Mat J_gradx, Mat I_gradx){
    
    if(this->trackingMode==1){
    float J_irr = J_origin.at<float>(y,x);
    float I_irr = I_origin.at<float>(y,x);
    float J_grad = J_gradx.at<float>(y,x);
    float I_grad = I_gradx.at<float>(y,x);
    int J_idx= getIdxForRF(J_irr);
    int I_idx= getIdxForRF(I_irr);
    float a = (g0_[J_idx]*J_grad + g0_[I_idx]*I_grad)/2;
    return a;
    }
}
float JointRadiometicCalib::get_b(int x, int y,float g0_[], Mat J_origin, Mat I_origin,Mat J_grady, Mat I_grady){
    
    if(this->trackingMode==1){
    float J_irr = J_origin.at<float>(y,x);
    float I_irr = I_origin.at<float>(y,x);
    float J_grad = J_grady.at<float>(y,x);
    float I_grad = I_grady.at<float>(y,x);
    int J_idx= getIdxForRF(J_irr);
    int I_idx= getIdxForRF(I_irr);
    float b = (g0_[J_idx]*J_grad + g0_[I_idx]*I_grad)/2;
    return b;
    }
}
float JointRadiometicCalib::get_r_k(int x, int y, float h[], Mat J_origin, Mat I_origin){
    float J_irr = J_origin.at<float>(y,x);
    float I_irr = I_origin.at<float>(y,x);
    int J_idx= getIdxForRF(J_irr);
    int I_idx= getIdxForRF(I_irr);
    float r = h[J_idx]-h[I_idx];
    return r;
}
float JointRadiometicCalib::get_p_k(int x, int y, float h_[], Mat J_origin, Mat I_origin,Mat J_gradx, Mat I_gradx){
    float J_irr = J_origin.at<float>(y,x);
    float I_irr = I_origin.at<float>(y,x);
    float J_grad = J_gradx.at<float>(y,x);
    float I_grad = I_gradx.at<float>(y,x);
    int J_idx= getIdxForRF(J_irr);
    int I_idx= getIdxForRF(I_irr);
    float p = (h_[J_idx]*J_grad + h_[I_idx]*I_grad)/2;
    return p;
}
float JointRadiometicCalib::get_q_k(int x, int y, float h_[], Mat J_origin, Mat I_origin,Mat J_grady, Mat I_grady){
    float J_irr = J_origin.at<float>(y,x);
    float I_irr = I_origin.at<float>(y,x);
    float J_grad = J_grady.at<float>(y,x);
    float I_grad = I_grady.at<float>(y,x);
    int J_idx= getIdxForRF(J_irr);
    int I_idx= getIdxForRF(I_irr);
    float q = (h_[J_idx]*J_grad + h_[I_idx]*I_grad)/2;
    return q;
}
float JointRadiometicCalib::get_d(int x, int y, float g0[],Mat J_origin, Mat I_origin){
    float J_irr = J_origin.at<float>(y,x);
    float I_irr = I_origin.at<float>(y,x);
    int J_idx= getIdxForRF(J_irr);
    int I_idx= getIdxForRF(I_irr);
    float d = g0[J_idx]-g0[I_idx];
    return d;
}

}