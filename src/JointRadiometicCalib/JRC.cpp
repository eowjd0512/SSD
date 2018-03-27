#include "cv.hpp"
#include <iostream>
//#include <Eigen/Dense>
#include "JRC.hpp"
#include <iostream>
#include <string>
#include <fstream>
#include <math.h>
#include "eigen3/Eigen/Dense"
using namespace cv;
using namespace std;
namespace JRC{
float get_numerical_derivative(float RF[], float B[], int x){
    float dx = (RF[x+1]-RF[x])/(B[x+1]-B[x]);
    return dx;
}
void JointRadiometicCalib::setRFderivatives(){
    for(int i=0;i<1023;i++)
        this->g0_[i]=get_numerical_derivative(this->g0, this->B, i);
    this->g0_[1023] = this->g0_[1022];
    for(int i=0;i<1023;i++)
        this->h_[0][i]=get_numerical_derivative(this->h[0],this->B,i);
    this->h_[0][1023] = this->h_[0][1022];
    for(int i=0;i<1023;i++)
        this->h_[1][i]=get_numerical_derivative(this->h[1],this->B,i);
    this->h_[1][1023] = this->h_[1][1022];
    for(int i=0;i<1023;i++)
        this->h_[2][i]=get_numerical_derivative(this->h[2],this->B,i);
    this->h_[2][1023] = this->h_[2][1022];
    for(int i=0;i<1024;i++) cout<<this->g0_[i]<<" ";
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
        float val = log(stof(number)*255.0);
        if(stof(number) ==0.0){val=log(1e-10*255.0);}
        this->g0[i] = val;
    }else if(k>1){
        float val = log(stof(number)*255.0);
        if(stof(number) ==0.0){val=log(1e-10*255.0);}
        this->h[k-2][i]=log(stof(number)*255.0);
    }
    i++;
}
//cout<< "i: "<<i<<endl;
//for(int i=0;i<1024;i++) if(i==514)cout<<this->g0[i]<<" ";
}
int JointRadiometicCalib::getIdxForRF(float x){
    int idx= int(x/255.0*1023.0);
    if (idx ==0){cout<<"idx: 0 , it's possible to be error"<<endl;}
    return idx;
}
float JointRadiometicCalib::getRF_Value(int idx){
    int M = this->M;
    float val = this->g0[idx];
    for(int i=0; i<M;i++){
        val+=this->h[i][idx]*this->c[i];
    }
    return val;
}
float JointRadiometicCalib::getRF_DerivValue(int idx){
    int M = this->M;
    float val = this->g0_[idx];
    for(int i=0; i<M;i++){
        val+=this->h_[i][idx]*this->c[i];
    }
    return val;
}
float JointRadiometicCalib::get_beta(int x, int y, Mat J_origin, Mat I_origin){
    float J_irr = J_origin.at<float>(y,x);
    float I_irr = I_origin.at<float>(y,x);
    int J_idx= getIdxForRF(J_irr);
    int I_idx= getIdxForRF(I_irr);
    float g_J = getRF_Value(J_idx);
    float g_I = getRF_Value(I_idx);
    float beta = g_J-g_I;
    return beta;
}

float JointRadiometicCalib::get_a(int x, int y, float g0_[], Mat J_origin, Mat I_origin,Mat J_gradx, Mat I_gradx,bool JRCtrackingMode){
    float J_irr = J_origin.at<float>(y,x);
    float I_irr = I_origin.at<float>(y,x);
    float J_grad = J_gradx.at<float>(y,x);
    float I_grad = I_gradx.at<float>(y,x);
    int J_idx= getIdxForRF(J_irr);
    int I_idx= getIdxForRF(I_irr);
    if(JRCtrackingMode==0){
        float g_J = getRF_DerivValue(J_idx);
        float g_I = getRF_DerivValue(I_idx);
        float a = g_J*J_grad-g_I*I_grad;
        return a;
    }
    else if(JRCtrackingMode==1){
        float a = (g0_[J_idx]*J_grad + g0_[I_idx]*I_grad)/2;
        return a;
    }
}
float JointRadiometicCalib::get_b(int x, int y,float g0_[], Mat J_origin, Mat I_origin,Mat J_grady, Mat I_grady,bool JRCtrackingMode){
    float J_irr = J_origin.at<float>(y,x);
    float I_irr = I_origin.at<float>(y,x);
    float J_grad = J_grady.at<float>(y,x);
    float I_grad = I_grady.at<float>(y,x);
    int J_idx= getIdxForRF(J_irr);
    int I_idx= getIdxForRF(I_irr);
    if(JRCtrackingMode==0){
        float g_J = getRF_DerivValue(J_idx);
        float g_I = getRF_DerivValue(I_idx);
        float b = g_J*J_grad-g_I*I_grad;
        return b;
    }
    else if(JRCtrackingMode==1){
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

void JointRadiometicCalib::updateByKalmanFilter(){
    Eigen::MatrixXf c_prior = this->c_new;
    Eigen::MatrixXf P_prior = this->P_new;
    Eigen::MatrixXf identity = Eigen::MatrixXf::Identity(4,4);
    Eigen::MatrixXf temp = (this->D.transpose()*this->D).inverse();
    Eigen::MatrixXf temp2 = this->D*this->u_-this->b;
    Eigen::MatrixXf temp3 = temp2*temp2.transpose();
    Eigen::MatrixXf R = temp*temp3;

    Eigen::MatrixXf k = P_prior*(P_prior+R).inverse();
    this->c_new = c_prior+k*(this->u_-c_prior);
    this->P_new = (identity-k)*P_prior;

    static int stableCount=0;
    if(abs(c_new(0)-c_prev(0))<1 &&abs(c_new(1)-c_prev(1))<1&& abs(c_new(2)-c_prev(2))<1 )
        stableCount++;
        if(stableCount >3){
            this->trackingMode =0;
            this->c[0] = this->c_new(0);
            this->c[1] = this->c_new(1);
            this->c[2] = this->c_new(2);
        }
}

}