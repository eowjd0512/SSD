#include "cv.hpp"
#include <iostream>
//#include <Eigen/Dense>
#include "JRC.hpp"
#include <iostream>
#include <math.h>
//#include "eigen3/Eigen/Dense"
using namespace cv;
using namespace std;
namespace JRC{

Eigen::MatrixXf JointRadiometicCalib::get_U(int window_size, int x, int y, Mat a,Mat b){
    int hh = window_size/2;
    int hw = window_size/2;
    float a_=0;
    float b_=0;
    Eigen::Matrix2f U = Eigen::Matrix2f::Zero();
    try{

    for (int j = -hh ; j <= hh ; j++)
        for (int i = -hw ; i <= hw ; i++)  {
            a_= a.at<float>(y+j,x+i);
            b_= b.at<float>(y+j,x+i);
            U(0,0) += a_*a_;
            U(1,1) += b_*b_;
            U(0,1) = U(1,0) += a_*b_;
        }
    }catch(int exception){
        cerr<< "u border error"<<endl;
    }
    
    U(0,0) /= 2;
    U(0,1) = U(1,0) /= 2;
    U(1,1) /= 2;
    return U;
    
}
Eigen::MatrixXf JointRadiometicCalib::get_U(int window_size, int x, int y, Mat aM,Mat bM,Mat pM[],Mat qM[]){
    int hh = window_size/2;
    int hw = window_size/2;
    float a=0;
    float b=0;
    float p[3]={0};
    float q[3]={0};
    Eigen::MatrixXf U = Eigen::MatrixXf::Zero(8,8);
    try{

    for (int j = -hh ; j <= hh ; j++)
        for (int i = -hw ; i <= hw ; i++)  {
            a= aM.at<float>(y+j,x+i);
            b= bM.at<float>(y+j,x+i);
            for(int k=0; k<3;k++){
                p[k]=pM[k].at<float>(y+j,x+i);
                q[k]=qM[k].at<float>(y+j,x+i);
            }
            //first row and first cloumn
            U(0,0)+= a*a; U(0,1) = U(1,0) += a*p[0]; U(0,2) = U(2,0) += a*p[1]; U(0,3) = U(3,0) += a*p[2];
            U(0,4) = U(4,0) += a*b; U(0,5) = U(5,0) += a*q[0]; U(0,6) = U(6,0) += a*q[1]; U(0,7) = U(7,0) += a*q[2]; 
            //second row and second column
            U(1,1)+= p[0]*p[0]; U(1,2) = U(2,1) += p[0]*p[1]; U(1,3) = U(3,1) += p[0]*p[2]; U(1,4) = U(4,1) += p[0]*b;
            U(1,5) = U(5,1) += p[0]*q[0]; U(1,6) = U(6,1) += p[0]*q[1]; U(1,7) = U(7,1) += p[0]*q[2];
            //third row and column
            U(2,2)+= p[1]*p[1]; U(2,3) = U(3,2) += p[1]*p[2]; U(2,4) = U(4,2) += p[1]*b; 
            U(2,5) = U(5,2) += p[1]*q[0]; U(2,6) = U(6,2) += p[1]*q[1]; U(2,7) = U(7,2) += p[1]*q[2];
            //fourth row and column
            U(3,3)+= p[2]*p[2]; U(3,4) = U(4,3) += p[2]*b; U(3,5) = U(5,3) += p[2]*q[0]; 
            U(3,6) = U(6,3) += p[2]*q[1]; U(3,7) = U(7,3) += p[2]*q[2];
            //fifth row and column
            U(4,4)+= b*b; U(4,5) = U(5,4) += b*q[0]; U(4,6) = U(6,4) += b*q[1]; U(4,7) = U(7,4) += b*q[2];
            //sixth row and column
            U(5,5)+= q[0]*q[0]; U(5,6) = U(6,5) += q[0]*q[1]; U(5,7) = U(7,5) += q[0]*q[2];
            //seventh row and column,  eight
            U(6,6)+= q[1]*q[1]; U(6,7) = U(7,6) += q[1]*q[2]; U(7,7) += q[2]*q[2];
        }
            
    }catch(int exception){
        cerr<< "U border error"<<endl;
    }
    return U;
    
}
Eigen::MatrixXf JointRadiometicCalib::get_w(int window_size, int x, int y, Mat aM, Mat bM){
    int hh = window_size/2;
    int hw = window_size/2;
    float a=0;
    float b=0;
    Eigen::VectorXf w= Eigen::MatrixXf::Zero(2,1);
    try{
        
    for (int j = -hh ; j <= hh ; j++)
        for (int i = -hw ; i <= hw ; i++)  {
            a= aM.at<float>(y+j,x+i);
            b= bM.at<float>(y+j,x+i);
            w(0,0) = -a;
            w(1,0) = -b;
        }
    }catch(int exception){
        cerr<< "w border error"<<endl;
    }
    return w;
}
Eigen::MatrixXf JointRadiometicCalib::get_w(int window_size, int x, int y, Mat aM, Mat bM,Mat rM[],Mat pM[],Mat qM[]){
    int hh = window_size/2;
    int hw = window_size/2;
    float a=0;
    float b=0;
    float p[3]={0};
    float q[3]={0};
    float r[3]={0};
    Eigen::MatrixXf W = Eigen::MatrixXf::Zero(8,4);
    try{

    for (int j = -hh ; j <= hh ; j++)
        for (int i = -hw ; i <= hw ; i++)  {
            a= aM.at<float>(y+j,x+i);
            b= bM.at<float>(y+j,x+i);
            for(int k=0; k<3;k++){
                p[k]=pM[k].at<float>(y+j,x+i);
                q[k]=qM[k].at<float>(y+j,x+i);
                r[k]=rM[k].at<float>(y+j,x+i);
            }
            W(0,0)+= a*r[0]; W(0,1)+= a*r[1]; W(0,2)+= a*r[2]; W(0,3)-= a;
            W(1,0)+= p[0]*r[0]; W(1,1)+= p[0]*r[1]; W(1,2)+= p[0]*r[2]; W(1,3)-= p[0];
            W(2,0)+= p[1]*r[0]; W(2,1)+= p[1]*r[1]; W(2,2)+= p[1]*r[2]; W(2,3)-= p[1];
            W(3,0)+= p[2]*r[0]; W(3,1)+= p[2]*r[1]; W(3,2)+= p[2]*r[2]; W(3,3)-= p[2];
            W(4,0)+= b*r[0]; W(4,1)+= b*r[1]; W(4,2)+= b*r[2]; W(4,3)-= b;
            W(5,0)+= q[0]*r[0]; W(5,1)+= q[0]*r[1]; W(5,2)+= q[0]*r[2]; W(5,3)-= q[0];
            W(6,0)+= q[1]*r[0]; W(6,1)+= q[1]*r[1]; W(6,2)+= q[1]*r[2]; W(6,3)-= q[1];
            W(7,0)+= q[2]*r[0]; W(7,1)+= q[2]*r[1]; W(7,2)+= q[2]*r[2]; W(7,3)-= q[2];
        }
            
    }catch(int exception){
        cerr<< "W border error"<<endl;
    }
    return W;
}
Eigen::VectorXf JointRadiometicCalib::get_v(int window_size, int x, int y, Mat aM, Mat bM, Mat betaM){
    int hh = window_size/2;
    int hw = window_size/2;
    float a=0;
    float b=0;
    float beta=0;
    Eigen::Vector2f v;
    try{
        
    for (int j = -hh ; j <= hh ; j++)
        for (int i = -hw ; i <= hw ; i++)  {
            beta = betaM.at<float>(y+j,x+i);
            a= aM.at<float>(y+j,x+i);
            b= bM.at<float>(y+j,x+i);
            v(0)-=beta*a;
            v(1)-=beta*b;
        }

    }catch(int exception){
        cerr<< "v border error"<<endl;
    }
    return v;
}
Eigen::VectorXf JointRadiometicCalib::get_v(int window_size, int x, int y, Mat aM, Mat bM, Mat dM, Mat pM[], Mat qM[]){
    int hh = window_size/2;
    int hw = window_size/2;
    float a=0;
    float b=0;
    float d=0;
    float p[3]={0};
    float q[3]={0};
    Eigen::VectorXf v = Eigen::VectorXf::Zero(8);
    try{
    for (int j = -hh ; j <= hh ; j++)
        for (int i = -hw ; i <= hw ; i++)  {
            a= aM.at<float>(y+j,x+i);
            b= bM.at<float>(y+j,x+i);
            d = dM.at<float>(y+j,x+i);
            for(int k=0; k<3;k++){
                p[k]=pM[k].at<float>(y+j,x+i);
                q[k]=qM[k].at<float>(y+j,x+i);
            }
            v(0) -= a*d; v(1) -= p[0]*d; v(2) -= p[1]*d; v(3) -= p[2]*d; 
            v(4) -= b*d; v(5) -= q[0]*d; v(6) -= q[1]*d; v(7) -= q[2]*d;
        }
            
    }catch(int exception){
        cerr<< "V border error"<<endl;
    }
    return v;
    
}
Eigen::VectorXf JointRadiometicCalib::get_z(bool JRCtrackingMode){}
Eigen::MatrixXf JointRadiometicCalib::get_lamda(int window_size,int x, int y){
    int hh = window_size/2;
    int hw = window_size/2;
    Eigen::MatrixXf lamda(1,1);
    lamda(0,0)=0;
    for (int j = -hh ; j <= hh ; j++)
        for (int i = -hw ; i <= hw ; i++)  {
            lamda(0,0) +=2;
        }
    return lamda;
}
Eigen::MatrixXf JointRadiometicCalib::get_lamda(int window_size,int x, int y, Mat rM[]){
    int hh = window_size/2;
    int hw = window_size/2;
    float r[3]={0};
    Eigen::MatrixXf lamda = Eigen::MatrixXf::Zero(4,4);
    try{
    for (int j = -hh ; j <= hh ; j++)
        for (int i = -hw ; i <= hw ; i++)  {
            for(int k=0; k<3;k++){
                r[k]=rM[k].at<float>(y+j,x+i);
            }
            lamda(0,0) += r[0]*r[0]; lamda(0,1) += r[0]*r[1]; lamda(0,2) += r[0]*r[2]; lamda(0,3) -= r[0];
            lamda(1,0) += r[1]*r[0]; lamda(1,1) += r[1]*r[1]; lamda(1,2) += r[1]*r[2]; lamda(1,3) -= r[1];
            lamda(2,0) += r[2]*r[0]; lamda(2,1) += r[2]*r[1]; lamda(2,2) += r[2]*r[2]; lamda(2,3) -= r[2];
            lamda(3,0) -= r[0]; lamda(3,1) -= r[1]; lamda(3,2) -= r[2]; lamda(3,3) += 1;
        }
            
    }catch(int exception){
        cerr<< "LAMDA border error"<<endl;
    }
    return lamda;
}
Eigen::VectorXf JointRadiometicCalib::get_m(int window_size, int x, int y, Mat betaM){
    int hh = window_size/2;
    int hw = window_size/2;
    Eigen::VectorXf beta(1);
    beta(0)=0;
    try{
        
    for (int j = -hh ; j <= hh ; j++)
        for (int i = -hw ; i <= hw ; i++)  {
            beta(0) += betaM.at<float>(y+j,x+i);
        }
    }catch(int exception){
        cerr<< "m border error"<<endl;
    }
    beta *=2;
    return beta;
}
Eigen::VectorXf JointRadiometicCalib::get_m(int window_size, int x, int y, Mat dM, Mat rM[]){
    int hh = window_size/2;
    int hw = window_size/2;
    float d=0;
    float r[3]={0};
    Eigen::VectorXf m = Eigen::VectorXf::Zero(4);
    try{
    for (int j = -hh ; j <= hh ; j++)
        for (int i = -hw ; i <= hw ; i++)  {
            d = dM.at<float>(y+j,x+i);
            for(int k=0; k<3;k++){
                r[k]=rM[k].at<float>(y+j,x+i);   
            }
            m(0)-=r[0]; m(1)-=r[1]; m(2)-=r[2]; m(3)+=d;
        }
            
    }catch(int exception){
        cerr<< "M border error"<<endl;
    }
    return m;
}
float JointRadiometicCalib::get_K(kltFeature f,Eigen::MatrixXf Uinv_all,Eigen::MatrixXf w_all,Eigen::VectorXf v_all,Eigen::MatrixXf lamda_all,Eigen::VectorXf m_all,bool JRCtrackingMode){
    if (JRCtrackingMode ==0 ){ // known RF
        Eigen::MatrixXf A = -w_all.transpose()*Uinv_all*w_all+lamda_all;
        Eigen::MatrixXf b = -w_all.transpose()*Uinv_all*w_all+m_all;
        float A_ = A(0);
        float b_ = b(0);
        float K = b_/A_;
        return K;
    }else if(JRCtrackingMode ==1){// unknown RF
        int w=1;
        int tou= getIdxForRF(128);
        Eigen::MatrixXf A = -w_all.transpose()*Uinv_all*w_all+lamda_all;
        Eigen::MatrixXf b = -w_all.transpose()*Uinv_all*w_all+m_all;
        Eigen::MatrixXf A_(5,4);
        Eigen::MatrixXf b_(5,1);
        A_.block(0,0,4,4) = A;
        b_.block(0,0,4,1) = b;
        A_(4,0)=w*this->h[0][tou]; A_(4,1)=w*this->h[1][tou];A_(4,2)=w*this->h[2][tou];A_(4,3)=0;
        b_(4,0)=w*(log(128)-this->g0[tou]);
        
        Eigen::Vector4f x = A_.llt().solve(b_);
        //Eigen::Vector4f x = A_.ldlt().solve(b_);
        //Eigen::Vector4f x = A_.jacobiSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(b_);
        //float K = right[0]/left[0];
        this->c[0] = x[0]; this->c[1] = x[1]; this->c[2] = x[2];
        return x[3];
    }
}
void JointRadiometicCalib::initialization(kltFeature f){
    //if (JRCtrakingMode == 0){
        Eigen::MatrixXf U;
        Eigen::VectorXf w,v;
        Eigen::VectorXf z;
        Eigen::MatrixXf lamda;
        Eigen::VectorXf m;
        for (int i=0; i< this->M; i++){
            f.U.push_back(U);
            f.w.push_back(w);
            f.v.push_back(v);
            f.z.push_back(z);
            f.lamda.push_back(lamda);
            f.m.push_back(m);
        }
    //}//else if(JRCtrakingMode==1){}
}

void JointRadiometicCalib::blockAllMatrix(int numOfTrackFeature,Eigen::MatrixXf &Uinv_all,Eigen::MatrixXf &w_all,Eigen::VectorXf &v_all,Eigen::MatrixXf &lamda_all,Eigen::VectorXf &m_all, bool JRCtrackingMode){
    if (JRCtrackingMode ==0 ){ // known RF
        Uinv_all = Uinv_all.block(0, 0, numOfTrackFeature*2, numOfTrackFeature*2);
        w_all = w_all.block(0, 0, numOfTrackFeature*2, 2);
        v_all = v_all.block(0, 0, numOfTrackFeature*2, 2);
        lamda_all = lamda_all.block(0, 0, 1, 1);
        m_all = m_all.block(0, 0, 1, 1);

    }else if(JRCtrackingMode ==1){// unknown RF
        Uinv_all = Uinv_all.block(0, 0, numOfTrackFeature*8, numOfTrackFeature*8);
        w_all = w_all.block(0, 0, numOfTrackFeature*8, 4);
        v_all = v_all.block(0, 0, numOfTrackFeature*8, 1);
    }
}

void JointRadiometicCalib::constructAllMatrix(kltFeature f, int numOfTrackFeature, Eigen::MatrixXf &Uinv_all,Eigen::MatrixXf &w_all,Eigen::VectorXf &v_all,Eigen::MatrixXf &lamda_all,Eigen::VectorXf &m_all){
    Uinv_all.block(numOfTrackFeature*f.U[0].rows(), numOfTrackFeature*f.U[0].cols(), f.U[0].rows(), f.U[0].cols()) = f.U[0].inverse();
    w_all.block(numOfTrackFeature*f.w[0].rows(), 0, f.w[0].rows(), f.w[0].cols()) = f.w[0];
    v_all.block(numOfTrackFeature*f.v[0].rows(), 0, f.v[0].rows(), f.v[0].cols()) = f.v[0];
    //lamda and m adapt summation
    lamda_all += f.lamda[0];
    m_all += f.m[0];

}
void JointRadiometicCalib::cal_g_and_g_(){
    int M = this->M;
    for(int i=0;i<1024;i++){
        float val = this->g0[i];
        float val_ = this->g0_[i];
        for(int j=0; j<M;j++){
            val+=this->h[j][i]*this->c[j];
            val_+=this->h_[j][i]*this->c[j];
        }
        this->g[i] = val;
        this->g_[i] = val_;
    }
}
void JointRadiometicCalib::constructMatrix(kltFeature f, int pylevel, int window_size, int x, int y, Mat J_origin, Mat I_origin,Mat J_gradx, Mat I_gradx, Mat J_grady, Mat I_grady, bool JRCtrackingMode){
    if (JRCtrackingMode ==0){
    Mat a(J_origin.size(),CV_32FC1);
    Mat b(J_origin.size(),CV_32FC1);
    Mat beta(J_origin.size(),CV_32FC1);
    float* data_J_origin = (float*)J_origin.data;
    float* data_I_origin = (float*)I_origin.data;
    float* data_J_gradx = (float*)J_gradx.data;
    float* data_I_gradx = (float*)I_gradx.data;
    float* data_J_grady = (float*)J_grady.data;
    float* data_I_grady = (float*)I_grady.data;
    float* data_a = (float*)a.data;
    float* data_b = (float*)b.data;
    float* data_beta = (float*)beta.data;

    for(int i=0; i<J_origin.rows*J_origin.cols;i++){
        int J_index= getIdxForRF(data_J_origin[i]);
        int I_index= getIdxForRF(data_I_origin[i]);
        data_a[i] = this->g_[J_index]*data_J_gradx[i] + this->g_[I_index]*data_I_gradx[i];
        data_b[i] = this->g_[J_index]*data_J_grady[i] + this->g_[I_index]*data_I_grady[i];
        data_beta[i] = this->g[J_index] - this->g[I_index];
    }


    Eigen::MatrixXf U = get_U(window_size,x,y,a,b);
    Eigen::VectorXf w = get_w(window_size,x,y,a,b);
    Eigen::VectorXf v = get_v(window_size,x,y,a,b,beta);
    //Eigen::Vector3f z = get_z_knownRF(window_size,x,y,J_origin,I_origin,J_gradx,I_gradx,J_grady,I_grady);
    Eigen::MatrixXf lamda = get_lamda(window_size,x,y);
    Eigen::VectorXf m  = get_m(window_size,x,y,beta);
    f.U[pylevel]=U;
    f.w[pylevel]=w;
    f.v[pylevel]=v;
    f.lamda[pylevel]=lamda;
    f.m[pylevel]=m;

    }else if(JRCtrackingMode==1){
        Mat a(J_origin.size(),CV_32FC1);
        Mat b(J_origin.size(),CV_32FC1);
        //Mat beta(J_origin.size(),CV_32FC1);
        Mat r[3],p[3],q[3];
        for(int i=0;i<3;i++){r[i] = Mat(J_origin.size(),CV_32FC1);p[i] = Mat(J_origin.size(),CV_32FC1);q[i] = Mat(J_origin.size(),CV_32FC1);}
        Mat d(J_origin.size(),CV_32FC1);
        float* data_J_origin = (float*)J_origin.data;
        float* data_I_origin = (float*)I_origin.data;
        float* data_J_gradx = (float*)J_gradx.data;
        float* data_I_gradx = (float*)I_gradx.data;
        float* data_J_grady = (float*)J_grady.data;
        float* data_I_grady = (float*)I_grady.data;
        float* data_a = (float*)a.data;
        float* data_b = (float*)b.data;
        float* data_r0 = (float*)r[0].data;float* data_p0 = (float*)p[0].data;float* data_q0 = (float*)q[0].data;
        float* data_r1 = (float*)r[1].data;float* data_p1 = (float*)p[1].data;float* data_q1 = (float*)q[1].data;
        float* data_r2 = (float*)r[2].data;float* data_p2 = (float*)p[2].data;float* data_q2 = (float*)q[2].data;
        float* data_d = (float*)d.data;
        //float* data_beta = (float*)beta.data;
        
        for(int i=0; i<J_origin.rows*J_origin.cols;i++){
            int J_index= getIdxForRF(data_J_origin[i]);
            int I_index= getIdxForRF(data_I_origin[i]);
            data_a[i] = (this->g0_[J_index]*data_J_gradx[i] + this->g0_[I_index]*data_I_gradx[i])/2;
            data_b[i] = (this->g0_[J_index]*data_J_grady[i] + this->g0_[I_index]*data_I_grady[i])/2;
            data_r0[i]=this->h[0][J_index]-this->h[0][I_index];
            data_r1[i]=this->h[1][J_index]-this->h[1][I_index];
            data_r2[i]=this->h[2][J_index]-this->h[2][I_index];
            data_p0[i]=(this->h_[0][J_index]*data_J_gradx[i] + this->h_[0][I_index]*data_I_gradx[i])/2;
            data_p1[i]=(this->h_[1][J_index]*data_J_gradx[i] + this->h_[1][I_index]*data_I_gradx[i])/2;
            data_p2[i]=(this->h_[2][J_index]*data_J_gradx[i] + this->h_[2][I_index]*data_I_gradx[i])/2;
            data_q0[i]=(this->h_[0][J_index]*data_J_grady[i] + this->h_[0][I_index]*data_I_grady[i])/2;
            data_q1[i]=(this->h_[1][J_index]*data_J_grady[i] + this->h_[1][I_index]*data_I_grady[i])/2;
            data_q2[i]=(this->h_[2][J_index]*data_J_grady[i] + this->h_[2][I_index]*data_I_grady[i])/2;
            data_d[i] = this->g0[J_index] - this->g0[I_index];
        }


        Eigen::MatrixXf U = get_U(window_size,x,y,a,b,p,q);
        Eigen::VectorXf w = get_w(window_size,x,y,a,b,r,p,q);
        Eigen::VectorXf v = get_v(window_size,x,y,a,b,d,p,q);
        //Eigen::Vector3f z = get_z_knownRF(window_size,x,y,J_origin,I_origin,J_gradx,I_gradx,J_grady,I_grady);
        Eigen::MatrixXf lamda = get_lamda(window_size,x,y,r);
        Eigen::VectorXf m  = get_m(window_size,x,y,d,r);
        f.U[pylevel]=U;
        f.w[pylevel]=w;
        f.v[pylevel]=v;
        f.lamda[pylevel]=lamda;
        f.m[pylevel]=m;
    }

}

void JointRadiometicCalib::solveEquation(){}
void JointRadiometicCalib::decomposition(){}

}