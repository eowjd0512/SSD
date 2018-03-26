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

Eigen::MatrixXf JointRadiometicCalib::get_U(int window_size, int x, int y, Mat J_origin, Mat I_origin,Mat J_gradx, Mat I_gradx,Mat J_grady, Mat I_grady, bool JRCtrackingMode){
    if (JRCtrackingMode==0){
        int hh = window_size/2;
        int hw = window_size/2;
        float a=0;
        float b=0;
        float sum_of_a2=0;
        float sum_of_b2=0;
        float sum_of_ab=0;
        
        try{

        for (int j = -hh ; j <= hh ; j++)
            for (int i = -hw ; i <= hw ; i++)  {
                a= get_a(x+i, y+j, this->g0_, J_origin, I_origin, J_gradx, I_gradx);
                b= get_b(x+i, y+j, this->g0_, J_origin, I_origin, J_grady, I_grady);
                sum_of_a2 += a*a;
                sum_of_b2 += b*b;
                sum_of_ab += a*b;
            }
        }catch(int exception){
            cerr<< "U border error"<<endl;
        }
        Eigen::Matrix2f U;
        U(0,0) = 1/2*sum_of_a2;
        U(0,1) = 1/2*sum_of_ab;
        U(1,0) = 1/2*sum_of_ab;
        U(1,1) = 1/2*sum_of_b2;
        return U;
    }
}
Eigen::MatrixXf JointRadiometicCalib::get_w(int window_size, int x, int y, Mat J_origin, Mat I_origin,Mat J_gradx, Mat I_gradx,Mat J_grady, Mat I_grady,bool JRCtrackingMode){
    if (JRCtrackingMode==0){
        int hh = window_size/2;
        int hw = window_size/2;
        float a=0;
        float b=0;
        try{
            
        for (int j = -hh ; j <= hh ; j++)
            for (int i = -hw ; i <= hw ; i++)  {
                a += get_a(x+i, y+j, this->g0_, J_origin, I_origin, J_gradx, I_gradx);
                b += get_b(x+i, y+j, this->g0_, J_origin, I_origin, J_grady, I_grady);
            }
        }catch(int exception){
            cerr<< "w border error"<<endl;
        }
        Eigen::Vector2f w;
        w(0) = -a;
        w(1) = -b;
        return w;
    }
}
Eigen::VectorXf JointRadiometicCalib::get_v(int window_size, int x, int y, Mat J_origin, Mat I_origin,Mat J_gradx, Mat I_gradx, Mat J_grady, Mat I_grady,bool JRCtrackingMode){
    if (JRCtrackingMode==0){
        int hh = window_size/2;
        int hw = window_size/2;
        float a=0;
        float b=0;
        float beta=0;
        float beta_a=0;
        float beta_b=0;
        try{
            
        for (int j = -hh ; j <= hh ; j++)
            for (int i = -hw ; i <= hw ; i++)  {
                beta = get_beta(x+i, y+j, J_origin, I_origin);
                a = get_a(x+i, y+j, this->g0_, J_origin, I_origin, J_gradx, I_gradx);
                b = get_b(x+i, y+j, this->g0_, J_origin, I_origin, J_grady, I_grady);
                beta_a+=beta*a;
                beta_b+=beta*b;
            }

        }catch(int exception){
            cerr<< "v border error"<<endl;
        }
        Eigen::Vector2f v;
        v(0) = -beta_a;
        v(1) = -beta_b;
        return v;
    }
}
Eigen::VectorXf JointRadiometicCalib::get_z(bool JRCtrackingMode){}
Eigen::MatrixXf JointRadiometicCalib::get_lamda(int window_size,bool JRCtrackingMode){
    if (JRCtrackingMode==0){
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
}
Eigen::VectorXf JointRadiometicCalib::get_m(int window_size, int x, int y, Mat J_origin, Mat I_origin,bool JRCtrackingMode){
    if (JRCtrackingMode==0){
        int hh = window_size/2;
        int hw = window_size/2;
        Eigen::VectorXf beta(1);
        beta(0)=0;
        try{
            
        for (int j = -hh ; j <= hh ; j++)
            for (int i = -hw ; i <= hw ; i++)  {
                beta(0) += get_beta(x+i, y+j,  J_origin, I_origin);
            }
        }catch(int exception){
            cerr<< "m border error"<<endl;
        }
        beta *=2;
        return beta;
    }
}
float JointRadiometicCalib::get_K(kltFeature f,Eigen::MatrixXf Uinv_all,Eigen::MatrixXf w_all,Eigen::VectorXf v_all,Eigen::MatrixXf lamda_all,Eigen::VectorXf m_all,bool JRCtrackingMode){
    if (JRCtrackingMode ==0 ){ // known RF
        
    }else if(JRCtrackingMode ==1){// unknown RF

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
        lamda_all = lamda_all.block(0, 0, 4, 4);
        m_all = m_all.block(0, 0, 4, 1);
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
void JointRadiometicCalib::constructMatrix(kltFeature f, int pylevel, int window_size, int x, int y, Mat J_origin, Mat I_origin,Mat J_gradx, Mat I_gradx, Mat J_grady, Mat I_grady, bool JRCtrackingMode){
    
    //if(JRCtrackingMode==0){
    Eigen::MatrixXf U = get_U(window_size,x,y,J_origin,I_origin,J_gradx,I_gradx,J_grady,I_grady,JRCtrackingMode);
    Eigen::VectorXf w = get_w(window_size,x,y,J_origin,I_origin,J_gradx,I_gradx,J_grady,I_grady,JRCtrackingMode);
    Eigen::VectorXf v = get_v(window_size,x,y,J_origin,I_origin,J_gradx,I_gradx,J_grady,I_grady,JRCtrackingMode);
    //Eigen::Vector3f z = get_z_knownRF(window_size,x,y,J_origin,I_origin,J_gradx,I_gradx,J_grady,I_grady);
    Eigen::MatrixXf lamda = get_lamda(window_size,JRCtrackingMode);
    Eigen::VectorXf m  = get_m(window_size,x,y,J_origin,I_origin,JRCtrackingMode);
    f.U[pylevel]=U;
    f.w[pylevel]=w;
    f.v[pylevel]=v;
    f.lamda[pylevel]=lamda;
    f.m[pylevel]=m;
    //}else if(JRCtrackingMode==1){

    //}

}

void JointRadiometicCalib::solveEquation(){}
void JointRadiometicCalib::decomposition(){}

}