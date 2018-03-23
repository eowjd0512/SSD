/*********************************************************************
 * convolve.h
 *********************************************************************/

#ifndef _CONVOLVE_H_
#define _CONVOLVE_H_
#include <iostream>
#include "cv.hpp"
//#include "klt.h"
//#include "klt_util.h"
using namespace cv;
#define MAX_KERNEL_WIDTH 	71
namespace klt{

class convolution{
public:

typedef struct  {
  int width;
  float data[MAX_KERNEL_WIDTH];
}  ConvolutionKernel;

/* Kernels */
ConvolutionKernel gauss_kernel;
ConvolutionKernel gaussderiv_kernel;
float sigma_last;

convolution(){
  sigma_last = -10.0;
}
~convolution(){}
void _computeKernels(
  float sigma,
  ConvolutionKernel *gauss,
  ConvolutionKernel *gaussderiv);

void _KLTComputeGradients(
  Mat img,
  float sigma,
  Mat &gradx,
  Mat &grady);

void _KLTGetKernelWidths(
  float sigma,
  int *gauss_width,
  int *gaussderiv_width);

void _KLTComputeSmoothedImage(
  Mat img,
  float sigma,
  Mat &smooth);

void _convolveImageHoriz(
  Mat imgin,
  ConvolutionKernel kernel,
  Mat &imgout);
void _convolveImageVert(
  Mat imgin,
  ConvolutionKernel kernel,
  Mat &imgout);
void _convolveSeparate(
  Mat imgin,
  ConvolutionKernel horiz_kernel,
  ConvolutionKernel vert_kernel,
  Mat &imgout);

};

}
#endif
