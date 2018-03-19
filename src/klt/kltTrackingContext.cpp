#include <iostream>
#include "kltTrackingContext.h"

namespace klt{
void kltTrackingContext::KLTUpdateTCBorder(){
    float val;
  int pyramid_gauss_hw;
  int smooth_gauss_hw;
  int gauss_width, gaussderiv_width;
  int num_levels = this->nPyramidLevels;
  int n_invalid_pixels;
  int window_hw;
  int ss = this->subsampling;
  int ss_power;
  int border;
  int i;

  /* Check window size (and correct if necessary) */
  if (this->window_width % 2 != 1) {
    this->window_width = this->window_width+1;
  }
  if (this->window_height % 2 != 1) {
    this->window_height = this->window_height+1;
  }
  if (this->window_width < 3) {
    this->window_width = 3;
  }
  if (this->window_height < 3) {
    this->window_height = 3;
  }
  window_hw = max(this->window_width, this->window_height)/2;

  /* Find widths of convolution windows */
  float smoothSigma = this->smooth_sigma_fact * max(this->window_width, this->window_height);
  static float pyramidSigma = this->pyramid_sigma_fact * this->subsampling;
  
  Mat smoothkernel = getGaussianKernel(this->window_width,smoothSigma);
  smooth_gauss_hw = smoothkernel.cols/2;
  Mat pyramidkernel = getGaussianKernel(this->window_width,pyramidSigma);
  pyramid_gauss_hw = pyramidkernel.cols/2;
  /*_KLTGetKernelWidths(smoothSigma,
                      &gauss_width, &gaussderiv_width);
  smooth_gauss_hw = gauss_width/2;
  _KLTGetKernelWidths(pyramidSigma,
                      &gauss_width, &gaussderiv_width);
  pyramid_gauss_hw = gauss_width/2;*/

  /* Compute the # of invalid pixels at each level of the pyramid.
     n_invalid_pixels is computed with respect to the ith level   
     of the pyramid.  So, e.g., if n_invalid_pixels = 5 after   
     the first iteration, then there are 5 invalid pixels in   
     level 1, which translated means 5*subsampling invalid pixels   
     in the original level 0. */
  n_invalid_pixels = smooth_gauss_hw;
  for (i = 1 ; i < num_levels ; i++)  {
    val = ((float) n_invalid_pixels + pyramid_gauss_hw) / ss;
    n_invalid_pixels = (int) (val + 0.99);  /* Round up */
  }

  /* ss_power = ss^(num_levels-1) */
  ss_power = 1;
  for (i = 1 ; i < num_levels ; i++)
    ss_power *= ss;

  /* Compute border by translating invalid pixels back into */
  /* original image */
  border = (n_invalid_pixels + window_hw) * ss_power;

  this->borderx = border;
  this->bordery = border;
}

void kltTrackingContext::KLTChangeTCPyramid(){
    float window_halfwidth;
  float subsampling;

  /* Check window size (and correct if necessary) */
  if (this->window_width % 2 != 1) {
    this->window_width = this->window_width+1;
    std::cout<< "(KLTChangeTCPyramid) Window width must be odd."<<std::endl;
  }
  if (this->window_height % 2 != 1) {
    this->window_height = this->window_height+1;
    std::cout<< "(KLTChangeTCPyramid) Window height must be odd."<<std::endl;
  }
  if (this->window_width < 3) {
    this->window_width = 3;
    std::cout<< "(KLTChangeTCPyramid) Window width must be at least three."<<std::endl;
  }
  if (this->window_height < 3) {
    this->window_height = 3;
    std::cout<< "(KLTChangeTCPyramid) Window height must be at least three."<<std::endl;
  
  }
  window_halfwidth = min(this->window_width,this->window_height)/2.0f;

  subsampling = ((float) search_range) / window_halfwidth;

  if (subsampling < 1.0)  {		/* 1.0 = 0+1 */
    this->nPyramidLevels = 1;
  } else if (subsampling <= 3.0)  {	/* 3.0 = 2+1 */
    this->nPyramidLevels = 2;
    this->subsampling = 2;
  } else if (subsampling <= 5.0)  {	/* 5.0 = 4+1 */
    this->nPyramidLevels = 2;
    this->subsampling = 4;
  } else if (subsampling <= 9.0)  {	/* 9.0 = 8+1 */
    this->nPyramidLevels = 2;
    this->subsampling = 8;
  } else {
    /* The following lines are derived from the formula:
       search_range = 
       window_halfwidth * \sum_{i=0}^{nPyramidLevels-1} 8^i,
       which is the same as:
       search_range = 
       window_halfwidth * (8^nPyramidLevels - 1)/(8 - 1).
       Then, the value is rounded up to the nearest integer. */
    float val = (float) (log(7.0*subsampling+1.0)/log(8.0));
    this->nPyramidLevels = (int) (val + 0.99);
    this->subsampling = 8;
  }
}


}