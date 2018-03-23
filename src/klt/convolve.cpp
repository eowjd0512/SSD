/*********************************************************************
 * convolve.c
 *********************************************************************/

/* Standard includes */
#include <assert.h>
#include <math.h>
#include <iostream>
#include <stdlib.h>   /* malloc(), realloc() */
#include "cv.hpp"
/* Our includes */
//#include "base.h"
//#include "error.h"
#include "convolve.hpp"
//#include "klt_util.h"   /* printing */


using namespace std;
using namespace cv;
namespace klt{
#define MAX_KERNEL_WIDTH 	71



/*********************************************************************
 * _KLTToFloatImage
 *
 * Given a pointer to image data (probably unsigned chars), copy
 * data to a float image.
 */



/*********************************************************************
 * _computeKernels
 */

void convolution::_computeKernels(
  float sigma,
  ConvolutionKernel *gauss,
  ConvolutionKernel *gaussderiv)
{
  const float factor = 0.01f;   /* for truncating tail */
  int i;

  assert(MAX_KERNEL_WIDTH % 2 == 1);
  assert(sigma >= 0.0);

  /* Compute kernels, and automatically determine widths */
  {
    const int hw = MAX_KERNEL_WIDTH / 2;
    float max_gauss = 1.0f, max_gaussderiv = (float) (sigma*exp(-0.5f));
	
    /* Compute gauss and deriv */
    for (i = -hw ; i <= hw ; i++)  {
      gauss->data[i+hw]      = (float) exp(-i*i / (2*sigma*sigma));
      gaussderiv->data[i+hw] = -i * gauss->data[i+hw];
    }

    /* Compute widths */
    gauss->width = MAX_KERNEL_WIDTH;
    for (i = -hw ; fabs(gauss->data[i+hw] / max_gauss) < factor ; 
         i++, gauss->width -= 2);
    gaussderiv->width = MAX_KERNEL_WIDTH;
    for (i = -hw ; fabs(gaussderiv->data[i+hw] / max_gaussderiv) < factor ; 
         i++, gaussderiv->width -= 2);
    if (gauss->width == MAX_KERNEL_WIDTH || 
        gaussderiv->width == MAX_KERNEL_WIDTH)
      //KLTError("(_computeKernels) MAX_KERNEL_WIDTH %d is too small for "
               //"a sigma of %f", MAX_KERNEL_WIDTH, sigma);
               cerr<<"err"<<endl;
  }

  /* Shift if width less than MAX_KERNEL_WIDTH */
  for (i = 0 ; i < gauss->width ; i++)
    gauss->data[i] = gauss->data[i+(MAX_KERNEL_WIDTH-gauss->width)/2];
  for (i = 0 ; i < gaussderiv->width ; i++)
    gaussderiv->data[i] = gaussderiv->data[i+(MAX_KERNEL_WIDTH-gaussderiv->width)/2];
  /* Normalize gauss and deriv */
  {
    const int hw = gaussderiv->width / 2;
    float den;
			
    den = 0.0;
    for (i = 0 ; i < gauss->width ; i++)  den += gauss->data[i];
    for (i = 0 ; i < gauss->width ; i++)  gauss->data[i] /= den;
    den = 0.0;
    for (i = -hw ; i <= hw ; i++)  den -= i*gaussderiv->data[i+hw];
    for (i = -hw ; i <= hw ; i++)  gaussderiv->data[i+hw] /= den;
  }

  this->sigma_last = sigma;
}
	

/*********************************************************************
 * _KLTGetKernelWidths
 *
 */

void convolution::_KLTGetKernelWidths(
  float sigma,
  int *gauss_width,
  int *gaussderiv_width)
{
  _computeKernels(sigma, &gauss_kernel, &gaussderiv_kernel);
  *gauss_width = gauss_kernel.width;
  *gaussderiv_width = gaussderiv_kernel.width;
}


/*********************************************************************
 * _convolveImageHoriz
 */

void convolution::_convolveImageHoriz(
  Mat imgin,
  ConvolutionKernel kernel,
  Mat &imgout)
{
  float *ptrrow = (float*)imgin.data;           /* Points to row's first pixel */
  float *ptrout = (float*)imgout.data, /* Points to next output pixel */
    *ppp;
  float sum;
  int radius = kernel.width / 2;
  int ncols = imgin.cols, nrows = imgin.rows;
  int i, j, k;

  /* Kernel width must be odd */
  assert(kernel.width % 2 == 1);

  /* Must read from and write to different images */
  //assert(imgin != imgout);

  /* Output image must be large enough to hold result */
  assert(imgout.cols >= imgin.cols);
  assert(imgout.rows >= imgin.rows);

  /* For each row, do ... */
  for (j = 0 ; j < nrows ; j++)  {

    /* Zero leftmost columns */
    for (i = 0 ; i < radius ; i++)
      *ptrout++ = 0.0;

    /* Convolve middle columns with kernel */
    for ( ; i < ncols - radius ; i++)  {
      ppp = ptrrow + i - radius;
      sum = 0.0;
      for (k = kernel.width-1 ; k >= 0 ; k--)
        sum += *ppp++ * kernel.data[k];
      *ptrout++ = sum;
    }

    /* Zero rightmost columns */
    for ( ; i < ncols ; i++)
      *ptrout++ = 0.0;

    ptrrow += ncols;
  }
}


/*********************************************************************
 * _convolveImageVert
 */

void convolution::_convolveImageVert(
  Mat imgin,
  ConvolutionKernel kernel,
  Mat &imgout)
{
  float *ptrcol = (float*)imgin.data;            /* Points to row's first pixel */
  float *ptrout = (float*)imgout.data,  /* Points to next output pixel */
    *ppp;
  float sum;
  int radius = kernel.width / 2;
  int ncols = imgin.cols, nrows = imgin.rows;
  int i, j, k;

  /* Kernel width must be odd */
  assert(kernel.width % 2 == 1);
  /* Must read from and write to different images */
  //assert(imgin != imgout);

  /* Output image must be large enough to hold result */
  assert(imgout.cols >= imgin.cols);
  assert(imgout.rows >= imgin.rows);
  /* For each column, do ... */
  for (i = 0 ; i < ncols ; i++)  {
    /* Zero topmost rows */
    for (j = 0 ; j < radius ; j++)  {
      *ptrout = 0.0;
      
      ptrout += ncols;
    }

    /* Convolve middle rows with kernel */
    for ( ; j < nrows - radius ; j++)  {
      ppp = ptrcol + ncols * (j - radius);
      sum = 0.0;
      for (k = kernel.width-1 ; k >= 0 ; k--)  {
        sum += *ppp * kernel.data[k];
        ppp += ncols;
      }
      *ptrout = sum;
      ptrout += ncols;
    }

    /* Zero bottommost rows */
    for ( ; j < nrows ; j++)  {
      *ptrout = 0.0;
      ptrout += ncols;
    }
    ptrcol++;
    ptrout -= nrows * ncols - 1;
  }
}


/*********************************************************************
 * _convolveSeparate
 */

void convolution::_convolveSeparate(
  Mat imgin,
  ConvolutionKernel horiz_kernel,
  ConvolutionKernel vert_kernel,
  Mat &imgout)
{
  /* Create temporary image */
  Mat tmpimg = Mat::zeros(imgin.rows,imgin.cols,CV_32FC1);
  
  //tmpimg = _KLTCreateFloatImage(imgin.cols, imgin.rows);
  /* Do convolution */
  _convolveImageHoriz(imgin, horiz_kernel, tmpimg);
 // imwrite("/home/jun/SSD_SLAM/debug/tmpimg.jpg", tmpimg);
 //cout<<"222"<<endl;

  _convolveImageVert(tmpimg, vert_kernel, imgout);
 //   imwrite("/home/jun/SSD_SLAM/debug/imgout.jpg", imgout);
 //cout<<"222"<<endl;
 //cout<<"333"<<endl;

  /* Free memory */
  //_KLTFreeFloatImage(tmpimg);
}

	
/*********************************************************************
 * _KLTComputeGradients
 */

void convolution::_KLTComputeGradients(
  Mat img,
  float sigma,
  Mat &gradx,
  Mat &grady)
{
				
  /* Output images must be large enough to hold result */
  assert(gradx.cols >= img.cols);
  assert(gradx.rows >= img.rows);
  assert(grady.cols >= img.cols);
  assert(grady.rows >= img.rows);

  /* Compute kernels, if necessary */
  if (fabs(sigma - sigma_last) > 0.05)
    _computeKernels(sigma, &gauss_kernel, &gaussderiv_kernel);
//cout<<"111"<<endl;
  _convolveSeparate(img, gaussderiv_kernel, gauss_kernel, gradx);
  // cout<<"222"<<endl;
  _convolveSeparate(img, gauss_kernel, gaussderiv_kernel, grady);
 //cout<<"333"<<endl;
}
	

/*********************************************************************
 * _KLTComputeSmoothedImage
 */

void convolution::_KLTComputeSmoothedImage(
  Mat img,
  float sigma,
  Mat &smooth)
{
  /* Output image must be large enough to hold result */
  assert(smooth.cols >= img.cols);
  assert(smooth.rows >= img.rows);

  /* Compute kernel, if necessary; gauss_deriv is not used */
  if (fabs(sigma - sigma_last) > 0.05)
    _computeKernels(sigma, &gauss_kernel, &gaussderiv_kernel);

  _convolveSeparate(img, gauss_kernel, gauss_kernel, smooth);
}



}