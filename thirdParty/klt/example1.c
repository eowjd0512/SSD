/**********************************************************************
Finds the 100 best features in an image, and tracks these
features to the next image.  Saves the feature
locations (before and after tracking) to text files and to PPM files, 
and prints the features to the screen.
**********************************************************************/

#include "pnmio.h" // for reading an image 
#include "klt.h" //here, all of implementation are included

#ifdef WIN32
int RunExample1()
#else
int main()
#endif
{
  unsigned char *img1, *img2;
  KLT_TrackingContext tc;
  KLT_FeatureList fl;
  int nFeatures = 100;
  int ncols, nrows;
  int i;

  tc = KLTCreateTrackingContext(); //creates a tracking context, which is a structure containing various parameters (such as window size, sigma for Gaussian smoothing, minimum distance between features, etc.) that govern the computation. The nature of the structure is not important for now.
  KLTPrintTrackingContext(tc);

  //creates a feature list, which is a structure containing all the features in a given image. The parameter of the function is the number of features desired. 
  fl = KLTCreateFeatureList(nFeatures);

//The helper function pgmReadFile() has been provided for reading PGM images, but it can be removed if the raw image data is obtained in another manner. Passing NULL as the second parameter causes pgmReadFile() to allocate memory for the image.
  img1 = pgmReadFile("img0.pgm", NULL, &ncols, &nrows);
  img2 = pgmReadFile("img1.pgm", NULL, &ncols, &nrows);

//finds the N best features in an image and stores them in a feature list in descending order of goodness. N is the size of the feature list (in this case 100). (Note: It is also possible to select all the features above a certain level of goodness, rather than just the N best features, by simply changing the min_eigenvalue field of the tracking context, as described in Chapter 7.)
  KLTSelectGoodFeatures(tc, img1, ncols, nrows, fl); 

  printf("\nIn first image:\n");

//The for loop demonstrates how to access data in the feature list. A feature list is simply an array of fl->nFeatures features, where each feature has three members: x location, y location, and value. The first two are floats, and the second is an int. The location is given in pixels with the origin in the upper-left corner. The value is a constant multiplied by the minimum eigenvalue of the window surrounding the feature -- the larger the value, the better the feature. (NOTE: Because fl->nFeatures indicates the amount of memory allocated, it must not be changed by hand.)
  for (i = 0 ; i < fl->nFeatures ; i++)  {
    printf("Feature #%d:  (%f,%f) with value of %d\n",
           i, fl->feature[i]->x, fl->feature[i]->y,
           fl->feature[i]->val);
  }
//writes the grey-level image to a PPM file and overlays the features in red. This image can then be viewed with a tool like xv.
  KLTWriteFeatureListToPPM(fl, img1, ncols, nrows, "feat1.ppm");

  //writes the features to a file (If the second parameter is NULL, then the features are written to stderr instead). You will probably want to keep the third parameter NULL, which causes the file to be binary. Binary files are better because they are smaller and cause no truncation error. However, if a text file is desired, then the third parameter must be a string beginning with '%' and ending with either 'd' or 'f', as in "%3d" or "%5.1f", which specifies the format of the feature locations to be either decimal or floating point. Once the feature list is written, the file can be viewed off-line (if it is text) or read back into a feature list using the command KLTReadFeatureList().
  KLTWriteFeatureList(fl, "feat1.txt", "%3d");

//tracks the features from the first image to the second image and overwrites the feature list. If a feature is successfully tracked, then its value becomes zero; otherwise, it becomes negative. Note that this function does not resort the features, but rather keeps them in place so that correspondences between the images are maintained. Chapter 5 describes the KLT_FeatureTable structure, which is useful for storing the feature list before it is overwritten.
  KLTTrackFeatures(tc, img1, img2, ncols, nrows, fl);

  printf("\nIn second image:\n");
  for (i = 0 ; i < fl->nFeatures ; i++)  {
    printf("Feature #%d:  (%f,%f) with value of %d\n",
           i, fl->feature[i]->x, fl->feature[i]->y,
           fl->feature[i]->val);
  }

  KLTWriteFeatureListToPPM(fl, img2, ncols, nrows, "feat2.ppm");
  KLTWriteFeatureList(fl, "feat2.fl", NULL);      /* binary file */
  KLTWriteFeatureList(fl, "feat2.txt", "%5.1f");  /* text file   */

  return 0;
}

