
#include <iostream>
#include "cv.hpp"
#include "kltFeature.h"
#include "kltTrackingContext.h"
#include "convolve.hpp"
#include <vector>
#include "../JointRadiometicCalib/JRC.hpp"
using namespace cv;
using namespace klt;
using namespace std;
using namespace JRC;
namespace klt{
    
    #define KLT_TRACKED           0
    #define KLT_NOT_FOUND        -1
    #define KLT_SMALL_DET        -2
    #define KLT_MAX_ITERATIONS   -3
    #define KLT_OOB              -4
    #define KLT_LARGE_RESIDUE    -5

    class KLTtracker{
    public:
    kltTrackingContext tracker;
    JointRadiometicCalib jrc;
    vector<kltFeature> featureList;
    int nfeatures;
    vector<Mat> prevPyr, currPyr;
    vector<Mat> prevGradx, currGradx;
    vector<Mat> prevGrady, currGrady;
    typedef enum {SELECTING_ALL, REPLACING_SOME} selectionMode;
    KLTtracker(int nfeatures){
        this->tracker.sequentialMode = true;
        this->tracker.writeInternalImages = false;
        this->tracker.affineConsistencyCheck = -1;  /* set this to 2 to turn on affine consistency check */
        this->nfeatures = nfeatures;
        //jrc.setRFs();
    }
    void get_image();
    void createFeatureTable();

    void selectGoodFeatures(Mat Img, vector<kltFeature> &fl);
    void storeFeatureList();
    void _enforceMinimumDistance(int *pointlist, int npoints,vector<kltFeature> &fl,int cols, int rows,bool overwriteAllFeatures);
    void _KLTSelectGoodFeatures(Mat Img, vector<kltFeature> &fl, selectionMode mode);
    void trackFeatures(Mat prevImg, Mat currImg, vector<kltFeature> prevfl, vector<kltFeature> &currfl);
    void JRCtrackFeatures(Mat prevImg, Mat currImg, vector<kltFeature> prevfl, vector<kltFeature> &currfl);
    int _trackFeature(float x1, float y1, float *x2, float *y2,
                      Mat img1,Mat gradx1,Mat grady1,Mat img2,Mat gradx2,Mat grady2);
    void replaceLostFeatures(Mat Img, vector<kltFeature> &fl);

    ~KLTtracker(){

    }
    };
}