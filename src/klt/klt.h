
#include <iostream>
#include "cv.hpp"
#include "kltFeature.h"
#include "kltTrackingContext.h"
#include <vector>

using namespace cv;
using namespace klt;
using namespace std;
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
    }
    void get_image();
    void createFeatureTable();

    void selectGoodFeatures(Mat Img, vector<kltFeature> &fl);
    void storeFeatureList();
    void _enforceMinimumDistance(int *pointlist, int npoints,vector<kltFeature> &fl,int cols, int rows,bool overwriteAllFeatures);
    void _KLTSelectGoodFeatures(Mat Img, vector<kltFeature> &fl, selectionMode mode);
    void trackFeatures(Mat currImg, Mat prevImg, vector<kltFeature> currfl, vector<kltFeature> &prevfl);
    void replaceLostFeatures(Mat Img, vector<kltFeature> &fl);

    ~KLTtracker(){

    }
    };
}