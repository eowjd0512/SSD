
#include <iostream>
#include "cv.hpp"

using namespace cv;
using namespace std;
namespace klt{

    #define KLT_BOOL int

class kltTrackingContext{
    public:
    /* Available to user */
    int mindist;			/* min distance b/w features */
    int window_width, window_height;
    bool sequentialMode;	/* whether to save most recent image to save time */
    /* can set to TRUE manually, but don't set to */
    /* FALSE manually */
    bool smoothBeforeSelecting;	/* whether to smooth image before */
    /* selecting features */
    bool writeInternalImages;	/* whether to write internal images */
    /* tracking features */
    bool lighting_insensitive;  /* whether to normalize for gain and bias (not in original algorithm) */
    
    /* Available, but hopefully can ignore */
    int min_eigenvalue;		/* smallest eigenvalue allowed for selecting */
    float min_determinant;	/* th for determining lost */
    float min_displacement;	/* th for stopping tracking when pixel changes little */
    int max_iterations;		/* th for stopping tracking when too many iterations */
    float max_residue;		/* th for stopping tracking when residue is large */
    float grad_sigma;
    float smooth_sigma_fact;
    float pyramid_sigma_fact;
    float step_factor;  /* size of Newton steps; 2.0 comes from equations, 1.0 seems to avoid overshooting */
    int nSkippedPixels;		/* # of pixels skipped when finding features */
    int borderx;			/* border in which features will not be found */
    int bordery;
    int nPyramidLevels;		/* computed from search_ranges */
    int subsampling;		/* 		" */
    static const int search_range = 15;
    
    /* for affine mapping */ 
    int affine_window_width, affine_window_height;
    int affineConsistencyCheck; /* whether to evaluates the consistency of features with affine mapping 
                                -1 = don't evaluates the consistency
                                0 = evaluates the consistency of features with translation mapping
                                1 = evaluates the consistency of features with similarity mapping
                                2 = evaluates the consistency of features with affine mapping
    */
    int affine_max_iterations;  
    float affine_max_residue;
    float affine_min_displacement;        
    float affine_max_displacement_differ; /* th for the difference between the displacement calculated 
    by the affine tracker and the frame to frame tracker in pel*/

    /* User must not touch these */
    vector<Mat> pyramid_last;
    vector<Mat> pyramid_last_gradx;
    vector<Mat> pyramid_last_grady;
    
    //Mat kernal;
    kltTrackingContext(){
         /* Set values to default values */
        mindist = 10;
        window_width = 7;
        window_height = 7;
        sequentialMode = true;
        smoothBeforeSelecting = true;
        writeInternalImages = true;
        lighting_insensitive = false;
        min_eigenvalue = 1;
        min_determinant = 0.01f;
        max_iterations = 10;
        min_displacement = 0.1f;
        max_residue = 10.0f;
        grad_sigma = 1.0f;
        smooth_sigma_fact = 0.1f;
        pyramid_sigma_fact = 0.9f;
        step_factor = 1.0f;
        nSkippedPixels = 0;
        
        /* for affine mapping */
        affineConsistencyCheck = -1;
        affine_window_width = 15;
        affine_window_height = 15;
        affine_max_iterations = 10;
        affine_max_residue = 10.0;
        affine_min_displacement = 0.2f;
        affine_max_displacement_differ = 1.5f;
        KLTChangeTCPyramid();
        KLTUpdateTCBorder();
    }
    void KLTChangeTCPyramid();
    void KLTUpdateTCBorder();
    ~kltTrackingContext(){}
};

};