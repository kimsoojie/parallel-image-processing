#include "test.h"
#include "Ipp.h"
#include "Interpolation.h"
#include "video.h"
#include "Opencv.h"
#include "Openmp.h"
#include "Bayer.h"
#include "opencv2/highgui.hpp"
#include "opencv2/opencv.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"
#include "InstructionSet.h"
#include "SSE.h"

#include <omp.h>
#include<ipp.h>
#include <iostream>
#include <stdio.h>
#include <bitset>

#include <vector>   
#include <array>  
#include <string>  
#include <intrin.h>  

using namespace cv;
using namespace std;



int main(int ac, char** av) {
    //Bayer bayer;
    //bayer.Interpolation();

    //InstructionSet SSE_CHECK;
    //SSE_CHECK.check();

    //SSE sse;
    //sse.ArraySum();
    //sse.CalcSumSSE();
    //sse.CalcSumAVX();
    //sse.CalcSqrt();
    //sse.MeanFilter();
    //sse.Practice();
    //sse.MeanFilter();
    
    //Ipp ipp;
    //ipp.GaussianFilter();

    //Openmp omp;
    //omp.CompareFilter2DCV_2DMP();
    
    //Mat img_pro = imread("./panorama/r1.jpg", 0);
    //Mat img_auto = imread("./panorama/r2.jpg", 0);
    //Mat img_apap = imread("./panorama/easy.jpg", 0);
    //Mat dst_pro, dst_auto, dst_apap;
    //
    //medianBlur(img_pro, img_pro, 3);
    //medianBlur(img_auto, img_auto, 3);
    //medianBlur(img_apap, img_apap, 3);
    //
    //Canny(img_pro, dst_pro, 20, 40);
    //Canny(img_auto, dst_auto, 20, 40);
    //Canny(img_apap, dst_apap, 0, 50);
    //
    //imwrite("./panorama/dr1.jpg", dst_pro);
    //imwrite("./panorama/dr2.jpg", dst_auto);
    //imwrite("./panorama/dst_easy.jpg", dst_apap);

    return 0;

}
