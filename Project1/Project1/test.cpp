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

    InstructionSet SSE_CHECK;
    SSE_CHECK.check();

    SSE sse;
    sse.ArraySum();
    sse.CalcSumSSE();
    sse.CalcSumAVX();
    sse.CalcSqrt();

    return 0;

}
