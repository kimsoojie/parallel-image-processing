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

#include <omp.h>
#include<ipp.h>
#include <iostream>
#include <stdio.h>
#include <bitset>

using namespace cv;
using namespace std;

int main(int ac, char** av) {
    Bayer bayer;
    bayer.Interpolation();

    return 0;

}
