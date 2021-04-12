#include "test.h"
#include "Ipp.h"
#include "Interpolation.h"
#include "video.h"
#include "Opencv.h"
#include "Openmp.h"
#include "opencv2/highgui.hpp"
#include "opencv2/opencv.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"

#include <omp.h>
#include<ipp.h>
#include <iostream>
#include <stdio.h>

using namespace cv;
using namespace std;


int main(int ac, char** av) {
    video v;
    v.Detection("person.mp4");

    //v.MultipleVideoProcessingTest();
    return 0;
}
