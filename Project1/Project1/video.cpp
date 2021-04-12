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

void video::MultipleVideoProcessingTest()
{
#pragma omp parallel sections
    {
#pragma omp section
        DisplayVideo("testvideo.mp4", "original");
#pragma omp section
        DisplayVideo("testvideo.mp4", "gaussian");
#pragma omp section
        DisplayVideo("testvideo.mp4", "sobel");
#pragma omp section
        DisplayVideo("testvideo.mp4", "gabor");
    }
}

int video::DisplayVideo(string strVideo, string windowName)
{
    VideoCapture cap(strVideo);
    if (!cap.isOpened()) return -1;

    Mat edges;

    namedWindow(windowName, 1);

    double fstart, fend, fprocTime;
    double fps;

    for (;;)
    {
        fstart = omp_get_wtime();

        Mat frame;
        cap >> frame;
        if (frame.empty())
        {
            destroyWindow(windowName);
            break;
        }

        if (windowName == "original")
            ;
        else if (windowName == "gaussian")
            GaussianBlur(frame, frame, Size(5, 5), 10, 10);
        else if (windowName == "sobel")
            Sobel(frame, frame, -1, 0, 1);
        else if (windowName == "gabor")
        {
            Mat kernel = getGaborKernel(Size(21, 21), 5, 1, 10, 1, 0, CV_32F);
            filter2D(frame, frame, -1, kernel);
        }

        fend = omp_get_wtime();
        fprocTime = fend - fstart;
        fps = 1 / fprocTime;
        putText(frame, "fps: " + to_string(fps), Point(50, 50), FONT_HERSHEY_SIMPLEX, 0.8, Scalar(0, 0, 255), 3);
        imshow(windowName, frame);
        waitKey(10);
    }
    return 0;
}