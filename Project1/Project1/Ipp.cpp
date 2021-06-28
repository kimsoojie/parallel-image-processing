#include "Ipp.h"
#include "opencv2/highgui.hpp"
#include "opencv2/opencv.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"

#include<ipp.h>
#include <iostream>
#include <stdio.h>

using namespace cv;
using namespace std;

void Ipp::GaussianFilter()
{
    Mat image = imread("Grab_Image.bmp",0);
    //int imsize = 2048;
    //resize(image, image, Size(imsize, imsize));
    
    TickMeter tm;
    tm.start();

    //ipp size
    IppiSize size, tsize;
    size.width = image.cols;
    size.height = image.rows;
    
    // source(S_img), buffer(T_img), target(T) image malloc
    Ipp8u* S_img = (Ipp8u*)ippsMalloc_8u(size.width * size.height);
    unsigned char* buffer = 0;
    Ipp8u* T = (Ipp8u*)ippsMalloc_8u(size.width * size.height);
    
    // copy : image -> S_img
    ippiCopy_8u_C1R((const Ipp8u*)image.data, size.width, S_img, size.width, size);
    
    // sobel s_img -> T
    tsize.width = image.cols;
    tsize.height = image.rows;
    
    int specSize = 0;
    int bufSize = 0;
    int flag = 0;
    
    unsigned int kernelSize = 5;
    IppFilterGaussianSpec* pSpec = NULL;
    flag = ippiFilterGaussianGetBufferSize(tsize, kernelSize, ipp8u, 1, &specSize, &bufSize);
    pSpec = (IppFilterGaussianSpec*)malloc(specSize);
    buffer = (unsigned char*)malloc(bufSize);
    
    flag = ippiFilterGaussianInit(tsize, kernelSize, 3.0f, ippBorderMirror, ipp8u, 1, pSpec, buffer);
    ippiFilterGaussianBorder_8u_C1R(S_img, size.width, T, size.width, tsize, ippBorderConst, pSpec, buffer);
    
    
    //show T->s
    Size s;
    s.width = image.cols;
    s.height = image.rows;
    cv::Mat dst(s, CV_8U, T); 

    tm.stop();
    cout << "Ipp : ";
    cout << tm.getTimeMilli();
    
    imwrite("ipp.bmp", dst);
    imshow("image", image);
    imshow("gaussian", dst);
    waitKey(0);
}

void Ipp::MedianFilter()
{
    Mat image = imread("hw1_2.jpg", 0);
    int imsize = 2048;
    resize(image, image, Size(imsize, imsize));

    TickMeter tm;
    tm.start();

    //ipp size
    IppiSize size, tsize;
    size.width = image.cols;
    size.height = image.rows;

    // source(S_img), buffer(T_img), target(T) image malloc
    Ipp8u* S_img = (Ipp8u*)ippsMalloc_8u(size.width * size.height);
    Ipp8u* T_img = (Ipp8u*)ippsMalloc_8u(size.width * size.height);
    Ipp8u* T = (Ipp8u*)ippsMalloc_8u(size.width * size.height);

    // copy : image -> S_img
    ippiCopy_8u_C1R((const Ipp8u*)image.data, size.width, S_img, size.width, size);

    // sobel s_img -> T
    tsize.width = image.cols;
    tsize.height = image.rows;

    IppiSize maskSize = { 5, 5 };
    int numChannels = 1;
    ippiFilterMedianBorder_8u_C1R(S_img, size.width, T, size.width, tsize, maskSize, ippBorderConst, numChannels, T_img);

    //show T->s
    Size s;
    s.width = image.cols;
    s.height = image.rows;
    cv::Mat dst(s, CV_8U, T);

    tm.stop();
    cout << "Ipp : ";
    cout << tm.getTimeMilli();

    imshow("image", image);
    imshow("median", dst);
    waitKey(0);
}

