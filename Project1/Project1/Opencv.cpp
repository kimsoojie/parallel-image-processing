#include "Opencv.h"
#include "opencv2/highgui.hpp"
#include "opencv2/opencv.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"

#include <iostream>
#include <stdio.h>

using namespace cv;
using namespace std;

void Opencv::GaussianFilter()
{
	int size = 2048;
	Mat img = imread("hw1_2.jpg", 0);
	resize(img, img, Size(size, size));

	TickMeter tm;
	tm.start();

	Mat img_blur;
	GaussianBlur(img, img_blur, Size(5, 5), 10, 10);

	tm.stop();
	cout << "\nOpencv : ";
	cout << tm.getTimeMilli();

	imshow("opencv_gaussian", img_blur);
	waitKey(0);
}

void Opencv::MedianFilter()
{
	int size = 2048;
	Mat img = imread("hw1_2.jpg", 0);
	resize(img, img, Size(size, size));

	TickMeter tm;
	tm.start();

	Mat img_blur;
	medianBlur(img, img_blur, 5);

	tm.stop();
	cout << "\nOpencv : ";
	cout << tm.getTimeMilli();

	imshow("opencv_median", img_blur);
	waitKey(0);
}