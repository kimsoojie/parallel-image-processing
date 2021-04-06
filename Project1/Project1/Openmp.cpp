#include "Ipp.h"
#include "Openmp.h"
#include "opencv2/opencv.hpp"

#include <omp.h>
#include <ipp.h>
#include <iostream>
#include <stdio.h>
#include <math.h>

using namespace cv;
using namespace std;

void Openmp::Sum()
{
	int sum = 0;
	int i;

	TickMeter tm;
	tm.start();

#pragma omp parallel for private(i) reduction(+:sum)
		for (i = 0; i < 1000000; ++i)
		{
			sum += i;
		}

		tm.stop();
		cout << "Parallel sum : ";
		cout << tm.getTimeMilli();
		cout << "\n";

// Serial sum
		tm.reset();
		tm.start();
		sum = 0;
		for (i = 0; i < 1000000; ++i)
		{
			sum += i;
		}
		tm.stop();
		cout << "Serial sum : ";
		cout << tm.getTimeMilli();
		cout << "\n";
}

void Openmp::Multiply()
{
	double multiple = 1;
	int  i;

	TickMeter tm;
	tm.start();

#pragma omp parallel 
	{
#pragma omp for private(i) reduction(+:multiple)
		for (int i = 1; i <= 1000000; ++i)
		{
			multiple *= i;
		}
	}

	tm.stop();
	cout << "Parallel multiplication : ";
	cout << tm.getTimeMilli();
	cout << "\n";

// Serial multiply
	tm.reset();
	tm.start();
	multiple = 1;
	for (int i = 0; i < 1000000; ++i)
	{
		multiple *= i;
	}
	tm.stop();
	cout << "Serial multiplication : ";
	cout << tm.getTimeMilli();
	cout << "\n";
}

void Openmp::fnc()
{
	int sum = 0;
	double multiple = 1;

	TickMeter tm;
	tm.start();

#pragma omp parallel sections
	{
#pragma omp section
		{
			for (int i = 0; i < 1000000; ++i)
			{
				sum += i;
			}
		}

#pragma omp section
		{
			for (int i = 1; i <= 1000000; ++i)
			{
				multiple *= i;
			}
	    }

    }

	tm.stop();
	cout << "using parallel sections : ";
	cout << tm.getTimeMilli();
	cout << "\n";

	tm.reset();
	tm.start();

	sum = 0;
	multiple = 1;

	for (int i = 0; i < 1000000; ++i)
	{
		sum += i;
	}

	for (int i = 1; i <= 1000000; ++i)
	{
		multiple *= i;
	}

	tm.stop();
	cout << "using serial : ";
	cout << tm.getTimeMilli();
	cout << "\n";
}

void Openmp::Filter2DCV(Mat src, int w, int h, Mat dst, Mat element, int we, int he)
{
	for (int i = 0; i < w - we; i++)
	{
		for (int j = 0; j < h - he; j++)
		{
			float val = 0;
			for (int fw = 0; fw < we; fw++)
			{
				for (int fh = 0; fh < he; fh++)
				{
					val += src.at<char>(i, j) * element.at<float>(fw, fh);
				}
			}
			dst.at<char>(i, j) = val;
		}
	}

	imshow("dst_cv", dst);
}

void Openmp::Filter2DMP(Mat src, int w, int h, Mat dst, Mat element, int we, int he)
{
	float val = 0;

#pragma omp parallel for private(val)
	for (int i = 0; i < w - we; i++)
	{
		for (int j = 0; j < h - he; j++)
		{
			val = 0;
			for (int fw = 0; fw < we; fw++)
			{
				for (int fh = 0; fh < he; fh++)
				{
					val += src.at<char>(i, j) * element.at<float>(fw, fh);
				}
			}
			dst.at<char>(i, j) = val;
		}
	}

	imshow("dst_openmp", dst);
}

void Openmp::CompareFilter2DCV_2DMP()
{
	TickMeter tm;

	// kernel
	Mat element(3, 3, CV_32F);
	float FilterElm = (float)1 / (element.rows * element.cols);
	element.at<float>(0, 0) = FilterElm;	element.at<float>(0, 1) = FilterElm;	element.at<float>(0, 2) = FilterElm;
	element.at<float>(1, 0) = FilterElm;	element.at<float>(1, 1) = FilterElm;	element.at<float>(1, 2) = FilterElm;
	element.at<float>(2, 0) = FilterElm;	element.at<float>(2, 1) = FilterElm;	element.at<float>(2, 2) = FilterElm;

	//Mat element(5, 5, CV_32F);
	//float FilterElm = (float)1 / (element.rows * element.cols);
	//element.at<float>(0, 0) = FilterElm;	element.at<float>(0, 1) = FilterElm;	element.at<float>(0, 2) = FilterElm; element.at<float>(0, 3) = FilterElm;   element.at<float>(0, 4) = FilterElm;
	//element.at<float>(1, 0) = FilterElm;	element.at<float>(1, 1) = FilterElm;	element.at<float>(1, 2) = FilterElm; element.at<float>(1, 3) = FilterElm;	element.at<float>(1, 4) = FilterElm;
	//element.at<float>(2, 0) = FilterElm;	element.at<float>(2, 1) = FilterElm;	element.at<float>(2, 2) = FilterElm; element.at<float>(2, 3) = FilterElm;	element.at<float>(2, 4) = FilterElm;
	//element.at<float>(3, 0) = FilterElm;	element.at<float>(3, 1) = FilterElm;	element.at<float>(3, 2) = FilterElm; element.at<float>(3, 3) = FilterElm;	element.at<float>(3, 4) = FilterElm;
	//element.at<float>(4, 0) = FilterElm;	element.at<float>(4, 1) = FilterElm;	element.at<float>(4, 2) = FilterElm; element.at<float>(4, 3) = FilterElm;	element.at<float>(4, 4) = FilterElm;

	cout << "<filter>" << endl;
	for (int i = 0; i < element.rows; i++) {
		for (int j = 0; j < element.cols; j++) {
			cout << element.at<float>(i, j) << "   ";
		}
		cout << endl;
	}
	cout << endl;

	// load src image
	Mat src = imread("hw1_2.jpg", 0);
	Mat dstCV(src.size().width, src.size().height, CV_8UC1);
	Mat dstMP(src.size().width, src.size().height, CV_8UC1);

	// src , kernel size
	int width = src.size().width;		int height = src.size().height;
	int eWidht = element.cols;	int eHeight = element.rows;
	cout << "<image size>" << endl;
	cout << "width : " << width << "   height : " << height << endl;
	cout << endl;
	cout << "<filter size>" << endl;
	cout << "eWidht : " << eWidht << "   eHeight : " << eHeight << endl;
	cout << endl;

	tm.start();
	//Serial 
	Filter2DCV(src, width, height, dstCV, element, eWidht, eHeight);
	tm.stop();
	cout << "\nProcessTime-Serial : ";
	cout << tm.getTimeMilli();

	tm.reset();
	tm.start();
	//OpenMP
	Filter2DMP(src, width, height, dstMP, element, eWidht, eHeight);
	tm.stop();
	cout << "\nProcessTime-OpenMP : ";
	cout << tm.getTimeMilli();


	imshow("src", src);
	waitKey(0);
}
