#pragma once
#include "Ipp.h"
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


class Openmp
{
public:
	// test operations
	void Sum();
	void Multiply();
	void fnc();

	// filter: Compare Filter2DV, Filter2DMP
	void CompareFilter2DCV_2DMP(); 

	// bilinear interpolation : compare
	void CompareBilinearInterpolation();

	// bicubic
	void CompareBicubicInterpolation();

private:
	// filter
	void Filter2DCV(Mat src, int w, int h, Mat dst, Mat element, int we, int he);
	void Filter2DMP(Mat src, int w, int h, Mat dst, Mat element, int we, int he);

	// bilinear interpolation
	void wInter(int x, int y, float* w);
	void Interp(unsigned char* src, int h, int width, float* w, int x, int y, unsigned char* output);
	void Interp_omp(unsigned char* src, int h, int width, float* w, int x, int y, unsigned char* output);

	//bicubic
	void wInter_bicubic(int x, float* w, int bicubic_num);
	void Interp_bicubic(Mat src, int h, int width, float* w, int x, Mat output, int bicubic_num);
	void Interp_bicubic_omp(Mat src, int h, int width, float* w, int x, Mat output, int bicubic_num);
};

