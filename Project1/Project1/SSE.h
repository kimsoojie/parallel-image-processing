#pragma once
#include "opencv2/highgui.hpp"
#include "opencv2/opencv.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"

#include <omp.h>
#include<ipp.h>
#include <iostream>
#include <stdio.h>

#include <emmintrin.h>
#include <immintrin.h>
#include <zmmintrin.h>

using namespace cv;
using namespace std;


class SSE
{
public:
	// practice
	void ArraySum();
	void CalcSumSSE();
	void CalcSumAVX();
	void CalcSqrt();
	int Practice();

	// hw5
	void MeanFilter();
	void Filter2DCV(Mat src, int w, int h, Mat dst, Mat element, int we, int he);
	void Filter2DMP(Mat src, int w, int h, Mat dst, Mat element, int we, int he);
	void Filter2DSSE(Mat src, int w, int h, Mat dst, Mat element, int we, int he);

	// hw6
	void SSEmean_8bit(Mat src, int w, int h, Mat dst, Mat element, int we, int he);
};

