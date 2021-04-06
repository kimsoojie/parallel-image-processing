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


private:
	// filter
	void Filter2DCV(Mat src, int w, int h, Mat dst, Mat element, int we, int he);
	void Filter2DMP(Mat src, int w, int h, Mat dst, Mat element, int we, int he);

};

