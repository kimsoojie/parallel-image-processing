#pragma once
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


class video
{
public:
	void MultipleVideoProcessingTest(); //test
	void Detection(string str);

private:
	int DisplayVideo(string str, string windowName); //test
	
	void Display_Original(Mat cap_frame, string windowName);
	void Display_Face(Mat cap_frame, string windowName);
	void Display_Eye(Mat cap_frame, string windowName);
	void FaceDetect(Mat cap_frame);
	void EyeDetect(Mat cap_frame);

};

