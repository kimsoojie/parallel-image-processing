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
	void MultipleVideoProcessingTest(); //practice
	void Detection(string str);

private:
	int DisplayVideo(string str, string windowName); //practice
	
	bool Grab(string video, vector<Mat>& cap_frames);
	void Display_Original(Mat cap_frame, string windowName);
	void Display_Face(Mat cap_frame, string windowName);
	void Display_Body(Mat cap_frame, string windowName);
	void FaceDetect(Mat cap_frame);
	void BodyDetect(Mat cap_frame);

};

