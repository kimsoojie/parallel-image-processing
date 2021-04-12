#pragma once
#include <iostream>
#include <stdio.h>
using namespace std;

class video
{
public:
	void MultipleVideoProcessingTest();
	void FaceDetectTest();

private:
	int DisplayVideo(string str, string windowName);
	void FaceDetect(string str, string windowName);
	void BodyDetect(string str, string windowName);
};

