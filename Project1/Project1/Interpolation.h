#pragma once
#include "opencv2/opencv.hpp"

using namespace cv;

class Interpolation
{
public:
	void CompareBilinear(const char* image,int img_size, int nx, int ny, int number_of_pixels);
	void CompareBicubic(const char* image, int img_size, int nx, int ny, int number_of_pixels);
	void CompareLagrange(const char* image, int img_size, int nx, int ny, int number_of_pixels);
	void CompareBspline(const char* image, int img_size, int nx, int ny, int number_of_pixels);

private:
	// Weight
	void wInter_bilinear(int x, int y, float* w);
	void wInter_bicubic(int x, float* w, int bicubic_num);
	void wInter_lagrange(int x, float* w, int num);
	void wInter_bspline(int x, float* w, int num);

	// Interpolation
	void Interp(unsigned char* src, int h, int width, float* w, int x, int y, int num, unsigned char* output);
	void Interp_omp(unsigned char* src, int h, int width, float* w, int x, int y, int num, unsigned char* output);

	
};

