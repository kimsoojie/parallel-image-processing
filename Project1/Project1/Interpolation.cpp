#include "Interpolation.h"
#include "Openmp.h"
#include "opencv2/highgui.hpp"
#include "opencv2/opencv.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"

#include <omp.h>
#include<ipp.h>
#include <iostream>
#include <stdio.h>
#include <math.h>

using namespace cv;
using namespace std;

void Interpolation::Interp(unsigned char* src, int hg, int wd, float* w, int x, int y, int num, unsigned char* output) {

	x = x - 1;
	y = y - 1;
	int r, c, i, j, nc, nr, size;
	size = num / 2;

	int nwd = wd * (x + 1);

	float temp;

	for (r = 0; r < hg; r++) {
		for (c = 0 + size - 1; c < wd - size; c++)
		{
			nr = r * (y + 1);
			nc = c * (x + 1);

			output[nr * nwd + nc] = src[r * wd + c];

			for (i = 0; i < x; i++)
			{
				nc = c * (x + 1) + i + 1;
				temp = 0;
				for (j = 0; j < size * 2; j++)
					temp += w[i * (size * 2) + j] * (float)src[r * wd + c - size + j + 1];
				
				output[nr * nwd + nc] = (unsigned char)((int)(temp + 0.5));
			}
		}
	}


	int ntemp;

	for (r = 0 + size - 1; r < hg - size; r++) {
		for (c = 0 * (x + 1); c < wd * (x + 1) + x; c++)
		{
			for (i = 0; i < y; i++)
			{
				nr = r * (y + 1) + i + 1;
				temp = 0;
				for (j = 0; j < size * 2; j++)
				{
					ntemp = (r - size + j + 1) * (y + 1);
					temp += w[i * (size * 2) + j] * (float)output[ntemp * nwd + c];
				}

				output[nr * nwd + c] = (unsigned char)((int)(temp + 0.5));
			}
		}
	}

}

void Interpolation::Interp_omp(unsigned char* src, int hg, int wd, float* w, int x, int y, int num, unsigned char* output) {

	x = x - 1;
	y = y - 1;
	int size;
	size = num / 2;

	int nwd = wd * (x + 1);

#pragma omp parallel for
	for (int r = 0; r < hg; r++) {
		for (int c = 0 + size - 1; c < wd - size; c++)
		{
			int nr = r * (y + 1);
			int nc = c * (x + 1);

			output[nr * nwd + nc] = src[r * wd + c];

			for (int i = 0; i < x; i++)
			{
				nc = c * (x + 1) + i + 1;
				float temp = 0;
				for (int j = 0; j < size * 2; j++)
					temp += w[i * (size * 2) + j] * (float)src[r * wd + c - size + j + 1];

				output[nr * nwd + nc] = (unsigned char)((int)(temp + 0.5));
			}
		}
	}

#pragma omp parallel for
	for (int r = 0 + size - 1; r < hg - size; r++) {
		for (int c = 0 * (x + 1); c < wd * (x + 1) + x; c++)
		{
			for (int i = 0; i < y; i++)
			{
				int nr = r * (y + 1) + i + 1;
				float temp = 0;
				for (int j = 0; j < size * 2; j++)
				{
					int ntemp = (r - size + j + 1) * (y + 1);
					temp += w[i * (size * 2) + j] * (float)output[ntemp * nwd + c];
				}

				output[nr * nwd + c] = (unsigned char)((int)(temp + 0.5));
			}
		}
	}

}

void Interpolation::wInter_bilinear(int x, int y, float* w)
{
	x = x - 1;
	y = y - 1;
	int i;

	for (i = 0; i < x; i++)
	{
		w[i * 2 + 0] = 1 - (float)(i + 1) / (float)(x + 1); // 0, 2 -> 1-1/3, 1-2/3 
		w[i * 2 + 1] = (float)(i + 1) / (float)(x + 1);	// 1, 3 -> 1/3, 2/3
	}
}

void Interpolation::wInter_bicubic(int nx, float* w, int bicubic_num = 4)
{
	double a = -0.5;

	for (int i = 0; i < (nx - 1) * bicubic_num; i += bicubic_num)
	{
		int new_idx = i / bicubic_num;

		for (int j = 0; j < bicubic_num; j++)
		{
			double x = abs((j - (1 + (double)(new_idx + 1) / (double)nx)));

			if (x >= 0 && x < 1)
				w[i + j] = (a + 2) * pow(x, 3) - (a + 3) * pow(x, 2) + 1;
			else if (x >= 1 && x < 2)
				w[i + j] = a * pow(x, 3) - 5 * a * pow(x, 2) + 8 * a * x - 4 * a;
			else
				w[i + j] = 0;

			cout << w[i + j] << "   ";
		}
		cout << "\n";
	}
}

void Interpolation::wInter_lagrange(int nx, float* w, int num = 4)
{
	for (int i = 0; i < (nx - 1) * num; i += num)
	{
		int new_idx = i / num;

		for (int j = 0; j < num; j++)
		{
			double x = abs((j - (1 + (double)(new_idx + 1) / (double)nx)));

			if (x >= 0 && x < 1)
				w[i + j] = (1.0 / 2.0) * pow(x, 3) - pow(x, 2) - (1.0 / 2.0) * x + 1;
			else if (x >= 1 && x < 2)
				w[i + j] = -(1.0 / 6.0) * pow(x, 3) + pow(x, 2) - (11.0 / 6.0) * x + 1;
			else
				w[i + j] = 0;

			cout << w[i + j] << "   ";
		}
		cout << "\n";
	}
}

void Interpolation::wInter_bspline(int nx, float* w, int num = 4)
{
	for (int i = 0; i < (nx - 1) * num; i += num)
	{
		int new_idx = i / num;

		for (int j = 0; j < num; j++)
		{
			double x = abs((j - (1 + (double)(new_idx + 1) / (double)nx)));

			if (x >= 0 && x < 1)
				w[i + j] = 0.5 * pow(x, 3) - pow(x, 2) + (2.0 / 3.0);
			else if (x >= 1 && x < 2)
				w[i + j] = -(1.0 / 6.0) * pow(x, 3) + pow(x, 2) - 2.0 * x + (4.0 / 3.0);
			else
				w[i + j] = 0;

			cout << w[i + j] << "   ";
		}
		cout << "\n";
	}
}

void Interpolation::CompareBilinear(const char* image, int img_size, int nx = 3, int ny = 3, int number_of_pixels = 2)
{
	TickMeter tm;

	int n = nx;
	int bilinear = number_of_pixels;

	// load src image
	Mat src = imread(image, 0);
	resize(src, src, Size(img_size, img_size));
	Mat dst_Serial(src.size().width * n, src.size().height * n, CV_8UC1);
	Mat dst_Openmp(src.size().width * n, src.size().height * n, CV_8UC1);

	// calculate weight
	float* w = new float[bilinear * (n - 1)];
	wInter_bilinear(n, n, w);

	int iteration = 10;
	double avg_time = 0.0;

	// Interpolation (Bilinear)
	tm.start();
	for (int i = 0; i < iteration; ++i)
		Interp((unsigned char*)src.data, src.size().height, src.size().width, w, n, n, bilinear, (unsigned char*)dst_Serial.data);
	tm.stop();
	avg_time = tm.getTimeMilli() / iteration;

	cout << "\nProcessTime-Serial : ";
	cout << avg_time;

	// Interpolation omp (Bilinear)
	avg_time = 0.0;
	tm.reset();
	tm.start();
	for (int i = 0; i < iteration; ++i)
		Interp_omp(src.data, src.size().height, src.size().width, w, n, n, bilinear, dst_Openmp.data);
	tm.stop();
	avg_time = tm.getTimeMilli() / iteration;

	cout << "\nProcessTime-Openmp : ";
	cout << avg_time << "\n";

	// Show result image
	imshow("src", src);
	imshow("dst_Serial", dst_Serial);
	imshow("dst_Openmp", dst_Openmp);
	waitKey(0);
}

void Interpolation::CompareBicubic(const char* image, int img_size, int nx = 3, int ny = 3, int number_of_pixels = 4)
{
	TickMeter tm;

	int n = nx;
	int bicubic = number_of_pixels;

	// load src image
	Mat src = imread(image, 0);
	resize(src, src, Size(img_size, img_size));
	Mat dst_Serial(src.size().width * n, src.size().height * n, CV_8UC1);
	Mat dst_Openmp(src.size().width * n, src.size().height * n, CV_8UC1);

	// calculate weight
	float* w = new float[bicubic * (n - 1)];
	wInter_bicubic(n, w, bicubic);

	int iteration = 10;
	double avg_time = 0.0;

	// Interpolation (Bicubic)
	tm.start();
	for (int i = 0; i < iteration; ++i)
		Interp((unsigned char*)src.data, src.size().height, src.size().width, w, n, n, bicubic, (unsigned char*)dst_Serial.data);
	tm.stop();
	avg_time = tm.getTimeMilli() / iteration;

	cout << "\nProcessTime-Serial : ";
	cout << avg_time;

	// Interpolation omp (Bicubic)
	avg_time = 0.0;
	tm.reset();
	tm.start();
	for (int i = 0; i < iteration; ++i)
		Interp_omp(src.data, src.size().height, src.size().width, w, n, n, bicubic, dst_Openmp.data);
	tm.stop();
	avg_time = tm.getTimeMilli() / iteration;

	cout << "\nProcessTime-Openmp : ";
	cout << avg_time << "\n";

	// Show result image
	imshow("src", src);
	imshow("dst_Serial", dst_Serial);
	imshow("dst_Openmp", dst_Openmp);
	waitKey(0);
}

void Interpolation::CompareLagrange(const char* image, int img_size, int nx = 3, int ny = 3, int number_of_pixels = 4)
{
	TickMeter tm;

	int n = nx;
	int lagrange = number_of_pixels;

	// load src image
	Mat src = imread(image, 0);
	resize(src, src, Size(img_size, img_size));
	Mat dst_Serial(src.size().width * n, src.size().height * n, CV_8UC1);
	Mat dst_Openmp(src.size().width * n, src.size().height * n, CV_8UC1);

	// calculate weight
	float* w = new float[lagrange * (n - 1)];
	wInter_lagrange(n, w, lagrange);

	int iteration = 10;
	double avg_time = 0.0;

	// Interpolation (Lagrange)
	tm.start();
	for (int i = 0; i < iteration; ++i)
		Interp((unsigned char*)src.data, src.size().height, src.size().width, w, n, n, lagrange, (unsigned char*)dst_Serial.data);
	tm.stop();
	avg_time = tm.getTimeMilli() / iteration;

	cout << "\nProcessTime-Serial : ";
	cout << avg_time;

	// Interpolation omp (Lagrange)
	avg_time = 0.0;
	tm.reset();
	tm.start();
	for (int i = 0; i < iteration; ++i)
		Interp_omp(src.data, src.size().height, src.size().width, w, n, n, lagrange, dst_Openmp.data);
	tm.stop();
	avg_time = tm.getTimeMilli() / iteration;

	cout << "\nProcessTime-Openmp : ";
	cout << avg_time << "\n";

	// Show result image
	imshow("src", src);
	imshow("dst_Serial", dst_Serial);
	imshow("dst_Openmp", dst_Openmp);
	waitKey(0);
}

void Interpolation::CompareBspline(const char* image, int img_size, int nx = 3, int ny = 3, int number_of_pixels = 4)
{
	TickMeter tm;

	int n = nx;
	int bspline = number_of_pixels;

	// load src image
	Mat src = imread(image, 0);
	resize(src, src, Size(img_size, img_size));
	Mat dst_Serial(src.size().width * n, src.size().height * n, CV_8UC1);
	Mat dst_Openmp(src.size().width * n, src.size().height * n, CV_8UC1);

	// calculate weight
	float* w = new float[bspline * (n - 1)];
	wInter_bspline(n, w, bspline);

	int iteration = 10;
	double avg_time = 0.0;

	// Interpolation (B-Spline)
	tm.start();
	for (int i = 0; i < iteration; ++i)
		Interp((unsigned char*)src.data, src.size().height, src.size().width, w, n, n, bspline, (unsigned char*)dst_Serial.data);
	tm.stop();
	avg_time = tm.getTimeMilli() / iteration;

	cout << "\nProcessTime-Serial : ";
	cout << avg_time;

	// Interpolation omp (B-Spline)
	avg_time = 0.0;
	tm.reset();
	tm.start();
	for (int i = 0; i < iteration; ++i)
		Interp_omp(src.data, src.size().height, src.size().width, w, n, n, bspline, dst_Openmp.data);
	tm.stop();
	avg_time = tm.getTimeMilli() / iteration;

	cout << "\nProcessTime-Openmp : ";
	cout << avg_time << "\n";

	// Show result image
	imshow("src", src);
	imshow("dst_Serial", dst_Serial);
	imshow("dst_Openmp", dst_Openmp);
	waitKey(0);
}