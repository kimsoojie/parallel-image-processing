#include "Ipp.h"
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

void Openmp::wInter(int x, int y, float* w)
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

void Openmp::Interp(unsigned char* src, int hg, int wd, float* w, int x, int y, unsigned char* output) {

	x = x - 1;
	y = y - 1;
	int r, c, i, j, nc, nr, size;
	size = 1;

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

void Openmp::Interp_omp(unsigned char* src, int hg, int wd, float* w, int x, int y, unsigned char* output) {

	x = x - 1;
	y = y - 1;
	int size;
	size = 1;

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

void Openmp::CompareBilinearInterpolation()
{
	TickMeter tm;

	int nx = 3;
	int ny = 3;

	// load src image
	Mat src = imread("hw1_2.jpg", 0);
	Mat dst_Serial(src.size().width * nx, src.size().height * ny, CV_8UC1);
	Mat dst_Openmp(src.size().width * nx, src.size().height * ny, CV_8UC1);

	int wd = src.size().width;
	int hg = src.size().height;

	int row, col;

	float* input = new float[wd * hg];
	memset(input, 0, wd * hg * sizeof(float));

	float* output = new float[(wd * nx) * (hg * ny)];
	memset(output, 0, (wd * nx) * (hg * ny) * sizeof(float));

	for (row = 0; row < hg; row++)
		for (col = 0; col < wd; col++)
			input[row * wd + col] = (float)src.at<char>(row, col);

	float* w = new float[nx * 8];
	memset(w, 0, nx * 8 * sizeof(float));

	wInter(nx, ny, w);
	
	int iteration = 10;
	double avg_time = 0.0;

	tm.start();
	for (int i = 0; i < iteration; ++i)
		Interp((unsigned char*)src.data, src.size().height, src.size().width, w, nx, ny, (unsigned char*)dst_Serial.data);
	tm.stop();
	avg_time = tm.getTimeMilli()/iteration;

	cout << "\nProcessTime-Serial : ";
	cout << avg_time;
	
	avg_time = 0.0;
	tm.reset();
	tm.start();
	for (int i = 0; i < iteration; ++i)
		Interp_omp((unsigned char*)src.data, src.size().height, src.size().width, w, nx, ny, (unsigned char*)dst_Openmp.data);
	tm.stop();
	avg_time = tm.getTimeMilli()/ iteration;

	cout << "\nProcessTime-Openmp : ";
	cout << avg_time;

	imshow("src", src);
	imshow("dst_Serial", dst_Serial);
	imshow("dst_Openmp", dst_Openmp);
	waitKey(0);
}

void Openmp::CompareBicubicInterpolation()
{
	int nx = 3;
	int bicubic_num = 4;

	float* w = new float[bicubic_num * (nx - 1)];
	wInter_bicubic(nx, w, bicubic_num);
}


void Openmp::wInter_bicubic(int nx, float* w, int bicubic_num=4)
{
	double a = -0.5;
	
	for (int i = 0; i < (nx - 1) * bicubic_num; i += bicubic_num)
	{
		int new_idx = i / bicubic_num;
		
		for (int j = 0; j < bicubic_num; j++)
		{
			double x = abs(((1 + (double)(new_idx + 1) / (double)nx) - j));
			
			if (x >= 0 && x < 1)
				w[i + j] = (a + 2) * x * x * x - (a + 3) * x * x + 1;
			else if (x >= 1 && x < 2)
				w[i + j] = a * x * x * x - 5 * a * x * x + 8 * a * x - 4 * a;
			else
				w[i + j] = 0;

			//cout << w[i + j] << "   ";
		}
		//cout << "\n";
	}
}

void Openmp::Interp_bicubic(unsigned char* src, int hg, int wd, float* w, int x, int y, unsigned char* output)
{

}