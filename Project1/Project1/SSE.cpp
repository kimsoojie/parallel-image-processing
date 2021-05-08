#include "SSE.h"
#include <stdio.h>
#include <iostream>
#include <emmintrin.h>
#include <immintrin.h>
#include <zmmintrin.h>

using namespace std;

void SSE::ArraySum()
{
	short A[8] = { 1,2,3,4,5,6,7,8 };
	short B[8] = { 1,2,3,4,5,6,7,8 };
	short C[8] = { 0};
	short D[8] = { 0};

	// c program
	//for (int i = 0; i < 8; i++)
	//{
	//	C[i] = A[i] + B[i];
	//	cout << C[i] << " ";
	//}

	// SIMD program
	__m128i xmmA = _mm_loadu_si128((__m128i*)A);
	__m128i xmmB = _mm_loadu_si128((__m128i*)B);
	__m128i xmmC = _mm_add_epi16(xmmA, xmmB);
	_mm_storeu_si128((__m128i*)D, xmmC);
	
	printf("%d %d %d %d %d %d %d %d \n", D[0], D[1], D[2], D[3], D[4], D[5], D[6], D[7]);
}

void SSE::CalcSumSSE()
{
	float sum = 0;
	__m128 avxSum = _mm_setzero_ps();
	__m128 avxCurVal = _mm_setzero_ps();

	float data[8] = { 1,2,3,4,5,6,7,8 };
	float* pData = data;
	int size = 8;

	for (int i = 0; i < size; i += 4)
	{
		avxCurVal = _mm_loadu_ps(pData + i);
		avxSum = _mm_add_ps(avxSum, avxCurVal);
	}
	for (int i = 0; i < 4; i++)
	{
		sum += *((float*)(&avxSum) + i);
	}
	cout << sum << endl;
}

void SSE::CalcSumAVX()
{
	float sum = 0;

	float data[8] = { 1,2,3,4,5,6,7,8 };
	float* pData = data;
	int size = 8;

	__m256 avxSum = _mm256_setzero_ps();
	__m256 avxCurVal = _mm256_setzero_ps();

	for (int i = 0; i < size; i += 8)
	{
		avxCurVal = _mm256_loadu_ps(pData + i);
		avxSum = _mm256_add_ps(avxSum, avxCurVal);
	}
	for (int i = 0; i < 8; i++)
	{
		sum += *((float*)(&avxSum) + i);
	}
	cout << sum << endl;

}

void SSE::CalcSqrt()
{
	float data1[8] = { 1,2,3,4,5,6,7,8 };
	float data2[8] = { 1,2,3,4,5,6,7,8 };
	float* pData1 = data1;
	float* pData2 = data2;
	int size = 8;

	__m256 avxSqrt = _mm256_setzero_ps();
	__m256 avxMul1 = _mm256_setzero_ps();
	__m256 avxMul2 = _mm256_setzero_ps();
	__m256 avxSum = _mm256_setzero_ps();

	__m256 avxCurVal1 = _mm256_setzero_ps();
	__m256 avxCurVal2 = _mm256_setzero_ps();

	for (int i = 0; i < size; i += 8)
	{
		avxCurVal1 = _mm256_loadu_ps(pData1 + i);
		avxCurVal2 = _mm256_loadu_ps(pData2 + i);
		avxMul1 = _mm256_mul_ps(avxCurVal1,avxCurVal1);
		avxMul2 = _mm256_mul_ps(avxCurVal2,avxCurVal2);
		avxSum = _mm256_add_ps(avxMul1, avxMul2);
		avxSqrt = _mm256_sqrt_ps(avxSum);
	}


	for (int i = 0; i < 8; i++)
	{
		cout << *((float*)(&avxSqrt) + i) << " ";
	}
	cout << endl;
}

void SSE::MeanFilter()
{
	TickMeter tm;

	// kernel
	Mat element(3, 3, CV_32F);
	float FilterElm = (float)1 / (element.rows * element.cols);
	element.at<float>(0, 0) = FilterElm;	element.at<float>(0, 1) = FilterElm;	element.at<float>(0, 2) = FilterElm;
	element.at<float>(1, 0) = FilterElm;	element.at<float>(1, 1) = FilterElm;	element.at<float>(1, 2) = FilterElm;
	element.at<float>(2, 0) = FilterElm;	element.at<float>(2, 1) = FilterElm;	element.at<float>(2, 2) = FilterElm;

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
	Mat dstSSE(src.size().width, src.size().height, CV_8UC1);

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

	tm.reset();
	tm.start();
	//SSE
	Filter2DSSE(src, width, height, dstSSE, element, eWidht, eHeight);
	tm.stop();
	cout << "\nProcessTime-SSE : ";
	cout << tm.getTimeMilli();

	imshow("src", src);
	waitKey(0);
}

void SSE::Filter2DCV(Mat src, int w, int h, Mat dst, Mat element, int we, int he)
{
	for (int i = 1; i < h - 1; i++)
	{
		for (int j = 1; j < w - 1; j++)
		{
			float val = 0;
			for (int fh = 0; fh < he; fh++)
			{
				for (int fw = 0; fw < we; fw++)
				{
					val += (src.at<uchar>(i - 1 + fh, j - 1 + fw) * element.at<float>(fh, fw));
				}
			}
			dst.at<char>(i, j) = val;
		}
	}

	imshow("dst_cv", dst);
}

void SSE::Filter2DMP(Mat src, int w, int h, Mat dst, Mat element, int we, int he)
{
	float val = 0;

#pragma omp parallel for private(val)

	for (int i = 1; i < h - 1; i++)
	{
		for (int j = 1; j < w - 1; j++)
		{
			float val = 0;
			for (int fh = 0; fh < he; fh++)
			{
				for (int fw = 0; fw < we; fw++)
				{
					val += (src.at<uchar>(i - 1 + fh, j - 1 + fw) * element.at<float>(fh, fw));
				}
			}
			dst.at<char>(i, j) = val;
		}
	}

	imshow("dst_openmp", dst);
}

void SSE::Filter2DSSE(Mat src, int w, int h, Mat dst, Mat element, int we, int he)
{
	uchar* data = src.ptr(0);
	uchar* pData = data;

	__m256i xmmData[3];
	__m256i xmmSum;
	
	float kernel[3][32];
	for (int i = 0; i < 3; i++)
	{
		for (int j = 0; j < 32; j++)
		{
			kernel[i][j] = element.at<float>(i, j % 3);
		}
	}

	for (int i = 0; i < h-2; i++)
	{
		for (int j = 0; j < w-2; j += 30)
		{
			xmmData[0] = _mm256_loadu_epi32((__m256i*)(pData + ((i + 0) * w + j)));
			xmmData[1] = _mm256_loadu_epi32((__m256i*)(pData + ((i + 1) * w + j)));
			xmmData[2] = _mm256_loadu_epi32((__m256i*)(pData + ((i + 2) * w + j)));
			
			for (int k = 0; k < 32; k++)
			{
				*((uchar*)(&xmmData[0]) + k) *= kernel[0][k];
				*((uchar*)(&xmmData[1]) + k) *= kernel[1][k];
				*((uchar*)(&xmmData[2]) + k) *= kernel[2][k];
			}

			xmmSum = _mm256_add_epi32(xmmData[0], xmmData[1]);
			xmmSum = _mm256_add_epi32(xmmSum, xmmData[2]);
	
			for (int k = 1; k < 31; k++)
			{
				dst.at<uchar>(i + 1, j + k) = ((*((uchar*)(&xmmSum) + (k-1)))+ (*((uchar*)(&xmmSum) + (k)))+ (*((uchar*)(&xmmSum) + (k + 1))));
			}
		}
	}

	imshow("dst_sse", dst);
}
