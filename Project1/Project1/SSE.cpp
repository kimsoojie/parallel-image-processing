#include "SSE.h"
#include <stdio.h>
#include <iostream>
#include <emmintrin.h>
#include <immintrin.h>

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
	float data[8] = { 1,4,9,16,25,36,49,64 };
	float* pData = data;
	int size = 8;

	__m256 avxSqrt = _mm256_setzero_ps();
	__m256 avxCurVal = _mm256_setzero_ps();

	for (int i = 0; i < size; i += 8)
	{
		avxCurVal = _mm256_loadu_ps(pData + i);
		avxSqrt = _mm256_sqrt_ps(avxCurVal);
	}


	for (int i = 0; i < 8; i++)
	{
		cout << *((float*)(&avxSqrt) + i) << " ";
	}
	cout << endl;
}