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


//#define _UNSIGNED_UNPACK_
//#define _PALIGNR_
#define _SIGNED_UNPACK_
//#define _FLOAT_ABS_   //???
//#define _PACK_WITH_SAT_
//#define _INTERLEAVED_PACK_WITH_SATURATION_
//#define _INTERLEAVED_PACK_WITHOUT_SATURATION_
//#define _NON_INTERLEAVED_UNPACK_
//#define _EXTWORD_
//#define _SHUFFLE_WORD_
//#define _SHUFFLE_SWAP_6_1_

int SSE::Practice()
{
	
	// Method for Signed unpack
#ifdef _UNSIGNED_UNPACK_

	__m128i dst1;
	__m128i dst2;
	__m128i m0 = _mm_set_epi16(0, 0, 0, 0, 0, 0, 0, 0);
	__m128i source = _mm_set_epi16(1, 2, 3, 4, 5, 6, 7, 8);  //1,2,3,4: high 5,6,7,8: low
	dst1 = _mm_unpacklo_epi16(source, m0);
	dst2 = _mm_unpackhi_epi16(source, m0);

	for (int i = 0; i < 8; i++) {
		cout << dst1.m128i_i16[i] << " ";
	}

	for (int i = 0; i < 8; i++) {
		cout << dst2.m128i_i16[i] << " ";
	}

#endif

	// Method for palignr
#ifdef _PALIGNR_
	__m128i a = _mm_set_epi16(1, 2, 3, 4, 5, 6, 7, 8);
	__m128i b = _mm_set_epi16(9, 10, 11, 12, 13, 14, 15, 16);
	__m128i c = _mm_alignr_epi8(a, b, 4);// 16byte a, b 를 32byte로 바꿔서 right시프트

	for (int i = 0; i < 8; i++)
	{
		cout << c.m128i_i16[i] << " ";
	}
#endif

	// Method for Unsigned unpack
#ifdef _SIGNED_UNPACK_	
	//uchar data[18] = { 2,4,6,8,10,12,14,16,
	//	              18,20,22,24,26,28,30,32,34,36 };
	
	uchar data[18] = { 250, 182, 174 ,163, 159, 164, 183, 185,
	181, 160 ,146 ,146, 154 ,149, 137, 131 };
	
	uchar* pData = data;
	
	__m128i tmp1;
	__m128i tmp2;
	__m128i tmp3;

	__m128i dst_16bit[2];
	__m128i dst_32bit[4];

	__m128i div = _mm_set_epi32(1,1,1,1);

	__m128i source;
	__m128i source_0 = _mm_loadu_epi8((__m128i*)pData);
	__m128i source_1 = _mm_loadu_epi8((__m128i*)(pData + 1));
	__m128i source_2 = _mm_loadu_epi8((__m128i*)(pData + 2));
	
	tmp1 = _mm_unpacklo_epi8(source_0, source_0); // 2 2 4 4 6 6 8 8 10 10 12 12 14 14 16 16
	tmp2 = _mm_unpackhi_epi8(source_0, source_0); // 18 18 20 20 22 22 24 24 26 26 28 28 30 30 32 32

	//dst_16bit[0] = _mm_srai_epi16(tmp1, 8);  // 2 4 6 8 10 12 14 16
	//dst_16bit[1] = _mm_srai_epi16(tmp2, 8); // 18 20 22 24 26 28 30 32.

	dst_16bit[0] = _mm_set_epi16(tmp1.m128i_u8[14], tmp1.m128i_u8[12], tmp1.m128i_u8[10], tmp1.m128i_u8[8],
		tmp1.m128i_u8[6], tmp1.m128i_u8[4], tmp1.m128i_u8[2], tmp1.m128i_u8[0]);

	dst_16bit[1] = _mm_set_epi16(tmp2.m128i_u8[14], tmp2.m128i_u8[12], tmp2.m128i_u8[10], tmp2.m128i_u8[8],
		tmp2.m128i_u8[6], tmp2.m128i_u8[4], tmp2.m128i_u8[2], tmp2.m128i_u8[0]);

	tmp1 = _mm_unpacklo_epi16(dst_16bit[0], dst_16bit[0]); // 2 2 4 4 6 6 8 8
	tmp2 = _mm_unpackhi_epi16(dst_16bit[0], dst_16bit[0]); //10 10 12 12 14 14 16 16

	dst_32bit[0] = _mm_srai_epi32(tmp1, 16); // 2 4 6 8
	dst_32bit[1] = _mm_srai_epi32(tmp2, 16); // 10 12 14 16

	
	tmp1 = _mm_unpacklo_epi16(dst_16bit[1], dst_16bit[1]); //18 18 20 20 22 22 24 24
	tmp2 = _mm_unpackhi_epi16(dst_16bit[1], dst_16bit[1]); //26 26 28 28 30 30 32 32
	
	dst_32bit[2] = _mm_srai_epi32(tmp1, 16); // 18 20 22 24
	dst_32bit[3] = _mm_srai_epi32(tmp2, 16); // 26 28 30 32 

	dst_32bit[0] = _mm_div_epi32(dst_32bit[0], div); // 1 2 3 4
	dst_32bit[1] = _mm_div_epi32(dst_32bit[1], div); // 5 6 7 8
	dst_32bit[2] = _mm_div_epi32(dst_32bit[2], div); // 9 10 11 12
	dst_32bit[3] = _mm_div_epi32(dst_32bit[3], div); // 13 14 15 16


	tmp1 = _mm_packs_epi32(dst_32bit[0], dst_32bit[0]); // 1 2 3 4 1 2 3 4
	tmp2 = _mm_packs_epi32(dst_32bit[1], dst_32bit[1]); // 5 6 7 8 5 6 7 8
	tmp3 = _mm_unpacklo_epi16(tmp1, tmp2); // 1 5 2 6 3 7 4 8

	
	tmp3 = _mm_shufflehi_epi16(tmp3, _MM_SHUFFLE(3, 1, 2, 0)); // 1 2 5 6 3 7 4 8
	tmp3 = _mm_shufflelo_epi16(tmp3, _MM_SHUFFLE(3, 1, 2, 0)); // 1 2 5 6 3 4 7 8
	dst_16bit[0] = _mm_shuffle_epi32(tmp3, _MM_SHUFFLE(3, 1, 2, 0)); // 1 2 3 4 5 6 7 8
	
	tmp1 = _mm_packs_epi32(dst_32bit[2], dst_32bit[2]); // 9 10 11 12 9 10 11 12
	tmp2 = _mm_packs_epi32(dst_32bit[3], dst_32bit[3]); // 13 14 15 16 13 14 15 16
	tmp3 = _mm_unpacklo_epi16(tmp1, tmp2); // 9 13 10 14 11 15 12 16
	
	tmp3 = _mm_shufflehi_epi16(tmp3, _MM_SHUFFLE(3, 1, 2, 0)); // 9 10 13 14 11 15 12 16
	tmp3 = _mm_shufflelo_epi16(tmp3, _MM_SHUFFLE(3, 1, 2, 0)); // 9 10 13 14 11 12 15 16
	dst_16bit[1] = _mm_shuffle_epi32(tmp3, _MM_SHUFFLE(3, 1, 2, 0)); // 9 10 11 12 13 14 15 16 

	tmp1 = _mm_packus_epi16(dst_16bit[0], dst_16bit[0]); // 1 2 3 4 5 6 7 8 1 2 3 4 5 6 7 8
	tmp2 = _mm_packus_epi16(dst_16bit[1], dst_16bit[1]); // 9 10 11 12 13 14 15 16 9 10 11 12 13 14 15 16
	tmp3 = _mm_unpacklo_epi8(tmp1, tmp2); // 1 9 2 10 3 11 4 12 5 13 6 14 8 16
	
	__m128i cof = _mm_setr_epi8(0, 2, 4, 6, 8, 10, 12, 14, 1, 3, 5, 7, 9, 11, 13, 15);
	source = _mm_shuffle_epi8(tmp3, cof);
	
	cout << endl;
	for (int i = 0; i < 16; i++)
	{
		printf("%d ", source.m128i_u8[i]);
	}

#endif

#ifdef _FLOAT_ABS_   
	int x = 0x7fffffff;
	__m128 cof = _mm_set1_ps(*(float*)&x);
	__m128 src = _mm_set_ps(-1, -4, -6, -255500);
	__m128 result = _mm_and_ps(cof, src);
	
	//for (int i = 0; i < 4; i++)
	//{
	//	cout << result.m128_i32[i] << " ";
	//}
#endif

	// Method for Interleaved Pack with Saturation
#ifdef _PACK_WITH_SAT_	
	__m128i dst1;
	__m128i source1 = _mm_set_epi16(-300, -1, -298, -2, -296, -3, -294, -4);
	__m128i source2 = _mm_set_epi16(300, 1, 298, 2, 296, 3, 294, 4);
	dst1 = _mm_packs_epi16(source1, source2); // 16 bit integer to 8bit integer

	// -4,-128....-128,4,127...127
	for (int i = 0; i < 16; i++)
	{
		printf("%d ", dst1.m128i_i8[i]);
	}

#endif

#ifdef _INTERLEAVED_PACK_WITH_SATURATION_	
	__m128i dst1;
	__m128i dst2;
	__m128i dst3;

	__m128i source1 = _mm_set_epi32(-1, -2, -3, -4);  //(32bit 4개)
	__m128i source2 = _mm_set_epi32(1, 2, 3, 4);
	//dst1 = _mm_packs_epi16(source1, source1); //-4 -1 -3 -1 ...    -> 32bit to 8bit (8bit 16개)
	//dst2 = _mm_packs_epi16(source2, source2); // 4 0 3 0 2 0 1 0  4 0 3 0 2 0 1 0
	//dst3 = _mm_unpacklo_epi16(dst1, dst2); // unpack 16bit from 8bit

	dst1 = _mm_packs_epi32(source1, source1); //-4 -1 -3 -1 ...    -> 32bit to 8bit (8bit 16개)
	dst2 = _mm_packs_epi32(source2, source2); // 4 0 3 0 2 0 1 0  4 0 3 0 2 0 1 0
	dst3 = _mm_unpacklo_epi16(dst1, dst2); // unpack 16bit from 8bit

	////-4 4 -3 3 -2 2 -1 1
	//for (int i = 0; i < 16; i++)
	//{
	//	printf("%d ", dst3.m128i_i16[i]);
	//}

	for (int i = 0; i < 8; i++)
	{
		printf("%d ", dst3.m128i_i16[i]);
	}
#endif

#ifdef _INTERLEAVED_PACK_WITHOUT_SATURATION_	
	__m128i dst1;
	__m128i dst2;
	__m128i dst3;

	__m128i source1 = _mm_set_epi16(-300, -1, -298, -2, -296, -3, -294, -4);
	__m128i source2 = _mm_set_epi16(300, 1, 298, 2, 296, 3, 294, 4);
	__m128i mask = _mm_set_epi8(0, 0xff, 0, 0xff, 0, 0xff, 0, 0xff, 0, 0xff, 0, 0xff, 0, 0xff, 0, 0xff);

	dst1 = _mm_slli_epi16(source1, 8);  // 16bit integer left shift
	dst2 = _mm_and_si128(source2, mask); // 0
	dst3 = _mm_or_si128(dst2, dst1); // dst1


	for (int i = 0; i < 8; i++)
	{
		printf("%d ", dst3.m128i_i16[i]);
	}
#endif

#ifdef _NON_INTERLEAVED_UNPACK_	 
	__m128i dst1;
	__m128i dst2;
	__m128i dst3;

	__m128i source1 = _mm_set_epi32(1, 2, 3, 4);
	__m128i source2 = _mm_set_epi32(5, 6, 7, 8);

	dst1 = _mm_unpacklo_epi32(source1, source2);
	dst2 = _mm_unpackhi_epi32(source1, source2);


	for (int i = 0; i < 4; i++)
	{
		printf("%d ", dst2.m128i_i32[i]);
	}
#endif

#ifdef _EXTWORD_

	int result = 0;
	__m128i source = _mm_set_epi16(1, 2, 3, 4, 5, 6, 7, 8);

	result = _mm_extract_epi16(source, 3);
	printf("%d\n", result);//5

	source = _mm_insert_epi16(source, result, 2);

	//8 7 5 5 4 3 2 1
	for (int i = 0; i < 8; i++)
	{
		printf("%d,", source.m128i_i16[i]);
	}

#endif

#ifdef _SHUFFLE_WORD_	
	__m128i dst1;
	__m128i source1 = _mm_set_epi16(7, 6, 5, 4, 3, 2, 1, 0);

	//dst1 = _mm_shufflehi_epi16(source1, ((3<<6) | (2<<4) | (1<<2) | (1)));
	//dst1 = _mm_shuffle_epi32(dst1, ((2<<6) | (2<<4) | (2<<2) | (2)));
	// 위 주석과 똑같은 구현임.
	dst1 = _mm_shufflehi_epi16(source1, _MM_SHUFFLE(3, 2, 1, 1));//4,5,6,7 -> 7,6,5,5 (0,1,2,3,5,5,6,7)
	dst1 = _mm_shuffle_epi32(dst1, _MM_SHUFFLE(2,2,2,2)); //5,5/5,5/5,5/5,5

	for (int i = 0; i < 8; i++)
	{
		printf("%d,", dst1.m128i_i16[i]);
	}

#endif


#ifdef _SHUFFLE_SWAP_6_1_	
	//__m128i dst1;
	//__m128i source1 = _mm_set_epi16(7,6,5,4,3,2,1,0);
	//dst1 = _mm_shuffle_epi32(source1, ((3<<6) | (0<<4) | (1<<2) | (2)));
	//dst1 = _mm_shufflehi_epi16(dst1, ((3<<6) | (1<<4) | (2<<2) | (0)));
	//dst1 = _mm_shuffle_epi32(dst1, ((3<<6) | (0<<4) | (1<<2) | (2)));

	// Reverse the order of the words
	__m128i dst1;
	__m128i source1 = _mm_set_epi16(7, 6, 5, 4, 3, 2, 1, 0);
	dst1 = _mm_shufflelo_epi16(source1, ((0 << 6) | (1 << 4) | (2 << 2) | 3));//7,6,5,4,0,1,2,3
	//dst1 = _mm_shufflehi_epi16(dst1, ((0<<6) | (1<<4) | (2<<2) | 3));
	//dst1 = _mm_shuffle_epi32(dst1, ((1<<6) | (0<<4) | (3<<2) | (2))); 

	// SSSE3 shffle instruction
	//__m128i dst1;
	//__m128i source = _mm_setr_epi8(10,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15);
	//__m128i cof = _mm_setr_epi8(8,0,9,1,10,2,11,3,12,4,13,5,14,6,15,7);// 이 순서대로 데이터를 정렬
	//dst1 = _mm_shuffle_epi8(source,cof);

	//3,0,2,0,...
	for (int i = 0; i < 16; i++)
	{
		printf("%d,", dst1.m128i_i8[i]);
	}
#endif
	return 0;
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
	Mat dstOpencv(src.size().width, src.size().height, CV_8UC1);

	// src , kernel size
	int width = src.size().width;		int height = src.size().height;
	int eWidht = element.cols;	int eHeight = element.rows;
	cout << "<image size>" << endl;
	cout << "width : " << width << "   height : " << height << endl;
	cout << endl;
	cout << "<filter size>" << endl;
	cout << "eWidht : " << eWidht << "   eHeight : " << eHeight << endl;
	cout << endl;

	//tm.start();
	////Serial 
	//Filter2DCV(src, width, height, dstCV, element, eWidht, eHeight);
	//tm.stop();
	//cout << "\nProcessTime-Serial : ";
	//cout << tm.getTimeMilli();
	//
	//opencv
	tm.reset();
	tm.start();
	cv::filter2D(src, dstOpencv, -1, element);
	tm.stop();
	cout << "\nProcessTime-OpenCV : ";
	cout << tm.getTimeMilli();
	
	tm.reset();
	tm.start();
	//OpenMP
	Filter2DMP(src, width, height, dstMP, element, eWidht, eHeight);
	tm.stop();
	cout << "\nProcessTime-OpenMP : ";
	cout << tm.getTimeMilli();
	
	//tm.reset();
	//tm.start();
	////SSE (256bit)
	//Filter2DSSE(src, width, height, dstSSE, element, eWidht, eHeight);
	//tm.stop();
	//cout << "\nProcessTime-SSE : ";
	//cout << tm.getTimeMilli();

	tm.reset();
	tm.start();
	//SSE (128bit-8bit)
	SSEmean_8bit(src, width, height, dstSSE, element, eWidht, eHeight);
	tm.stop();
	cout << "\nProcessTime-SSE(8bit) : ";
	cout << tm.getTimeMilli();

	//imshow("src", src);
	imshow("dst_opencv", dstOpencv);
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

	//imshow("dst_cv", dst);
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

	//imshow("dst_openmp", dst);
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

	//imshow("dst_sse", dst);
}

void SSE::SSEmean_8bit(Mat src, int w, int h, Mat dst, Mat element, int we, int he)
{
	uchar* data = src.data;
	uchar* pData = data;

	__m128i row;
	__m128i row_0[3];
	__m128i row_1[3];
	__m128i row_2[3];

	int sum_0[16] = { 0 };
	int sum_1[16] = { 0 };
	int sum_2[16] = { 0 };
	int sum[16] = { 0 };

	__m128i xmmResult;

	__m128i div = _mm_set_epi32(9, 9, 9, 9);
	__m128i tmp1;
	__m128i tmp2;
	__m128i tmp3;

	__m128i dst_16bit_0;
	__m128i dst_16bit_1;

	__m128i dst_32bit_0;
	__m128i dst_32bit_1;
	__m128i dst_32bit_2;
	__m128i dst_32bit_3;

	for (int i = 0; i < h - 2; i++)
	{
		for (int j = 0; j < w; j += 16)
		{
			// 8bit data load (16 members assigned)
			row_0[0] = _mm_loadu_epi8((__m128i*)(pData + ((i + 0) * w + j)));
			row_0[1] = _mm_loadu_epi8((__m128i*)(pData + ((i + 0) * w + j + 1)));
			row_0[2] = _mm_loadu_epi8((__m128i*)(pData + ((i + 0) * w + j + 2)));

			row_1[0] = _mm_loadu_epi8((__m128i*)(pData + ((i + 1) * w + j)));
			row_1[1] = _mm_loadu_epi8((__m128i*)(pData + ((i + 1) * w + j + 1)));
			row_1[2] = _mm_loadu_epi8((__m128i*)(pData + ((i + 1) * w + j + 2)));

			row_2[0] = _mm_loadu_epi8((__m128i*)(pData + ((i + 2) * w + j)));
			row_2[1] = _mm_loadu_epi8((__m128i*)(pData + ((i + 2) * w + j + 1)));
			row_2[2] = _mm_loadu_epi8((__m128i*)(pData + ((i + 2) * w + j + 2)));

			// calculate sum
			for (int k = 0; k < 16; k++)
			{
				sum_0[k] = (int)row_0[0].m128i_u8[k] + (int)row_0[1].m128i_u8[k] + (int)row_0[2].m128i_u8[k];
				sum_1[k] = (int)row_1[0].m128i_u8[k] + (int)row_1[1].m128i_u8[k] + (int)row_1[2].m128i_u8[k];
				sum_2[k] = (int)row_2[0].m128i_u8[k] + (int)row_2[1].m128i_u8[k] + (int)row_2[2].m128i_u8[k];
				sum[k] = sum_0[k] + sum_1[k] + sum_2[k];
			}

			// set 16bit 
			dst_16bit_0 = _mm_set_epi16((short)sum[7], (short)sum[6], (short)sum[5], (short)sum[4],
				(short)sum[3], (short)sum[2], (short)sum[1], (short)sum[0]);

			dst_16bit_1 = _mm_set_epi16((short)sum[15], (short)sum[14], (short)sum[13], (short)sum[12],
				(short)sum[11], (short)sum[10], (short)sum[9], (short)sum[8]);

			//  16bit -> 32bit
			tmp1 = _mm_unpacklo_epi16(dst_16bit_0, dst_16bit_0); 
			tmp2 = _mm_unpackhi_epi16(dst_16bit_0, dst_16bit_0); 

			dst_32bit_0 = _mm_srai_epi32(tmp1, 16); 
			dst_32bit_1 = _mm_srai_epi32(tmp2, 16); 

			tmp1 = _mm_unpacklo_epi16(dst_16bit_1, dst_16bit_1); 
			tmp2 = _mm_unpackhi_epi16(dst_16bit_1, dst_16bit_1); 

			dst_32bit_2 = _mm_srai_epi32(tmp1, 16); 
			dst_32bit_3 = _mm_srai_epi32(tmp2, 16); 

			// calculate average (divide 9)
			dst_32bit_0 = _mm_div_epi32(dst_32bit_0, div); 
			dst_32bit_1 = _mm_div_epi32(dst_32bit_1, div); 
			dst_32bit_2 = _mm_div_epi32(dst_32bit_2, div); 
			dst_32bit_3 = _mm_div_epi32(dst_32bit_3, div); 

			//  32bit -> 16bit
			tmp1 = _mm_packs_epi32(dst_32bit_0, dst_32bit_0); 
			tmp2 = _mm_packs_epi32(dst_32bit_1, dst_32bit_1); 
			tmp3 = _mm_unpacklo_epi16(tmp1, tmp2); 

			// shuffle
			tmp3 = _mm_shufflehi_epi16(tmp3, _MM_SHUFFLE(3, 1, 2, 0)); 
			tmp3 = _mm_shufflelo_epi16(tmp3, _MM_SHUFFLE(3, 1, 2, 0)); 
			dst_16bit_0 = _mm_shuffle_epi32(tmp3, _MM_SHUFFLE(3, 1, 2, 0)); 

			tmp1 = _mm_packs_epi32(dst_32bit_2, dst_32bit_2); 
			tmp2 = _mm_packs_epi32(dst_32bit_3, dst_32bit_3); 
			tmp3 = _mm_unpacklo_epi16(tmp1, tmp2); 

			tmp3 = _mm_shufflehi_epi16(tmp3, _MM_SHUFFLE(3, 1, 2, 0)); 
			tmp3 = _mm_shufflelo_epi16(tmp3, _MM_SHUFFLE(3, 1, 2, 0)); 
			dst_16bit_1 = _mm_shuffle_epi32(tmp3, _MM_SHUFFLE(3, 1, 2, 0));  

			//  16bit -> 8bit
			tmp1 = _mm_packus_epi16(dst_16bit_0, dst_16bit_0); 
			tmp2 = _mm_packus_epi16(dst_16bit_1, dst_16bit_1); 
			tmp3 = _mm_unpacklo_epi8(tmp1, tmp2); 

			__m128i cof = _mm_setr_epi8(0, 2, 4, 6, 8, 10, 12, 14, 1, 3, 5, 7, 9, 11, 13, 15);
			xmmResult = _mm_shuffle_epi8(tmp3, cof);

			// assign 8bit data -> uchar dst Mat
			for (int k = 0; k < 16; k++)
			{
				dst.at<uchar>(i + 1, j + k) = *((uchar*)(&xmmResult) + (k));
			}
		}
	}

	imshow("dst_sse(8bit)", dst);
	//imwrite("./result/dst_sse(8bit).jpg", dst);
}
