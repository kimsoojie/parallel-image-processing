#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "opencv2/highgui.hpp"
#include "opencv2/opencv.hpp"

#include <ipp.h>

#include <omp.h>

#include <stdio.h>
#include <iostream>

#include <emmintrin.h>
#include <immintrin.h>
#include <zmmintrin.h>

extern "C" void gpu_update_board(float* pcuSrc, float* pcuDst, int frame_w, int frame_h, int board_w, int board_h, int start_i, int start_j);


using namespace cv;
using namespace std;

#define NUM_VIDEO 16

//#define _OPENMP_
//#define _SSE_
//#define _GPU_
#define _IPP_


struct VideoInfo
{
	int _index;
	VideoCapture _cap;
	Size _size;
	int _idx_i;
	int _idx_j;
	int _image_process_type;
};

void process_time(TickMeter* t);

void display(Mat board, VideoInfo v, int* c, TickMeter* t);
void display(Mat board, VideoInfo v1, VideoInfo v2, int* c, TickMeter* t);

void display_sse(Mat board, VideoInfo v, int* c, TickMeter* t);
void display_sse(Mat board, VideoInfo v1, VideoInfo v2, int* c, TickMeter* t);

void display_gpu(Mat* board, VideoInfo v, int* c, TickMeter* t);
void display_gpu(Mat* board, VideoInfo v1, VideoInfo v2, int* c, TickMeter* t);

void display_ipp(Mat* board, VideoInfo v, int* c, TickMeter* t);
void display_ipp(Mat* board, VideoInfo v1, VideoInfo v2, int* c, TickMeter* t);

int main()
{
	Mat board(800, 1000, CV_8UC3, Scalar(0,0,0));
	namedWindow("mosaic", 0);
	Size re_size;
	re_size.width = 250;
	re_size.height = 200;

	int count = 0;
	
	TickMeter tm;
	tm.start();

#pragma omp parallel sections
	{
#pragma omp section
			{
				VideoCapture cap("./video/video_0.avi");
				VideoInfo v = { 0, cap,re_size,0,0,0 };
				
#ifdef _OPENMP_
				display(board, v, &count, &tm);
#endif _OPENMP_

#ifdef _SSE_
				display_sse(board, v, &count, &tm);
#endif _SSE_

#ifdef _GPU_
				display_gpu(&board, v, &count, &tm);
#endif _GPU_

#ifdef _IPP_
				display_ipp(&board, v, &count, &tm);
#endif _IPP_

			}
#pragma omp section
			{
				VideoCapture cap("./video/video_1.avi");
				VideoInfo v = { 1, cap,re_size,0,1,0 };
		
#ifdef _OPENMP_
				display(board, v, &count, &tm);
#endif _OPENMP_

#ifdef _SSE_
				display_sse(board, v, &count, &tm);
#endif _SSE_

#ifdef _GPU_
				display_gpu(&board, v, &count, &tm);
#endif _GPU_

#ifdef _IPP_
				display_ipp(&board, v, &count, &tm);
#endif _IPP_
			}
#pragma omp section
			{
				VideoCapture cap("./video/video_2.avi");
				VideoInfo v = { 2, cap,re_size,0,2,0 };
#ifdef _OPENMP_
				display(board, v, &count, &tm);
#endif _OPENMP_

#ifdef _SSE_
				display_sse(board, v, &count, &tm);
#endif _SSE_

#ifdef _GPU_
				display_gpu(&board, v, &count, &tm);
#endif _GPU_

#ifdef _IPP_
				display_ipp(&board, v, &count, &tm);
#endif _IPP_
			}
#pragma omp section
			{
				VideoCapture cap("./video/video_3.avi");
				VideoInfo v = { 3, cap,re_size,0,3,0 };
#ifdef _OPENMP_
				display(board, v, &count, &tm);
#endif _OPENMP_

#ifdef _SSE_
				display_sse(board, v, &count, &tm);
#endif _SSE_

#ifdef _GPU_
				display_gpu(&board, v, &count, &tm);
#endif _GPU_

#ifdef _IPP_
				display_ipp(&board, v, &count, &tm);
#endif _IPP_
			}
#pragma omp section
			{
				VideoCapture cap("./video/video_4.avi");
				VideoInfo v = { 4, cap,re_size,1,0,0 };
#ifdef _OPENMP_
				display(board, v, &count, &tm);
#endif _OPENMP_

#ifdef _SSE_
				display_sse(board, v, &count, &tm);
#endif _SSE_

#ifdef _GPU_
				display_gpu(&board, v, &count, &tm);
#endif _GPU_

#ifdef _IPP_
				display_ipp(&board, v, &count, &tm);
#endif _IPP_
			}
#pragma omp section
			{
				VideoCapture cap("./video/video_5.avi");
				VideoInfo v = { 5, cap,re_size,1,1,0 };
				
#ifdef _OPENMP_
				display(board, v, &count, &tm);
#endif _OPENMP_

#ifdef _SSE_
				display_sse(board, v, &count, &tm);
#endif _SSE_

#ifdef _GPU_
				display_gpu(&board, v, &count, &tm);
#endif _GPU_

#ifdef _IPP_
				display_ipp(&board, v, &count, &tm);
#endif _IPP_
			}
#pragma omp section
			{
				VideoCapture cap("./video/video_6.avi");
				VideoInfo v = { 6, cap,re_size,1,2,0 };
#ifdef _OPENMP_
				display(board, v, &count, &tm);
#endif _OPENMP_

#ifdef _SSE_
				display_sse(board, v, &count, &tm);
#endif _SSE_

#ifdef _GPU_
				display_gpu(&board, v, &count, &tm);
#endif _GPU_

#ifdef _IPP_
				display_ipp(&board, v, &count, &tm);
#endif _IPP_
			}
#pragma omp section
			{
				VideoCapture cap("./video/video_7.avi");
				VideoInfo v = { 7, cap,re_size,1,3,0 };
#ifdef _OPENMP_
				display(board, v, &count, &tm);
#endif _OPENMP_

#ifdef _SSE_
				display_sse(board, v, &count, &tm);
#endif _SSE_

#ifdef _GPU_
				display_gpu(&board, v, &count, &tm);
#endif _GPU_

#ifdef _IPP_
				display_ipp(&board, v, &count, &tm);
#endif _IPP_
			}
#pragma omp section
			{
				VideoCapture cap1("./video/video_8.avi");
				VideoCapture cap2("./video/video_12.avi");
				VideoInfo v1 = { 8, cap1,re_size,2,0,0 };
				VideoInfo v2 = { 12, cap2,re_size,3,0,0 };
#ifdef _OPENMP_
				display(board, v1,v2, &count, &tm);
#endif _OPENMP_

#ifdef _SSE_
				display_sse(board, v1, v2, &count, &tm);
#endif _SSE_

#ifdef _GPU_
				display_gpu(&board, v1, v2, &count, &tm);
#endif _GPU_

#ifdef _IPP_
				display_ipp(&board, v1,v2, &count, &tm);
#endif _IPP_
			}
#pragma omp section
			{
				VideoCapture cap1("./video/video_9.avi");
				VideoCapture cap2("./video/video_13.avi");
				VideoInfo v1 = { 9, cap1,re_size,2,1,0 };
				VideoInfo v2 = { 13, cap2,re_size,3,1,0 };
#ifdef _OPENMP_
				display(board, v1, v2, &count, &tm);
#endif _OPENMP_

#ifdef _SSE_
				display_sse(board, v1, v2, &count, &tm);
#endif _SSE_

#ifdef _GPU_
				display_gpu(&board, v1,v2, &count, &tm);
#endif _GPU_

#ifdef _IPP_
				display_ipp(&board, v1, v2, &count, &tm);
#endif _IPP_
			}
#pragma omp section
			{
				VideoCapture cap1("./video/video_10.avi");
				VideoCapture cap2("./video/video_14.avi");
				VideoInfo v1 = { 10, cap1,re_size,2,2,0 };
				VideoInfo v2 = { 14, cap2,re_size,3,2,0 };
#ifdef _OPENMP_
				display(board, v1, v2, &count, &tm);
#endif _OPENMP_

#ifdef _SSE_
				display_sse(board, v1, v2, &count, &tm);
#endif _SSE_

#ifdef _GPU_
				display_gpu(&board, v1, v2, &count, &tm);
#endif _GPU_

#ifdef _IPP_
				display_ipp(&board, v1, v2, &count, &tm);
#endif _IPP_
			}
#pragma omp section
			{
				VideoCapture cap1("./video/video_11.avi");
				VideoCapture cap2("./video/video_15.avi");
				VideoInfo v1 = { 11, cap1,re_size,2,3,0 };
				VideoInfo v2 = { 15, cap2,re_size,3,3,0 };
#ifdef _OPENMP_
				display(board, v1, v2, &count, &tm);
#endif _OPENMP_

#ifdef _SSE_
				display_sse(board, v1, v2, &count, &tm);
#endif _SSE_

#ifdef _GPU_
				display_gpu(&board, v1, v2, &count, &tm);
#endif _GPU_

#ifdef _IPP_
				display_ipp(&board, v1, v2, &count, &tm);
#endif _IPP_
			}
	}

	return 0;
}

void process_time(TickMeter* t)
{
	(*t).stop();
	cout << "process time (msec) : " << (*t).getTimeMilli() << endl;
	cout << "process time (sec) : " << (*t).getTimeSec() << endl;
}

void display(Mat board, VideoInfo v, int* c, TickMeter* t)
{
	int interval = 0;
	int start_i = v._idx_i * v._size.height + interval;
	int start_j = v._idx_j * v._size.width + interval;

	double fstart, fend, fprocTime, fps;
	
	for (;;)
	{
		fstart = omp_get_wtime();

		Mat frame;
		v._cap >> frame;

		if (frame.empty())
		{
			v._cap.release();
			cout << "finish: " << v._index << endl;
			(*c)++;
			if ((*c) == NUM_VIDEO)
			{
				cout << "finished all" << endl;
				process_time(t);
				break;
			}
			imshow("mosaic", board);
			waitKey(0);
			break;
		}

		resize(frame, frame, v._size);

		for (int i = 0; i < frame.size().height; i++)
		{
			for (int j = 0; j < frame.size().width; j++)
			{
				board.at<Vec3b>(i + start_i, j + start_j) = frame.at<Vec3b>(i, j);
			}
		}

		fend = omp_get_wtime();
		fprocTime = fend - fstart;
		fps = 1 / fprocTime;
		putText(board, "fps: " + to_string(fps), Point(start_j+10, start_i+20), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 255, 0), 2);

		imshow("mosaic", board);
		waitKey(1000/fps);
	}
}

void display(Mat board, VideoInfo v1, VideoInfo v2, int* c, TickMeter* t)
{
	int interval = 0;
	int start_i_0 = v1._idx_i * v1._size.height + interval;
	int start_j_0 = v1._idx_j * v1._size.width + interval;
	int start_i_1 = v2._idx_i * v2._size.height + interval;
	int start_j_1 = v2._idx_j * v2._size.width + interval;

	double fstart, fend, fprocTime, fps;
	
	bool empty1 = false;
	bool empty2 = false;

	for (;;)
	{
		fstart = omp_get_wtime();

		if (!empty1)
		{
			Mat frame1;
			v1._cap >> frame1;
			empty1 = frame1.empty();
			
			if (!empty1)
			{
				resize(frame1, frame1, v1._size);
				for (int ii = 0; ii < v1._size.height; ii++)
				{
					for (int jj = 0; jj < v1._size.width; jj++)
					{
						board.at<Vec3b>(ii + start_i_0, jj + start_j_0) = frame1.at<Vec3b>(ii, jj);
		
					}
				}
		
				fend = omp_get_wtime();
				fprocTime = fend - fstart;
				fps = 1 / fprocTime;
				putText(board, "fps: " + to_string(fps), Point(start_j_0 + 10, start_i_0 + 20), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 255, 0), 2);
		
				imshow("mosaic", board);
				waitKey(1000/ (fps*2));
			}
			else
			{
				v1._cap.release();
				cout << "finish: " << v1._index << endl;
				(*c)++;
				if ((*c) == NUM_VIDEO)
				{
					cout << "finished all" << endl;
					process_time(t);
					break;
				}
			}
		}


		if (!empty2)
		{
			Mat frame2;
			v2._cap >> frame2;
			empty2 = frame2.empty();
		
			if (!empty2)
			{
				resize(frame2, frame2, v2._size);
				for (int ii = 0; ii < v1._size.height; ii++)
				{
					for (int jj = 0; jj < v1._size.width; jj++)
					{
						board.at<Vec3b>(ii + start_i_1, jj + start_j_1) = frame2.at<Vec3b>(ii, jj);
		
					}
				}
		
				fend = omp_get_wtime();
				fprocTime = fend - fstart;
				fps = 1 / fprocTime;
				putText(board, "fps: " + to_string(fps), Point(start_j_1 + 10, start_i_1 + 20), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 255, 0), 2);
		
				imshow("mosaic", board);
				waitKey(1000/ (fps*2));
			}
			else
			{
				v2._cap.release();
				cout << "finish: " << v2._index << endl;
				(*c)++;
				if ((*c) == NUM_VIDEO)
				{
					cout << "finished all" << endl;
					process_time(t);
					break;
				}
			}
		}
		if (empty1 && empty2) break;
	}
	imshow("mosaic", board);
	waitKey(0);
}



void display_sse(Mat board, VideoInfo v, int* c, TickMeter* t)
{
	int interval = 0;
	int start_i = v._idx_i * v._size.height + interval;
	int start_j = v._idx_j * v._size.width + interval;

	double fstart, fend, fprocTime, fps;

	__m128i _row[10];

	for (;;)
	{
		fstart = omp_get_wtime();

		Mat frame;
		v._cap >> frame;

		if (frame.empty())
		{
			v._cap.release();
			cout << "finish: " << v._index << endl;
			(*c)++;
			if ((*c) == NUM_VIDEO)
			{
				cout << "finished all" << endl;
				process_time(t);
				break;
			}
			imshow("mosaic", board);
			waitKey(0);
			break;
		}

		resize(frame, frame, v._size);

		for (int i = 0; i < frame.size().height; i+=10)
		{
			for (int j = 0; j < frame.size().width; j+=5)
			{
				_row[0] = _mm_loadu_epi8((__m128i*)(&frame.at<Vec3b>(i + 0, j)));
				_row[1] = _mm_loadu_epi8((__m128i*)(&frame.at<Vec3b>(i + 1, j)));
				_row[2] = _mm_loadu_epi8((__m128i*)(&frame.at<Vec3b>(i + 2, j)));
				_row[3] = _mm_loadu_epi8((__m128i*)(&frame.at<Vec3b>(i + 3, j)));
				_row[4] = _mm_loadu_epi8((__m128i*)(&frame.at<Vec3b>(i + 4, j)));
				_row[5] = _mm_loadu_epi8((__m128i*)(&frame.at<Vec3b>(i + 5, j)));
				_row[6] = _mm_loadu_epi8((__m128i*)(&frame.at<Vec3b>(i + 6, j)));
				_row[7] = _mm_loadu_epi8((__m128i*)(&frame.at<Vec3b>(i + 7, j)));
				_row[8] = _mm_loadu_epi8((__m128i*)(&frame.at<Vec3b>(i + 8, j)));
				_row[9] = _mm_loadu_epi8((__m128i*)(&frame.at<Vec3b>(i + 9, j)));
				
				_mm_storeu_si128((__m128i*)(&board.at<Vec3b>(0 + i + start_i, j + start_j)), _row[0]);
				_mm_storeu_si128((__m128i*)(&board.at<Vec3b>(1 + i + start_i, j + start_j)), _row[1]);
				_mm_storeu_si128((__m128i*)(&board.at<Vec3b>(2 + i + start_i, j + start_j)), _row[2]);
				_mm_storeu_si128((__m128i*)(&board.at<Vec3b>(3 + i + start_i, j + start_j)), _row[3]);
				_mm_storeu_si128((__m128i*)(&board.at<Vec3b>(4 + i + start_i, j + start_j)), _row[4]);
				_mm_storeu_si128((__m128i*)(&board.at<Vec3b>(5 + i + start_i, j + start_j)), _row[5]);
				_mm_storeu_si128((__m128i*)(&board.at<Vec3b>(6 + i + start_i, j + start_j)), _row[6]);
				_mm_storeu_si128((__m128i*)(&board.at<Vec3b>(7 + i + start_i, j + start_j)), _row[7]);
				_mm_storeu_si128((__m128i*)(&board.at<Vec3b>(8 + i + start_i, j + start_j)), _row[8]);
				_mm_storeu_si128((__m128i*)(&board.at<Vec3b>(9 + i + start_i, j + start_j)), _row[9]);
			}
		}

		fend = omp_get_wtime();
		fprocTime = fend - fstart;
		fps = 1 / fprocTime;
		putText(board, "fps: " + to_string(fps), Point(start_j + 10, start_i + 20), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 255, 0), 2);

		imshow("mosaic", board);
		waitKey(1000 / fps);
	}
}

void display_sse(Mat board, VideoInfo v1, VideoInfo v2, int* c, TickMeter* t)
{
	int interval = 0;
	int start_i_0 = v1._idx_i * v1._size.height + interval;
	int start_j_0 = v1._idx_j * v1._size.width + interval;
	int start_i_1 = v2._idx_i * v2._size.height + interval;
	int start_j_1 = v2._idx_j * v2._size.width + interval;

	double fstart, fend, fprocTime, fps;

	bool empty1 = false;
	bool empty2 = false;

	__m128i _row[10];

	for (;;)
	{
		fstart = omp_get_wtime();

		if (!empty1)
		{
			Mat frame1;
			v1._cap >> frame1;
			empty1 = frame1.empty();

			if (!empty1)
			{
				resize(frame1, frame1, v1._size);
				for (int ii = 0; ii < v1._size.height; ii+=10)
				{
					for (int jj = 0; jj < v1._size.width; jj+=5)
					{
						_row[0] = _mm_loadu_epi8((__m128i*)(&frame1.at<Vec3b>(ii + 0, jj)));
						_row[1] = _mm_loadu_epi8((__m128i*)(&frame1.at<Vec3b>(ii + 1, jj)));
						_row[2] = _mm_loadu_epi8((__m128i*)(&frame1.at<Vec3b>(ii + 2, jj)));
						_row[3] = _mm_loadu_epi8((__m128i*)(&frame1.at<Vec3b>(ii + 3, jj)));
						_row[4] = _mm_loadu_epi8((__m128i*)(&frame1.at<Vec3b>(ii + 4, jj)));
						_row[5] = _mm_loadu_epi8((__m128i*)(&frame1.at<Vec3b>(ii + 5, jj)));
						_row[6] = _mm_loadu_epi8((__m128i*)(&frame1.at<Vec3b>(ii + 6, jj)));
						_row[7] = _mm_loadu_epi8((__m128i*)(&frame1.at<Vec3b>(ii + 7, jj)));
						_row[8] = _mm_loadu_epi8((__m128i*)(&frame1.at<Vec3b>(ii + 8, jj)));
						_row[9] = _mm_loadu_epi8((__m128i*)(&frame1.at<Vec3b>(ii + 9, jj)));

						_mm_storeu_si128((__m128i*)(&board.at<Vec3b>(0 + ii + start_i_0, jj + start_j_0)), _row[0]);
						_mm_storeu_si128((__m128i*)(&board.at<Vec3b>(1 + ii + start_i_0, jj + start_j_0)), _row[1]);
						_mm_storeu_si128((__m128i*)(&board.at<Vec3b>(2 + ii + start_i_0, jj + start_j_0)), _row[2]);
						_mm_storeu_si128((__m128i*)(&board.at<Vec3b>(3 + ii + start_i_0, jj + start_j_0)), _row[3]);
						_mm_storeu_si128((__m128i*)(&board.at<Vec3b>(4 + ii + start_i_0, jj + start_j_0)), _row[4]);
						_mm_storeu_si128((__m128i*)(&board.at<Vec3b>(5 + ii + start_i_0, jj + start_j_0)), _row[5]);
						_mm_storeu_si128((__m128i*)(&board.at<Vec3b>(6 + ii + start_i_0, jj + start_j_0)), _row[6]);
						_mm_storeu_si128((__m128i*)(&board.at<Vec3b>(7 + ii + start_i_0, jj + start_j_0)), _row[7]);
						_mm_storeu_si128((__m128i*)(&board.at<Vec3b>(8 + ii + start_i_0, jj + start_j_0)), _row[8]);
						_mm_storeu_si128((__m128i*)(&board.at<Vec3b>(9 + ii + start_i_0, jj + start_j_0)), _row[9]);
					}
				}

				fend = omp_get_wtime();
				fprocTime = fend - fstart;
				fps = 1 / fprocTime;
				putText(board, "fps: " + to_string(fps), Point(start_j_0 + 10, start_i_0 + 20), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 255, 0), 2);

				imshow("mosaic", board);
				if(!empty2)waitKey(1000 / (fps *2));
				else waitKey(1000 / (fps));
			}
			else
			{
				v1._cap.release();
				cout << "finish: " << v1._index << endl;
				(*c)++;
				if ((*c) == NUM_VIDEO)
				{
					cout << "finished all" << endl;
					process_time(t);
					break;
				}
			}
		}


		if (!empty2)
		{
			Mat frame2;
			v2._cap >> frame2;
			empty2 = frame2.empty();

			if (!empty2)
			{
				resize(frame2, frame2, v2._size);
				for (int ii = 0; ii < v1._size.height; ii+=10)
				{
					for (int jj = 0; jj < v1._size.width; jj+=5)
					{
						_row[0] = _mm_loadu_epi8((__m128i*)(&frame2.at<Vec3b>(ii + 0, jj)));
						_row[1] = _mm_loadu_epi8((__m128i*)(&frame2.at<Vec3b>(ii + 1, jj)));
						_row[2] = _mm_loadu_epi8((__m128i*)(&frame2.at<Vec3b>(ii + 2, jj)));
						_row[3] = _mm_loadu_epi8((__m128i*)(&frame2.at<Vec3b>(ii + 3, jj)));
						_row[4] = _mm_loadu_epi8((__m128i*)(&frame2.at<Vec3b>(ii + 4, jj)));
						_row[5] = _mm_loadu_epi8((__m128i*)(&frame2.at<Vec3b>(ii + 5, jj)));
						_row[6] = _mm_loadu_epi8((__m128i*)(&frame2.at<Vec3b>(ii + 6, jj)));
						_row[7] = _mm_loadu_epi8((__m128i*)(&frame2.at<Vec3b>(ii + 7, jj)));
						_row[8] = _mm_loadu_epi8((__m128i*)(&frame2.at<Vec3b>(ii + 8, jj)));
						_row[9] = _mm_loadu_epi8((__m128i*)(&frame2.at<Vec3b>(ii + 9, jj)));

						_mm_storeu_si128((__m128i*)(&board.at<Vec3b>(0 + ii + start_i_1, jj + start_j_1)), _row[0]);
						_mm_storeu_si128((__m128i*)(&board.at<Vec3b>(1 + ii + start_i_1, jj + start_j_1)), _row[1]);
						_mm_storeu_si128((__m128i*)(&board.at<Vec3b>(2 + ii + start_i_1, jj + start_j_1)), _row[2]);
						_mm_storeu_si128((__m128i*)(&board.at<Vec3b>(3 + ii + start_i_1, jj + start_j_1)), _row[3]);
						_mm_storeu_si128((__m128i*)(&board.at<Vec3b>(4 + ii + start_i_1, jj + start_j_1)), _row[4]);
						_mm_storeu_si128((__m128i*)(&board.at<Vec3b>(5 + ii + start_i_1, jj + start_j_1)), _row[5]);
						_mm_storeu_si128((__m128i*)(&board.at<Vec3b>(6 + ii + start_i_1, jj + start_j_1)), _row[6]);
						_mm_storeu_si128((__m128i*)(&board.at<Vec3b>(7 + ii + start_i_1, jj + start_j_1)), _row[7]);
						_mm_storeu_si128((__m128i*)(&board.at<Vec3b>(8 + ii + start_i_1, jj + start_j_1)), _row[8]);
						_mm_storeu_si128((__m128i*)(&board.at<Vec3b>(9 + ii + start_i_1, jj + start_j_1)), _row[9]);
					}
				}

				fend = omp_get_wtime();
				fprocTime = fend - fstart;
				fps = 1 / fprocTime;
				putText(board, "fps: " + to_string(fps), Point(start_j_1 + 10, start_i_1 + 20), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 255, 0), 2);

				imshow("mosaic", board);
				if (!empty1)waitKey(1000 / (fps * 2));
				else waitKey(1000 / (fps));
			}
			else
			{
				v2._cap.release();
				cout << "finish: " << v2._index << endl;
				(*c)++;
				if ((*c) == NUM_VIDEO)
				{
					cout << "finished all" << endl;
					process_time(t);
					break;
				}
			}
		}
		if (empty1 && empty2) break;
	}
	imshow("mosaic", board);
	waitKey(0);
}


void display_gpu(Mat* board, VideoInfo v, int* c, TickMeter* t)
{
	double fstart, fend, fprocTime, fps;

	Mat pfWhite;
	Mat pfInput;
	Mat pfInput_board;
	int frame_arr_size = v._size.width * v._size.height * 3;
	int board_arr_size = (*board).size().width * (*board).size().height * 3;
	float* pDst = new float[board_arr_size];
	float* pWhite = new float[board_arr_size];

	Mat white(v._size, CV_8UC3, Scalar(255, 255, 255));
	
	// Allocate cuda device memory (memory alloc: cuda src, dst)
	float* pcuWhite;
	float* pcuWhited;
	float* pcuSrc;
	float* pcuDst;
	(cudaMalloc((void**)&pcuWhite, frame_arr_size * sizeof(float)));
	(cudaMalloc((void**)&pcuWhited, board_arr_size * sizeof(float)));
	(cudaMalloc((void**)&pcuSrc, frame_arr_size * sizeof(float)));
	(cudaMalloc((void**)&pcuDst, board_arr_size * sizeof(float)));

	Mat frame;

	int interval = 0;
	int start_i = v._idx_i * v._size.height + interval;
	int start_j = v._idx_j * v._size.width + interval;

	white.convertTo(pfWhite, CV_32FC3);
	(cudaMemcpy(pcuWhite, pfWhite.data, frame_arr_size * sizeof(float), cudaMemcpyHostToDevice));
	gpu_update_board(pcuWhite, pcuWhited, v._size.width, v._size.height, (*board).size().width, (*board).size().height, start_i, start_j);
	(cudaMemcpy(pWhite, pcuWhited, board_arr_size * sizeof(float), cudaMemcpyDeviceToHost));
	Mat board_white((*board).size(), CV_32FC3, pWhite);
	board_white.convertTo(board_white, CV_8UC3);

	for (;;)
	{
		fstart = omp_get_wtime();

		v._cap >> frame;

		if (frame.empty())
		{
			v._cap.release();
			cout << "finish: " << v._index << endl;
			(*c)++;
			if ((*c) == NUM_VIDEO)
			{
				cout << "finished all" << endl;
				process_time(t);
				break;
			}
			imshow("mosaic", (*board));
			waitKey(0);
			break;
		}
		resize(frame, frame, v._size);

		// convert frame -> float
		frame.convertTo(pfInput, CV_32FC3);
		(*board).convertTo(pfInput_board, CV_32FC3);
		
		// copy input image across to the device (input image -> pcuSrc, board -> pcuDst)
		(cudaMemcpy(pcuSrc, pfInput.data, frame_arr_size * sizeof(float), cudaMemcpyHostToDevice));
		
		gpu_update_board(pcuSrc, pcuDst, frame.size().width, frame.size().height, (*board).size().width, (*board).size().height, start_i, start_j);
		
		(cudaMemcpy(pDst, pcuDst, board_arr_size * sizeof(float), cudaMemcpyDeviceToHost));
		
		Mat board_update((*board).size(), CV_32FC3, pDst);
		
		board_update.convertTo(board_update, CV_8UC3);
		cv::subtract((*board), board_white, (*board));
		cv::add((*board), board_update, (*board));
		
		fend = omp_get_wtime();
		fprocTime = fend - fstart;
		fps = 1 / fprocTime;
		putText((*board), "fps: " + to_string(fps), Point(start_j + 10, start_i + 20), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 255, 0), 2);

		imshow("mosaic", (*board));
		waitKey(1);
	}

	// free memory
	delete pDst;
	delete pcuSrc;
	delete pcuDst;
}

void display_gpu(Mat* board, VideoInfo v1, VideoInfo v2, int* c, TickMeter* t)
{
	Mat pfWhite;
	Mat pfInput;
	Mat pfInput_board;
	int frame_arr_size = v1._size.width * v1._size.height * 3;
	int board_arr_size = (*board).size().width * (*board).size().height * 3;
	float* pDst = new float[board_arr_size];
	float* pWhite = new float[board_arr_size];

	Mat white(v1._size, CV_8UC3, Scalar(255, 255, 255));

	// Allocate cuda device memory (memory alloc: cuda src, dst)
	float* pcuWhite;
	float* pcuWhited;
	float* pcuSrc;
	float* pcuDst;
	(cudaMalloc((void**)&pcuWhite, frame_arr_size * sizeof(float)));
	(cudaMalloc((void**)&pcuWhited, board_arr_size * sizeof(float)));
	(cudaMalloc((void**)&pcuSrc, frame_arr_size * sizeof(float)));
	(cudaMalloc((void**)&pcuDst, board_arr_size * sizeof(float)));

	int interval = 0;
	int start_i_0 = v1._idx_i * v1._size.height + interval;
	int start_j_0 = v1._idx_j * v1._size.width + interval;
	int start_i_1 = v2._idx_i * v2._size.height + interval;
	int start_j_1 = v2._idx_j * v2._size.width + interval;

	white.convertTo(pfWhite, CV_32FC3);
	(cudaMemcpy(pcuWhite, pfWhite.data, frame_arr_size * sizeof(float), cudaMemcpyHostToDevice));

	double fstart, fend, fprocTime, fps;

	bool empty1 = false;
	bool empty2 = false;

	for (;;)
	{
		fstart = omp_get_wtime();

		if (!empty1)
		{
			Mat frame1;
			v1._cap >> frame1;
			empty1 = frame1.empty();

			if (!empty1)
			{
				resize(frame1, frame1, v1._size);

				// convert frame -> float
				frame1.convertTo(pfInput, CV_32FC3);
				(*board).convertTo(pfInput_board, CV_32FC3);

				// copy input image across to the device (input image -> pcuSrc, board -> pcuDst)
				(cudaMemcpy(pcuSrc, pfInput.data, frame_arr_size * sizeof(float), cudaMemcpyHostToDevice));

				gpu_update_board(pcuSrc, pcuDst, frame1.size().width, frame1.size().height, (*board).size().width, (*board).size().height, start_i_0, start_j_0);
				gpu_update_board(pcuWhite, pcuWhited, frame1.size().width, frame1.size().height, (*board).size().width, (*board).size().height, start_i_0, start_j_0);

				(cudaMemcpy(pDst, pcuDst, board_arr_size * sizeof(float), cudaMemcpyDeviceToHost));
				(cudaMemcpy(pWhite, pcuWhited, board_arr_size * sizeof(float), cudaMemcpyDeviceToHost));

				Mat board_update((*board).size(), CV_32FC3, pDst);
				Mat board_white((*board).size(), CV_32FC3, pWhite);

				board_white.convertTo(board_white, CV_8UC3);
				board_update.convertTo(board_update, CV_8UC3);
				cv::subtract((*board), board_white, (*board));
				cv::add((*board), board_update, (*board));

				fend = omp_get_wtime();
				fprocTime = fend - fstart;
				fps = 1 / fprocTime;
				putText((*board), "fps: " + to_string(fps), Point(start_j_0 + 10, start_i_0 + 20), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 255, 0), 2);

				imshow("mosaic", (*board));
				waitKey(1);
			}
			else
			{
				v1._cap.release();
				cout << "finish: " << v1._index << endl;
				(*c)++;
				if ((*c) == NUM_VIDEO)
				{
					cout << "finished all" << endl;
					process_time(t);
					break;
				}
			}
		}


		if (!empty2)
		{
			Mat frame2;
			v2._cap >> frame2;
			empty2 = frame2.empty();

			if (!empty2)
			{
				resize(frame2, frame2, v2._size);

				// convert frame -> float
				frame2.convertTo(pfInput, CV_32FC3);
				(*board).convertTo(pfInput_board, CV_32FC3);

				// copy input image across to the device (input image -> pcuSrc, board -> pcuDst)
				(cudaMemcpy(pcuSrc, pfInput.data, frame_arr_size * sizeof(float), cudaMemcpyHostToDevice));

				gpu_update_board(pcuSrc, pcuDst, frame2.size().width, frame2.size().height, (*board).size().width, (*board).size().height, start_i_1, start_j_1);
				gpu_update_board(pcuWhite, pcuWhited, frame2.size().width, frame2.size().height, (*board).size().width, (*board).size().height, start_i_1, start_j_1);

				(cudaMemcpy(pDst, pcuDst, board_arr_size * sizeof(float), cudaMemcpyDeviceToHost));
				(cudaMemcpy(pWhite, pcuWhited, board_arr_size * sizeof(float), cudaMemcpyDeviceToHost));

				Mat board_update((*board).size(), CV_32FC3, pDst);
				Mat board_white((*board).size(), CV_32FC3, pWhite);

				board_white.convertTo(board_white, CV_8UC3);
				board_update.convertTo(board_update, CV_8UC3);
				cv::subtract((*board), board_white, (*board));
				cv::add((*board), board_update, (*board));

				fend = omp_get_wtime();
				fprocTime = fend - fstart;
				fps = 1 / fprocTime;
				putText((*board), "fps: " + to_string(fps), Point(start_j_1 + 10, start_i_1 + 20), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 255, 0), 2);

				imshow("mosaic", (*board));
				waitKey(1);
			}
			else
			{
				v2._cap.release();
				cout << "finish: " << v2._index << endl;
				(*c)++;
				if ((*c) == NUM_VIDEO)
				{
					cout << "finished all" << endl;
					process_time(t);
					break;
				}
			}
		}
		if (empty1 && empty2) break;
	}
	imshow("mosaic", (*board));
	waitKey(0);

	// free memory
	delete pDst;
	delete pcuSrc;
	delete pcuDst;
}

void display_ipp(Mat* board, VideoInfo v, int* c, TickMeter* t)
{
	int interval = 0;
	int start_i = v._idx_i * v._size.height + interval;
	int start_j = v._idx_j * v._size.width + interval;

	double fstart, fend, fprocTime, fps;
	

	//ipp size
	IppiSize size, tsize;
	size.width = v._size.width;
	size.height = v._size.height;
	tsize.width = (*board).size().width;
	tsize.height = (*board).size().height;

	for (;;)
	{
		fstart = omp_get_wtime();

		Mat frame;
		v._cap >> frame;

		if (frame.empty())
		{
			v._cap.release();
			cout << "finish: " << v._index << endl;
			(*c)++;
			if ((*c) == NUM_VIDEO)
			{
				cout << "finished all" << endl;
				process_time(t);
				break;
			}
			imshow("mosaic", *board);
			waitKey(0);
			break;
		}

		resize(frame, frame, v._size);

		// copy 
		int idx = start_i * tsize.width * 3 + start_j * 3;
		ippiCopy_8u_C3R((const Ipp8u*)frame.data, size.width * 3, (*board).data + idx, tsize.width * 3, size);
		
		fend = omp_get_wtime();
		fprocTime = fend - fstart;
		fps = 1 / fprocTime;
		putText(*board, "fps: " + to_string(fps), Point(start_j + 10, start_i + 20), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 255, 0), 2);

		imshow("mosaic", *board);
		waitKey(1000/fps);
		
	}
}

void display_ipp(Mat* board, VideoInfo v1, VideoInfo v2, int* c, TickMeter* t)
{
	int interval = 0;
	int start_i_0 = v1._idx_i * v1._size.height + interval;
	int start_j_0 = v1._idx_j * v1._size.width + interval;
	int start_i_1 = v2._idx_i * v2._size.height + interval;
	int start_j_1 = v2._idx_j * v2._size.width + interval;

	double fstart, fend, fprocTime, fps;

	bool empty1 = false;
	bool empty2 = false;

	//ipp size
	IppiSize size, tsize;
	size.width = v1._size.width;
	size.height = v1._size.height;
	tsize.width = (*board).size().width;
	tsize.height = (*board).size().height;

	for (;;)
	{
		fstart = omp_get_wtime();

		if (!empty1)
		{
			Mat frame1;
			v1._cap >> frame1;
			empty1 = frame1.empty();

			if (!empty1)
			{
				resize(frame1, frame1, v1._size);
				
				// copy 
				int idx = start_i_0 * tsize.width * 3 + start_j_0 * 3;
				ippiCopy_8u_C3R((const Ipp8u*)frame1.data, size.width * 3, (*board).data + idx, tsize.width * 3, size);

				fend = omp_get_wtime();
				fprocTime = fend - fstart;
				fps = 1 / fprocTime;
				putText(*board, "fps: " + to_string(fps), Point(start_j_0 + 10, start_i_0 + 20), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 255, 0), 2);

				imshow("mosaic", *board);
				waitKey(1000 / (fps * 2));
			}
			else
			{
				v1._cap.release();
				cout << "finish: " << v1._index << endl;
				(*c)++;
				if ((*c) == NUM_VIDEO)
				{
					cout << "finished all" << endl;
					process_time(t);
					break;
				}
			}
		}


		if (!empty2)
		{
			Mat frame2;
			v2._cap >> frame2;
			empty2 = frame2.empty();

			if (!empty2)
			{
				resize(frame2, frame2, v2._size);
				
				// copy 
				int idx = start_i_1 * tsize.width * 3 + start_j_1 * 3;
				ippiCopy_8u_C3R((const Ipp8u*)frame2.data, size.width * 3, (*board).data + idx, tsize.width * 3, size);

				fend = omp_get_wtime();
				fprocTime = fend - fstart;
				fps = 1 / fprocTime;
				putText(*board, "fps: " + to_string(fps), Point(start_j_1 + 10, start_i_1 + 20), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 255, 0), 2);

				imshow("mosaic", *board);
				waitKey(1000 / (fps * 2));
			}
			else
			{
				v2._cap.release();
				cout << "finish: " << v2._index << endl;
				(*c)++;
				if ((*c) == NUM_VIDEO)
				{
					cout << "finished all" << endl;
					process_time(t);
					break;
				}
			}
		}
		if (empty1 && empty2) break;
	}
	imshow("mosaic", *board);
	waitKey(0);
}