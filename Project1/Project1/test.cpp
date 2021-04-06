#include "test.h"
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

using namespace cv;
using namespace std;


int main(int ac, char** av) {

    Openmp mp;
    //mp.CompareBilinearInterpolation();
	
	mp.CompareBicubicInterpolation();
    //mp.CompareFilter2DCV_2DMP();
    //mp.Sum();
    //mp.Multiply();
    //mp.fnc();

    //Ipp _Ipp;
    ////_Ipp.GaussianFilter();
    //_Ipp.MedianFilter();
    //
    //Opencv _Opencv;
    ////_Opencv.GaussianFilter();
    //_Opencv.MedianFilter();

    return 0;
}

void wInter(int x, int y,  float* w)
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

void Interp(unsigned char* src, int hg, int wd, float* w, int x, int y, unsigned char* output) {

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