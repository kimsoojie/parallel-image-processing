
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include<iostream>

extern "C" void gpu_Gabor(float* pcuSrc, float* pcuDst, int w, int h, float* cuGkernel, int kernel_size);
extern "C" void gpu_Gaussian(float* pcuSrc, float* pcuDst, int w, int h, float* cuGkernel, int kernel_size);

extern "C" const int KERNEL_SIZE = 5;

using namespace std;

__constant__ float constKernel[KERNEL_SIZE * KERNEL_SIZE];

__global__ void cuda_Filter2D(float* pSrcImage, int SrcWidth, int SrcHeight, float* pKernel, int KWidth, int KHeight, float* pDstImage)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    int index = y * SrcWidth + x;
    float temp;
    int _x, _y;

    if (x >= KWidth / 2 && y >= KHeight / 2 && x < SrcWidth - KWidth / 2 && y < SrcHeight - KHeight / 2)
    {
        for (int i = 0; i < KHeight; i++)
        {
            for (int j = 0; j < KWidth; j++)
            {
                _x = j - (KWidth / 2);
                _y = i - (KHeight / 2);
                
                temp += pKernel[i * KWidth + j] * pSrcImage[(_y * SrcWidth + _x) + index];
            }
        }
        pDstImage[index] = temp;
    }
    else
    {
        pDstImage[index] = 0;
    }
}

__global__ void cuda_Shared_Filter2D(float* pSrcImage, int SrcWidth, int SrcHeight, float* pKernel, int KWidth, int KHeight, float* pDstImage)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    int index = y * SrcWidth + x;
    float temp;
    int _x, _y;

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    extern __shared__ float gmat[];
    if (tx < KWidth && ty < KHeight)
    {
        gmat[ty * KWidth + tx] = pKernel[ty * KWidth + tx];
    }
    //__syncthreads();

    if (x >= KWidth / 2 && y >= KHeight / 2 && x < SrcWidth - KWidth / 2 && y < SrcHeight - KHeight / 2)
    {
        for (int i = 0; i < KHeight; i++)
        {
            for (int j = 0; j < KWidth; j++)
            {
                _x = j - (KWidth / 2);
                _y = i - (KHeight / 2);

                temp += gmat[i * KWidth + j] * pSrcImage[(_y * SrcWidth + _x) + index];
            }
        }
        pDstImage[index] = temp;
    }
    else
    {
        pDstImage[index] = 0;
    }
}

__global__ void cuda_const_Filter2D(float* pSrcImage, int SrcWidth, int SrcHeight, int KWidth, int KHeight, float* pDstImage)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    int index = y * SrcWidth + x;
    float temp;
    int _x, _y;

    if (x >= KWidth / 2 && y >= KHeight / 2 && x < SrcWidth - KWidth / 2 && y < SrcHeight - KHeight / 2)
    {
        for (int i = 0; i < KHeight; i++)
        {
            for (int j = 0; j < KWidth; j++)
            {
                _x = j - (KWidth / 2);
                _y = i - (KHeight / 2);

                temp += constKernel[i * KWidth + j] * pSrcImage[(_y * SrcWidth + _x) + index];
            }
        }
        pDstImage[index] = temp;
    }
    else
    {
        pDstImage[index] = 0;
    }
}


void gpu_Gabor(float* pcuSrc, float* pcuDst, int w, int h, float* cuGkernel, int kernel_size)
{
    dim3 block = dim3(16, 16);
    dim3 grid = dim3(w / 16, h / 16); 

    // <global>
    cuda_Filter2D << <grid, block >> > (pcuSrc, w, h, cuGkernel, kernel_size, kernel_size, pcuDst);

    // <shared>
    //cuda_Shared_Filter2D << < grid, block, sizeof(float)* kernel_size* kernel_size >> > (pcuSrc, w, h, cuGkernel, kernel_size, kernel_size, pcuDst);
    
    // <const>
    //cudaMemcpyToSymbol(constKernel, cuGkernel, sizeof(float) * kernel_size * kernel_size);
    //cuda_const_Filter2D << < grid, block >> > (pcuSrc, w, h, kernel_size, kernel_size, pcuDst);

    cudaThreadSynchronize();

}


void gpu_Gaussian(float* pcuSrc, float* pcuDst, int w, int h, float* cuGkernel, int kernel_size)
{
    dim3 block = dim3(16, 16);
    dim3 grid = dim3(w / 16, h / 16);

    // <global>
    cuda_Filter2D <<<grid, block >>> (pcuSrc, w, h, cuGkernel, kernel_size, kernel_size, pcuDst);
     
    // <shared>
    //cuda_Shared_Filter2D << < grid, block, sizeof(float)* kernel_size* kernel_size >> > (pcuSrc, w, h, cuGkernel, kernel_size, kernel_size, pcuDst);

    // <const>
    //cudaMemcpyToSymbol(constKernel, cuGkernel, sizeof(float) * kernel_size * kernel_size);
    //cuda_const_Filter2D << < grid, block >> > (pcuSrc, w, h, kernel_size, kernel_size, pcuDst);

    cudaThreadSynchronize();

}