
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include<iostream>

extern "C" void gpu_update_board(float* pcuSrc, float* pcuDst, int frame_w, int frame_h, int board_w, int board_h, int start_i, int start_j);

using namespace std;

__global__ void cuda_Update(float* pSrcImage, float* pDstImage, int frame_w, int frame_h, int board_w, int board_h, int start_i, int start_j)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    int index_src = y * frame_w * 3 + x * 3;
    int index_dst = (y+ start_i) * board_w * 3 + (x+ start_j) * 3;

    
    pDstImage[index_dst + 0] = pSrcImage[index_src + 0];
    pDstImage[index_dst + 1] = pSrcImage[index_src + 1];
    pDstImage[index_dst + 2] = pSrcImage[index_src + 2];

}

void gpu_update_board(float* pcuSrc, float* pcuDst, int frame_w, int frame_h, int board_w, int board_h, int start_i, int start_j)
{
    dim3 block = dim3(25, 20);
    dim3 grid = dim3(frame_w / 25, frame_h / 20);

    cuda_Update << <grid, block >> > (pcuSrc, pcuDst, frame_w, frame_h, board_w, board_h,  start_i,  start_j);

    cudaThreadSynchronize();

}