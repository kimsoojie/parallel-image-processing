#include "Bayer.h"
#include "Image.h"
#include "opencv2/highgui.hpp"
#include "opencv2/opencv.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"

#include <omp.h>
#include <ipp.h>
#include <iostream>
#include <stdio.h>

#define MASK_WIDTH 2
#define MASK_HEIGHT 2

#define IMG_WIDTH 3264
#define IMG_HEIGHT 2448

using namespace std;
using namespace cv;

void Bayer::Interpolation()
{
    // read file
    FILE* pFile; 
    long lSize;
    unsigned char* raw;
    unsigned short* RGB_serial;
    unsigned short* RGB_parallel;
    size_t result;
    fopen_s(&pFile, "raw.raw", "rb");
    if (pFile == NULL) { fputs("File error", stderr); exit(1); }

    // obtain file size
    fseek(pFile, 0, SEEK_END);
    lSize = ftell(pFile);
    rewind(pFile);

    // allocate memory to contain the whole file
    raw = (unsigned char*)malloc(sizeof(unsigned char) * lSize);
    if (raw == NULL) { fputs("Memory error", stderr); exit(2); }

    // copy the file into the buffer
    result = fread(raw, 1, lSize, pFile);
    if (result != lSize) { fputs("Reading error", stderr); exit(3); }
    unsigned short* data = (unsigned short*)malloc(sizeof(unsigned short) * IMG_HEIGHT * IMG_WIDTH);
    RGB_serial = (unsigned short*)malloc(sizeof(unsigned short) * IMG_HEIGHT * IMG_WIDTH * 3);
    RGB_parallel = (unsigned short*)malloc(sizeof(unsigned short) * IMG_HEIGHT * IMG_WIDTH * 3);

    // 8bit data to 10 bit data( raw -> data)
    seq_data_copy(raw, data, lSize);

    // make mask
    char pattern[MASK_HEIGHT * MASK_WIDTH] = {'r','g','g','b'};
    char* mask = (char*)malloc(sizeof(char*) * IMG_HEIGHT * IMG_WIDTH);
    create_mask(IMG_WIDTH, IMG_HEIGHT, MASK_WIDTH, MASK_HEIGHT, mask, pattern);
    
    // interpolation (Serial)
    TickMeter tm;
    tm.start();
    //for(int i = 0 ; i < 50; i++)
        interpolation_serial(data, RGB_serial, IMG_WIDTH, IMG_HEIGHT, mask);
    tm.stop();
    cout << "serial process time : " << tm.getTimeMilli() << endl;

    // interpolation (Parallel)
    tm.reset();
    tm.start();
    //for (int i = 0; i < 50; i++)
        interpolation_parallel(data, RGB_parallel, IMG_WIDTH, IMG_HEIGHT, mask);
    tm.stop();
    cout << "parallel process time : " << tm.getTimeMilli() << endl;

    int sum = 0;
    for (int i = 0; i < IMG_HEIGHT; i++)
    {
        for (int j = 0; j < IMG_WIDTH; j++)
        {
            int idx = i * IMG_WIDTH + j;
            sum += (RGB_serial[idx] - RGB_parallel[idx]);
        }
    }
    cout << "\nsum of difference: " << sum << endl;
    
    // save raw file (10 bit)
    save_raw_file_10bit("10bit_raw_serial.raw", RGB_serial,IMG_HEIGHT,IMG_WIDTH*3);
    save_raw_file_10bit("10bit_raw_parallel.raw", RGB_parallel,IMG_HEIGHT,IMG_WIDTH*3);
    
    // save bmp file (10 bit)
    save_bmp("result_serial.bmp", RGB_serial, IMG_WIDTH, IMG_HEIGHT);
    save_bmp("result_parallel.bmp", RGB_parallel, IMG_WIDTH, IMG_HEIGHT);

    fclose(pFile);
    free(raw);

}

void Bayer::interpolation_serial(unsigned short* data, unsigned short* rgb, int width, int height, char* mask_arr)
{
    int start_r = width * height * 0;
    int start_g = width * height * 1;
    int start_b = width * height * 2;

    //for (int i = 0; i < height; i++)
    //{
    //    for (int j = 0; j < width; j++)
    //    {
    //        int idx_r = i * width + j + start_r;
    //        int idx_g = i * width + j + start_g;
    //        int idx_b = i * width + j + start_b;
    //        rgb[idx_r] = averaging(data, mask_arr, 'r', width, height, i, j);
    //        rgb[idx_g] = averaging(data, mask_arr, 'g', width, height, i, j);
    //        rgb[idx_b] = averaging(data, mask_arr, 'b', width, height, i, j);
    //    }
    //}

    
    for (int i = 1; i < height - 1; i += 2)
    {
        for (int j = 1; j < width - 1; j+=2)
        {
            int idx_r = (i+1) * width + (j+1);
            int idx_g1 = i * width + (j+1);
            int idx_g2 = (i+1) * width + j ;
            int idx_b = i * width + j;

            rgb[idx_r + start_r] = data[idx_r];
            rgb[idx_g1 + start_r] = (data[idx_g1 - width] + data[idx_g1 + width]) / 2;
            rgb[idx_g2 + start_r] = (data[idx_g2 - 1] + data[idx_g2 + 1]) / 2;
            rgb[idx_b + start_r] = (data[idx_b - width - 1] + data[idx_b - width + 1] + data[idx_b + width - 1] + data[idx_b + width + 1]) / 4;

            rgb[idx_r + start_g] = (data[idx_r - width] + data[idx_r + width] + data[idx_r - 1] + data[idx_r + 1]) / 4;
            rgb[idx_g1 + start_g] = data[idx_g1];
            rgb[idx_g2 + start_g] = data[idx_g2];
            rgb[idx_b + start_g] = (data[idx_b - width] + data[idx_b + width] + data[idx_b - 1] + data[idx_b + 1]) / 4;

            rgb[idx_r + start_b] = (data[idx_r - width - 1] + data[idx_r - width + 1] + data[idx_r + width - 1] + data[idx_r + width + 1]) / 4;
            rgb[idx_g1 + start_b] = (data[idx_g1 - 1] + data[idx_g1 + 1]) / 2;
            rgb[idx_g2 + start_b] = (data[idx_g2 - width] + data[idx_g2 + width]) / 2;
            rgb[idx_b + start_b] = data[idx_b];
        }
    }
   


    rgb[start_r] = data[0];
    rgb[start_g] = (data[1] + data[width]) / 2;
    rgb[start_b] = data[width + 1];

    rgb[width - 1 + start_r] = data[width - 2];
    rgb[width - 1 + start_g] = data[width - 1];
    rgb[width - 1 + start_b] = data[2 * width - 1];

    rgb[width * (height - 1) + start_r] = data[width * (height - 1) - width];
    rgb[width * (height - 1) + start_g] = data[width * (height - 1)];
    rgb[width * (height - 1) + start_b] = data[width * (height - 1) + 1];

    rgb[width * (height - 1) + width - 1 + start_r] = data[width * (height - 1) - 2];
    rgb[width * (height - 1) + width - 1 + start_g] = (data[width * (height - 1) - 1] + data[width * (height - 1) + width - 2]) / 2;
    rgb[width * (height - 1) + width - 1 + start_b] = data[width * (height - 1) + width - 1];

    for (int j = 1; j < width - 1; j += 2)
    {

        int top = 0;
        int bottom = height - 1;

        int idx_r_top_1 = top * width + j + start_r;
        int idx_g_top_1 = top * width + j + start_g;
        int idx_b_top_1 = top * width + j + start_b;

        int idx_r_bottom_1 = bottom * width + j + start_r;
        int idx_g_bottom_1 = bottom * width + j + start_g;
        int idx_b_bottom_1 = bottom * width + j + start_b;

        int cur_t_1 = j;
        int cur_b_1 = bottom * width + j;

        int idx_r_top_2 = top * width + j + start_r+1;
        int idx_g_top_2 = top * width + j + start_g+1;
        int idx_b_top_2 = top * width + j + start_b+1;

        int idx_r_bottom_2 = bottom * width + j + start_r+1;
        int idx_g_bottom_2 = bottom * width + j + start_g+1;
        int idx_b_bottom_2 = bottom * width + j + start_b+1;

        int cur_t_2 = j+1;
        int cur_b_2 = bottom * width + j+1;

        rgb[idx_r_top_1] = (data[cur_t_1 - 1] + data[cur_t_1 + 1]) / 2;
        rgb[idx_g_top_1] = data[cur_t_1];
        rgb[idx_b_top_1] = data[width + cur_t_1];

        rgb[idx_r_bottom_1] = (data[cur_b_1 - width - 1] + data[cur_b_1 - width + 1]) / 2;
        rgb[idx_g_bottom_1] = (data[cur_b_1 - 1] + data[cur_b_1 + 1]) / 2;
        rgb[idx_b_bottom_1] = data[cur_b_1];

        rgb[idx_r_top_2] = data[cur_t_2];
        rgb[idx_g_top_2] = (data[width + cur_t_2] + data[cur_t_2 + 1]) / 2;
        rgb[idx_b_top_2] = cur_t_2 > 0 ? (data[width + cur_t_2 - 1] + data[width + cur_t_2 + 1]) / 2 : data[width + cur_t_2 + 1];

        rgb[idx_r_bottom_2] = data[cur_b_2 - width];
        rgb[idx_g_bottom_2] = data[cur_b_2];
        rgb[idx_b_bottom_2] = data[cur_b_2 + 1];
    }

    for (int i = 1; i < height - 1; i+=2)
    {
        int left = 0;
        int right = width - 1;

        int idx_r_left = i * width + left + start_r;
        int idx_g_left = i * width + left + start_g;
        int idx_b_left = i * width + left + start_b;

        int idx_r_right = i * width + right + start_r;
        int idx_g_right = i * width + right + start_g;
        int idx_b_right = i * width + right + start_b;

        int cur_l = i * width;
        int cur_r = i * width + right;

        int idx_r_left_2 = (i+1) * width + left + start_r;
        int idx_g_left_2 = (i+1) * width + left + start_g;
        int idx_b_left_2 = (i+1) * width + left + start_b;

        int idx_r_right_2 = (i+1) * width + right + start_r;
        int idx_g_right_2 = (i+1) * width + right + start_g;
        int idx_b_right_2 = (i+1) * width + right + start_b;

        int cur_l_2 = (i+1) * width;
        int cur_r_2 = (i+1) * width + right;

        rgb[idx_r_left] = (data[cur_l - width] + data[cur_l + width]) / 2;
        rgb[idx_g_left] = data[cur_l];
        rgb[idx_b_left] = data[cur_l + 1];

        rgb[idx_r_right] = (data[cur_r - width - 1] + data[cur_r + width - 1]) / 2;
        rgb[idx_g_right] = (data[cur_r - width] + data[cur_r + width] + data[cur_r - 1]) / 3;
        rgb[idx_b_right] = data[cur_r];

        rgb[idx_r_left_2] = data[cur_l_2];
        rgb[idx_g_left_2] = (data[cur_l_2 - width] + data[cur_l_2 + width] + data[cur_l_2 + 1]) / 3;
        rgb[idx_b_left_2] = (data[cur_l_2 - width + 1] + data[cur_l_2 + width + 1]) / 2;

        rgb[idx_r_right_2] = data[cur_r_2 - 1];
        rgb[idx_g_right_2] = data[cur_r_2];
        rgb[idx_b_right_2] = (data[cur_r_2 - width] + data[cur_r_2 + width]) / 2;
    }
    
}

void Bayer::interpolation_parallel(unsigned short* data, unsigned short* rgb, int width, int height, char* mask_arr)
{
    int start_r = width * height * 0;
    int start_g = width * height * 1;
    int start_b = width * height * 2;

//#pragma omp parallel for
//    for (int i = 0; i < height; i++)
//    {
//        for (int j = 0; j < width; j++)
//        {
//            int idx_r = i * width + j + start_r;
//            int idx_g = i * width + j + start_g;
//            int idx_b = i * width + j + start_b;
//            rgb[idx_r] = averaging(data, mask_arr, 'r', width, height, i, j);
//            rgb[idx_g] = averaging(data, mask_arr, 'g', width, height, i, j);
//            rgb[idx_b] = averaging(data, mask_arr, 'b', width, height, i, j);
//        }
//    }

    
#pragma omp parallel for
    for (int i = 1; i < height - 1; i += 2)
    {
        for (int j = 1; j < width - 1; j += 2)
        {
            int idx_r = (i + 1) * width + (j + 1);
            int idx_g1 = i * width + (j + 1);
            int idx_g2 = (i + 1) * width + j;
            int idx_b = i * width + j;

            rgb[idx_r + start_r] = data[idx_r];
            rgb[idx_g1 + start_r] = (data[idx_g1 - width] + data[idx_g1 + width]) / 2;
            rgb[idx_g2 + start_r] = (data[idx_g2 - 1] + data[idx_g2 + 1]) / 2;
            rgb[idx_b + start_r] = (data[idx_b - width - 1] + data[idx_b - width + 1] + data[idx_b + width - 1] + data[idx_b + width + 1]) / 4;

            rgb[idx_r + start_g] = (data[idx_r - width] + data[idx_r + width] + data[idx_r - 1] + data[idx_r + 1]) / 4;
            rgb[idx_g1 + start_g] = data[idx_g1];
            rgb[idx_g2 + start_g] = data[idx_g2];
            rgb[idx_b + start_g] = (data[idx_b - width] + data[idx_b + width] + data[idx_b - 1] + data[idx_b + 1]) / 4;

            rgb[idx_r + start_b] = (data[idx_r - width - 1] + data[idx_r - width + 1] + data[idx_r + width - 1] + data[idx_r + width + 1]) / 4;
            rgb[idx_g1 + start_b] = (data[idx_g1 - 1] + data[idx_g1 + 1]) / 2;
            rgb[idx_g2 + start_b] = (data[idx_g2 - width] + data[idx_g2 + width]) / 2;
            rgb[idx_b + start_b] = data[idx_b];
        }
    }

#pragma omp parallel
    {
        rgb[start_r] = data[0];
        rgb[start_g] = (data[1] + data[width]) / 2;
        rgb[start_b] = data[width + 1];

        rgb[width - 1 + start_r] = data[width - 2];
        rgb[width - 1 + start_g] = data[width - 1];
        rgb[width - 1 + start_b] = data[2 * width - 1];

        rgb[width * (height - 1) + start_r] = data[width * (height - 1) - width];
        rgb[width * (height - 1) + start_g] = data[width * (height - 1)];
        rgb[width * (height - 1) + start_b] = data[width * (height - 1) + 1];

        rgb[width * (height - 1) + width - 1 + start_r] = data[width * (height - 1) - 2];
        rgb[width * (height - 1) + width - 1 + start_g] = (data[width * (height - 1) - 1] + data[width * (height - 1) + width - 2]) / 2;
        rgb[width * (height - 1) + width - 1 + start_b] = data[width * (height - 1) + width - 1];
    }

#pragma omp parallel for
    for (int j = 1; j < width - 1; j += 2)
    {

        int top = 0;
        int bottom = height - 1;

        int idx_r_top_1 = top * width + j + start_r;
        int idx_g_top_1 = top * width + j + start_g;
        int idx_b_top_1 = top * width + j + start_b;

        int idx_r_bottom_1 = bottom * width + j + start_r;
        int idx_g_bottom_1 = bottom * width + j + start_g;
        int idx_b_bottom_1 = bottom * width + j + start_b;

        int cur_t_1 = j;
        int cur_b_1 = bottom * width + j;

        int idx_r_top_2 = top * width + j + start_r + 1;
        int idx_g_top_2 = top * width + j + start_g + 1;
        int idx_b_top_2 = top * width + j + start_b + 1;

        int idx_r_bottom_2 = bottom * width + j + start_r + 1;
        int idx_g_bottom_2 = bottom * width + j + start_g + 1;
        int idx_b_bottom_2 = bottom * width + j + start_b + 1;

        int cur_t_2 = j + 1;
        int cur_b_2 = bottom * width + j + 1;

        rgb[idx_r_top_1] = (data[cur_t_1 - 1] + data[cur_t_1 + 1]) / 2;
        rgb[idx_g_top_1] = data[cur_t_1];
        rgb[idx_b_top_1] = data[width + cur_t_1];

        rgb[idx_r_bottom_1] = (data[cur_b_1 - width - 1] + data[cur_b_1 - width + 1]) / 2;
        rgb[idx_g_bottom_1] = (data[cur_b_1 - 1] + data[cur_b_1 + 1]) / 2;
        rgb[idx_b_bottom_1] = data[cur_b_1];

        rgb[idx_r_top_2] = data[cur_t_2];
        rgb[idx_g_top_2] = (data[width + cur_t_2] + data[cur_t_2 + 1]) / 2;
        rgb[idx_b_top_2] = cur_t_2 > 0 ? (data[width + cur_t_2 - 1] + data[width + cur_t_2 + 1]) / 2 : data[width + cur_t_2 + 1];

        rgb[idx_r_bottom_2] = data[cur_b_2 - width];
        rgb[idx_g_bottom_2] = data[cur_b_2];
        rgb[idx_b_bottom_2] = data[cur_b_2 + 1];
    }

#pragma omp parallel for
    for (int i = 1; i < height - 1; i += 2)
    {
        int left = 0;
        int right = width - 1;

        int idx_r_left = i * width + left + start_r;
        int idx_g_left = i * width + left + start_g;
        int idx_b_left = i * width + left + start_b;

        int idx_r_right = i * width + right + start_r;
        int idx_g_right = i * width + right + start_g;
        int idx_b_right = i * width + right + start_b;

        int cur_l = i * width;
        int cur_r = i * width + right;

        int idx_r_left_2 = (i + 1) * width + left + start_r;
        int idx_g_left_2 = (i + 1) * width + left + start_g;
        int idx_b_left_2 = (i + 1) * width + left + start_b;

        int idx_r_right_2 = (i + 1) * width + right + start_r;
        int idx_g_right_2 = (i + 1) * width + right + start_g;
        int idx_b_right_2 = (i + 1) * width + right + start_b;

        int cur_l_2 = (i + 1) * width;
        int cur_r_2 = (i + 1) * width + right;

        rgb[idx_r_left] = (data[cur_l - width] + data[cur_l + width]) / 2;
        rgb[idx_g_left] = data[cur_l];
        rgb[idx_b_left] = data[cur_l + 1];

        rgb[idx_r_right] = (data[cur_r - width - 1] + data[cur_r + width - 1]) / 2;
        rgb[idx_g_right] = (data[cur_r - width] + data[cur_r + width] + data[cur_r - 1]) / 3;
        rgb[idx_b_right] = data[cur_r];

        rgb[idx_r_left_2] = data[cur_l_2];
        rgb[idx_g_left_2] = (data[cur_l_2 - width] + data[cur_l_2 + width] + data[cur_l_2 + 1]) / 3;
        rgb[idx_b_left_2] = (data[cur_l_2 - width + 1] + data[cur_l_2 + width + 1]) / 2;

        rgb[idx_r_right_2] = data[cur_r_2 - 1];
        rgb[idx_g_right_2] = data[cur_r_2];
        rgb[idx_b_right_2] = (data[cur_r_2 - width] + data[cur_r_2 + width]) / 2;
    }
    
}


unsigned short Bayer::averaging(unsigned short* data, char* mask, char rgb, int img_width, int img_height, int row, int col)
{
    if (mask[row * img_width + col] == rgb)
        return data[row * img_width + col];
    
    int count = 0;
    unsigned short sum = 0;

    for (int i = row - 1; i <= row + 1; i++)
    {
        if (i < 0 || i >= img_height) continue;

        for (int j = col - 1; j <= col + 1; j++)
        {
            if (j < 0 || j >= img_width) continue;

            int idx = i * img_width + j;
            if (mask[idx] == rgb)
            {
                sum += data[idx];
                count++;
            }
        }
    }

    return sum / count;
}

void Bayer::create_mask(int img_width, int img_height, int mask_width, int mask_height, char* mask_data, char* mask_pattern)
{
    for (int i = 0; i < img_height; i += mask_height)
    {
        for (int j = 0; j < img_width; j += mask_width)
        {
            for (int h = 0; h < mask_height; h++)
            {
                for (int w = 0; w < mask_width; w++)
                {
                    int idx = (i * img_width + j) + (h * img_width + w);
                    int idx_mask = h * mask_width + w;
                    mask_data[idx] = mask_pattern[idx_mask];
                }
            }
        }
    }
}

void Bayer::seq_data_copy(unsigned char* buffer, unsigned short* data, int size)
{
    for (int i = 0, j = 0; i < size; i += 5, j += 4)
    {
        unsigned short buffer_0 = buffer[i];
        unsigned short buffer_1 = buffer[i + 1];
        unsigned short buffer_2 = buffer[i + 2];
        unsigned short buffer_3 = buffer[i + 3];
        unsigned short buffer_4 = buffer[i + 4];

        // for 10 bit 
        data[j] = (buffer_0 << 2) + ((buffer_4 >> 0) & 3);
        data[j + 1] = (buffer_1 << 2) + ((buffer_4 >> 2) & 3);
        data[j + 2] = (buffer_2 << 2) + ((buffer_4 >> 4) & 3);
        data[j + 3] = (buffer_3 << 2) + ((buffer_4 >> 6) & 3);
    }
}

void Bayer::save_raw_file_10bit(const char* filename, unsigned short* data, int width, int height)
{
    FILE* fp;
    fopen_s(&fp, filename, "wb");
    fwrite(data, sizeof(unsigned short), height * width, fp);
    fclose(fp);
}

void Bayer::save_bmp(const char* filename, unsigned short* rgb_data, int img_width, int img_height)
{
    Image image(img_width, img_height);

    int start_r = img_width * img_height * 0;
    int start_g = img_width * img_height * 1;
    int start_b = img_width * img_height * 2;
    
    for (int i = 0; i < img_height; i++)
    {
        for (int j = 0; j < img_width; j++)
        {
            int idx_r = i * img_width + j + start_r;
            int idx_g = i * img_width + j + start_g;
            int idx_b = i * img_width + j + start_b;
    
            float r = (float)rgb_data[idx_r] / 1023.0f;
            float g = (float)rgb_data[idx_g] / 1023.0f;
            float b = (float)rgb_data[idx_b] / 1023.0f;
           
            image.SetColor(Color(r, g, b), j, i);
        }
    }
    image.Export(filename);

}