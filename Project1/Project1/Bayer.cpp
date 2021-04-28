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
    interpolation_serial(data, RGB_serial, IMG_WIDTH, IMG_HEIGHT, mask);
    tm.stop();
    cout << "serial process time : " << tm.getTimeMilli() << endl;

    // interpolation (Parallel)
    tm.reset();
    tm.start();
    interpolation_parallel(data, RGB_parallel, IMG_WIDTH, IMG_HEIGHT, mask);
    tm.stop();
    cout << "parallel process time : " << tm.getTimeMilli() << endl;
    
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

    for (int i = 0; i < height; i++)
    {
        for (int j = 0; j < width; j++)
        {
            int idx_r = i * width + j + start_r;
            int idx_g = i * width + j + start_g;
            int idx_b = i * width + j + start_b;
            rgb[idx_r] = averaging(data, mask_arr, 'r', width, height, i, j);
            rgb[idx_g] = averaging(data, mask_arr, 'g', width, height, i, j);
            rgb[idx_b] = averaging(data, mask_arr, 'b', width, height, i, j);
        }
    }
}

void Bayer::interpolation_parallel(unsigned short* data, unsigned short* rgb, int width, int height, char* mask_arr)
{
    int start_r = width * height * 0;
    int start_g = width * height * 1;
    int start_b = width * height * 2;

#pragma omp parallel for
    for (int i = 0; i < height; i++)
    {
        for (int j = 0; j < width; j++)
        {
            int idx_r = i * width + j + start_r;
            int idx_g = i * width + j + start_g;
            int idx_b = i * width + j + start_b;
            rgb[idx_r] = averaging(data, mask_arr, 'r', width, height, i, j);
            rgb[idx_g] = averaging(data, mask_arr, 'g', width, height, i, j);
            rgb[idx_b] = averaging(data, mask_arr, 'b', width, height, i, j);
        }
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