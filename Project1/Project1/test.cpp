#include "test.h"
#include "Ipp.h"
#include "Interpolation.h"
#include "video.h"
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
#include <bitset>

using namespace cv;
using namespace std;

void seq_data_copy(unsigned char* buffer, unsigned short* data, int size);
void SaveRawFile10bit(const char* filename, unsigned short* data, int dataWidth, int dataHeight);

int main(int ac, char** av) {
    int w = 3264; // image width
    int h = 2448; // image height 

    //******************* File Read ***********************//
    FILE* pFile; // File pointer
    long lSize;
    unsigned char* raw;
    unsigned char* RGB;
    size_t result;
    fopen_s(&pFile, "raw.raw", "rb");
    if (pFile == NULL) { fputs("File error", stderr); exit(1); }

    // obtain file size:
    fseek(pFile, 0, SEEK_END);
    lSize = ftell(pFile);
    rewind(pFile);

    // allocate memory to contain the whole file:
    raw = (unsigned char*)malloc(sizeof(unsigned char) * lSize);
    if (raw == NULL) { fputs("Memory error", stderr); exit(2); }

    // copy the file into the buffer:
    result = fread(raw, 1, lSize, pFile);
    if (result != lSize) { fputs("Reading error", stderr); exit(3); }
    unsigned short* data = (unsigned short*)malloc(sizeof(unsigned short) * h * w);

    // 8bit data to 10 bit data( raw -> data)
    seq_data_copy(raw, data, lSize);

    int s_data = _msize(data) / sizeof(unsigned short);
    int s_raw = _msize(raw) / sizeof(unsigned char);
    cout << "raw: " << s_raw << endl;
    cout << "data: " << s_data << endl;
    cout << s_raw / 5 << endl;
    cout << s_data / 4 << endl;

    //interpolation 
    
    //char output_file_name[150];
    
    //strcpy(output_file_name,"rgb.raw"); 
    
    //SaveRawFile10bit(output_file_name,RGB,h,w*3);

    fclose(pFile);
    free(raw);

    return 0;

}

void seq_data_copy(unsigned char* buffer, unsigned short* data, int size)
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

void SaveRawFile10bit(const char* filename, unsigned short* data, int dataWidth, int dataHeight)
{
    FILE* fp;
    fopen_s(&fp, filename, "wb");
    fwrite(data, sizeof(unsigned short), dataHeight * dataWidth, fp);
    fclose(fp);
}