#pragma once

class Bayer
{
public:
	void Interpolation();

private:
	void interpolation_serial(unsigned short* data, unsigned short* rgb, int img_width, int img_height, char* mask_arr);
	void interpolation_parallel(unsigned short* data, unsigned short* rgb, int img_width, int img_height, char* mask_arr);
	void seq_data_copy(unsigned char* buffer, unsigned short* data, int size);
	void create_mask(int img_width, int img_height, int mask_width, int mask_height, char* mask_data, char* mask_pattern);
	unsigned short averaging(unsigned short* data, char* mask,char rgb,int img_width, int img_height, int row, int col);

	void save_raw_file_10bit(const char* filename, unsigned short* data, int img_width, int img_height);
	void save_bmp(const char* filename, unsigned short* rgb_data, int img_width, int img_height);
};

