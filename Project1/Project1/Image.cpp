#include "Image.h"

#include <iostream>
#include <fstream>

Color::Color()
	: r(0), g(0), b(0)
{
}

Color::Color(float r, float g, float b)
	: r(r), g(g), b(b)
{
}

Color::~Color()
{
}

Image::Image(int width, int height)
	: _width(width), _height(height), _colors(std::vector<Color>(width*height))
{
}

Image::~Image()
{
}

Color Image::GetColor(int x, int y) const
{
	return _colors[y * _width + x];
}

void Image::SetColor(const Color& color, int x, int y)
{
	_colors[y * _width + x].r = color.r;
	_colors[y * _width + x].g = color.g;
	_colors[y * _width + x].b = color.b;
}

void Image::Export(const char* path) const
{
	std::ofstream f;
	f.open(path, std::ios::out | std::ios::binary);
	
	if (!f.is_open())
	{
		std::cout << "File could not be opened.\n";
		return;
	}

	unsigned char bmpPad[3] = { 0,0,0 };
	const int paddingAmount = { (4 - (_width * 3) % 4) % 4 };

	const int fileHeaderSize = 14;
	const int informationHeaderSize = 40;
	const int fileSize = fileHeaderSize + informationHeaderSize + _width * _height * 3 + paddingAmount * _width;

	unsigned char fileHeader[fileHeaderSize];

	// File type
	fileHeader[0] = 'B';
	fileHeader[1] = 'M';
	// File size
	fileHeader[2] = fileSize;
	fileHeader[3] = fileSize >> 8;
	fileHeader[4] = fileSize >> 16;
	fileHeader[5] = fileSize >> 24;
	// Reserved 1 (Not used)
	fileHeader[6] = 0;
	fileHeader[7] = 0;
	// Reserved 2 (Not used)
	fileHeader[8] = 0;
	fileHeader[9] = 0;
	// Pixel data offset
	fileHeader[10] = fileHeaderSize + informationHeaderSize;
	fileHeader[11] = 0;
	fileHeader[12] = 0;
	fileHeader[13] = 0;

	unsigned char informationHeader[informationHeaderSize];

	// Header size
	informationHeader[0] = informationHeaderSize;
	informationHeader[1] = 0;
	informationHeader[2] = 0;
	informationHeader[3] = 0;
	// Image width
	informationHeader[4] = _width;
	informationHeader[5] = _width >> 8;
	informationHeader[6] = _width >> 16;
	informationHeader[7] = _width >> 24;
	// Image height
	informationHeader[8] = _height;
	informationHeader[9] = _height >> 8;
	informationHeader[10] = _height >> 16;
	informationHeader[11] = _height >> 24;
	// Planes
	informationHeader[12] = 1;
	informationHeader[13] = 0;
	// Bits per pixel (RGB)
	informationHeader[14] = 24;
	informationHeader[15] = 0;
	// Compressino (no compression)
	informationHeader[16] = 0;
	informationHeader[17] = 0;
	informationHeader[18] = 0;
	informationHeader[19] = 0;
	// Image Size (no compression)
	informationHeader[20] = 0;
	informationHeader[21] = 0;
	informationHeader[22] = 0;
	informationHeader[23] = 0;
	// x pixels per meter (not specified)
	informationHeader[24] = 0;
	informationHeader[25] = 0;
	informationHeader[26] = 0;
	informationHeader[27] = 0;
	// y pixels per meter (not specified)
	informationHeader[28] = 0;
	informationHeader[29] = 0;
	informationHeader[30] = 0;
	informationHeader[31] = 0;
	// total colors (color palette not used)
	informationHeader[32] = 0;
	informationHeader[33] = 0;
	informationHeader[34] = 0;
	informationHeader[35] = 0;
	// Important colors (generally ignore)
	informationHeader[36] = 0;
	informationHeader[37] = 0;
	informationHeader[38] = 0;
	informationHeader[39] = 0;

	f.write(reinterpret_cast<char*>(fileHeader), fileHeaderSize);
	f.write(reinterpret_cast<char*>(informationHeader), informationHeaderSize);
	
	for (int y = 0; y < _height; y++)
	{
		for (int x = 0; x < _width; x++)
		{
			unsigned char r = static_cast<unsigned char>(GetColor(x, y).r * 255.0f);
			unsigned char g = static_cast<unsigned char>(GetColor(x, y).g * 255.0f);
			unsigned char b = static_cast<unsigned char>(GetColor(x, y).b * 255.0f);

			unsigned char color[] = { b,g,r };

			f.write(reinterpret_cast<char*>(color), 3);
		}
		f.write(reinterpret_cast<char*>(bmpPad), paddingAmount);
	}
	f.close();
	std::cout << "File created.\n";
}
