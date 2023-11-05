
#pragma once

#include <fstream>
#include <stdlib.h>
#include <uchar.h>

typedef unsigned char uchar;

class MNIST
{
public:
	static uchar** readImages(std::string path, size_t& imageCount, size_t& imageSize)
	{
		// Helper function
		auto reverseInt = [](int i)
		{
			unsigned char c1, c2, c3, c4;
			c1 = i & 255, c2 = (i >> 8) & 255, c3 = (i >> 16) & 255, c4 = (i >> 24) & 255;
			return ((int)c1 << 24) + ((int)c2 << 16) + ((int)c3 << 8) + c4;
		};

		// Open file
		std::ifstream file(path, std::ios::binary);
		if (file.is_open())
		{
			int magic_number = 0, n_rows = 0, n_cols = 0;

			// Read magic number
			file.read((char*)&magic_number, sizeof(magic_number));
			magic_number = reverseInt(magic_number);
			if (magic_number != 2051) throw std::runtime_error("Invalid MNIST image file!");

			// Read dataset parameters
			file.read((char*)&imageCount, sizeof(imageCount)), imageCount = reverseInt(imageCount);
			file.read((char*)&n_rows, sizeof(n_rows)), n_rows = reverseInt(n_rows);
			file.read((char*)&n_cols, sizeof(n_cols)), n_cols = reverseInt(n_cols);
			imageSize = n_rows * n_cols;

			// Read in dataset
			uchar** _dataset = new uchar * [imageCount];
			for (size_t i = 0; i < imageCount; i++)
			{
				_dataset[i] = new uchar[imageSize];
				file.read((char*)_dataset[i], imageSize);
			}
			return _dataset;
		}

		// File reading error
		else throw std::runtime_error("Cannot open file `" + path + "`!");
	}

	static uchar* readLabels(std::string path, size_t& labelCount)
	{
		// Helper function
		auto reverseInt = [](int i)
		{
			unsigned char c1, c2, c3, c4;
			c1 = i & 255, c2 = (i >> 8) & 255, c3 = (i >> 16) & 255, c4 = (i >> 24) & 255;
			return ((int)c1 << 24) + ((int)c2 << 16) + ((int)c3 << 8) + c4;
		};

		// Open file
		std::ifstream file(path, std::ios::binary);
		if (file.is_open())
		{
			int magic_number = 0;

			// Read magic number
			file.read((char*)&magic_number, sizeof(magic_number));
			magic_number = reverseInt(magic_number);
			if (magic_number != 2049) throw std::runtime_error("Invalid MNIST label file!");

			// Read dataset parameters
			file.read((char*)&labelCount, sizeof(labelCount)), labelCount = reverseInt(labelCount);

			// Read in dataset
			uchar* _dataset = new uchar[labelCount];
			for (size_t i = 0; i < labelCount; i++)
				file.read((char*)&_dataset[i], 1);
			return _dataset;
		}

		// File reading error
		else throw std::runtime_error("Unable to open file `" + path + "`!");
	}
};
