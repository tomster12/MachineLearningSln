
#pragma once

#include <stdlib.h>


typedef unsigned char uchar;

class MNIST {

public:
	static uchar** readImages(std::string path, int& imageCount, int& imageSize);
	static uchar* readLabels(std::string path, int& labelCount);
};
