
#pragma once

#include "Matrix.h"


namespace tbml
{
	float sigmoid(float x);
	float sigmoidPd(float x);
	float tanh(float x);
	float tanhPd(float x);
	float relu(float x);
	float reluPd(float x);
	float sign(float x);
	float signPd(float x);
	//float softmax(float x);
	//float softmaxPd(float x);

	float calcErrSqDiff(Matrix predicted, Matrix expected);
	Matrix calcErrSqDiffPd(Matrix predicted, Matrix expected);

	float getRandomFloat();
}
