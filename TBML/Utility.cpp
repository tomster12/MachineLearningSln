
#include "stdafx.h"
#include "Utility.h"
#include "Matrix.h"

namespace tbml
{
	float sigmoid(float x) { return 1.0f / (1.0f + exp(-x)); }

	float sigmoidPd(float x) { return x * (1 - x); }

	float tanh(float x) { return tanhf(x); }

	float tanhPd(float x) { float th = tanh(x); return 1 - th * th; }

	float relu(float x) { return x > 0.0f ? x : 0.0f; }

	float reluPd(float x) { return x > 0.0f ? 1.0f : 0.0f; }

	float sign(float x) { return x > 0.0f ? 1.0f : -1.0f; }

	float signPd(float x) { return x > 0.0f ? 1.0f : -1.0f; }

	//float actSoftmax(float x) { return 0; }

	//float actSoftmaxPd(float x) { return 0; }

	float calcErrSqDiff(const Matrix& predicted, const Matrix& expected)
	{
		float error = 0;
		for (size_t i = 0; i < predicted.getRowCount(); i++)
		{
			for (size_t j = 0; j < predicted.getColCount(); j++)
			{
				float diff = expected(i, j) - predicted(i, j);
				error += diff * diff;
			}
		}
		return 0.5f * error;
	}

	Matrix calcErrSqDiffPd(const Matrix& predicted, const Matrix& expected) { return predicted - expected; }

	float getRandomFloat() { return static_cast<float>(rand()) / static_cast<float>(RAND_MAX); }
}
