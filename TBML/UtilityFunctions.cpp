
#include "stdafx.h"
#include "UtilityFunctions.h"
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
		for (size_t row = 0; row < predicted.getRows(); row++)
		{
			for (size_t col = 0; col < predicted.getCols(); col++)
			{
				float diff = expected.get(row, col) - predicted.get(row, col);
				error += diff * diff;
			}
		}
		return 0.5f * error;
	}

	Matrix calcErrSqDiffPd(const Matrix& predicted, const Matrix& expected) { return predicted.sub(expected); }

	float getRandomFloat() { return static_cast<float>(rand()) / static_cast<float>(RAND_MAX); }
}
