
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
	// 
	//float actSoftmaxPd(float x) { return 0; }


	float calcErrSqDiff(Matrix predicted, Matrix expected)
	{
		std::vector<std::vector<float>>& pData = predicted.getData();
		std::vector<std::vector<float>>& eData = expected.getData();
		float error = 0;
		for (size_t row = 0; row < predicted.getRows(); row++)
		{
			for (size_t col = 0; col < predicted.getCols(); col++)
			{
				float diff = eData[row][col] - pData[row][col];
				error += diff * diff;
			}
		}
		return 0.5f * error;
	}
	
	Matrix calcErrSqDiffPd(Matrix predicted, Matrix expected) { return predicted.sub(expected); }


	float getRandomFloat() { return static_cast<float>(rand()) / static_cast<float>(RAND_MAX); }
}
