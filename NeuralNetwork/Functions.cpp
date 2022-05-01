
#include <ctgmath>
#include "Functions.h"
#include "Matrix.h"


float actSigmoid(float x) { return 1.0f / (1.0f + exp(-x)); }
float actSigmoidPd(float x) { return x * (1 - x); }

float actTanh(float x) { return tanhf(x); }
float actTanhPd(float x) { float th = actTanh(x); return 1 - th * th; }

float actRelu(float x) { return x > 0 ? x : 0; }
float actReluPd(float x) { return x > 0 ? 1 : 0; }

//float actSoftmax(float x) { return 0; }
//float actSoftmaxPd(float x) { return 0; }


float calcErrSqDiff(Matrix predicted, Matrix expected) {
	std::vector<std::vector<float>>* pData = predicted.getData();
	std::vector<std::vector<float>>* eData = expected.getData();
	float error = 0;
	for (int row = 0; row < predicted.getRows(); row++) {
		for (int col = 0; col < predicted.getCols(); col++) {
			float diff = (*eData)[row][col] - (*pData)[row][col];
			error += diff * diff;
		}
	}
	return 0.5f * error;
}
Matrix calcErrSqDiffPd(Matrix predicted, Matrix expected) { return predicted.sub(expected); }
