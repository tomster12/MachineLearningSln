
#include <ctgmath>
#include "Activators.h"


// TODO: Change to be performed on each (1 x n) row of an (m x n) matrix


float actSigmoid(float x) { return 1.0f / (1.0f + exp(-x)); }
float actSigmoidPd(float x) { return actSigmoid(x) * (1 - actSigmoid(x)); }

float actTanh(float x) { return tanhf(x); }
float actTanhPd(float x) { float th = actTanh(x);  return 1 - th * th; }

float actRelu(float x) { return x > 0 ? x : 0; }
float actReluPd(float x) { return x > 0 ? 1 : 0; }

float actSoftmax(float x) { return 0; }
float actSoftmaxPd(float x) { return 0; }
