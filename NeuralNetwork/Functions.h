
#pragma once

#include "Matrix.h"


float actSigmoid(float x);
float actSigmoidPd(float x);

float actTanh(float x);
float actTanhPd(float x);

float actRelu(float x);
float actReluPd(float x);

//float actSoftmax(float x);
//float actSoftmaxPd(float x);


float calcErrSqDiff(Matrix predicted, Matrix expected);
Matrix calcErrSqDiffPd(Matrix predicted, Matrix expected);
