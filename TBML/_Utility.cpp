#include "stdafx.h"
#include "_Utility.h"

float tbml::fn::_classificationAccuracy(const tbml::_Tensor& predicted, const tbml::_Tensor& expected)
{
	assert(predicted.getShape() == expected.getShape());
	assert(predicted.getDims() == 2);
	size_t rows = predicted.getShape(0);
	size_t cols = predicted.getShape(1);

	float accuracy = 0.0f;
	for (size_t row = 0; row < rows; row++)
	{
		int predictedClass = 0;
		float predictedValue = predicted(row, 0);
		for (size_t i = 1; i < cols; i++)
		{
			float v = predicted(row, i);
			if (v > predictedValue)
			{
				predictedClass = (int)i;
				predictedValue = v;
			}
		}

		int expectedClass = 0;
		float expectedValue = expected(row, 0);
		for (size_t i = 1; i < cols; i++)
		{
			float v = expected(row, i);
			if (v > expectedValue)
			{
				expectedClass = (int)i;
				expectedValue = v;
			}
		}

		accuracy += ((predictedClass == expectedClass) ? 1.0f : 0.0f) / rows;
	}

	return accuracy;
}
