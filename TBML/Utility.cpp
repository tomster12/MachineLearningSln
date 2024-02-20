#include "stdafx.h"
#include "Utility.h"

float tbml::fn::getRandomFloat()
{
	return (float)rand() / (float)RAND_MAX;
}

float tbml::fn::classificationAccuracy(const tbml::Tensor& output, const tbml::Tensor& expected)
{
	assert(output.getShape() == expected.getShape());
	assert(output.getDims() == 2);
	size_t rows = output.getShape(0);
	size_t cols = output.getShape(1);

	float accuracy = 0.0f;
	for (size_t row = 0; row < rows; row++)
	{
		int predictedClass = 0;
		float predictedValue = output(row, 0);
		for (size_t i = 1; i < cols; i++)
		{
			float v = output(row, i);
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
