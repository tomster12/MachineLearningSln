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

std::shared_ptr<tbml::fn::ActivationFunction> tbml::fn::ActivationFunction::deserialize(std::istream& is)
{
	std::string type;
	is >> type;

	if (type == "ReLU") return std::make_shared<ReLU>();
	if (type == "Sigmoid") return std::make_shared<Sigmoid>();
	if (type == "TanH") return std::make_shared<TanH>();
	if (type == "SoftMax") return std::make_shared<SoftMax>();

	assert(false);
	return nullptr;
};

std::shared_ptr<tbml::fn::LossFunction> tbml::fn::LossFunction::deserialize(std::istream& is)
{
	std::string type;
	is >> type;

	if (type == "SquareError") return std::make_shared<SquareError>();
	if (type == "CrossEntropy") return std::make_shared<CrossEntropy>();

	assert(false);
	return nullptr;
};
