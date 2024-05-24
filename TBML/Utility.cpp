#include "stdafx.h"
#include "Utility.h"

float tbml::fn::getRandomFloat()
{
	return (float)rand() / (float)RAND_MAX;
}

size_t tbml::fn::argmax(const tbml::Tensor& tensor, float row)
{
	assert(tensor.getDims() == 2);
	assert(row < tensor.getShape(0));

	size_t maxIndex = 0;
	float maxValue = tensor(row, 0);
	for (size_t i = 1; i < tensor.getShape(1); i++)
	{
		float v = tensor(row, i);
		if (v > maxValue)
		{
			maxIndex = i;
			maxValue = v;
		}
	}

	return maxIndex;
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
		size_t predictedClass = argmax(output, row);
		size_t expectedClass = argmax(expected, row);
		accuracy += ((predictedClass == expectedClass) ? 1.0f : 0.0f) / rows;
	}

	return accuracy;
}

std::shared_ptr<tbml::fn::LossFunction> tbml::fn::LossFunction::deserialize(std::istream& is)
{
	std::string type;
	is >> type;

	if (type == "SquareError") return std::make_shared<SquareError>();
	if (type == "CrossEntropy") return std::make_shared<CrossEntropy>();

	assert(false);
	return nullptr;
};
